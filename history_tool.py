from __future__ import annotations

import asyncio
import re
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx

from core.chat.message_utils import KiraMessageBatchEvent

from . import locate
from .onebot_compat import build_payload, resolve_impl


# Placeholder raw_message produced by some OneBot implementations (e.g.
# SnowLuma) when the reply segment conversion fails - the message content
# is actually empty and must be rebuilt from segments or get_msg.
# NOTE: raw_message is a CQ-coded string, so literal "[", "]", ",", "&" in
# text arrive escaped as "&#91;", "&#93;", "&#44;", "&amp;" (SnowLuma
# helper/cq.ts cqEscape). Every placeholder comparison must unescape first,
# otherwise SnowLuma's "[引用消息]" placeholder arrives as "&#91;引用消息&#93;"
# and slips through the filter.
#
# Only the synthetic reply-target placeholder and the empty-message marker are
# real placeholders. "[引用]" and "[转发消息]" are the *renderings of real
# segments* (a reply without an id / a forward card) - filtering them would
# hide the very messages (and their message_id) the bot needs to re-forward.
_PLACEHOLDER_TOKENS = {"[引用消息]", "[空消息]"}
_PLACEHOLDER_RAW = _PLACEHOLDER_TOKENS | {""}
_CQ_ENTITIES = (("&#91;", "["), ("&#93;", "]"), ("&#44;", ","), ("&amp;", "&"))


def cq_unescape(text: str) -> str:
    """Decode OneBot CQ entities; "&amp;" must be last (see SnowLuma cq.ts)."""
    if not text:
        return text
    for entity, char in _CQ_ENTITIES:
        text = text.replace(entity, char)
    return text
# Segment types whose source (url/file) may be missing in stored history
# and needs a get_msg refresh (SnowLuma refreshes image URLs on get_msg).
_MEDIA_TYPES = {"image", "record", "video"}
# Max messages to refresh per call (get_msg is one round-trip each).
_MAX_REFRESH = 10


class HistoryToolService:
    """
    跨会话历史查询（对齐新版 history_plugin v1.3.2 强解析能力）。

    - WS 通道优先（复用适配器连接，与转发/撤回同一 ID 命名空间，
      SnowLuma 下 get_msg 可反查），HTTP 通道兜底。
    - 强解析：raw_message 为占位（如 SnowLuma 的 [引用消息]）时改用
      message 段数组；reply 段显示 [引用 msg_id:xxx]；媒体缺源标记待刷新。
    - get_msg 批量刷新（最多 10 条/次）恢复媒体 URL。
    - 空引用占位消息渲染后判定过滤，不污染 LLM 上下文。
    - 保留 KSM 特有：全局熔断、同回合调用限制、缓存、权限控制、截断。
    任何失败只 return str，绝不抛异常。
    """

    # 本地 OneBot 拉取历史消息（尤其群聊大 count）可能耗时数秒，
    # 参考可用的 history_plugin 显式 timeout=10，这里按阶段拆分并留足余量。
    CONNECT_TIMEOUT = 3.0
    READ_TIMEOUT = 15.0
    ERROR_CACHE_TTL = 90.0
    # 同一 agent 回合内：同一目标会话最多成功返回几次（再调用直接拒绝，不塞大段历史）
    MAX_CALLS_PER_TARGET_PER_EVENT = 2
    # 同一 agent 回合内：历史工具总调用上限（含被拒绝的）
    MAX_CALLS_PER_EVENT = 3
    # 单次返回正文最大字符，避免 tool_result 把上下文撑爆
    MAX_RESULT_CHARS = 3500
    # 单次调用内最多翻多少页（硬兜底，防止实现异常导致死循环）
    MAX_SCAN_STEPS = 80
    # 「涉及」行最多列出几个不同的人（每条约 15-25 字符，与正文共享 3500 字预算）
    MAX_PEOPLE_SHOWN = 5
    # 旧路径的超取下限：至少请求这么多条，避免占位行挤掉真实消息
    FETCH_CEILING = 80

    def __init__(
        self,
        http_host: str = "localhost",
        http_port: int = 3000,
        access_token: str = "",
        master_id: str = "",
        allowed_users: Optional[List[str]] = None,
        restricted_groups: Optional[List[str]] = None,
        cache_ttl_sec: int = 120,
        circuit_fail_threshold: int = 2,
        circuit_open_sec: float = 60.0,
        use_ws: bool = True,
        ctx=None,
        logger=None,
        locate_cfg: Optional[Dict[str, Any]] = None,
    ):
        self.http_host = http_host or "localhost"
        self.http_port = int(http_port or 3000)
        self.base_url = f"http://{self.http_host}:{self.http_port}"
        self.access_token = access_token or ""
        self.master_id = str(master_id or "").strip()
        self.allowed_users = [str(u).strip() for u in (allowed_users or []) if str(u).strip()]
        self.restricted_groups = [str(g).strip() for g in (restricted_groups or []) if str(g).strip()]
        self.cache_ttl_sec = max(0, int(cache_ttl_sec or 0))
        self.circuit_fail_threshold = max(1, int(circuit_fail_threshold or 2))
        self.circuit_open_sec = max(0.0, float(circuit_open_sec or 60.0))
        self.use_ws = bool(use_ws)
        # Plugin context: needed by _get_client to resolve the adapter's WS
        # client (same ID namespace as the adapter, so message IDs work).
        self.ctx = ctx
        self.logger = logger
        self._call_cache: Dict[str, Dict[str, Any]] = {}
        self._fail_streak = 0
        self._circuit_open_until = 0.0

        # ---------- locate (time / user / keyword) ----------
        locate_cfg = locate_cfg or {}
        self.enable_locate = bool(locate_cfg.get("enable_locate", True))
        self.enable_keyword = bool(locate_cfg.get("enable_keyword", True))
        self.enable_time_range = bool(locate_cfg.get("enable_time_range", True))
        self.enable_user_filter = bool(locate_cfg.get("enable_user_filter", True))
        self.default_scan_limit = max(1, int(locate_cfg.get("default_scan_limit", 300) or 300))
        self.max_scan_limit = max(0, int(locate_cfg.get("max_scan_limit", 0) or 0))
        self.scan_max_seconds = max(1.0, float(locate_cfg.get("scan_max_seconds", 25) or 25))
        self.max_fetch_per_request = max(1, int(locate_cfg.get("max_fetch_per_request", 50) or 50))
        self.fetch_timeout_sec = max(1.0, float(locate_cfg.get("fetch_timeout_sec", 15) or 15))
        self.max_scanned_per_turn = max(0, int(locate_cfg.get("max_scanned_per_turn", 3000) or 3000))
        # 调用次数预算（可配置；类常量仅作为默认值）
        self.max_calls_per_turn = max(
            1, int(locate_cfg.get("max_calls_per_turn", self.MAX_CALLS_PER_EVENT)
                   or self.MAX_CALLS_PER_EVENT))
        self.max_calls_per_target_per_turn = max(
            1, int(locate_cfg.get("max_calls_per_target_per_turn",
                                  self.MAX_CALLS_PER_TARGET_PER_EVENT)
                   or self.MAX_CALLS_PER_TARGET_PER_EVENT))
        self.early_stop_on_enough = bool(locate_cfg.get("early_stop_on_enough", True))
        self.detect_boundary = bool(locate_cfg.get("detect_boundary", True))
        self.keyword_case_sensitive = bool(locate_cfg.get("keyword_case_sensitive", False))
        self.max_keywords = max(1, int(locate_cfg.get("max_keywords", 5) or 5))
        self.offset_max = max(0, int(locate_cfg.get("offset_max", 1000) or 1000))
        self.max_return_count = max(1, int(locate_cfg.get("max_return_count", 80) or 80))
        self.locate_fallback_on_error = bool(locate_cfg.get("locate_fallback_on_error", True))
        self.locate_head_meta = bool(locate_cfg.get("locate_head_meta", True))
        self.locate_cache_ttl_sec = max(0, int(locate_cfg.get("locate_cache_ttl_sec", 120) or 120))
        self._locate_cache: Dict[str, Dict[str, Any]] = {}
        self._impl_cache: Dict[str, Dict[str, Any]] = {}

    def _check_permission(self, user_id: str, session_type: str, session_id: str) -> bool:
        if not self.master_id:
            return True
        if user_id == self.master_id:
            return True
        if user_id in self.allowed_users:
            if session_type == "gm" and session_id in self.restricted_groups:
                return False
            return True
        if session_type == "dm":
            return session_id == user_id
        if session_type == "gm":
            return session_id not in self.restricted_groups
        return False

    # Session-type tokens that may appear in the middle position. A 2-part ref
    # like "gm:123" is ambiguous (adapter:entity? or type:entity?), so we
    # resolve it by checking this set rather than assuming it is adapter:id.
    _TYPE_TOKENS = {
        "gm": "gm", "group": "gm", "g": "gm",
        "dm": "dm", "private": "dm", "p": "dm", "friend": "dm",
    }

    @classmethod
    def parse_session_ref(cls, session_id: str, session_type: Optional[str] = None) -> Dict[str, str]:
        sid = (session_id or "").strip()
        st = (session_type or "").strip().lower()

        if ":" in sid:
            parts = sid.split(":")
            if len(parts) >= 3:
                adapter = parts[0] or "qq"
                typ = parts[1] or "dm"
                entity = ":".join(parts[2:])
            elif len(parts) == 2:
                head, tail = parts[0].strip(), parts[1].strip()
                if head.lower() in cls._TYPE_TOKENS:
                    # "gm:123" -> type + entity (no adapter segment)
                    adapter, typ, entity = "qq", head.lower(), tail
                else:
                    # "qq:123" -> adapter + entity, type unknown
                    adapter, typ, entity = head or "qq", "dm", tail
            else:  # len == 1 (a bare leading ':')
                adapter, typ, entity = "qq", "dm", sid
            if typ in ("group", "g"):
                typ = "gm"
            if typ in ("private", "p", "friend"):
                typ = "dm"
            return {"adapter": adapter, "session_type": typ,
                    "session_id": entity, "full": sid}

        if st in ("group", "gm", "g"):
            typ = "gm"
        elif st in ("private", "dm", "p", "friend"):
            typ = "dm"
        else:
            typ = "dm"
        return {"adapter": "qq", "session_type": typ, "session_id": sid, "full": f"qq:{typ}:{sid}"}

    # ---------- 强解析（对齐 history_plugin v1.3.2） ----------

    # ---------- locate: 实现探测 / 分页 ----------

    async def _resolve_impl(self, client, adapter_name: str) -> Dict[str, Any]:
        """Probe the OneBot implementation once per adapter (see onebot_compat).

        NapCat / LLOneBot / SnowLuma disagree on the anchor parameter, so the
        scanner needs to know which one it is talking to. Failure falls back to
        the generic knob set (no anchor) - degraded, never broken.
        """
        cached = self._impl_cache.get(adapter_name)
        if cached is not None:
            return cached
        impl = resolve_impl("")
        if client is not None:
            try:
                resp = await client.send_action("get_version_info", {}, timeout=8)
                data = (resp or {}).get("data") or {}
                app_name = str(data.get("app_name") or "")
                if app_name:
                    impl = resolve_impl(app_name)
                    if self.logger:
                        self.logger.info(
                            "[history_tool] OneBot impl=%s anchor=%s",
                            app_name, impl.get("anchor_param"))
            except Exception as e:
                if self.logger:
                    self.logger.warning(
                        "[history_tool] get_version_info failed, generic knobs: %s", e)
        if self.max_scan_limit > 0:
            impl["max_scan_limit"] = self.max_scan_limit
        impl["max_page"] = max(1, min(int(impl.get("max_page") or 30),
                                      self.max_fetch_per_request))
        self._impl_cache[adapter_name] = impl
        return impl

    async def _fetch_page_ws(self, client, session_type: str, session_id: str,
                             impl: Dict[str, Any], anchor, count: int):
        """One page via the adapter WS channel (oldest->newest)."""
        if session_type == "gm":
            action, base = "get_group_msg_history", {"group_id": str(session_id)}
        else:
            action, base = "get_friend_msg_history", {"user_id": str(session_id)}
        payload = build_payload(impl, base, anchor, count)
        try:
            resp = await client.send_action(action, payload, timeout=self.fetch_timeout_sec)
        except Exception as e:
            if self.logger:
                self.logger.error(f"[history_tool] WS page failed: {e}")
            return None
        if not isinstance(resp, dict) or resp.get("status") != "ok":
            return None
        messages = (resp.get("data") or {}).get("messages") or []
        return messages if isinstance(messages, list) else None

    async def _fetch_page_http(self, session_type: str, session_id: str,
                               impl: Dict[str, Any], anchor, count: int):
        """One page via the HTTP service, same payload shape as the WS path."""
        if session_type == "gm":
            api, base = "get_group_msg_history", {"group_id": str(session_id)}
        else:
            api, base = "get_friend_msg_history", {"user_id": str(session_id)}
        payload = build_payload(impl, base, anchor, count)
        headers = {}
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        timeout = httpx.Timeout(
            connect=self.CONNECT_TIMEOUT,
            read=self.READ_TIMEOUT,
            write=self.READ_TIMEOUT,
            pool=self.CONNECT_TIMEOUT,
        )
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(f"{self.base_url}/{api}", json=payload,
                                         headers=headers)
                if resp.status_code >= 400:
                    return None
                result = resp.json()
        except Exception as e:
            if self.logger:
                self.logger.error(f"[history_tool] HTTP page failed: {e}")
            return None
        if not isinstance(result, dict) or result.get("status") != "ok":
            return None
        messages = (result.get("data") or {}).get("messages") or []
        return messages if isinstance(messages, list) else None

    def _make_page_fetcher(self, client, session_type: str, session_id: str,
                           impl: Dict[str, Any]):
        """fetch_page(anchor, count) for the scanner.

        A WS failure on the FIRST page falls back to HTTP for the whole walk;
        a failure later on is raised so the caller can retry from the newest
        page (a mid-walk transport switch would repeat everything anyway).
        """
        state = {"fell_back": False}

        async def fetch_page(anchor, count):
            if client is not None and not state["fell_back"]:
                page = await self._fetch_page_ws(client, session_type, session_id,
                                                 impl, anchor, count)
                if page is not None:
                    return page
                if anchor is None:
                    state["fell_back"] = True
                    if self.logger:
                        self.logger.warning(
                            "[history_tool] WS page failed, falling back to HTTP")
                else:
                    raise RuntimeError("WS 翻页失败")
            page = await self._fetch_page_http(session_type, session_id, impl,
                                               anchor, count)
            if page is None:
                raise RuntimeError("HTTP 翻页失败")
            return page

        return fetch_page

    @staticmethod
    def _segments_to_text(msg_segments) -> str:
        """Render message segments to text, keeping media URLs and reply IDs."""
        parts = []
        for seg in msg_segments:
            seg_type = seg.get("type")
            seg_data = seg.get("data", {})
            if seg_type == "text":
                parts.append(seg_data.get("text", ""))
            elif seg_type == "at":
                parts.append(f"@{seg_data.get('qq', 'someone')}")
            elif seg_type == "face":
                parts.append("[表情]")
            elif seg_type == "image":
                img_url = seg_data.get("url", "")
                if img_url:
                    parts.append(f"[图片]({img_url})")
                else:
                    parts.append("[图片]")
            elif seg_type == "video":
                parts.append("[视频]")
            elif seg_type == "file":
                file_name = seg_data.get("name", "文件")
                parts.append(f"[文件]{file_name}")
            elif seg_type == "reply":
                rid = seg_data.get("id", "")
                parts.append(f"[引用 msg_id:{rid}]" if rid else "[引用]")
            elif seg_type == "forward":
                # Keep the forward's resource id: the outer message_id lets the
                # bot re-forward the card, the res id lets it inspect the
                # nested content.
                fid = seg_data.get("id", "")
                parts.append(f"[转发消息](id={fid})" if fid else "[转发消息]")
            else:
                parts.append(f"[{seg_type}]")
        return " ".join(parts)

    def _message_to_text(self, msg: dict) -> str:
        """Convert a message to formatted text. Uses raw_message only when it
        is real content; placeholder raw_message (e.g. SnowLuma's
        "[引用消息]") falls back to the segment array."""
        raw = cq_unescape((msg.get("raw_message") or "").strip())
        if raw and raw not in _PLACEHOLDER_RAW:
            content = raw
        else:
            msg_segments = msg.get("message", [])
            if not msg_segments:
                content = "[空消息]"
            else:
                content = self._segments_to_text(msg_segments)

        msg_id = msg.get("message_id")
        if msg_id:
            content += f" (msg_id:{msg_id})"
        return content

    def _is_placeholder(self, msg: dict) -> bool:
        """True only for the synthetic empty-quote rows SnowLuma stores for an
        unresolvable reply target (and genuinely empty messages).

        A real message - including a forward card or a bare reply - is kept:
        the bot needs its message_id to re-forward it.
        """
        raw = cq_unescape((msg.get("raw_message") or "").strip())
        segs = msg.get("message") or []
        # Genuinely empty (no raw text and no segments).
        if not raw and not segs:
            return True
        # SnowLuma's synthetic reply-target backfill (buildBackfillEvent) is a
        # single "[引用消息]" text whose sender identity is EMPTY: nickname and
        # card are always "", while user_id is the QUOTED message's sender uin
        # - frequently a real, non-zero uin - so an uid==0 test alone misses
        # most of them. A real user message keeps a nickname/card and is shown.
        sender = msg.get("sender") if isinstance(msg.get("sender"), dict) else {}
        uid = str(msg.get("user_id") or sender.get("user_id") or "").strip()
        nick = str(sender.get("nickname") or "").strip()
        card = str(sender.get("card") or "").strip()
        seg_text = self._segments_to_text(segs).strip() if segs else ""
        is_token = raw in _PLACEHOLDER_TOKENS or seg_text in _PLACEHOLDER_TOKENS
        if is_token and (uid in ("", "0") or (not nick and not card)):
            return True
        # Placeholder raw marker with no segments at all.
        if raw in _PLACEHOLDER_TOKENS and not segs:
            return True
        # Renders to nothing after stripping the trailing (msg_id:xxx).
        content = re.sub(r"\s*\(msg_id:-?\d+\)\s*$", "",
                         self._message_to_text(msg)).strip()
        return not content

    def _needs_refresh(self, msg: dict) -> bool:
        """True when the message needs a get_msg refresh: placeholder
        raw_message, or media segments without a usable source."""
        raw = cq_unescape((msg.get("raw_message") or "").strip())
        if raw in _PLACEHOLDER_RAW:
            return True
        for seg in msg.get("message") or []:
            if seg.get("type") in _MEDIA_TYPES:
                data = seg.get("data") or {}
                if not (data.get("url") or data.get("file") or data.get("file_id")):
                    return True
        return False

    # ---------- 通道 ----------

    def _get_client(self, event):
        """Get the adapter WS client from the event (same ID namespace as
        the adapter itself, so message IDs are usable by get_msg / forward)."""
        try:
            info = getattr(event, "adapter", None)
            if info is None:
                return None
            name = getattr(info, "name", None) or getattr(info, "adapter_id", None)
            if not name:
                return None
            adapter = self.ctx.adapter_mgr.get_adapter(name)
            if adapter is None:
                return None
            return adapter.get_client()
        except Exception as e:
            if self.logger:
                self.logger.error(f"[history_tool] get client failed: {e}")
            return None

    async def _fetch_ws(self, client, session_type: str, session_id: str, count: int):
        """Fetch history via the WS channel (adapter's own OneBot connection)."""
        try:
            if session_type == "gm":
                resp = await client.send_action(
                    "get_group_msg_history",
                    {"group_id": int(session_id), "count": count},
                    timeout=15,
                )
            else:
                resp = await client.send_action(
                    "get_friend_msg_history",
                    {"user_id": int(session_id), "count": count},
                    timeout=15,
                )
            if isinstance(resp, dict) and resp.get("status") == "ok":
                return resp.get("data", {}).get("messages") or []
        except Exception as e:
            if self.logger:
                self.logger.error(f"[history_tool] WS history failed: {e}")
        return None

    async def _fetch_http(self, session_type: str, session_id: str, count: int):
        """Fetch history via the HTTP service (legacy channel)."""
        try:
            if session_type == "gm":
                api = "get_group_msg_history"
                params = {"group_id": int(session_id), "count": count}
            else:
                api = "get_friend_msg_history"
                params = {"user_id": int(session_id), "count": count}

            headers = {}
            if self.access_token:
                headers["Authorization"] = f"Bearer {self.access_token}"

            timeout = httpx.Timeout(
                connect=self.CONNECT_TIMEOUT,
                read=self.READ_TIMEOUT,
                write=self.READ_TIMEOUT,
                pool=self.CONNECT_TIMEOUT,
            )

            url = f"{self.base_url}/{api}"
            if self.logger:
                self.logger.info("[history_tool] fetching %s params=%s", url, params)

            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, json=params, headers=headers)
                if resp.status_code >= 400:
                    err = (
                        f"Error: HTTP {resp.status_code} from {url}/{api}. "
                        "OneBot HTTP 不可用。请勿再次调用 get_session_history，"
                        "请基于当前对话上下文回答。"
                    )
                    if self.logger:
                        self.logger.error("Error fetching history: HTTP %s body=%s", resp.status_code, resp.text[:200])
                    return None, err
                try:
                    result = resp.json()
                except Exception as e:
                    err = f"Error: invalid JSON from OneBot ({e}) body={resp.text[:200]}"
                    return None, err

            if result.get("status") != "ok":
                err = f"Failed: {result.get('message', 'unknown error')}"
                return None, err
            return result.get("data", {}).get("messages", []), None
        except Exception as e:
            err = f"{type(e).__name__}: {str(e) or '(no message)'}"
            return None, err

    async def _get_msg_ws(self, client, message_id) -> dict | None:
        """Fetch a single message via get_msg (SnowLuma refreshes image URLs
        on get_msg, so this recovers media sources missing from history)."""
        try:
            resp = await client.send_action(
                "get_msg", {"message_id": message_id}, timeout=15
            )
            if isinstance(resp, dict) and resp.get("status") == "ok":
                return resp.get("data") or {}
        except Exception as e:
            if self.logger:
                self.logger.error(f"[history_tool] get_msg({message_id}) failed: {e}")
        return None

    # ---------- 缓存 / 熔断 / 限制（KSM 保留） ----------

    def _cache_put(self, key: str, count: int, data: str, is_error: bool = False):
        self._call_cache[key] = {
            "count": count,
            "data": data,
            "timestamp": time.time(),
            "is_error": is_error,
        }
        if len(self._call_cache) > 100:
            now = time.time()
            for k in [k for k, v in self._call_cache.items() if now - v.get("timestamp", 0) > 300]:
                del self._call_cache[k]

    def _cache_get_hit(self, key: str, count: int) -> Optional[str]:
        cached = self._call_cache.get(key)
        if not cached:
            return None
        now = time.time()
        is_error = bool(cached.get("is_error"))
        ttl = self.ERROR_CACHE_TTL if is_error else self.cache_ttl_sec
        if ttl <= 0:
            return None
        if (now - cached.get("timestamp", 0)) >= ttl:
            return None
        if (not is_error) and count > cached.get("count", 0):
            return None
        data = cached["data"]
        if is_error:
            return data
        # 缓存命中：只回短拒，不再把整段历史塞进 tool_result（否则每步 +数千 token）
        return (
            "Rejected: 该会话历史本回合已查询过（结果在上文 tool 记录中）。"
            "请直接基于当前对话上下文回复，禁止再次调用 get_session_history。"
        )

    @staticmethod
    def _event_extra(event):
        """Return the per-event scratch dict, or None when the event refuses it.

        An EMPTY dict is a normal state and must be returned as-is: several
        callers create keys on it. Returning {} here used to make the scan
        budget unchargeable.
        """
        try:
            extra = getattr(event, "extra", None)
            if isinstance(extra, dict):
                return extra
            extra = {}
            try:
                event.extra = extra
            except Exception:
                return None
            return extra
        except Exception:
            return None

    def _track_and_limit(self, event, target_key: str) -> Optional[str]:
        """
        同一 KiraMessageBatchEvent（一次 agent 回合）内限制历史工具调用。
        返回非 None 则应直接 return 该字符串，不再打 HTTP。
        """
        extra = self._event_extra(event)
        if extra is None:
            return None
        total = int(extra.get("merger_hist_total", 0) or 0)
        by_target = extra.get("merger_hist_by_target")
        if not isinstance(by_target, dict):
            by_target = {}
            extra["merger_hist_by_target"] = by_target

        if total >= self.max_calls_per_turn:
            return (
                "Rejected: 本回合 get_session_history 调用次数已达上限。"
                "请直接回复，禁止再查历史。"
            )
        n = int(by_target.get(target_key, 0) or 0)
        if n >= self.max_calls_per_target_per_turn:
            return (
                f"Rejected: 本回合已查询过 {target_key} 的历史。"
                "请直接基于上下文回复，禁止再次 get_session_history。"
            )

        by_target[target_key] = n + 1
        extra["merger_hist_total"] = total + 1
        return None

    def _truncate_result(self, text: str) -> str:
        if not text or len(text) <= self.MAX_RESULT_CHARS:
            return text
        # 保留末尾（更新）
        cut = text[-self.MAX_RESULT_CHARS :]
        return "…(truncated older)…\n" + cut

    def _note_failure(self):
        """记录失败并进入熔断。

        窗口期内再次失败时不再把窗口续到未来（否则会像「永远打不开」）。

        注意 `_circuit_open_until == 0` 表示「熔断从未打开」：此时
        `now >= 0` 恒为真，若照旧重置计数，`_fail_streak` 会永远停在 1，
        永远达不到阈值 —— 熔断实际从未生效。只有在窗口**真的开过又结束**
        （`0 < open_until <= now`）时才重置计数。
        """
        now = time.time()
        if 0 < self._circuit_open_until <= now:
            # 窗口刚刚结束：这是一段新的失败序列，从零开始计
            self._fail_streak = 0
            self._circuit_open_until = 0.0
        self._fail_streak += 1
        if self._fail_streak >= self.circuit_fail_threshold:
            if now >= self._circuit_open_until:
                self._circuit_open_until = now + self.circuit_open_sec
                if self.logger:
                    self.logger.warning(
                        "history circuit OPEN for %.0fs after %d failures",
                        self.circuit_open_sec,
                        self._fail_streak,
                    )

    def _note_success(self):
        self._fail_streak = 0
        self._circuit_open_until = 0.0

    def _circuit_blocked(self) -> Optional[str]:
        now = time.time()
        if now < self._circuit_open_until:
            left = int(self._circuit_open_until - now)
            return (
                f"Error: OneBot HTTP circuit open ({left}s left). "
                "请勿再次调用 get_session_history，请基于当前对话上下文回答。"
            )
        return None

    async def get_session_history(
        self,
        event: KiraMessageBatchEvent,
        session_id: str,
        count: int = 20,
        session_type: Optional[str] = None,
        *,
        merge_enabled: bool = False,
        since: Optional[str] = None,
        until: Optional[str] = None,
        user_id: Optional[str] = None,
        keyword: Optional[str] = None,
        offset: int = 0,
        scan_limit: Optional[int] = None,
    ) -> str:
        try:
            blocked = self._circuit_blocked()
            if blocked:
                return blocked

            caller_id = "unknown"
            if event.messages and event.messages[0].sender:
                caller_id = str(event.messages[0].sender.user_id)

            ref = self.parse_session_ref(session_id, session_type)
            st = ref["session_type"]
            entity = ref["session_id"]

            if not self._check_permission(caller_id, st, entity):
                return "抱歉，您没有权限查看此会话的历史消息。"

            # Models routinely send numbers as strings ("20") or as floats;
            # coerce before comparing, otherwise `count < 5` raises TypeError
            # and the tool call dies instead of returning anything.
            try:
                count = int(float(count))
            except (TypeError, ValueError):
                count = 20
            # 与 history_plugin 对齐：最少 5，最多 max_return_count
            if count < 5:
                count = 5
            elif count > self.max_return_count:
                count = self.max_return_count

            cache_key = f"{st}:{entity}"
            target_key = cache_key

            wants_locate = self.enable_locate and any(
                arg not in (None, "", 0)
                for arg in (since, until, user_id, keyword, offset, scan_limit)
            )
            if wants_locate:
                return await self._locate_session_history(
                    event, st, entity, count, since, until, user_id, keyword,
                    offset, scan_limit, target_key)

            # 本回合调用次数硬限制（在缓存命中之前也计数，防止刷拒绝）
            limited = self._track_and_limit(event, target_key)
            if limited:
                return limited

            hit = self._cache_get_hit(cache_key, count)
            if hit is not None:
                return hit

            # ---------- 拉取历史：WS 通道优先，HTTP 兜底 ----------
            # Over-fetch so placeholder rows (which sit at the newest end)
            # cannot crowd real messages out of the window. The ceiling follows
            # `count` but never below FETCH_CEILING, so raising `count` (up to
            # max_return_count) actually fetches enough instead of silently
            # capping at 80.
            fetch_count = min(max(count, self.FETCH_CEILING),
                              max(count, count * 3))
            messages = None
            err = None
            client = None
            if self.use_ws:
                client = self._get_client(event)
                if client is not None:
                    messages = await self._fetch_ws(client, st, entity, fetch_count)
                    if messages is None:
                        if self.logger:
                            self.logger.warning(
                                "[history_tool] WS fetch failed for %s; falling back to HTTP",
                                cache_key,
                            )
            if messages is None:
                messages, err = await self._fetch_http(st, entity, fetch_count)

            if err is not None:
                self._cache_put(cache_key, 80, err, is_error=True)
                self._note_failure()
                return err

            if not messages:
                empty = "该会话暂无历史消息。"
                self._cache_put(cache_key, count, empty, is_error=False)
                self._note_success()
                return empty

            # ---------- get_msg 批量刷新（最多 10 条/次，并行） ----------
            if client is not None:
                target = messages[-fetch_count:]
                # 先收集待刷新 (下标, message_id)，再 gather 并行：
                # 串行最坏 _MAX_REFRESH × 单次超时（10×15s），并行后 ≈ 单次超时
                jobs = []
                for i, m in enumerate(target):
                    if len(jobs) >= _MAX_REFRESH:
                        break
                    if self._needs_refresh(m):
                        mid = m.get("message_id")
                        if mid is not None:
                            jobs.append((i, mid))
                refreshed = 0
                if jobs:
                    results = await asyncio.gather(
                        *(self._get_msg_ws(client, mid) for _, mid in jobs),
                        return_exceptions=True,
                    )
                    # 按下标映射回原位置，保持消息顺序不变
                    for (i, _mid), fresh in zip(jobs, results):
                        if isinstance(fresh, dict) and fresh.get("message"):
                            target[i] = fresh
                            refreshed += 1
                if refreshed and self.logger:
                    self.logger.info(
                        "[history_tool] refreshed %d messages via get_msg", refreshed
                    )

            # ---------- 格式化 + 空引用占位过滤 ----------
            # NOTE: iterate `target` (the refreshed slice) - the old code
            # formatted `messages[-count:]` again, so every get_msg refresh
            # was silently discarded.
            formatted = []
            skipped = 0
            real = []
            for msg in (target if client is not None else messages[-fetch_count:]):
                if self._is_placeholder(msg):
                    skipped += 1
                    continue
                real.append(msg)
            for msg in real[-count:]:
                formatted.append(self._format_line(msg))

            if skipped and self.logger:
                self.logger.info(
                    "[history_tool] filtered %d unresolvable placeholder messages",
                    skipped,
                )

            if not formatted:
                empty = "该会话暂无有效历史消息。"
                self._cache_put(cache_key, count, empty, is_error=False)
                self._note_success()
                return empty

            result_text = self._truncate_result("\n".join(formatted))
            self._cache_put(cache_key, count, result_text, is_error=False)
            self._note_success()
            return result_text

        except Exception as e:
            tb = traceback.format_exc()
            err_msg = f"{type(e).__name__}: {str(e) or '(no message)'}"
            if self.logger:
                self.logger.error("Error fetching history: %s\n%s", err_msg, tb)
            err = (
                f"Error: {err_msg}. "
                "请勿再次调用 get_session_history，请基于当前对话上下文回答。"
            )
            try:
                ref = self.parse_session_ref(session_id, session_type)
                self._cache_put(
                    f"{ref['session_type']}:{ref['session_id']}",
                    80,
                    err,
                    is_error=True,
                )
            except Exception:
                pass
            self._note_failure()
            return err

    # ---------- locate path: scan backwards + filter ----------

    def _format_line(self, msg: dict) -> str:
        """`昵称(QQ): 内容`.

        The QQ number is not decoration: nickname and group card both change,
        and in a group the two can differ from each other. Without a stable id
        the model cannot tell that two names refer to the same person.
        """
        label = locate.person_of(msg).sender_label()
        content = self._message_to_text(msg)
        return f"{label}: {content}"

    def _scan_limit_cap(self) -> int:
        caps = [impl.get("max_scan_limit") for impl in self._impl_cache.values()]
        caps = [int(c) for c in caps if c]
        cap = min(caps) if caps else (self.max_scan_limit or 600)
        if self.max_scan_limit:
            cap = min(cap, self.max_scan_limit)
        return max(1, cap)

    def _build_locale_query(self, since, until, user_id, keyword, offset, scan_limit):
        """Parse raw tool args into (query, notes, error)."""
        now = datetime.now()
        query = locate.LocateQuery()
        notes: List[str] = []

        if self.enable_time_range:
            ts, err = locate.parse_time_arg(since, now)
            if err:
                return None, notes, err
            query.since = ts
            ts, err = locate.parse_time_arg(until, now)
            if err:
                return None, notes, err
            query.until = ts
        elif since or until:
            notes.append("时间过滤已在配置中关闭（enable_time_range）")

        if self.enable_user_filter and user_id:
            query.user_ids, query.user_names = locate.normalize_user(user_id)
        elif user_id:
            notes.append("用户过滤已在配置中关闭（enable_user_filter）")

        if self.enable_keyword and keyword:
            words, warn = locate.normalize_keywords(keyword, self.max_keywords)
            query.keywords = words
            if warn:
                notes.append(warn)
        elif keyword:
            notes.append("关键词过滤已在配置中关闭（enable_keyword）")

        if not (query.since or query.until or query.user_ids or query.user_names
                or query.keywords):
            notes.append("定位参数均无效，本次按最近消息返回")

        try:
            query.offset = max(0, int(offset or 0))
        except (TypeError, ValueError):
            query.offset = 0
        if self.offset_max and query.offset > self.offset_max:
            notes.append(f"offset 超过上限 {self.offset_max}，已截断")
            query.offset = self.offset_max

        cap = self._scan_limit_cap()
        try:
            requested = int(scan_limit) if scan_limit else self.default_scan_limit
        except (TypeError, ValueError):
            requested = self.default_scan_limit
        # 下限：模型笔误（0 / -5）不应把扫描缩到 1 条然后报「没找到」。
        floor = min(cap, max(20, self.max_fetch_per_request))
        query.scan_limit = max(floor, min(requested, cap))
        # 两个方向都要说明：静默抬高会让模型以为请求值生效了。
        if requested > cap:
            notes.append(f"scan_limit 超过上限 {cap}，已截断为 {query.scan_limit}")
        elif requested < query.scan_limit:
            notes.append(f"scan_limit 低于下限 {query.scan_limit}，已提升到该值")

        query.keyword_case_sensitive = self.keyword_case_sensitive
        return query, notes, None

    def _locate_conditions(self, session_key: str, query):
        return (session_key, query.since, query.until, tuple(query.user_ids),
                tuple(query.user_names), tuple(query.keywords))

    def _locate_signature(self, session_key: str, query, count: int) -> str:
        import hashlib
        raw = "|".join(str(x) for x in (
            self._locate_conditions(session_key, query), query.offset,
            query.scan_limit, count,
        ))
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def _locate_cache_put(self, signature, text, query, conds, reached_start, oldest_ts):
        self._locate_cache[signature] = {
            "data": text,
            "timestamp": time.time(),
            "scan_limit": query.scan_limit,
            "conditions": conds,
            "reached_start": bool(reached_start),
            "oldest_ts": oldest_ts,
        }
        if len(self._locate_cache) > 100:
            now = time.time()
            for k in [k for k, v in self._locate_cache.items()
                      if now - v.get("timestamp", 0) > 300]:
                del self._locate_cache[k]

    def _locate_start_reached(self, conds):
        """A previous run with identical conditions already walked back to the
        oldest message: nothing older exists, so re-scanning cannot help.

        Scoped to the cache TTL: new messages arrive at the NEWEST end, so the
        claim "walked to the very beginning" stays true for the older part but
        the *answer* can go stale at any moment. Outside the TTL we let the
        query through and rescan instead of refusing forever.
        """
        now = time.time()
        for entry in self._locate_cache.values():
            if entry.get("conditions") != conds or not entry.get("reached_start"):
                continue
            if self.locate_cache_ttl_sec > 0 and \
                    (now - entry.get("timestamp", 0)) >= self.locate_cache_ttl_sec:
                continue
            return locate.fmt_ts(entry.get("oldest_ts"))
        return None

    def _locate_subset_prior(self, conds, current_limit: int):
        for entry in self._locate_cache.values():
            if entry.get("conditions") != conds:
                continue
            prior = int(entry.get("scan_limit", 0) or 0)
            if prior > current_limit:
                return prior
        return None

    def _charge_scan_budget(self, event, scanned: int) -> None:
        """累计本回合扫描条数（跨调用）。

        注意：空 dict 是正常状态 —— 若把空 extra 当成「无法计费」直接返回，
        预算就永远不会被消耗，等于把防循环关掉了。
        """
        if scanned <= 0 or self.max_scanned_per_turn <= 0:
            return
        extra = self._event_extra(event)
        if extra is None:
            return
        extra["merger_hist_scanned"] = int(extra.get("merger_hist_scanned", 0) or 0) + scanned

    def _scan_budget_left(self, event) -> int:
        if self.max_scanned_per_turn <= 0:
            return 1 << 30
        extra = self._event_extra(event)
        used = int(extra.get("merger_hist_scanned", 0) or 0) if extra is not None else 0
        return max(0, self.max_scanned_per_turn - used)

    async def _locate_session_history(self, event, st, entity, count,
                                      since, until, user_id, keyword,
                                      offset, scan_limit, target_key) -> str:
        # 先探测 OneBot 实现：扫描上限取决于它（NapCat 2000 / LLOneBot 1200 /
        # SnowLuma 800），若放到后面探测，首次调用会被钳到保守的通用上限。
        client = None
        if self.use_ws:
            client = self._get_client(event)
        adapter_name = ""
        try:
            info = getattr(event, "adapter", None)
            adapter_name = str(getattr(info, "name", None)
                               or getattr(info, "adapter_id", "") or "")
        except Exception:
            adapter_name = ""
        impl = await self._resolve_impl(client, adapter_name or st)

        query, notes, error = self._build_locale_query(
            since, until, user_id, keyword, offset, scan_limit)
        if error:
            return f"Error: {error}"

        # 防循环：调用预算 -> 精确签名缓存 -> 已到最早 -> 子集拒绝
        limited = self._track_and_limit(event, target_key)
        if limited:
            return limited

        requested_scan_limit = query.scan_limit
        conds = self._locate_conditions(target_key, query)
        signature = self._locate_signature(target_key, query, count)
        cached = self._locate_cache.get(signature)
        if cached and (time.time() - cached.get("timestamp", 0)) < self.locate_cache_ttl_sec:
            return (cached["data"]
                    + "\n\n---\n⚠️ 本次定位条件与刚才完全相同，结果见上。"
                    "请直接基于已有内容回答，不要重复查询；"
                    "如需更早的消息请加大 scan_limit。")
        reached = self._locate_start_reached(conds)
        if reached:
            return (f"Rejected: 上文中相同条件的查询已扫到该会话最早"
                    f"（{reached}），更早没有消息了。请直接使用已有结果。")
        prior = self._locate_subset_prior(conds, query.scan_limit)
        if prior:
            return (f"Rejected: 本次条件与上文某次查询相同，但扫描范围更小"
                    f"（本次 {query.scan_limit} < 上次 {prior}），"
                    "结果必然是上次的子集。请直接使用上文已有结果；"
                    "若需更多命中，请加大 scan_limit 或收窄条件。")

        budget_left = self._scan_budget_left(event)
        if budget_left <= 0:
            return ("Rejected: 本回合的扫描预算已用尽，请基于已有信息回答，"
                    "不要再次调用 get_session_history。")
        if budget_left < query.scan_limit:
            # 说明清楚：否则后续「加大 scan_limit」的建议会引用一个本次
            # 实际没有用到的数字。
            notes.append(
                f"本回合剩余扫描预算只有 {budget_left} 条，"
                f"本次按 {budget_left} 条执行（原定 {query.scan_limit}）")
            requested_scan_limit = budget_left
        query.scan_limit = min(query.scan_limit, budget_left)

        session_label = f"{st}:{entity}"
        result = await self._run_locate_scan(client, impl, st, entity, query,
                                             count, session_label)
        if result is None:
            self._note_failure()
            return ("Error: 历史扫描失败（请勿重复调用，基于已有信息回答）。\n"
                    "（定位失败，已计入熔断；本回合请勿再次查询）")

        self._charge_scan_budget(event, result.scanned_count)
        if result.report.error and not result.messages:
            self._note_failure()
            return f"Error: {result.report.error}"

        self._note_success()
        text = self._render_locate_result(result, query, count, session_label,
                                          notes, requested_scan_limit)
        self._locate_cache_put(signature, text, query, conds,
                               result.report.reached_start, result.report.oldest_ts)
        return text

    async def _run_locate_scan(self, client, impl, st, entity, query, count,
                               session_label):
        """Run the scan, retrying once from the newest page if the anchor went
        stale mid-walk (NapCat's short-id map is an LRU and can evict)."""
        for attempt in (1, 2):
            fetcher = self._make_page_fetcher(client, st, entity, impl)
            local = locate.LocateQuery(**query.__dict__)
            result = await locate.scan_backwards(
                fetcher, impl, local,
                session_label=session_label,
                page_size=int(impl.get("max_page") or 30),
                max_seconds=self.scan_max_seconds,
                max_steps=self.MAX_SCAN_STEPS,
                stop_when_enough=self.early_stop_on_enough,
                detect_boundary=self.detect_boundary,
                want_matches=count + query.offset,
                # 复用与「最近消息」路径相同的占位判定，避免关键词命中
                # SnowLuma 的合成占位行并把它显示给模型。
                msg_filter=lambda m: not self._is_placeholder(m),
            )
            result = locate.finalize(result, count + query.offset)
            if not result.report.error:
                return result
            if attempt == 2 or not self.locate_fallback_on_error:
                return result
            if self.logger:
                self.logger.warning(
                    "[history_tool] scan failed (%s); retrying from newest page",
                    result.report.error)
        return None

    def _format_people(self, people) -> str:
        """`涉及: 昵称[群名片:X](QQ)、...`

        Each entry is ONE PERSON (aggregated by QQ) - a nickname and a group
        card that differ are shown together rather than as two separate names.
        Truncated at MAX_PEOPLE_SHOWN; the list is already sorted by message
        count so the busiest speakers survive the cut.
        """
        shown = people[:self.MAX_PEOPLE_SHOWN]
        text = "、".join(p.display for p in shown)
        if len(people) > self.MAX_PEOPLE_SHOWN:
            # "共N人" (total) rather than "等N人" (ambiguous: N total, or N more?)
            text += "…（共%d人）" % len(people)
        return "涉及: " + text

    def _render_locate_result(self, result, query, count, session_label, notes,
                              requested_scan_limit) -> str:
        report = result.report
        lines: List[str] = []

        # Trim to `count` BEFORE the header is rendered: the header reports how
        # many messages are actually being returned, and computing it from the
        # pre-trim list makes it disagree with the body.
        selected = result.messages
        if count > 0 and len(selected) > count:
            selected = selected[:count]
        report.returned = len(selected)

        if self.locate_head_meta:
            lines.append(report.header(query, session_label))
            if result.people:
                lines.append(self._format_people(result.people))
            lines.append("---")

        if not selected:
            if report.reached_start:
                lines.append("已扫到会话最早，未找到符合条件的消息。")
            else:
                lines.append(
                    f"扫描范围内未命中（已扫最近 {report.scanned} 条，未到会话最早）。")
                lines.append(
                    "如需继续向前，请加大 scan_limit 重试"
                    f"（本次 {requested_scan_limit}，可试 "
                    f"{min(max(requested_scan_limit * 3, 600), self._scan_limit_cap())}）。")
            if notes and self.locate_head_meta:
                lines.extend(f"（{n}）" for n in notes)
            return "\n".join(lines)

        for msg in selected:
            lines.append(self._format_line(msg))

        if report.matched_total > len(selected) and self.locate_head_meta:
            first = query.offset + 1
            last = query.offset + len(selected)
            lines.append("---")
            # With an offset these are NOT "the newest N" - say which slice of
            # the match list is actually shown.
            lines.append(
                f"提示: 命中 {report.matched_total} 条，此处列出第 {first}-{last} 条。"
                f"可用 offset={last} 查看后续命中，或用 since/until 收窄条件。")
        if notes and self.locate_head_meta:
            lines.extend(f"（{n}）" for n in notes)

        return self._truncate_result("\n".join(lines))
