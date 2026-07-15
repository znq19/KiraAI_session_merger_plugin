from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import httpx

from core.chat.message_utils import KiraMessageBatchEvent


class HistoryToolService:
    """
    OneBot HTTP 历史查询。

    对齐 history_plugin：任何失败只 return str，绝不抛异常。
    额外：全局熔断，连续失败后短时间直接拒绝，避免拖慢 agent。
    """

    CONNECT_TIMEOUT = 1.5
    READ_TIMEOUT = 3.0
    ERROR_CACHE_TTL = 90.0
    # 连续失败 N 次后进入熔断，直接返回错误不再打 HTTP
    CIRCUIT_FAIL_THRESHOLD = 2
    CIRCUIT_OPEN_SEC = 120.0
    # 同一 agent 回合内：同一目标会话最多成功返回几次（再调用直接拒绝，不塞大段历史）
    MAX_CALLS_PER_TARGET_PER_EVENT = 1
    # 同一 agent 回合内：历史工具总调用上限（含被拒绝的）
    MAX_CALLS_PER_EVENT = 2
    # 单次返回正文最大字符，避免 tool_result 把上下文撑爆
    MAX_RESULT_CHARS = 3500

    def __init__(
        self,
        http_host: str = "localhost",
        http_port: int = 3000,
        access_token: str = "",
        master_id: str = "",
        allowed_users: Optional[List[str]] = None,
        restricted_groups: Optional[List[str]] = None,
        cache_ttl_sec: int = 120,
        logger=None,
    ):
        self.http_host = http_host or "localhost"
        self.http_port = int(http_port or 3000)
        self.base_url = f"http://{self.http_host}:{self.http_port}"
        self.access_token = access_token or ""
        self.master_id = str(master_id or "").strip()
        self.allowed_users = [str(u).strip() for u in (allowed_users or []) if str(u).strip()]
        self.restricted_groups = [str(g).strip() for g in (restricted_groups or []) if str(g).strip()]
        self.cache_ttl_sec = max(0, int(cache_ttl_sec or 0))
        self.logger = logger
        self._call_cache: Dict[str, Dict[str, Any]] = {}
        self._fail_streak = 0
        self._circuit_open_until = 0.0

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

    @staticmethod
    def parse_session_ref(session_id: str, session_type: Optional[str] = None) -> Dict[str, str]:
        sid = (session_id or "").strip()
        st = (session_type or "").strip().lower()

        if ":" in sid:
            parts = sid.split(":", 2)
            adapter = parts[0] if len(parts) >= 1 else "qq"
            typ = parts[1] if len(parts) >= 2 else "dm"
            entity = parts[2] if len(parts) >= 3 else sid
            if typ in ("group", "g"):
                typ = "gm"
            if typ in ("private", "p", "friend"):
                typ = "dm"
            return {"adapter": adapter, "session_type": typ, "session_id": entity, "full": sid}

        if st in ("group", "gm", "g"):
            typ = "gm"
        elif st in ("private", "dm", "p", "friend"):
            typ = "dm"
        else:
            typ = "dm"
        return {"adapter": "qq", "session_type": typ, "session_id": sid, "full": f"qq:{typ}:{sid}"}

    def _message_to_text(self, msg: dict) -> str:
        if msg.get("raw_message"):
            content = msg["raw_message"]
        else:
            msg_segments = msg.get("message", [])
            if not msg_segments:
                content = "[空消息]"
            else:
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
                        parts.append(f"[图片]({img_url})" if img_url else "[图片]")
                    elif seg_type == "video":
                        parts.append("[视频]")
                    elif seg_type == "file":
                        parts.append(f"[文件]{seg_data.get('name', '文件')}")
                    elif seg_type == "reply":
                        parts.append("[回复]")
                    elif seg_type == "forward":
                        parts.append("[转发消息]")
                    else:
                        parts.append(f"[{seg_type}]")
                content = " ".join(parts)

        msg_id = msg.get("message_id")
        if msg_id:
            content += f" (msg_id:{msg_id})"
        return content

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
    def _event_extra(event) -> dict:
        try:
            extra = getattr(event, "extra", None)
            if not isinstance(extra, dict):
                extra = {}
                try:
                    event.extra = extra
                except Exception:
                    return {}
            return extra
        except Exception:
            return {}

    def _track_and_limit(self, event, target_key: str) -> Optional[str]:
        """
        同一 KiraMessageBatchEvent（一次 agent 回合）内限制历史工具调用。
        返回非 None 则应直接 return 该字符串，不再打 HTTP。
        """
        extra = self._event_extra(event)
        total = int(extra.get("merger_hist_total", 0) or 0)
        by_target = extra.get("merger_hist_by_target")
        if not isinstance(by_target, dict):
            by_target = {}
            extra["merger_hist_by_target"] = by_target

        if total >= self.MAX_CALLS_PER_EVENT:
            return (
                "Rejected: 本回合 get_session_history 调用次数已达上限。"
                "请直接回复，禁止再查历史。"
            )
        n = int(by_target.get(target_key, 0) or 0)
        if n >= self.MAX_CALLS_PER_TARGET_PER_EVENT:
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
        self._fail_streak += 1
        if self._fail_streak >= self.CIRCUIT_FAIL_THRESHOLD:
            self._circuit_open_until = time.time() + self.CIRCUIT_OPEN_SEC
            if self.logger:
                self.logger.warning(
                    "history circuit OPEN for %.0fs after %d failures",
                    self.CIRCUIT_OPEN_SEC,
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
    ) -> str:
        try:
            blocked = self._circuit_blocked()
            if blocked:
                return blocked

            user_id = "unknown"
            if event.messages and event.messages[0].sender:
                user_id = str(event.messages[0].sender.user_id)

            ref = self.parse_session_ref(session_id, session_type)
            st = ref["session_type"]
            entity = ref["session_id"]

            if not self._check_permission(user_id, st, entity):
                return "抱歉，您没有权限查看此会话的历史消息。"

            try:
                count = int(count)
            except Exception:
                count = 20
            # 与 history_plugin 对齐：最少 5，最多 80，默认 20
            if count < 5:
                count = 5
            elif count > 80:
                count = 80

            cache_key = f"{st}:{entity}"
            target_key = cache_key

            # 本回合调用次数硬限制（在缓存命中之前也计数，防止刷拒绝）
            limited = self._track_and_limit(event, target_key)
            if limited:
                return limited

            hit = self._cache_get_hit(cache_key, count)
            if hit is not None:
                return hit

            if st == "gm":
                api = "get_group_msg_history"
                params = {"group_id": int(entity), "count": count}
            else:
                api = "get_friend_msg_history"
                params = {"user_id": int(entity), "count": count}

            headers = {}
            if self.access_token:
                headers["Authorization"] = f"Bearer {self.access_token}"

            timeout = httpx.Timeout(
                connect=self.CONNECT_TIMEOUT,
                read=self.READ_TIMEOUT,
                write=self.READ_TIMEOUT,
                pool=self.CONNECT_TIMEOUT,
            )

            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    f"{self.base_url}/{api}",
                    json=params,
                    headers=headers,
                )
                if resp.status_code >= 400:
                    err = (
                        f"Error: HTTP {resp.status_code} from {self.base_url}/{api}. "
                        "OneBot HTTP 不可用。请勿再次调用 get_session_history，"
                        "请基于当前对话上下文回答。"
                    )
                    if self.logger:
                        self.logger.error("Error fetching history: HTTP %s", resp.status_code)
                    self._cache_put(cache_key, 80, err, is_error=True)
                    self._note_failure()
                    return err
                try:
                    result = resp.json()
                except Exception as e:
                    err = f"Error: invalid JSON from OneBot ({e})"
                    self._cache_put(cache_key, 80, err, is_error=True)
                    self._note_failure()
                    return err

            if result.get("status") != "ok":
                err = f"Failed: {result.get('message', 'unknown error')}"
                self._cache_put(cache_key, 80, err, is_error=True)
                self._note_failure()
                return err

            messages = result.get("data", {}).get("messages", [])
            if not messages:
                empty = "该会话暂无历史消息。"
                self._cache_put(cache_key, count, empty, is_error=False)
                self._note_success()
                return empty

            formatted = []
            for msg in messages[-count:]:
                sender = msg.get("sender", {}).get("nickname", "Unknown")
                content = self._message_to_text(msg)
                formatted.append(f"{sender}: {content}")

            result_text = self._truncate_result("\n".join(formatted))
            self._cache_put(cache_key, count, result_text, is_error=False)
            self._note_success()
            return result_text

        except Exception as e:
            if self.logger:
                self.logger.error("Error fetching history: %s", e)
            err = (
                f"Error: {str(e)}. "
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
