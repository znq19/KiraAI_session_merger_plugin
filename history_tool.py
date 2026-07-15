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
        return (
            data
            + "\n\n---\n⚠️ 系统提示：检测到重复查询同一会话。"
            "以上是已获取的完整历史消息，请直接基于此内容进行总结或回复，"
            "**请勿再次调用 get_session_history 工具**。"
        )

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
            if count < 5:
                count = 5
            elif count > 80:
                count = 80

            cache_key = f"{st}:{entity}"
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

            result_text = "\n".join(formatted)
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
