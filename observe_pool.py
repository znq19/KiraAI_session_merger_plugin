from __future__ import annotations

import asyncio
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .timeout_utils import is_expired, session_type_of

def _safe_display_name(value) -> str:
    """单行化显示名；名字里带方括号时把结构字符转全角，防止伪造消息头。"""
    s = " ".join(str(value or "").split())
    if "[" in s or "]" in s:
        s = s.replace("[", "［").replace("]", "］").replace("|", "｜")
    return s



class ObservePool:
    """
    未唤醒消息观察池（非真相源）。
    - 旁路记录 not mentioned 原文
    - 供偷看注入；受与冷会话相同的超时限制
    - 内存为主，节流落盘，避免卡事件循环
    """

    def __init__(
        self,
        data_dir: Optional[Path],
        max_per_session: int = 20,
        other_session_timeout: int = 20,
        other_session_timeout_private: int = 0,
        other_session_timeout_group: int = 0,
        flush_every: int = 20,
        logger=None,
    ):
        self.data_dir = Path(data_dir) if data_dir else None
        self.max_per_session = max(1, int(max_per_session or 20))
        self.other_session_timeout = int(other_session_timeout or 0)
        self.other_session_timeout_private = int(other_session_timeout_private or 0)
        self.other_session_timeout_group = int(other_session_timeout_group or 0)
        self.flush_every = max(1, int(flush_every or 20))
        self.logger = logger
        # sid -> list[dict]
        self._data: Dict[str, List[Dict[str, Any]]] = {}
        self._dirty = 0
        self._seen_ids: Dict[str, set] = {}
        # 节流触发的异步落盘任务（避免 ON_IM_MESSAGE 链路同步写盘）
        self._flush_task: "Optional[asyncio.Task]" = None
        self._load()

    def _path(self) -> Optional[Path]:
        if not self.data_dir:
            return None
        d = self.data_dir / "observe"
        d.mkdir(parents=True, exist_ok=True)
        return d / "observe.json"

    def _load(self):
        path = self._path()
        if not path or not path.exists():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                for sid, items in raw.items():
                    if isinstance(items, list):
                        self._data[sid] = [x for x in items if isinstance(x, dict)]
                        self._seen_ids[sid] = {
                            str(x.get("message_id"))
                            for x in self._data[sid]
                            if x.get("message_id")
                        }
        except Exception as e:
            if self.logger:
                self.logger.warning("[observe] load failed: %s", e)

    def clear(self):
        """清空观察池内存并落盘。"""
        self._data.clear()
        self._seen_ids.clear()
        self._dirty = self.flush_every
        try:
            self.flush(force=True)
        except Exception as e:
            if self.logger:
                self.logger.warning("[observe] clear flush failed: %s", e)

    def flush(self, force: bool = False):
        if not force and self._dirty < self.flush_every:
            return
        path = self._path()
        if not path:
            return
        try:
            # 先快照再序列化：flush 可能在 worker 线程执行，
            # 与事件循环里的 add 并发时避免"遍历时字典变更"
            snapshot = {sid: list(items) for sid, items in self._data.items()}
            path.write_text(
                json.dumps(snapshot, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self._dirty = 0
        except Exception as e:
            if self.logger:
                self.logger.warning("[observe] flush failed: %s", e)

    def _trim_sid(self, sid: str):
        items = self._data.get(sid) or []
        # drop expired
        kept = []
        for it in items:
            ts = int(it.get("timestamp") or 0)
            if is_expired(
                ts,
                session_type_of(sid),
                self.other_session_timeout,
                self.other_session_timeout_private,
                self.other_session_timeout_group,
            ):
                continue
            kept.append(it)
        if len(kept) > self.max_per_session:
            kept = kept[-self.max_per_session:]
        self._data[sid] = kept
        self._seen_ids[sid] = {
            str(x.get("message_id")) for x in kept if x.get("message_id")
        }

    def add(
        self,
        sid: str,
        content: str,
        message_id: str = "",
        timestamp: int = 0,
        sender_id: str = "",
        sender_name: str = "",
        group_name: str = "",
        sender_card: str = "",
    ) -> bool:
        if not sid or not (content or "").strip():
            return False
        mid = str(message_id or "")
        if mid:
            seen = self._seen_ids.setdefault(sid, set())
            if mid in seen:
                return False
            seen.add(mid)

        item = {
            "sid": sid,
            "message_id": mid,
            "timestamp": int(timestamp or time.time()),
            "sender_id": str(sender_id or ""),
            # 显示名统一压单行：群名片/昵称里的换行不能伪造新的状态行。
            "sender_name": _safe_display_name(sender_name),
            "sender_card": _safe_display_name(sender_card),
            "group_name": _safe_display_name(group_name),
            "content": content.strip(),
        }
        bucket = self._data.setdefault(sid, [])
        bucket.append(item)
        self._trim_sid(sid)
        self._dirty += 1
        if self._dirty >= self.flush_every:
            self._schedule_flush()
        return True

    def _schedule_flush(self):
        """节流落盘：事件循环里把同步 JSON 写盘挪到 worker 线程。

        已有任务在跑则不再重复调度（dirty 计数继续累积，下轮再触发）；
        不在事件循环上下文（如极端兜底）退化为同步 flush。
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self.flush()
            return
        task = self._flush_task
        if task is not None and not task.done():
            return
        self._flush_task = loop.create_task(self._flush_in_thread())

    async def _flush_in_thread(self):
        try:
            await asyncio.to_thread(self.flush)
        except Exception:
            pass

    def sample_peek(
        self,
        current_sid: str,
        max_messages: int = 3,
        source_tag_mode: str = "prefix",
    ) -> List[Dict[str, Any]]:
        """返回可注入的 OpenAI 风格 dict 列表（role=user）。"""
        max_messages = max(1, int(max_messages or 3))
        candidates: List[Dict[str, Any]] = []
        for sid, items in list(self._data.items()):
            if not sid or sid == current_sid:
                continue
            self._trim_sid(sid)
            for it in self._data.get(sid) or []:
                ts = int(it.get("timestamp") or 0)
                if is_expired(
                    ts,
                    session_type_of(sid),
                    self.other_session_timeout,
                    self.other_session_timeout_private,
                    self.other_session_timeout_group,
                ):
                    continue
                if not (it.get("content") or "").strip():
                    continue
                candidates.append(it)

        if not candidates:
            return []

        k = min(max_messages, len(candidates))
        picked = random.sample(candidates, k)
        # 时间序
        picked.sort(key=lambda x: int(x.get("timestamp") or 0))

        out: List[Dict[str, Any]] = []
        for it in picked:
            text = self._format_peek_content(it, source_tag_mode)
            if text:
                out.append({"role": "user", "content": text})
        return out

    @staticmethod
    def _has_session_prefix(content: str) -> bool:
        """只认真正的会话前缀：正文里恰好出现 user_nickname: 不算（用户可能自己打）。"""
        head = (content or "")[:120]
        return head.startswith("[source_sid:") or head.startswith("[session:")

    def _format_peek_prefix(
        self,
        sid: str,
        mode: str,
        group_name: str,
        sender_name: str,
        sender_card: str,
        sender_id: str,
        timestamp: int,
    ) -> str:
        """偷看来源标签：prefix 与官方消息头同口径（群名/群号/昵称/群名片/QQ 号）。"""
        if mode == "none" or not sid:
            return ""
        # 旧 observe.json 可能没有 sender_card，也可能带换行：渲染前再压一次单行。
        group_name = _safe_display_name(group_name)
        sender_name = _safe_display_name(sender_name)
        sender_card = _safe_display_name(sender_card)
        parts = sid.split(":", 2)
        st = parts[1] if len(parts) >= 2 else ""
        entity = parts[2] if len(parts) >= 3 else sid
        name = sender_name or "未知"
        card = sender_card if sender_card and sender_card != name else ""
        uid = str(sender_id or "").strip()
        if not uid and st == "dm":
            uid = entity
        uid = uid or "未知"
        try:
            ts = int(timestamp or time.time())
        except (TypeError, ValueError):
            ts = int(time.time())
        try:
            date_str = time.strftime("%b %d %Y %H:%M %a", time.localtime(ts))
        except (ValueError, OverflowError, OSError):
            date_str = "未知时间"

        if mode == "compact":
            if st == "gm":
                bits = [f"{st}:{entity}", group_name or "-", name, uid]
                return "[" + "|".join(bits) + "] "
            if st == "dm":
                bits = [f"{st}:{entity}", name]
                return "[" + "|".join(bits) + "] "
            return f"[{sid}] "

        # prefix：完整身份，与 kira-ai 的群/私聊消息头一致
        if st == "gm":
            identity = (
                f"group_name: {group_name or '未知'} group_id: {entity} "
                f"user_nickname: {name}"
            )
            if card:
                identity += f", group_nickname: {card}"
            identity += f", user_id: {uid}"
        elif st == "dm":
            identity = f"user_nickname: {name}, user_id: {uid}"
        else:
            identity = f"title: {group_name or name}"
        return f"[session: {sid}] [{date_str}] [{identity}] "

    def _format_peek_content(self, it: Dict[str, Any], mode: str) -> str:
        """
        偷看注入文案 = 与官方消息头同口径的来源标签 + peek 标记 + 原文。
        peek 标记仅偷看需要，正式合并历史没有。
        """
        sid = str(it.get("sid") or "")
        content = str(it.get("content") or "").strip()
        if not content:
            return ""
        if "[msg_type: peek]" in content[:120]:
            return content

        prefix = ""
        if self._has_session_prefix(content):
            if sid not in content[:120]:
                prefix = f"[session: {sid}] "
        else:
            prefix = self._format_peek_prefix(
                sid,
                mode or "prefix",
                str(it.get("group_name") or "").strip(),
                str(it.get("sender_name") or "").strip(),
                str(it.get("sender_card") or "").strip(),
                str(it.get("sender_id") or "").strip(),
                it.get("timestamp") or 0,
            )
        peek_tag = "[msg_type: peek] [not_addressed_to_you] "
        return f"{prefix}{peek_tag}{content}"

    def stats(self) -> Dict[str, Any]:
        return {
            "sessions": len(self._data),
            "messages": sum(len(v) for v in self._data.values()),
        }
