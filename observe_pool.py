from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .timeout_utils import is_expired, session_type_of


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

    def flush(self, force: bool = False):
        if not force and self._dirty < self.flush_every:
            return
        path = self._path()
        if not path:
            return
        try:
            path.write_text(
                json.dumps(self._data, ensure_ascii=False, indent=2),
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
            "sender_name": str(sender_name or ""),
            "group_name": str(group_name or ""),
            "content": content.strip(),
        }
        bucket = self._data.setdefault(sid, [])
        bucket.append(item)
        self._trim_sid(sid)
        self._dirty += 1
        if self._dirty >= self.flush_every:
            self.flush()
        return True

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

    def _format_peek_content(self, it: Dict[str, Any], mode: str) -> str:
        sid = str(it.get("sid") or "")
        content = str(it.get("content") or "").strip()
        if not content:
            return ""
        group_name = str(it.get("group_name") or "").strip()
        sender_name = str(it.get("sender_name") or "").strip()
        parts = sid.split(":", 2)
        st = parts[1] if len(parts) >= 2 else ""
        entity = parts[2] if len(parts) >= 3 else sid

        if mode == "none":
            prefix = "[msg_type: peek] [not_addressed_to_you] "
        elif mode == "compact":
            label = group_name or sender_name or entity
            prefix = f"[{st}:{entity}|{label}|peek] "
        else:
            # prefix — 官方风格
            bits = [f"[session: {sid}]"]
            if st == "gm" and group_name:
                bits.append(f"[group_name: {group_name}]")
            elif st == "dm" and sender_name:
                bits.append(f"[user_nickname: {sender_name}]")
            elif group_name:
                bits.append(f"[group_name: {group_name}]")
            elif sender_name:
                bits.append(f"[user_nickname: {sender_name}]")
            bits.append("[msg_type: peek]")
            bits.append("[not_addressed_to_you]")
            prefix = " ".join(bits) + " "

        # 避免双重 peek 标记
        if "[msg_type: peek]" in content[:80]:
            return content
        return prefix + content

    def stats(self) -> Dict[str, Any]:
        return {
            "sessions": len(self._data),
            "messages": sum(len(v) for v in self._data.values()),
        }
