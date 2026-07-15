from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from core.agent.message import OpenAIMessage

from .memory_access import safe_fetch_memory, safe_session_meta


@dataclass
class TimelineMessage:
    role: str
    content: Any = None
    tool_calls: Optional[List[dict]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    source_sid: str = ""
    order: int = 0
    sort_key: Tuple = field(default_factory=tuple)


class TimelineBuilder:
    """从多会话官方记忆构建合法、可截断的统一时间线。"""

    def __init__(
        self,
        session_mgr,
        chars_per_token: float = 2.0,
        include_tool_traces: bool = True,
        drop_unpaired_tools: bool = True,
        prefer_official_content: bool = True,
        cross_session_time_order: bool = True,
        debug: bool = False,
        logger=None,
    ):
        self.session_mgr = session_mgr
        self.chars_per_token = max(0.1, float(chars_per_token or 2.0))
        self.include_tool_traces = include_tool_traces
        self.drop_unpaired_tools = drop_unpaired_tools
        self.prefer_official_content = prefer_official_content
        self.cross_session_time_order = cross_session_time_order
        self.debug = debug
        self.logger = logger
        # 单次 build 内会话标题缓存，避免重复 get_session_info
        self._title_cache: Dict[str, str] = {}

    def _log(self, msg: str, *args):
        if self.debug and self.logger:
            self.logger.debug(msg, *args)


    def _count_tokens(self, text: Any) -> int:
        if text is None:
            return 0
        if not isinstance(text, str):
            text = str(text)
        if not text:
            return 0
        return max(1, int(len(text) / self.chars_per_token) + 1)

    def _msg_tokens(self, msg: TimelineMessage) -> int:
        total = self._count_tokens(msg.content) + 4
        if msg.tool_calls:
            total += self._count_tokens(str(msg.tool_calls))
        return total

    @staticmethod
    def _extract_ts_from_content(content: Any) -> int:
        """从官方 message_str 前缀尽量解析时间；失败返回 0。"""
        if not isinstance(content, str) or not content:
            return 0
        # [Jul 15 2026 00:44 Wed] or similar
        m = re.search(
            r"\[([A-Za-z]{3})\s+(\d{1,2})\s+(\d{4})\s+(\d{1,2}):(\d{2})",
            content[:80],
        )
        if not m:
            return 0
        try:
            from datetime import datetime

            mon_map = {
                "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
                "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
            }
            mon = mon_map.get(m.group(1), 0)
            if not mon:
                return 0
            dt = datetime(
                int(m.group(3)), mon, int(m.group(2)),
                int(m.group(4)), int(m.group(5)),
            )
            return int(dt.timestamp())
        except Exception:
            return 0

    def collect_from_sessions(
        self,
        sids: List[str],
        per_sid_max_chunks: Optional[int] = None,
    ) -> List[TimelineMessage]:
        result: List[TimelineMessage] = []
        order = 0
        for sid_index, sid in enumerate(sids):
            # 关键：禁止 session_mgr.fetch_memory / get_session_info(sid)
            # 二者会 _ensure_session_data → 整文件写盘，卡死事件循环
            try:
                flat = safe_fetch_memory(
                    self.session_mgr,
                    sid,
                    max_chunks=per_sid_max_chunks,
                )
            except Exception as e:
                self._log("safe_fetch_memory failed for %s: %s", sid, e)
                continue


            meta = safe_session_meta(self.session_mgr, sid)
            session_ts = int(meta.get("timestamp") or 0)
            # 预填标题缓存，避免后续再查
            if sid not in self._title_cache:
                self._title_cache[sid] = str(meta.get("title") or "")


            for in_idx, raw in enumerate(flat):
                if not isinstance(raw, dict):
                    if hasattr(raw, "to_dict"):
                        raw = raw.to_dict()
                    else:
                        continue
                role = raw.get("role") or "user"
                if role not in ("user", "assistant", "tool", "system"):
                    continue
                if role == "system":
                    continue

                content = raw.get("content")
                tool_calls = raw.get("tool_calls") or []
                tool_call_id = raw.get("tool_call_id")
                name = raw.get("name")

                if not self.include_tool_traces:
                    if role == "tool":
                        continue
                    if role == "assistant" and tool_calls and not (content or "").strip():
                        continue
                    tool_calls = []
                    tool_call_id = None

                if role == "tool" and not tool_call_id:
                    continue
                if role != "tool" and content is None and not tool_calls:
                    continue
                if role == "assistant" and not content and not tool_calls:
                    continue

                ts = raw.get("timestamp")
                if ts is None:
                    ts = self._extract_ts_from_content(content)
                try:
                    ts = int(ts or 0)
                except Exception:
                    ts = 0
                if ts <= 0:
                    ts = session_ts

                if self.cross_session_time_order:
                    sort_key = (ts, sid_index, in_idx, order)
                else:
                    sort_key = (sid_index, in_idx, order)

                result.append(
                    TimelineMessage(
                        role=role,
                        content=content,
                        tool_calls=list(tool_calls) if tool_calls else None,
                        tool_call_id=tool_call_id,
                        name=name,
                        source_sid=sid,
                        order=order,
                        sort_key=sort_key,
                    )
                )
                order += 1

        result.sort(key=lambda m: m.sort_key)
        # reassign sequential order after sort (preserve relative tool chains within same sid chunk)
        for i, m in enumerate(result):
            m.order = i
        return result

    def sanitize_tool_pairs(self, msgs: List[TimelineMessage]) -> List[TimelineMessage]:
        if not self.drop_unpaired_tools:
            return msgs

        # Collect answered tool_call ids
        answered = {
            m.tool_call_id
            for m in msgs
            if m.role == "tool" and m.tool_call_id
        }
        # Collect declared tool_call ids from assistants
        declared: set = set()
        for m in msgs:
            if m.role == "assistant" and m.tool_calls:
                for tc in m.tool_calls:
                    if isinstance(tc, dict) and tc.get("id"):
                        declared.add(tc["id"])

        out: List[TimelineMessage] = []
        for m in msgs:
            if m.role == "tool":
                if m.tool_call_id and m.tool_call_id in declared:
                    out.append(m)
                else:
                    self._log("drop orphan tool: %s", m.tool_call_id)
                continue

            if m.role == "assistant" and m.tool_calls:
                kept = [
                    tc for tc in m.tool_calls
                    if isinstance(tc, dict) and tc.get("id") in answered
                ]
                if not kept and not (m.content or "").strip():
                    self._log("drop unanswered tool-only assistant")
                    continue
                if len(kept) != len(m.tool_calls):
                    m = TimelineMessage(
                        role=m.role,
                        content=m.content,
                        tool_calls=kept or None,
                        tool_call_id=m.tool_call_id,
                        name=m.name,
                        source_sid=m.source_sid,
                        order=m.order,
                        sort_key=m.sort_key,
                    )
            out.append(m)
        return out

    def rebuild_chunks(self, msgs: List[TimelineMessage]) -> List[List[TimelineMessage]]:
        if not msgs:
            return []
        chunks: List[List[TimelineMessage]] = []
        current: List[TimelineMessage] = []
        for msg in msgs:
            if msg.role == "user":
                if current:
                    chunks.append(current)
                current = [msg]
            else:
                if not current:
                    # leading non-user: keep as its own soft chunk start
                    current = [msg]
                else:
                    current.append(msg)
        if current:
            chunks.append(current)
        return chunks

    def trim_chunks(
        self,
        chunks: List[List[TimelineMessage]],
        max_chunks: int,
        max_tokens: int,
        enable_trim: bool = True,
    ) -> List[TimelineMessage]:
        if not chunks:
            return []
        if not enable_trim:
            return [m for c in chunks for m in c]

        selected = chunks
        if max_chunks and max_chunks > 0 and len(selected) > max_chunks:
            selected = selected[-max_chunks:]

        if max_tokens and max_tokens > 0:
            while selected:
                flat = [m for c in selected for m in c]
                total = sum(self._msg_tokens(m) for m in flat)
                if total <= max_tokens or len(selected) <= 1:
                    break
                selected = selected[1:]

        return [m for c in selected for m in c]

    def _session_title(self, source_sid: str) -> str:
        if not source_sid:
            return ""
        if source_sid in self._title_cache:
            return self._title_cache[source_sid]
        # 只读内存，绝不 get_session_info（会写盘）
        title = str(safe_session_meta(self.session_mgr, source_sid).get("title") or "")
        self._title_cache[source_sid] = title
        return title


    def _content_already_has_source(self, content: str) -> bool:
        """官方 user message_str 已含 group_name / user_nickname 等，避免重复堆叠。"""
        if not content:
            return False
        head = content[:240]
        return (
            "group_name:" in head
            or "group_id:" in head
            or "user_nickname:" in head
            or "user_id:" in head
            or head.startswith("[source_sid:")
            or head.startswith("[session:")
        )

    def _format_source_prefix(self, source_sid: str, mode: str, role: str, content: Any) -> str:
        """
        官方风格括号前缀，尽量不破坏原文（尤其已含 group_name 的 user 行）。

        官方 user 示例：
        [Jul 15 ...] [message_id: x] [group_name: 绿岛酒吧 group_id: ... user_nickname: ...] | 正文

        我们只在需要时外包一层：
        - prefix:  [session: qq:gm:123][group_name: 绿岛酒吧]
        - compact: [gm:123|绿岛酒吧]
        - none:    不加

        tool 永不加前缀。assistant 的 <msg> 通常无群名，需要前缀。
        user 若已有 group_name，仅补极短 [session: sid]（prefix）或跳过（若 sid 已在文中）。
        """
        if mode == "none" or not source_sid:
            return ""
        if role == "tool":
            return ""

        parts = source_sid.split(":", 2)
        st = parts[1] if len(parts) >= 2 else ""
        entity = parts[2] if len(parts) >= 3 else source_sid
        title = self._session_title(source_sid)
        text = content if isinstance(content, str) else ""

        # 已带官方身份字段：user 只补 session sid，避免再写一遍群名
        if role == "user" and self._content_already_has_source(text):
            if source_sid in text[:120]:
                return ""
            if mode == "compact":
                return f"[{st}:{entity}] " if st else f"[{source_sid}] "
            return f"[session: {source_sid}] "

        if mode == "compact":
            if title:
                return f"[{st}:{entity}|{title}] "
            return f"[{st}:{entity}] " if st else f"[{source_sid}] "

        # prefix：贴近官方 [key: value] 风格
        if st == "gm":
            if title:
                return f"[session: {source_sid}] [group_name: {title}] "
            return f"[session: {source_sid}] [chat_type: group] "
        if st == "dm":
            if title:
                return f"[session: {source_sid}] [user_nickname: {title}] "
            return f"[session: {source_sid}] [chat_type: private] "
        if title:
            return f"[session: {source_sid}] [title: {title}] "
        return f"[session: {source_sid}] "

    def to_openai_messages(
        self,
        msgs: List[TimelineMessage],
        source_tag_mode: str = "prefix",
    ) -> List[OpenAIMessage]:
        result: List[OpenAIMessage] = []
        for m in msgs:
            # 默认原样保留官方 content（含 message_id / 图片描述 / file_path），不改写缓存相关字段
            content = m.content

            if m.role == "tool":
                result.append(
                    OpenAIMessage(
                        role="tool",
                        content=content if content is not None else "[tool result]",
                        tool_call_id=m.tool_call_id or "",
                        name=m.name,
                    )
                )
                continue

            if isinstance(content, str) and content:
                already = (
                    content.startswith("[session:")
                    or content.startswith("[source:")
                    or content.startswith("[source_sid:")
                )
                if not already:
                    prefix = self._format_source_prefix(
                        m.source_sid, source_tag_mode, m.role, content
                    )
                    if prefix:
                        content = prefix + content

            if m.role == "assistant":
                result.append(
                    OpenAIMessage(
                        role="assistant",
                        content=content,
                        tool_calls=m.tool_calls or [],
                    )
                )
            else:
                result.append(OpenAIMessage(role=m.role, content=content))
        return result

    def build(
        self,
        sids: List[str],
        max_chunks: int = 12,
        max_tokens: int = 12000,
        enable_trim: bool = True,
        source_tag_mode: str = "prefix",
        per_sid_max_chunks: Optional[int] = None,
    ) -> List[OpenAIMessage]:
        self._title_cache.clear()
        if per_sid_max_chunks is None:
            per_sid_max_chunks = max_chunks if max_chunks and max_chunks > 0 else None
        raw = self.collect_from_sessions(sids, per_sid_max_chunks=per_sid_max_chunks)
        cleaned = self.sanitize_tool_pairs(raw)
        chunks = self.rebuild_chunks(cleaned)
        trimmed = self.trim_chunks(chunks, max_chunks, max_tokens, enable_trim)
        return self.to_openai_messages(trimmed, source_tag_mode)

    def preview_text(
        self,
        sids: List[str],
        max_chunks: int = 12,
        max_tokens: int = 12000,
        enable_trim: bool = True,
        source_tag_mode: str = "prefix",
        limit: int = 30,
        per_sid_max_chunks: Optional[int] = None,
    ) -> str:
        msgs = self.build(
            sids,
            max_chunks,
            max_tokens,
            enable_trim,
            source_tag_mode,
            per_sid_max_chunks=per_sid_max_chunks,
        )
        lines = []
        for i, m in enumerate(msgs[:limit]):
            c = m.content if isinstance(m.content, str) else str(m.content)
            if c and len(c) > 160:
                c = c[:160] + "..."
            extra = ""
            if m.role == "assistant" and m.tool_calls:
                extra = f" tool_calls={len(m.tool_calls)}"
            if m.role == "tool":
                extra = f" id={m.tool_call_id}"
            lines.append(f"[{i}] {m.role}{extra}: {c}")
        if len(msgs) > limit:
            lines.append(f"... total {len(msgs)} messages")
        return "\n".join(lines) if lines else "(empty)"

