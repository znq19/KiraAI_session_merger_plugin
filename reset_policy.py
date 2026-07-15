from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

from core.agent.message import OpenAIMessage


def count_messages_tokens(messages: List[OpenAIMessage], chars_per_token: float = 2.0) -> int:
    cpt = max(0.1, float(chars_per_token or 2.0))
    total = 0
    for m in messages:
        content = getattr(m, "content", None)
        if content is None:
            text = ""
        elif isinstance(content, str):
            text = content
        else:
            text = str(content)
        total += max(1, int(len(text) / cpt) + 1) if text else 0
        total += 4
        tcs = getattr(m, "tool_calls", None) or []
        if tcs:
            total += max(1, int(len(str(tcs)) / cpt) + 1)
    return total


def rebuild_openai_chunks(messages: List[OpenAIMessage]) -> List[List[OpenAIMessage]]:
    if not messages:
        return []
    chunks: List[List[OpenAIMessage]] = []
    cur: List[OpenAIMessage] = []
    for m in messages:
        role = getattr(m, "role", None)
        if role == "user":
            if cur:
                chunks.append(cur)
            cur = [m]
        else:
            if not cur:
                cur = [m]
            else:
                cur.append(m)
    if cur:
        chunks.append(cur)
    return chunks


def flatten_chunks(chunks: List[List[OpenAIMessage]]) -> List[OpenAIMessage]:
    out: List[OpenAIMessage] = []
    for c in chunks:
        out.extend(c)
    return out


class SoftResetState:
    """
    按合并组维护动态保留轮数。

    重要：只有「确认超限并执行重开」时才调用 on_reset()，
    才会更新 last_reset / 可能减半 keep。未超限时绝不减半。
    """

    def __init__(self, keep_turns: int = 6, check_interval_sec: int = 60):
        self.default_keep = max(1, int(keep_turns or 6))
        self.check_interval = max(0, int(check_interval_sec or 0))
        self._dynamic_keep: Dict[str, int] = {}
        self._last_reset: Dict[str, float] = {}
        self._last_check: Dict[str, float] = {}

    def update_defaults(self, keep_turns: int, check_interval_sec: int):
        self.default_keep = max(1, int(keep_turns or 6))
        self.check_interval = max(0, int(check_interval_sec or 0))

    def should_check(self, group_id: str, now: Optional[float] = None) -> bool:
        """是否允许本轮做一次「超限检测并可能重开」（节流）。"""
        if self.check_interval <= 0:
            return True
        now = now if now is not None else time.time()
        last = self._last_check.get(group_id, 0)
        if now - last < self.check_interval:
            return False
        self._last_check[group_id] = now
        return True

    def current_keep(self, group_id: str) -> int:
        """当前动态保留轮数（未重开过则用配置默认）。"""
        return max(1, int(self._dynamic_keep.get(group_id, self.default_keep)))

    def on_reset(self, group_id: str, now: Optional[float] = None) -> int:
        """
        仅在「确认超限且即将执行软/硬重开」时调用。
        - 若距上次重开很近（连续超限）→ keep 减半（最少 1）
        - 否则 → 重置为配置的 default_keep
        返回本次应使用的 keep 轮数。
        """
        now = now if now is not None else time.time()
        last_reset = self._last_reset.get(group_id, 0)
        half_window = self.check_interval / 2 if self.check_interval > 0 else 30.0

        if last_reset and (now - last_reset) < half_window:
            old = self._dynamic_keep.get(group_id, self.default_keep)
            new = max(1, old // 2)
            self._dynamic_keep[group_id] = new
        else:
            self._dynamic_keep[group_id] = self.default_keep

        self._last_reset[group_id] = now
        return self.current_keep(group_id)


def apply_round_cap(
    messages: List[OpenAIMessage],
    max_chunks: int,
) -> List[OpenAIMessage]:
    """只按轮数截断到最近 max_chunks，不涉及 token。"""
    if not messages:
        return []
    chunks = rebuild_openai_chunks(messages)
    if max_chunks and max_chunks > 0 and len(chunks) > max_chunks:
        chunks = chunks[-max_chunks:]
    return flatten_chunks(chunks)


def apply_soft_trim(
    messages: List[OpenAIMessage],
    max_chunks: int,
    token_limit: int,
    keep_turns: int,
    chars_per_token: float,
) -> Tuple[List[OpenAIMessage], int, bool]:
    """
    先取最近 max_chunks 轮，再估 token；
    超限则只保留 keep_turns 轮。
    返回 (结果, 估算token, 是否因 token 超限而截到 keep)
    """
    chunks = rebuild_openai_chunks(messages)
    if max_chunks and max_chunks > 0 and len(chunks) > max_chunks:
        chunks = chunks[-max_chunks:]

    flat = flatten_chunks(chunks)
    tokens = count_messages_tokens(flat, chars_per_token)
    triggered = False

    if token_limit and token_limit > 0 and tokens > token_limit:
        triggered = True
        k = max(1, int(keep_turns or 1))
        if len(chunks) > k:
            chunks = chunks[-k:]
        flat = flatten_chunks(chunks)
        tokens = count_messages_tokens(flat, chars_per_token)

    return flat, tokens, triggered
