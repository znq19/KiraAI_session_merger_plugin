from __future__ import annotations

import asyncio
import random
import time
from typing import List, Optional

from core.agent.message import OpenAIMessage
from core.provider import LLMRequest

from .anchor import inject_window_anchor
from .group_resolver import GroupResolver
from .hard_reset import hard_reset_members
from .memory_access import pick_member_sids
from .observe_pool import ObservePool
from .reset_policy import (
    SoftResetState,
    apply_round_cap,
    apply_soft_trim,
    count_messages_tokens,
)
from .timeline import TimelineBuilder


class MergeEngine:
    """读时合并 + 软/硬重启 + 未唤醒偷看。"""

    MERGE_TIMEOUT_SEC = 3.0

    def __init__(
        self,
        session_mgr,
        resolver: GroupResolver,
        timeline: TimelineBuilder,
        observe_pool: Optional[ObservePool] = None,
        max_merged_chunks: int = 10,
        merge_token_limit: int = 15000,
        merge_keep_turns: int = 6,
        merge_reset_mode: str = "soft",
        merge_check_interval_sec: int = 60,
        chars_per_token: float = 2.0,
        max_merge_sessions: int = 8,
        other_session_timeout: int = 20,
        other_session_timeout_private: int = 0,
        other_session_timeout_group: int = 0,
        unmentioned_probability: float = 0.01,
        peek_max_messages: int = 3,
        source_tag_mode: str = "prefix",
        enable_window_anchor: bool = True,
        window_anchor_prompt: str = "",
        debug: bool = False,
        log_preview: bool = False,
        logger=None,
    ):
        self.session_mgr = session_mgr
        self.resolver = resolver
        self.timeline = timeline
        self.observe_pool = observe_pool
        self.max_merged_chunks = max(1, int(max_merged_chunks or 10))
        self.merge_token_limit = max(0, int(merge_token_limit or 15000))
        self.merge_keep_turns = max(1, int(merge_keep_turns or 6))
        self.merge_reset_mode = (merge_reset_mode or "soft").strip().lower()
        if self.merge_reset_mode not in ("soft", "hard"):
            self.merge_reset_mode = "soft"
        self.chars_per_token = float(chars_per_token or 2.0)
        self.max_merge_sessions = max(1, int(max_merge_sessions or 8))
        self.other_session_timeout = int(other_session_timeout or 0)
        self.other_session_timeout_private = int(other_session_timeout_private or 0)
        self.other_session_timeout_group = int(other_session_timeout_group or 0)
        self.unmentioned_probability = float(unmentioned_probability or 0)
        self.peek_max_messages = max(1, int(peek_max_messages or 3))
        self.source_tag_mode = source_tag_mode or "prefix"
        self.enable_window_anchor = enable_window_anchor
        self.window_anchor_prompt = window_anchor_prompt or ""
        self.debug = debug
        self.log_preview = log_preview
        self.logger = logger
        self._soft_state = SoftResetState(
            keep_turns=self.merge_keep_turns,
            check_interval_sec=int(merge_check_interval_sec or 0),
        )

    def _log(self, msg: str, *args):
        if self.logger and (self.debug or msg.startswith("[MERGER]")):
            try:
                self.logger.info(msg, *args)
            except Exception:
                pass

    def should_merge(self, sid: str) -> bool:
        return bool(self.resolver.resolve_group_id(sid))

    def _group_id(self, sid: str) -> str:
        return self.resolver.resolve_group_id(sid) or f"solo:{sid}"

    def _member_sids(self, current_sid: str) -> List[str]:
        raw = self.resolver.members_for_session(current_sid)
        if not raw:
            raw = [current_sid]
        return pick_member_sids(
            self.session_mgr,
            raw,
            current_sid=current_sid,
            max_sessions=self.max_merge_sessions,
            other_session_timeout=self.other_session_timeout,
            other_session_timeout_private=self.other_session_timeout_private,
            other_session_timeout_group=self.other_session_timeout_group,
        )

    def _build_raw_merged(self, current_sid: str, members: List[str]) -> List[OpenAIMessage]:
        """按 max_merged_chunks 取最近轮，不做 token 重开。"""
        per_sid = max(self.max_merged_chunks, 1)
        return self.timeline.build(
            sids=members,
            max_chunks=self.max_merged_chunks,
            max_tokens=0,
            enable_trim=True,
            source_tag_mode=self.source_tag_mode,
            per_sid_max_chunks=per_sid,
        )

    def _maybe_peek(self, current_sid: str, msgs: List[OpenAIMessage]) -> List[OpenAIMessage]:
        if not (
            self.observe_pool
            and self.unmentioned_probability > 0
            and random.random() < self.unmentioned_probability
        ):
            return msgs
        peeks = self.observe_pool.sample_peek(
            current_sid=current_sid,
            max_messages=self.peek_max_messages,
            source_tag_mode=self.source_tag_mode,
        )
        for p in peeks:
            try:
                msgs.append(OpenAIMessage(role="user", content=p.get("content", "")))
            except Exception:
                pass
        if peeks:
            self._log("[MERGER] peek injected %d msgs for %s", len(peeks), current_sid)
        return msgs

    def build_merged_history(self, current_sid: str) -> List[OpenAIMessage]:
        """
        流程：
        1. 拼合并历史（最多 max_merged_chunks 轮）
        2. 估 token；未超限 → 直接用
        3. 超限 → 仅此时可能 on_reset（减半/重置 keep），并：
           - soft：合并视图截到 keep 轮
           - hard：各 sid 磁盘重开 keep 轮 + 合并视图也截到 keep 轮
        keep 减半只发生在「确认超限且执行重开」时。
        """
        group_id = self._group_id(current_sid)
        members = self._member_sids(current_sid)

        msgs = self._build_raw_merged(current_sid, members)
        tokens = count_messages_tokens(msgs, self.chars_per_token)
        over = self.merge_token_limit > 0 and tokens > self.merge_token_limit

        if not over:
            # 未超限：只保留 max_merged_chunks，绝不动 keep 状态
            msgs = apply_round_cap(msgs, self.max_merged_chunks)
        else:
            # 超限：是否允许本轮「正式重开」（更新 keep / 硬删）
            allow_reset = self._soft_state.should_check(group_id)
            if allow_reset:
                # 仅在真正重开时更新动态 keep（可能减半）
                keep = self._soft_state.on_reset(group_id)
            else:
                # 检测间隔内：不减半，沿用当前 keep，仍截断合并视图
                keep = self._soft_state.current_keep(group_id)

            if self.merge_reset_mode == "hard" and allow_reset:
                self._log(
                    "[MERGER hard] token=%d > %d, hard-reset members keep=%d",
                    tokens,
                    self.merge_token_limit,
                    keep,
                )
                hard_reset_members(
                    self.session_mgr, members, keep, logger=self.logger
                )
                msgs = self._build_raw_merged(current_sid, members)
            elif self.merge_reset_mode == "soft" and allow_reset:
                self._log(
                    "[MERGER soft] token=%d > %d, soft-reset keep=%d group=%s",
                    tokens,
                    self.merge_token_limit,
                    keep,
                    group_id,
                )
            else:
                self._log(
                    "[MERGER] token=%d > %d, apply keep=%d (no keep-halve this turn)",
                    tokens,
                    self.merge_token_limit,
                    keep,
                )

            # soft / hard 统一：合并视图轮数收到 keep（与 ADS 保留轮一致）
            msgs = apply_round_cap(msgs, keep)
            tokens_after = count_messages_tokens(msgs, self.chars_per_token)
            self._log(
                "[MERGER] after reset view: keep=%d msgs=%d token≈%d mode=%s",
                keep,
                len(msgs),
                tokens_after,
                self.merge_reset_mode,
            )

        msgs = self._maybe_peek(current_sid, msgs)
        self._log(
            "[MERGER] built sid=%s mode=%s members=%d msgs=%d",
            current_sid,
            self.merge_reset_mode,
            len(members),
            len(msgs),
        )
        return msgs

    def preview(self, current_sid: str, limit: int = 30) -> str:
        members = self._member_sids(current_sid)
        return self.timeline.preview_text(
            sids=members,
            max_chunks=self.max_merged_chunks,
            max_tokens=0,
            enable_trim=True,
            source_tag_mode=self.source_tag_mode,
            limit=limit,
            per_sid_max_chunks=max(self.max_merged_chunks, 1),
        )

    @staticmethod
    def _role_of(msg) -> Optional[str]:
        if isinstance(msg, dict):
            return msg.get("role")
        return getattr(msg, "role", None)

    def _apply_sync(self, event, req: LLMRequest) -> bool:
        t0 = time.perf_counter()
        sid = getattr(event, "sid", None) or ""
        if not sid or not self.should_merge(sid):
            return False

        merged = self.build_merged_history(sid)

        systems = []
        for m in list(req.messages or []):
            if self._role_of(m) == "system":
                if isinstance(m, OpenAIMessage):
                    systems.append(m)
                elif isinstance(m, dict):
                    try:
                        systems.append(OpenAIMessage(**m))
                    except Exception:
                        pass
                else:
                    systems.append(m)

        req.messages.clear()
        req.messages.extend(list(systems) + list(merged))

        inject_window_anchor(
            req,
            event,
            self.window_anchor_prompt,
            enabled=self.enable_window_anchor,
        )

        self._log(
            "[MERGER] applied sid=%s msgs=%d elapsed=%.3fs",
            sid,
            len(req.messages),
            time.perf_counter() - t0,
        )
        return True

    async def apply_to_request(self, event, req: LLMRequest) -> bool:
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._apply_sync, event, req),
                timeout=self.MERGE_TIMEOUT_SEC if self.merge_reset_mode == "soft" else 8.0,
            )
        except asyncio.TimeoutError:
            if self.logger:
                self.logger.warning("[MERGER] merge timeout, skip merge")
            return False
        except Exception:
            if self.logger:
                self.logger.exception("[MERGER] apply failed, skip merge")
            return False
