from __future__ import annotations

import asyncio
import random
import time
from typing import List, Optional

from core.agent.message import OpenAIMessage
from core.provider import LLMRequest

from .anchor import inject_window_anchor
from .group_resolver import GroupResolver
from .hard_reset import compute_dropped_flat, hard_reset_members
from .memory_access import pick_member_sids
from .observe_pool import ObservePool
from .reset_policy import (
    SoftResetState,
    apply_round_cap,
    apply_soft_trim,
    count_messages_tokens,
)
from .summarizer import build_summary_chunk, summarize_history
from .timeline import TimelineBuilder


class MergeEngine:
    """读时合并 + 软/硬重启 + 未唤醒偷看。"""

    def __init__(
        self,
        session_mgr,
        resolver: GroupResolver,
        timeline: TimelineBuilder,
        observe_pool: Optional[ObservePool] = None,
        max_merged_chunks: int = 10,
        merge_token_limit: int = 30000,
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
        merge_build_timeout_sec: float = 5.0,
        ctx=None,
        summarize_mode: str = "off",
        summarize_model: str = "",
        summarize_timeout_sec: float = 5.0,
        summarize_max_input_chars: int = 6000,
        summarize_max_output_chars: int = 3000,
        summarize_prompt_template: str = "",
        enable_summary_logging: bool = False,
        debug: bool = False,
        log_preview: bool = False,
        logger=None,
    ):
        self.session_mgr = session_mgr
        self.resolver = resolver
        self.timeline = timeline
        self.observe_pool = observe_pool
        self.max_merged_chunks = max(1, int(max_merged_chunks or 10))
        self.merge_token_limit = max(0, int(merge_token_limit or 30000))
        self.merge_keep_turns = max(1, int(merge_keep_turns or 6))
        self.merge_reset_mode = (merge_reset_mode or "soft").strip().lower()
        if self.merge_reset_mode not in ("soft", "hard"):
            self.merge_reset_mode = "soft"
        self.merge_build_timeout_sec = max(1.0, float(merge_build_timeout_sec or 5.0))
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
        # 重开前摘要（仅 hard 模式生效）
        self.ctx = ctx
        self.summarize_mode = (summarize_mode or "off").strip().lower()
        if self.summarize_mode not in ("off", "sync", "async"):
            self.summarize_mode = "off"
        self.summarize_model = str(summarize_model or "")
        self.summarize_timeout_sec = float(summarize_timeout_sec or 5.0)
        # 0 是合法值（无上限），不能用 `or 默认值`
        self.summarize_max_input_chars = (
            int(summarize_max_input_chars)
            if summarize_max_input_chars is not None
            else 6000
        )
        self.summarize_max_output_chars = (
            int(summarize_max_output_chars)
            if summarize_max_output_chars is not None
            else 3000
        )
        self.summarize_prompt_template = str(summarize_prompt_template or "")
        self.enable_summary_logging = bool(enable_summary_logging)
        # async 补写任务：sid -> Task
        self._summary_tasks: dict = {}
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

    def build_merged_history(
        self,
        current_sid: str,
        summary_chunks: Optional[dict] = None,
        out_info: Optional[dict] = None,
    ) -> List[OpenAIMessage]:
        """
        流程：
        1. 拼合并历史（最多 max_merged_chunks 轮）
        2. 估 token；未超限 → 直接用
        3. 超限 → 仅此时可能 on_reset（减半/重置 keep），并：
           - soft：合并视图截到 keep 轮
           - hard：各 sid 磁盘重开 keep 轮 + 合并视图也截到 keep 轮
        keep 减半只发生在「确认超限且执行重开」时。

        summary_chunks: sid -> summary_chunk（重开前摘要，仅 hard 重开时写入）。
        out_info: 若传入 dict，hard 重开实际执行时回填 {"hard_reset_sids": [...]}。
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
                    self.session_mgr,
                    members,
                    keep,
                    logger=self.logger,
                    summary_chunks=summary_chunks,
                )
                if out_info is not None:
                    out_info["hard_reset_sids"] = list(members)
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
            # 硬重开写入的摘要在触发本轮也应可见：
            # 轮数截断从尾部保留，首条摘要 chunk 可能被截掉，这里补回（防重）
            if summary_chunks:
                prepend = []
                for _sid, chunk in (summary_chunks or {}).items():
                    for d in chunk or []:
                        c = str(d.get("content", "") or "")
                        if c and not any(
                            c in str(getattr(m, "content", "") or "") for m in msgs
                        ):
                            prepend.append(OpenAIMessage(role="user", content=c))
                if prepend:
                    msgs = prepend + list(msgs)
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

    def _apply_sync(
        self,
        event,
        req: LLMRequest,
        summary_chunks: Optional[dict] = None,
        out_info: Optional[dict] = None,
    ) -> bool:
        t0 = time.perf_counter()
        sid = getattr(event, "sid", None) or ""
        if not sid or not self.should_merge(sid):
            return False

        merged = self.build_merged_history(
            sid, summary_chunks=summary_chunks, out_info=out_info
        )

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

    def _merge_timeout(self) -> float:
        """soft 用配置值；hard 至少 8s（磁盘重开更慢）。"""
        base = float(self.merge_build_timeout_sec or 5.0)
        if self.merge_reset_mode == "hard":
            return max(base, 8.0)
        return max(1.0, base)

    # ── 硬重开前摘要 ─────────────────────────────────────────

    def _estimate_group_tokens(self, members: List[str]) -> int:
        """
        只读粗估合并组 token（近似 _build_raw_merged 的输入量）。
        与真实构建存在偏差，仅用于决定「要不要提前生成摘要」；
        偏差最坏结果 = 多/少一次摘要调用，不影响重开正确性。
        """
        from .memory_access import safe_fetch_memory

        total = 0
        cpt = max(0.1, float(self.chars_per_token or 2.0))
        per_sid = max(self.max_merged_chunks, 1)
        for sid in members:
            try:
                flat = safe_fetch_memory(self.session_mgr, sid, max_chunks=per_sid)
            except Exception:
                continue
            for m in flat:
                content = m.get("content")
                if isinstance(content, str) and content:
                    total += int(len(content) / cpt) + 1
                total += 4
                tcs = m.get("tool_calls")
                if tcs:
                    total += int(len(str(tcs)) / cpt) + 1
        return total

    def precheck_hard_reset(self, current_sid: str) -> Optional[dict]:
        """
        只读预检：本轮 apply 是否大概率触发 hard 重开。
        是 → 返回 {"keep": int, "dropped_map": {sid: dropped_flat}}；否 → None。
        不更新 SoftResetState（真正的状态更新仍在 build_merged_history 内）。
        """
        if self.merge_reset_mode != "hard" or self.summarize_mode == "off":
            return None
        if self.merge_token_limit <= 0:
            return None
        group_id = self._group_id(current_sid)
        if not self._soft_state.peek_should_check(group_id):
            return None
        members = self._member_sids(current_sid)
        if self._estimate_group_tokens(members) <= self.merge_token_limit:
            return None
        keep = self._soft_state.peek_reset_keep(group_id)
        dropped_map = {}
        for sid in members:
            dropped = compute_dropped_flat(self.session_mgr, sid, keep)
            if dropped:
                dropped_map[sid] = dropped
        if not dropped_map:
            return None
        return {"keep": keep, "dropped_map": dropped_map}

    async def _summarize_members(self, dropped_map: dict, group_id: str) -> dict:
        """
        并发为各 sid 生成摘要;整体受 summarize_timeout_sec 约束。
        连续降级时优先复用各 sid 首条已有摘要，不重复调 LLM。
        """
        from .summarizer import is_summary_chunk
        from .memory_access import safe_read_memory

        sids = list(dropped_map.keys())
        reused = {}
        needs_llm = []

        # 检查降级窗口（与 peek_reset_keep 里的 half_window 一致）
        now = time.time()
        last_reset = self._soft_state._last_reset.get(group_id, 0)
        half_window = (
            self._soft_state.check_interval / 2
            if self._soft_state.check_interval > 0
            else 30.0
        )
        in_degrade_window = last_reset and (now - last_reset) < half_window

        if self.enable_summary_logging and self.logger:
            self.logger.info(
                "[摘要调试] 降级窗口检查: in_window=%s, 间隔=%.2fs, half_window=%.2fs",
                in_degrade_window, now - last_reset if last_reset else -1, half_window
            )

        if in_degrade_window:
            # 窗口内，尝试从各 sid 首条提取已有摘要复用
            for sid in sids:
                try:
                    chunks = safe_read_memory(self.session_mgr, sid)
                    if chunks and is_summary_chunk(chunks[0]):
                        # 首条是摘要，提取文本
                        first_msg = chunks[0][0] if chunks[0] else None
                        if first_msg and first_msg.get("role") == "user":
                            content = first_msg.get("content", "")
                            lines = content.split("\n", 1)
                            if len(lines) > 1 and content.startswith("[前情摘要|系统注入]"):
                                summary_text = lines[1].strip()
                                if summary_text:
                                    reused[sid] = build_summary_chunk(summary_text)
                                    if self.logger:
                                        self.logger.info(
                                            f"♻️ [MERGER] 连续降级，复用 {sid} 上次摘要"
                                        )
                                    if self.enable_summary_logging and self.logger:
                                        self.logger.info(f"[摘要调试] 复用 {sid} 的摘要: {summary_text[:100]}...")
                                    continue
                except Exception:
                    pass
                needs_llm.append(sid)
        else:
            needs_llm = sids

        if self.enable_summary_logging and self.logger:
            self.logger.info(
                "[摘要调试] 复用 %d 个摘要，需要 LLM 的 %d 个: %s",
                len(reused), len(needs_llm), needs_llm
            )

        # 剩余需要 LLM 的
        async def _one(sid: str):
            return await summarize_history(
                self.ctx,
                sid,
                dropped_map[sid],
                model_id=self.summarize_model,
                prompt_template=self.summarize_prompt_template,
                timeout_sec=self.summarize_timeout_sec,
                max_input_chars=self.summarize_max_input_chars,
                max_output_chars=self.summarize_max_output_chars,
                logger=self.logger,
                enable_detail_log=self.enable_summary_logging,
            )

        if needs_llm:
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*[_one(s) for s in needs_llm], return_exceptions=True),
                    timeout=max(1.0, self.summarize_timeout_sec + 2.0),
                )
            except asyncio.TimeoutError:
                if self.logger:
                    self.logger.warning("[MERGER summary] gather timeout, skip new summaries")
                results = []
            for sid, r in zip(needs_llm, results):
                if isinstance(r, str) and r:
                    reused[sid] = build_summary_chunk(r)

        return reused

    def _schedule_async_summaries(self, dropped_map: dict, group_id: str):
        """
        async 模式：hard 重开已完成，后台生成摘要并补写各 sid 记忆首部。
        连续降级时优先复用已有摘要。
        """
        from .summarizer import is_summary_chunk
        from .memory_access import safe_read_memory

        # 检查降级窗口
        now = time.time()
        last_reset = self._soft_state._last_reset.get(group_id, 0)
        half_window = (
            self._soft_state.check_interval / 2
            if self._soft_state.check_interval > 0
            else 30.0
        )
        in_degrade_window = last_reset and (now - last_reset) < half_window

        if self.enable_summary_logging and self.logger:
            self.logger.info(
                "[摘要调试] async 降级窗口检查: in_window=%s, 间隔=%.2fs",
                in_degrade_window, now - last_reset if last_reset else -1
            )

        for sid, dropped in dropped_map.items():
            old = self._summary_tasks.get(sid)
            if old and not old.done():
                old.cancel()

            async def _run(sid=sid, dropped=dropped):
                try:
                    # 先尝试复用
                    reused = None
                    if in_degrade_window:
                        try:
                            chunks = safe_read_memory(self.session_mgr, sid)
                            if chunks and is_summary_chunk(chunks[0]):
                                first_msg = chunks[0][0] if chunks[0] else None
                                if first_msg and first_msg.get("role") == "user":
                                    content = first_msg.get("content", "")
                                    lines = content.split("\n", 1)
                                    if len(lines) > 1 and content.startswith("[前情摘要|系统注入]"):
                                        reused = lines[1].strip()
                                        if reused and self.logger:
                                            self.logger.info(
                                                f"♻️ [MERGER async] 连续降级，复用 {sid} 上次摘要"
                                            )
                                        if reused and self.enable_summary_logging and self.logger:
                                            self.logger.info(f"[摘要调试] async 复用 {sid}: {reused[:100]}...")
                        except Exception:
                            pass

                    if reused:
                        summary = reused
                    else:
                        if self.enable_summary_logging and self.logger:
                            self.logger.info(f"[摘要调试] async 后台生成 {sid} 摘要")
                        summary = await summarize_history(
                            self.ctx,
                            sid,
                            dropped,
                            model_id=self.summarize_model,
                            prompt_template=self.summarize_prompt_template,
                            timeout_sec=self.summarize_timeout_sec,
                            max_input_chars=self.summarize_max_input_chars,
                            max_output_chars=self.summarize_max_output_chars,
                            logger=self.logger,
                            enable_detail_log=self.enable_summary_logging,
                        )

                    if not summary or not self.session_mgr:
                        return
                    chunks = self.session_mgr.read_memory(sid) or []
                    if chunks and is_summary_chunk(chunks[0]):
                        return
                    new_chunks = [build_summary_chunk(summary)] + list(chunks)
                    self.session_mgr.write_memory(sid, new_chunks)
                    if self.logger:
                        self.logger.info(
                            "[MERGER summary] async summary written for %s", sid
                        )
                except asyncio.CancelledError:
                    pass
                except Exception:
                    if self.logger:
                        self.logger.exception(
                            "[MERGER summary] async write failed sid=%s", sid
                        )
                finally:
                    self._summary_tasks.pop(sid, None)

            self._summary_tasks[sid] = asyncio.create_task(_run())

    def cancel_summary_tasks(self):
        for t in list(self._summary_tasks.values()):
            if t and not t.done():
                t.cancel()
        self._summary_tasks.clear()

    async def manual_reset_with_summary(self, current_sid: str) -> dict:
        """
        压缩重开命令触发：立即对当前会话所在合并组执行「摘要 + 硬重开」。
        与 merge_reset_mode 无关（soft 模式下也可手动硬重开）；
        摘要仍受 summarize_mode 控制（off = 只重开不摘要）。
        返回 {"members", "keep", "ok", "fail", "summarized"}。
        """
        members = self._member_sids(current_sid)
        group_id = self._group_id(current_sid)
        keep = max(1, int(self.merge_keep_turns or 1))

        dropped_map = {}
        for s in members:
            dropped = compute_dropped_flat(self.session_mgr, s, keep)
            if dropped:
                dropped_map[s] = dropped

        summary_chunks: dict = {}
        if self.summarize_mode != "off" and dropped_map:
            try:
                summary_chunks = await self._summarize_members(dropped_map, group_id)
            except Exception:
                if self.logger:
                    self.logger.exception("[MERGER summary] manual summarize failed")

        results = hard_reset_members(
            self.session_mgr,
            members,
            keep,
            logger=self.logger,
            summary_chunks=summary_chunks or None,
        )
        ok = sum(1 for v in results.values() if isinstance(v, int) and v >= 0)
        fail = len(results) - ok

        # 手动重开后：清除组级动态 keep，记录重开时间（节流自动重开）
        try:
            self._soft_state._dynamic_keep.pop(group_id, None)
            self._soft_state._last_reset[group_id] = time.time()
            self._soft_state._last_check[group_id] = time.time()
        except Exception:
            pass

        if self.logger:
            self.logger.warning(
                "[MERGER] manual reset+summary by cmd | group=%s keep=%d ok=%d fail=%d summarized=%d",
                group_id, keep, ok, fail, len(summary_chunks),
            )
        return {
            "members": members,
            "keep": keep,
            "ok": ok,
            "fail": fail,
            "summarized": len(summary_chunks),
        }

    async def apply_to_request(self, event, req: LLMRequest) -> bool:
        # 硬重开前摘要预检（只读，不改状态；失败不影响主流程）
        summary_chunks: Optional[dict] = None
        pending_async_dropped: Optional[dict] = None
        try:
            sid = getattr(event, "sid", None) or ""
            if sid and self.should_merge(sid):
                pre = self.precheck_hard_reset(sid)
                if pre:
                    if self.enable_summary_logging and self.logger:
                        self.logger.info(
                            "[摘要调试] 预检判定需要重开，dropped_map: %s", list(pre["dropped_map"].keys())
                        )
                    group_id = self._group_id(sid)
                    if self.summarize_mode == "sync":
                        if self.enable_summary_logging and self.logger:
                            self.logger.info("[摘要调试] 进入 sync 模式，开始并发生成摘要")
                        summary_chunks = await self._summarize_members(
                            pre["dropped_map"], group_id
                        )
                        if self.enable_summary_logging and self.logger:
                            self.logger.info("[摘要调试] sync 摘要完成，成功生成: %s", list(summary_chunks.keys()) if summary_chunks else [])
                    elif self.summarize_mode == "async":
                        if self.enable_summary_logging and self.logger:
                            self.logger.info("[摘要调试] 进入 async 模式，先捕获 dropped 数据，重开后补写")
                        # dropped 必须在重开删数据前捕获；摘要在重开后补写
                        pending_async_dropped = pre["dropped_map"]
                else:
                    if self.enable_summary_logging and self.logger:
                        self.logger.info("[摘要调试] 预检判定无需重开")
        except Exception:
            if self.logger:
                self.logger.exception("[MERGER summary] precheck failed (swallowed)")

        out_info: dict = {}
        try:
            ok = await asyncio.wait_for(
                asyncio.to_thread(
                    self._apply_sync, event, req, summary_chunks, out_info
                ),
                timeout=self._merge_timeout(),
            )
        except asyncio.TimeoutError:
            if self.logger:
                self.logger.warning(
                    "[MERGER] merge timeout (%.1fs mode=%s), skip merge",
                    self._merge_timeout(),
                    self.merge_reset_mode,
                )
            return False
        except Exception:
            if self.logger:
                self.logger.exception("[MERGER] apply failed, skip merge")
            return False

        # async 摘要：仅对实际发生 hard 重开的 sid 补写
        if pending_async_dropped:
            reset_sids = set(out_info.get("hard_reset_sids") or [])
            if self.enable_summary_logging and self.logger:
                self.logger.info("[摘要调试] async 模式实际重开的 sid: %s", list(reset_sids))
            if reset_sids:
                to_summarize = {
                    s: d for s, d in pending_async_dropped.items() if s in reset_sids
                }
                if to_summarize:
                    if self.enable_summary_logging and self.logger:
                        self.logger.info("[摘要调试] 调度后台摘要任务: %s", list(to_summarize.keys()))
                    sid = getattr(event, "sid", None) or ""
                    group_id = self._group_id(sid)
                    self._schedule_async_summaries(to_summarize, group_id)
        return ok
