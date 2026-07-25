from __future__ import annotations

import asyncio
import random
import threading
import time
from typing import Dict, List, Optional, Tuple

from core.agent.message import OpenAIMessage
from core.provider import LLMRequest

from .anchor import inject_window_anchor
from .group_resolver import GroupResolver
from .hard_reset import (
    hard_reset_members,
    read_reset_parts,
)
from .memory_access import pick_member_sids
from .observe_pool import ObservePool
from .reset_policy import (
    SoftResetState,
    apply_round_cap,
    apply_soft_trim,
    count_messages_tokens,
)
from .summarizer import (
    SUMMARY_MARKER,
    build_summary_chunk,
    dropped_fingerprint,
    extract_summary_text,
    is_summary_chunk,
    merge_summaries,
    self_compress_summary,
    summarize_history,
)
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
        summarize_timeout_sec: float = 60.0,
        summarize_max_input_chars: int = 10000,
        summarize_max_output_chars: int = 5000,
        summarize_prompt_template: str = "",
        enable_summary_logging: bool = False,
        summary_store=None,
        cumulative_summary: bool = True,
        merge_prompt_template: str = "",
        self_compress_prompt_template: str = "",
        write_through: bool = True,
        merge_order_mode: str = "time",
        merge_trigger_mode: str = "tokens",
        merge_trigger_rounds: int = 0,
        preprocess_tools: bool = True,
        tool_max_chars: int = 2000,
        preprocess_in_sync: bool = False,
        merge_timeout_sec: float = 120.0,
        preheat_ratio: float = 0.7,
        sync_wait_timeout: float = 0.0,
        continuous_merge_strategy: str = "append_then_merge",
        background_merge_max_concurrent: int = 5,
        background_merge_timeout_sec: float = 120.0,
        replace_concat_after_merge: bool = True,
        background_merge_retry_interval_sec: float = 30.0,
        background_merge_max_retries: int = 3,
        max_concat_summary_chars: int = 5000,
        concat_overflow_strategy: str = "self_compress",
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
        self.summarize_timeout_sec = float(summarize_timeout_sec or 60.0)
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
        # 累积摘要（借鉴 CCS：旧摘要 + 新增量合并，不再层层有损叠加）
        self.summary_store = summary_store
        self.cumulative_summary = bool(cumulative_summary)
        self.merge_prompt_template = str(merge_prompt_template or "")
        self.self_compress_prompt_template = str(self_compress_prompt_template or "")
        # 写穿式硬重开（不 delete_session，保留会话元信息）
        self.write_through = bool(write_through)
        # 合并排序模式：time（时间交错）/ session_blocks（按会话分块，当前会话最后）
        self.merge_order_mode = (merge_order_mode or "time").strip().lower()
        if self.merge_order_mode not in ("time", "session_blocks"):
            self.merge_order_mode = "time"
        # 双触发：tokens=合并视图估算超限；rounds=成员磁盘轮数对齐框架窗口
        self.merge_trigger_mode = (merge_trigger_mode or "tokens").strip().lower()
        if self.merge_trigger_mode not in ("tokens", "rounds", "either"):
            self.merge_trigger_mode = "tokens"
        self.merge_trigger_rounds = max(0, int(merge_trigger_rounds or 0))
        # 摘要输入预处理（超长 tool 结果/图片描述先压缩再摘要）
        self.preprocess_tools = bool(preprocess_tools)
        self.tool_max_chars = max(200, int(tool_max_chars or 2000))
        # sync 关键路径默认不做预处理（可能耗时数分钟），只留在 async/手动路径
        self.preprocess_in_sync = bool(preprocess_in_sync)
        # 合并/自压缩独立超时（超时快速退化拼接，避免阻塞回复）
        self.merge_timeout_sec = max(1.0, float(merge_timeout_sec or 120.0))
        # 持续后台压缩：阈值与 sync 等待时间
        self.preheat_ratio = max(0.0, min(1.0, float(preheat_ratio if preheat_ratio is not None else 0.7)))
        self.sync_wait_timeout = max(0.0, float(sync_wait_timeout if sync_wait_timeout is not None else 0.0))
        # 持续压缩合并策略
        self.continuous_merge_strategy = str(continuous_merge_strategy or "append_then_merge").lower()
        if self.continuous_merge_strategy not in ("immediate", "append_then_merge", "append_only"):
            self.continuous_merge_strategy = "append_then_merge"
        self.background_merge_max_concurrent = max(1, int(background_merge_max_concurrent or 5))
        self.background_merge_timeout_sec = max(1.0, float(background_merge_timeout_sec or 120.0))
        self.replace_concat_after_merge = bool(replace_concat_after_merge if replace_concat_after_merge is not None else True)
        self.background_merge_retry_interval_sec = max(1.0, float(background_merge_retry_interval_sec or 30.0))
        self.background_merge_max_retries = max(0, int(background_merge_max_retries if background_merge_max_retries is not None else 3))
        self.max_concat_summary_chars = max(500, int(max_concat_summary_chars if max_concat_summary_chars is not None else 5000))
        self.concat_overflow_strategy = str(concat_overflow_strategy or "self_compress").lower()
        if self.concat_overflow_strategy not in ("self_compress", "truncate_fifo", "none"):
            self.concat_overflow_strategy = "self_compress"
        # per-group 重开锁：手动命令与自动 hard 重开互斥（threading.Lock，
        # 因为自动重开发生在 to_thread 的 _apply_sync 里）
        self._reset_locks: Dict[str, threading.Lock] = {}
        # 上次重开后视图仍超预算标记：驱动下次重开降级 keep（比旧时间窗可靠）
        self._still_over: Dict[str, bool] = {}
        # async 补写任务：sid -> Task
        self._summary_tasks: dict = {}
        # 持续后台压缩：sid -> Task / 结果暂存
        self._preheat_tasks: dict = {}
        self._preheat_pending: dict = {}
        # 后台合并/自压缩任务：sid -> Task
        self._background_merge_tasks: dict = {}
        # 后台合并成功后等待替换到框架记忆的摘要：sid -> (old_final, new_final)
        self._pending_replace: dict = {}
        # 限制并发后台压缩任务数，避免多个合并组成员同时抢 LLM 导致彼此超时
        self._preheat_sem = asyncio.Semaphore(self.background_merge_max_concurrent)
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

    def _get_reset_lock(self, group_id: str) -> threading.Lock:
        if group_id not in self._reset_locks:
            self._reset_locks[group_id] = threading.Lock()
        return self._reset_locks[group_id]

    def _rounds_limit(self) -> int:
        """rounds 触发的轮数上限：配置优先，0 = 对齐框架 max_memory_length。"""
        if self.merge_trigger_rounds > 0:
            return self.merge_trigger_rounds
        return max(1, int(getattr(self.session_mgr, "max_memory_length", 10) or 10))

    def _members_rounds_over(self, members: List[str]) -> bool:
        """任一成员会话磁盘轮数达到框架窗口（rounds 触发，防止框架直接截断丢信息）。"""
        if self.merge_trigger_mode not in ("rounds", "either"):
            return False
        from .memory_access import safe_session_meta

        limit = self._rounds_limit()
        for sid in members:
            try:
                if int(safe_session_meta(self.session_mgr, sid).get("chunk_count", 0)) >= limit:
                    return True
            except Exception:
                continue
        return False

    def _clamp_keep(self, keep: int) -> int:
        """钳制：rounds 触发下 keep 必须 < 窗口，否则重开后下一次 append
        会被框架抢先截掉融合头部（摘要丢失 + 前缀缓存逐轮失效）。"""
        keep = max(1, int(keep or 1))
        if self.merge_trigger_mode in ("rounds", "either"):
            cap = max(1, self._rounds_limit() - 1)
            if keep > cap:
                if self.logger:
                    self.logger.warning(
                        "[MERGER] keep=%d >= rounds 窗口 %d，钳制为 %d 防止框架截断头部摘要",
                        keep, self._rounds_limit(), cap,
                    )
                keep = cap
        return keep

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
        """按 max_merged_chunks 取最近轮，不做 token 重开。

        session_blocks 模式：成员块按 sid 字典序固定、当前会话块恒在最后，
        其他成员沉默时整个前缀逐字节稳定，利于命中模型上下文缓存。
        """
        if self.merge_order_mode == "session_blocks" and len(members) > 1:
            others = sorted(m for m in members if m != current_sid)
            members = others + [current_sid] if current_sid in members else others
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
        tokens_over = (
            self.merge_trigger_mode in ("tokens", "either")
            and self.merge_token_limit > 0
            and tokens > self.merge_token_limit
        )
        rounds_over = self._members_rounds_over(members)
        over = tokens_over or rounds_over

        if not over:
            # 未超限：只保留 max_merged_chunks，绝不动 keep 状态
            self._still_over[group_id] = False
            msgs = apply_round_cap(msgs, self.max_merged_chunks)
        else:
            # 超限：是否允许本轮「正式重开」（更新 keep / 硬删）
            allow_reset = self._soft_state.should_check(group_id)
            if allow_reset:
                # 仅在真正重开时更新动态 keep；degrade 信号 =
                # 上次重开后视图仍超预算（比旧 30s 时间窗可靠）
                keep = self._soft_state.on_reset(
                    group_id, degrade=self._still_over.get(group_id, False)
                )
            else:
                # 检测间隔内：不减半，沿用当前 keep，仍截断合并视图
                keep = self._soft_state.current_keep(group_id)
            keep = self._clamp_keep(keep)

            if self.merge_reset_mode == "hard" and allow_reset:
                # 风暴守卫：只重置「轮数 > keep」的成员；全部已在 keep 内时
                # 跳过硬重开（否则空转重写记忆+重写摘要，白白毁掉前缀缓存）
                from .memory_access import safe_session_meta

                reset_members = []
                for _s in members:
                    try:
                        if int(safe_session_meta(self.session_mgr, _s).get("chunk_count", 0)) > keep:
                            reset_members.append(_s)
                    except Exception:
                        reset_members.append(_s)
                if not reset_members:
                    self._log(
                        "[MERGER hard] token=%d > %d 但所有成员已在 keep=%d 轮内，跳过硬重开",
                        tokens,
                        self.merge_token_limit,
                        keep,
                    )
                else:
                    self._log(
                        "[MERGER hard] token=%d > %d, hard-reset %d/%d members keep=%d",
                        tokens,
                        self.merge_token_limit,
                        len(reset_members),
                        len(members),
                        keep,
                    )
                    sub_summaries = (
                        {s: c for s, c in summary_chunks.items() if s in reset_members}
                        if summary_chunks
                        else None
                    )
                    # per-group 锁：与手动压缩重开互斥，避免并发双写
                    with self._get_reset_lock(group_id):
                        hard_reset_members(
                            self.session_mgr,
                            reset_members,
                            keep,
                            logger=self.logger,
                            summary_chunks=sub_summaries,
                            write_through=self.write_through,
                        )
                    if out_info is not None:
                        out_info["hard_reset_sids"] = list(reset_members)
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
            # 记录截断后视图是否仍超预算：驱动下次重开降级 keep（5→2→1 收敛）
            self._still_over[group_id] = (
                self.merge_token_limit > 0 and tokens_after > self.merge_token_limit
            )
            if self._still_over[group_id]:
                self._log(
                    "[MERGER] 视图截到 keep=%d 后 token≈%d 仍超 %d，下次重开将降级 keep",
                    keep,
                    tokens_after,
                    self.merge_token_limit,
                )
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


    # ── 持续后台压缩 ───────────────────────────────────────────────

    def _should_compress_continuously(self, sid: str) -> bool:
        """当前会话是否已超过阈值，需要启动持续后台压缩。"""
        if self.preheat_ratio <= 0 or self.summarize_mode == "off" or not self.session_mgr:
            return False
        from .memory_access import safe_fetch_memory, safe_session_meta

        if self.merge_trigger_mode in ("rounds", "either"):
            meta = safe_session_meta(self.session_mgr, sid)
            rounds = int(meta.get("chunk_count", 0))
            rounds_limit = self._rounds_limit()
            if rounds_limit > 0 and rounds >= rounds_limit * self.preheat_ratio:
                return True
        if self.merge_trigger_mode in ("tokens", "either"):
            flat = safe_fetch_memory(self.session_mgr, sid) or []
            cpt = max(0.1, float(self.chars_per_token or 2.0))
            total = 0
            for m in flat:
                content = m.get("content")
                if isinstance(content, str) and content:
                    total += int(len(content) / cpt) + 1
                total += 4
                tcs = m.get("tool_calls")
                if tcs:
                    total += int(len(str(tcs)) / cpt) + 1
            if self.merge_token_limit > 0 and total >= self.merge_token_limit * self.preheat_ratio:
                return True
        return False

    def _schedule_continuous_compression(self, sid: str):
        """启动或维持后台持续压缩任务。"""
        if self.preheat_ratio <= 0 or self.summarize_mode == "off":
            return
        task = self._preheat_tasks.get(sid)
        if task is not None and not task.done():
            return
        self._preheat_tasks[sid] = asyncio.create_task(self._continuous_compress_run(sid))

    async def _continuous_compress_run(self, sid: str):
        """持续后台压缩：增量追加新 dropped 消息，避免每轮都重算整段历史。

        只读记忆 + LLM 调用，绝不写会话记忆；失败静默，下一轮继续。
        思路借鉴 CCS：把累计摘要拆成片段列表，优先两两 pair merge，落单先拼接；
        触发重开时直接拿当前 final 用，后台再慢慢把拼接部分改写成合并版。
        """
        try:
            async with self._preheat_sem:
                if self.session_mgr is None:
                    return
                parts = read_reset_parts(self.session_mgr, sid, self.merge_keep_turns)
                head_text = parts.get("head_text", "")
                dropped = parts.get("dropped", [])
                if not dropped:
                    self._preheat_pending.pop(sid, None)
                    return
                group_id = self._group_id(sid)
                base = self._resolve_base(sid, group_id, head_text)
                eff = self._effective_dropped(head_text, dropped, base)
                fp = dropped_fingerprint(eff)
                pending = self._preheat_pending.get(sid)

                # 已是最新：无需再压缩；append_then_merge 可趁机推进后台合并
                if pending and pending.get("fp") == fp and (pending.get("base") or "") == (base or ""):
                    if self.continuous_merge_strategy == "append_then_merge":
                        if len(pending.get("parts", [])) > 1:
                            self._schedule_background_merge(sid)
                    return

                # 判断能否增量：base 一致且已压缩长度 < 当前 dropped 长度
                is_incremental = False
                prev_parts = []
                delta_input = eff
                if pending and (pending.get("base") or "") == (base or ""):
                    prev_len = int(pending.get("compressed_len", 0))
                    if 0 < prev_len < len(dropped):
                        is_incremental = True
                        prev_parts = list(pending.get("parts", []))
                        delta_input = dropped[prev_len:]

                delta = await summarize_history(
                    self.ctx,
                    sid,
                    delta_input,
                    model_id=self.summarize_model,
                    prompt_template=self.summarize_prompt_template,
                    timeout_sec=self.summarize_timeout_sec,
                    max_input_chars=self.summarize_max_input_chars,
                    max_output_chars=self.summarize_max_output_chars,
                    logger=self.logger,
                    enable_detail_log=self.enable_summary_logging,
                    preprocess_tools=self.preprocess_tools,
                    tool_max_chars=self.tool_max_chars,
                )
                if not delta:
                    return

                delta = delta.strip()
                parts_list = list(prev_parts) if is_incremental else []
                if not is_incremental and base.strip():
                    parts_list.append(base.strip())
                parts_list.append(delta)

                # 按策略做 pair merge
                if self.continuous_merge_strategy == "immediate":
                    # 尽量合并到只剩一个片段
                    parts_list = await self._pair_merge_parts(
                        sid, parts_list, max_attempts=0, timeout=self.background_merge_timeout_sec
                    )
                elif self.continuous_merge_strategy == "append_then_merge":
                    # 每轮最多合并一对，避免单次任务过重；落单交给后台
                    parts_list = await self._pair_merge_parts(
                        sid, parts_list, max_attempts=1, timeout=self.background_merge_timeout_sec
                    )

                final = self._concat_parts(parts_list)

                # 兜底：拼接/混合版过长时自压缩或截断
                if self.continuous_merge_strategy in ("append_then_merge", "append_only"):
                    if len(final) > self.max_concat_summary_chars:
                        if self.concat_overflow_strategy == "truncate_fifo":
                            parts_list = self._truncate_parts(parts_list)
                            final = self._concat_parts(parts_list)
                        elif self.concat_overflow_strategy == "self_compress":
                            self._schedule_self_compress(sid)

                self._preheat_pending[sid] = {
                    "fp": fp,
                    "final": final,
                    "base": base,
                    "compressed_len": len(dropped),
                    "parts": parts_list,
                    "background_tries": 0,
                }

                if self.continuous_merge_strategy == "append_then_merge" and len(parts_list) > 1:
                    self._schedule_background_merge(sid)

                if self.enable_summary_logging and self.logger:
                    self.logger.info(
                        "[摘要调试] [持续压缩] %s 摘要已更新 (%d 字符, %s, 策略=%s, 片段=%d)",
                        sid,
                        len(final),
                        "增量" if is_incremental else "全量",
                        self.continuous_merge_strategy,
                        len(parts_list),
                    )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if self.enable_summary_logging and self.logger:
                self.logger.info("[摘要调试] [持续压缩] %s 异常（静默）: %s", sid, e)
        finally:
            self._preheat_tasks.pop(sid, None)

    def _concat_parts(self, parts):
        """把摘要片段列表拼接成最终文本（过滤空片段）。"""
        return "\n".join(p.strip() for p in parts if p and p.strip())

    async def _merge_two(self, sid, a, b, timeout=None):
        """合并两个摘要片段；失败返回 None。"""
        a = (a or "").strip()
        b = (b or "").strip()
        if not a:
            return b or None
        if not b:
            return a or None
        return await merge_summaries(
            self.ctx,
            sid,
            a,
            b,
            model_id=self.summarize_model,
            prompt_template=self.merge_prompt_template,
            timeout_sec=timeout if timeout is not None else self.background_merge_timeout_sec,
            logger=self.logger,
            enable_detail_log=self.enable_summary_logging,
        )

    async def _pair_merge_parts(self, sid, parts, max_attempts=0, timeout=None):
        """对片段列表做相邻 pair merge。

        max_attempts=0 表示一直合到只剩一个或失败为止；>0 表示最多合并这么对。
        优先合并总长度最小的一对，降低单次 LLM 输入规模。
        """
        parts = [p.strip() for p in parts if p and p.strip()]
        if len(parts) <= 1:
            return parts
        attempts = 0
        limit = max_attempts if max_attempts > 0 else len(parts) * 2
        while len(parts) > 1 and attempts < limit:
            best_i = -1
            best_len = None
            for i in range(len(parts) - 1):
                pair_len = len(parts[i]) + len(parts[i + 1])
                if best_len is None or pair_len < best_len:
                    best_len = pair_len
                    best_i = i
            if best_i < 0:
                break
            merged = await self._merge_two(sid, parts[best_i], parts[best_i + 1], timeout)
            if not merged:
                break
            parts = parts[:best_i] + [merged.strip()] + parts[best_i + 2:]
            attempts += 1
        return parts

    def _truncate_parts(self, parts):
        """丢弃最旧的片段，直到拼接长度低于阈值。"""
        parts = [p.strip() for p in parts if p and p.strip()]
        while len(self._concat_parts(parts)) > self.max_concat_summary_chars and len(parts) > 1:
            parts = parts[1:]
        return parts

    async def _replace_summary_in_memory(self, sid, old_final, new_final, group_id):
        """如果框架记忆头部仍是 old_final，则替换为 new_final。"""
        if not new_final or not self.session_mgr:
            return False
        try:
            with self._get_reset_lock(group_id):
                chunks = self.session_mgr.read_memory(sid) or []
                cur_head = ""
                if chunks and is_summary_chunk(chunks[0]):
                    cur_head = extract_summary_text(
                        str(chunks[0][0].get("content", "") or "")
                    )
                if (cur_head or "").strip() != (old_final or "").strip():
                    return False
                summary_msg = build_summary_chunk(new_final)[0]
                if chunks and is_summary_chunk(chunks[0]):
                    chunks[0] = [summary_msg] + list(chunks[0][1:])
                elif chunks:
                    chunks[0] = [summary_msg] + list(chunks[0])
                else:
                    chunks = [[summary_msg]]
                self.session_mgr.write_memory(sid, chunks)
            self._update_store_after_reset([sid], {sid: new_final})
            if self.logger:
                self.logger.info(
                    "[MERGER summary] 后台合并替换摘要 sid=%s (%d -> %d 字符)",
                    sid, len(old_final), len(new_final)
                )
            return True
        except Exception:
            if self.logger:
                self.logger.exception("[MERGER summary] 后台替换失败 sid=%s", sid)
            return False

    def _schedule_background_merge(self, sid, delay=0.0):
        """安排后台合并/自压缩任务；同一 sid 同时只跑一个。"""
        task = self._background_merge_tasks.get(sid)
        if task is not None and not task.done():
            return
        pending = self._preheat_pending.get(sid)
        if pending and pending.get("background_tries", 0) >= self.background_merge_max_retries:
            return

        async def _run():
            if delay > 0:
                await asyncio.sleep(delay)
            await self._run_background_merge(sid)

        self._background_merge_tasks[sid] = asyncio.create_task(_run())

    async def _run_background_merge(self, sid):
        """后台把拼接/混合版逐步 pair merge；成功则替换框架记忆中的摘要。"""
        try:
            async with self._preheat_sem:
                pending = self._preheat_pending.get(sid)
                if not pending:
                    return
                parts = list(pending.get("parts", []))
                old_final = pending.get("final") or self._concat_parts(parts)

                # 如果只剩一个片段但超长，尝试自压缩
                if len(parts) <= 1:
                    if len(old_final) > self.max_concat_summary_chars and self.concat_overflow_strategy == "self_compress":
                        compressed = await self_compress_summary(
                            self.ctx,
                            sid,
                            old_final,
                            model_id=self.summarize_model,
                            prompt_template=self.self_compress_prompt_template,
                            timeout_sec=self.background_merge_timeout_sec,
                            logger=self.logger,
                            enable_detail_log=self.enable_summary_logging,
                        )
                        if compressed:
                            pending["parts"] = [compressed.strip()]
                            pending["final"] = compressed.strip()
                            self._preheat_pending[sid] = pending
                            if self.replace_concat_after_merge:
                                await self._replace_summary_in_memory(
                                    sid, old_final, compressed.strip(), self._group_id(sid)
                                )
                    return

                new_parts = await self._pair_merge_parts(
                    sid, parts, max_attempts=0, timeout=self.background_merge_timeout_sec
                )
                new_final = self._concat_parts(new_parts)
                if new_final != old_final:
                    pending["parts"] = new_parts
                    pending["final"] = new_final
                    pending["background_tries"] = 0
                    self._preheat_pending[sid] = pending
                    if self.replace_concat_after_merge:
                        await self._replace_summary_in_memory(
                            sid, old_final, new_final, self._group_id(sid)
                        )
                    # 还有片段未合并，继续安排下一轮
                    if len(new_parts) > 1:
                        self._schedule_background_merge(
                            sid, delay=self.background_merge_retry_interval_sec
                        )
                else:
                    # 没进展，计一次重试
                    pending["background_tries"] = pending.get("background_tries", 0) + 1
                    self._preheat_pending[sid] = pending
                    if pending["background_tries"] < self.background_merge_max_retries:
                        self._schedule_background_merge(
                            sid, delay=self.background_merge_retry_interval_sec
                        )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if self.enable_summary_logging and self.logger:
                self.logger.info("[摘要调试] [后台合并] %s 异常（静默）: %s", sid, e)
            # 失败也重试
            pending = self._preheat_pending.get(sid)
            if pending:
                pending["background_tries"] = pending.get("background_tries", 0) + 1
                self._preheat_pending[sid] = pending
                if pending["background_tries"] < self.background_merge_max_retries:
                    self._schedule_background_merge(
                        sid, delay=self.background_merge_retry_interval_sec
                    )
        finally:
            self._background_merge_tasks.pop(sid, None)

    def _schedule_self_compress(self, sid):
        """安排对过长拼接版的后台自压缩。"""
        task = self._background_merge_tasks.get(sid)
        if task is not None and not task.done():
            return
        async def _run():
            await self._run_self_compress(sid)
        self._background_merge_tasks[sid] = asyncio.create_task(_run())

    async def _run_self_compress(self, sid):
        """对超长 final 做 LLM 自压缩。"""
        try:
            async with self._preheat_sem:
                pending = self._preheat_pending.get(sid)
                if not pending:
                    return
                old_final = pending.get("final") or ""
                if len(old_final) <= self.max_concat_summary_chars:
                    return
                compressed = await self_compress_summary(
                    self.ctx,
                    sid,
                    old_final,
                    model_id=self.summarize_model,
                    prompt_template=self.self_compress_prompt_template,
                    timeout_sec=self.background_merge_timeout_sec,
                    logger=self.logger,
                    enable_detail_log=self.enable_summary_logging,
                )
                if compressed:
                    compressed = compressed.strip()
                    pending["parts"] = [compressed]
                    pending["final"] = compressed
                    self._preheat_pending[sid] = pending
                    if self.replace_concat_after_merge:
                        await self._replace_summary_in_memory(
                            sid, old_final, compressed, self._group_id(sid)
                        )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if self.enable_summary_logging and self.logger:
                self.logger.info("[摘要调试] [自压缩] %s 异常（静默）: %s", sid, e)
        finally:
            self._background_merge_tasks.pop(sid, None)

    async def post_reply_continuous_compress(self, current_sid: str):
        """每次回复后检查当前合并组成员是否需要继续后台压缩。"""
        if self.preheat_ratio <= 0 or self.summarize_mode == "off" or not self.session_mgr:
            return
        try:
            members = self._member_sids(current_sid)
            for sid in members:
                if self._should_compress_continuously(sid):
                    self._schedule_continuous_compression(sid)
        except Exception:
            pass

    async def _harvest_continuous_compression(
        self, sid: str, timeout: Optional[float] = None
    ) -> Optional[dict]:
        """sync 路径：等待并收割持续后台压缩结果。

        返回 pending dict 或 None；调用方负责 fp/base 校验。
        """
        timeout = timeout if timeout is not None else self.sync_wait_timeout
        task = self._preheat_tasks.get(sid)
        if task is None:
            return None
        if not task.done():
            if timeout <= 0:
                return self._preheat_pending.get(sid)
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                pass
        self._preheat_tasks.pop(sid, None)
        return self._preheat_pending.get(sid)

    def _preheat_valid_final(
        self, sid: str, dropped: List[dict], head_text: str, group_id: str
    ) -> Optional[str]:
        """校验 preheat_pending 是否与当前 dropped/base 匹配。"""
        pending = self._preheat_pending.get(sid)
        if not pending:
            return None
        base = self._resolve_base(sid, group_id, head_text)
        eff = self._effective_dropped(head_text, dropped, base)
        fp = dropped_fingerprint(eff)
        if pending.get("fp") == fp and (pending.get("base") or "") == (base or ""):
            return pending.get("final")
        return None

    async def _harvest_for_group_sync(
        self,
        dropped_map: dict,
        group_id: str,
        head_map: Optional[dict] = None,
        timeout: Optional[float] = None,
    ) -> Tuple[dict, dict]:
        """sync 模式：并发收割各成员持续压缩结果；未就绪的成员：
        - immediate：fallback 现场生成（仍可能阻塞 LLM）
        - append_then_merge / append_only：不现场生成，由调用方调度 async/后台补写
        """
        head_map = head_map or {}
        timeout = timeout if timeout is not None else self.sync_wait_timeout
        sids = list(dropped_map.keys())
        raw_results = await asyncio.gather(
            *[self._harvest_continuous_compression(sid, timeout=timeout) for sid in sids],
            return_exceptions=True,
        )
        finals: dict = {}
        missing: dict = {}
        for sid, raw in zip(sids, raw_results):
            if isinstance(raw, dict):
                final = self._preheat_valid_final(
                    sid, dropped_map[sid], head_map.get(sid, ""), group_id
                )
                if final:
                    finals[sid] = final
                    # append_then_merge 保留 pending 供后台继续合并拼接部分
                    if self.continuous_merge_strategy == "append_then_merge":
                        pending = self._preheat_pending.get(sid)
                        parts = list(pending.get("parts", [])) if pending else []
                        if len(parts) > 1:
                            self._schedule_background_merge(sid)
                    else:
                        self._preheat_pending.pop(sid, None)
                    if self.enable_summary_logging and self.logger:
                        self.logger.info("[摘要调试] sync 收割 %s 持续压缩摘要", sid)
                    continue
            missing[sid] = dropped_map[sid]
        if missing and self.continuous_merge_strategy == "immediate":
            more = await self._summarize_members(
                missing, group_id, head_map, preprocess=self.preprocess_in_sync
            )
            finals.update(more)
        return finals, missing

    def _write_summary_to_memory(self, sid: str, final: str, group_id: str):
        """将已生成的累计摘要写回会话记忆首部（带重开锁）。"""
        if not final or not self.session_mgr:
            return
        try:
            with self._get_reset_lock(group_id):
                chunks = self.session_mgr.read_memory(sid) or []
                summary_msg = build_summary_chunk(final)[0]
                if chunks and is_summary_chunk(chunks[0]):
                    chunks[0] = [summary_msg] + list(chunks[0][1:])
                elif chunks:
                    chunks[0] = [summary_msg] + list(chunks[0])
                else:
                    chunks = [[summary_msg]]
                self.session_mgr.write_memory(sid, chunks)
            self._update_store_after_reset([sid], {sid: final})
            if self.logger:
                self.logger.info("[MERGER summary] async summary written for %s", sid)
        except Exception:
            if self.logger:
                self.logger.exception("[MERGER summary] async write failed sid=%s", sid)

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
        是 → 返回 {"keep": int, "dropped_map": {sid: dropped_flat},
        "head_map": {sid: 旧头部摘要正文}}；否 → None。
        不更新 SoftResetState（真正的状态更新仍在 build_merged_history 内）。
        """
        if self.merge_reset_mode != "hard" or self.summarize_mode == "off":
            return None
        group_id = self._group_id(current_sid)
        if not self._soft_state.peek_should_check(group_id):
            return None
        members = self._member_sids(current_sid)
        # 双触发预检：tokens 估算超限 或 成员磁盘轮数达框架窗口
        tokens_over = (
            self.merge_trigger_mode in ("tokens", "either")
            and self.merge_token_limit > 0
            and self._estimate_group_tokens(members) > self.merge_token_limit
        )
        if not tokens_over and not self._members_rounds_over(members):
            return None
        keep = self._soft_state.peek_reset_keep(group_id)
        dropped_map = {}
        head_map = {}
        for sid in members:
            parts = read_reset_parts(self.session_mgr, sid, keep)
            if parts["dropped"]:
                dropped_map[sid] = parts["dropped"]
            if parts["head_text"]:
                head_map[sid] = parts["head_text"]
        if not dropped_map:
            return None
        return {"keep": keep, "dropped_map": dropped_map, "head_map": head_map}

    # ── 累计摘要辅助 ─────────────────────────────────────────

    def _resolve_base(self, sid: str, group_id: str, head_text: str) -> str:
        """累计 base：store 与记忆头部对账；非累积模式仅降级窗口复用（旧语义）。"""
        if self.cumulative_summary and self.summary_store is not None:
            return self.summary_store.sync_with_head(sid, head_text)
        now = time.time()
        last_reset = self._soft_state._last_reset.get(group_id, 0)
        half_window = (
            self._soft_state.check_interval / 2
            if self._soft_state.check_interval > 0
            else 30.0
        )
        if last_reset and (now - last_reset) < half_window:
            return head_text or ""
        return ""

    async def _cap_summary(self, sid: str, text: str) -> str:
        """累计摘要超输出上限：先 LLM 自压缩，失败则硬截断。"""
        cap = self.summarize_max_output_chars
        text = (text or "").strip()
        if not text or cap <= 0 or len(text) <= cap:
            return text
        compressed = await self_compress_summary(
            self.ctx,
            sid,
            text,
            model_id=self.summarize_model,
            prompt_template=self.self_compress_prompt_template,
            timeout_sec=self.merge_timeout_sec,
            logger=self.logger,
            enable_detail_log=self.enable_summary_logging,
        )
        if compressed:
            text = compressed.strip()
        if len(text) > cap:
            text = text[:cap] + "…"
        return text

    async def _merge_final(self, sid: str, base: str, delta: Optional[str]) -> str:
        """base（旧累计）+ delta（新增量）→ 新累计摘要；LLM 合并失败退化为拼接。"""
        base = (base or "").strip()
        delta = (delta or "").strip()
        if not base:
            return await self._cap_summary(sid, delta)
        if not delta:
            return base
        if self.enable_summary_logging and self.logger:
            self.logger.info(
                "[摘要调试] 尝试合并摘要 sid=%s old=%d 字符 new=%d 字符 timeout=%.1fs",
                sid, len(base), len(delta), self.merge_timeout_sec,
            )
        merged = await merge_summaries(
            self.ctx,
            sid,
            base,
            delta,
            model_id=self.summarize_model,
            prompt_template=self.merge_prompt_template,
            timeout_sec=self.merge_timeout_sec,
            logger=self.logger,
            enable_detail_log=self.enable_summary_logging,
        )
        if not merged:
            merged = f"{base}\n{delta}"
            if self.enable_summary_logging and self.logger:
                self.logger.info("[摘要调试] LLM 合并失败，退化为拼接")
        return await self._cap_summary(sid, merged)

    @staticmethod
    def _effective_dropped(head_text: str, dropped: List[dict], base: str) -> List[dict]:
        """非复用场景把旧摘要并入摘要输入（对齐旧版「旧摘要随 dropped 再压缩」语义）。"""
        if not dropped or not head_text or (base or "").strip():
            return dropped
        return [
            {"role": "user", "content": f"[更早的旧摘要，供参考] {head_text}"}
        ] + list(dropped)

    def _update_store_after_reset(self, sids: List[str], finals: dict):
        """重开写回后同步累计摘要存储。"""
        if not (self.cumulative_summary and self.summary_store is not None):
            return
        try:
            for s in sids:
                final = (finals.get(s) or "").strip()
                if final:
                    self.summary_store.set(s, final)
                else:
                    self.summary_store.pop(s)
            self.summary_store.save()
        except Exception:
            pass

    async def _summarize_members(
        self,
        dropped_map: dict,
        group_id: str,
        head_map: Optional[dict] = None,
        preprocess: Optional[bool] = None,
    ) -> dict:
        """
        并发为各 sid 生成累计摘要：base（store/头部对账）+ delta（dropped）→ merge。
        preprocess=None 用 self.preprocess_tools；sync 关键路径调用方传
        self.preprocess_in_sync（默认 False，避免预处理阻塞回复）。
        返回 {sid: final_text}（纯文本，未包装 chunk）。
        """
        head_map = head_map or {}
        use_preprocess = self.preprocess_tools if preprocess is None else preprocess

        async def _one(sid: str) -> str:
            head_text = head_map.get(sid, "")
            base = self._resolve_base(sid, group_id, head_text)
            eff = self._effective_dropped(head_text, dropped_map[sid], base)
            delta = await summarize_history(
                self.ctx,
                sid,
                eff,
                model_id=self.summarize_model,
                prompt_template=self.summarize_prompt_template,
                timeout_sec=self.summarize_timeout_sec,
                max_input_chars=self.summarize_max_input_chars,
                max_output_chars=self.summarize_max_output_chars,
                logger=self.logger,
                enable_detail_log=self.enable_summary_logging,
                preprocess_tools=use_preprocess,
                tool_max_chars=self.tool_max_chars,
            )
            return await self._merge_final(sid, base, delta)

        sids = list(dropped_map.keys())
        finals: dict = {}
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*[_one(s) for s in sids], return_exceptions=True),
                timeout=max(1.0, self.summarize_timeout_sec + 2.0),
            )
        except asyncio.TimeoutError:
            if self.logger:
                self.logger.warning("[MERGER summary] gather timeout, skip new summaries")
            results = []
        for sid, r in zip(sids, results):
            if isinstance(r, str) and r:
                finals[sid] = r
        return finals

    def _schedule_async_summaries(
        self, dropped_map: dict, group_id: str, head_map: Optional[dict] = None
    ):
        """
        async 模式：hard 重开已完成（各 sid 带旧累计 base 先行），
        后台生成增量摘要并合并补写各 sid 记忆首部。
        """
        head_map = head_map or {}

        for sid, dropped in dropped_map.items():
            old = self._summary_tasks.get(sid)
            if old and not old.done():
                old.cancel()
            # 避免与旧后台合并任务竞争写记忆：以 async 全量摘要为准
            old_bg = self._background_merge_tasks.get(sid)
            if old_bg and not old_bg.done():
                old_bg.cancel()
            self._preheat_pending.pop(sid, None)

            async def _run(sid=sid, dropped=dropped):
                try:
                    head_text = head_map.get(sid, "")
                    base = self._resolve_base(sid, group_id, head_text)
                    eff = self._effective_dropped(head_text, dropped, base)
                    delta = await summarize_history(
                        self.ctx,
                        sid,
                        eff,
                        model_id=self.summarize_model,
                        prompt_template=self.summarize_prompt_template,
                        timeout_sec=self.summarize_timeout_sec,
                        max_input_chars=self.summarize_max_input_chars,
                        max_output_chars=self.summarize_max_output_chars,
                        logger=self.logger,
                        enable_detail_log=self.enable_summary_logging,
                        preprocess_tools=self.preprocess_tools,
                        tool_max_chars=self.tool_max_chars,
                    )
                    if not delta or not self.session_mgr:
                        return
                    final = await self._merge_final(sid, base, delta)
                    if not final:
                        return
                    # 写回：锁内只做「读头校验 + 覆写」，绝不在锁内 await LLM
                    # （threading.Lock 在事件循环里等待 LLM 会阻塞整个 loop）
                    for _attempt in range(2):
                        with self._get_reset_lock(group_id):
                            chunks = self.session_mgr.read_memory(sid) or []
                            cur_head = ""
                            if chunks and is_summary_chunk(chunks[0]):
                                cur_head = extract_summary_text(
                                    str(chunks[0][0].get("content", "") or "")
                                )
                            if (cur_head or "") == (base or ""):
                                summary_msg = build_summary_chunk(final)[0]
                                if chunks and is_summary_chunk(chunks[0]):
                                    chunks[0] = [summary_msg] + list(chunks[0][1:])
                                elif chunks:
                                    chunks[0] = [summary_msg] + list(chunks[0])
                                else:
                                    chunks = [[summary_msg]]
                                self.session_mgr.write_memory(sid, chunks)
                                self._update_store_after_reset([sid], {sid: final})
                                if self.logger:
                                    self.logger.info(
                                        "[MERGER summary] async summary written for %s",
                                        sid,
                                    )
                                return
                        # 锁外：期间又发生过重开 → 以当前头部为准重新对齐并再合并一次
                        new_base = self._resolve_base(sid, group_id, cur_head)
                        if (new_base or "") == (base or ""):
                            return  # 头部反复变化，放弃本轮补写（下轮对账修复）
                        base = new_base
                        final = await self._merge_final(sid, base, delta)
                        if not final:
                            return
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
        for t in list(self._preheat_tasks.values()):
            if t and not t.done():
                t.cancel()
        self._preheat_tasks.clear()
        self._preheat_pending.clear()
        for t in list(self._background_merge_tasks.values()):
            if t and not t.done():
                t.cancel()
        self._background_merge_tasks.clear()
        self._pending_replace.clear()

    async def manual_reset_with_summary(self, current_sid: str) -> dict:
        """
        压缩重开命令触发：立即对当前会话所在合并组执行「累计摘要 + 硬重开」。
        与 merge_reset_mode 无关（soft 模式下也可手动硬重开）；
        摘要仍受 summarize_mode 控制（off = 只重开不摘要）。
        返回 {"members", "keep", "ok", "fail", "summarized"}。
        """
        members = self._member_sids(current_sid)
        group_id = self._group_id(current_sid)
        keep = self._clamp_keep(self.merge_keep_turns)

        dropped_map = {}
        head_map = {}
        for s in members:
            parts = read_reset_parts(self.session_mgr, s, keep)
            if parts["dropped"]:
                dropped_map[s] = parts["dropped"]
            if parts["head_text"]:
                head_map[s] = parts["head_text"]

        finals: dict = {}
        missing: dict = {}
        if self.summarize_mode != "off" and dropped_map:
            try:
                # 手动命令也尝试收割后台持续压缩，未就绪再现场生成
                finals, missing = await self._harvest_for_group_sync(
                    dropped_map, group_id, head_map, timeout=self.sync_wait_timeout
                )
            except Exception:
                if self.logger:
                    self.logger.exception("[MERGER summary] manual summarize failed")

        summary_chunks = {
            s: build_summary_chunk(t) for s, t in finals.items() if t
        }

        # per-group 锁：与自动 hard 重开互斥（自动发生在 to_thread 的 _apply_sync 里）
        with self._get_reset_lock(group_id):
            results = hard_reset_members(
                self.session_mgr,
                members,
                keep,
                logger=self.logger,
                summary_chunks=summary_chunks or None,
                write_through=self.write_through,
            )
        ok = sum(1 for v in results.values() if isinstance(v, int) and v >= 0)
        fail = len(results) - ok

        # 累计摘要落盘（无摘要的成员清掉旧条目，防止旧摘要污染）
        self._update_store_after_reset(members, finals)

        # append 策略下未就绪成员走 async 补写，避免手动命令也阻塞 LLM
        if missing and self.continuous_merge_strategy in ("append_then_merge", "append_only"):
            if self.logger:
                self.logger.info(
                    "[MERGER summary] manual reset: append 策略未就绪，调度 async 补写 %s",
                    list(missing.keys()),
                )
            self._schedule_async_summaries(missing, group_id, head_map)

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
        sync_finals: dict = {}
        sync_missing: dict = {}
        pending_async: Optional[tuple] = None  # (group_id, dropped_map, head_map)
        current_sid = getattr(event, "sid", None) or ""
        group_id = ""
        try:
            if current_sid and self.should_merge(current_sid):
                pre = self.precheck_hard_reset(current_sid)
                if pre:
                    if self.enable_summary_logging and self.logger:
                        self.logger.info(
                            "[摘要调试] 预检判定需要重开，dropped_map: %s", list(pre["dropped_map"].keys())
                        )
                    group_id = self._group_id(current_sid)
                    head_map = pre.get("head_map") or {}
                    if self.summarize_mode == "sync":
                        if self.enable_summary_logging and self.logger:
                            self.logger.info("[摘要调试] 进入 sync 模式，尝试收割持续压缩 / append 策略 fallback 走 async")
                        # sync 关键路径默认不做预处理（preprocess_in_sync，避免阻塞）
                        sync_finals, sync_missing = await self._harvest_for_group_sync(
                            pre["dropped_map"], group_id, head_map,
                            timeout=self.sync_wait_timeout,
                        )
                        summary_chunks = {
                            s: build_summary_chunk(t)
                            for s, t in sync_finals.items()
                            if t
                        }
                        if self.enable_summary_logging and self.logger:
                            self.logger.info("[摘要调试] sync 摘要完成，成功生成: %s", list(summary_chunks.keys()) if summary_chunks else [])
                    elif self.summarize_mode == "async":
                        if self.enable_summary_logging and self.logger:
                            self.logger.info("[摘要调试] 进入 async 模式，先捕获 dropped 数据，重开后补写")
                        # dropped 必须在重开删数据前捕获；摘要在重开后补写
                        pending_async = (group_id, pre["dropped_map"], head_map)
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

        reset_sids = set(out_info.get("hard_reset_sids") or [])

        # sync 摘要：仅对实际发生 hard 重开的 sid 落盘累计摘要
        if sync_finals and reset_sids:
            self._update_store_after_reset(list(reset_sids), sync_finals)

        # async 摘要：仅对实际发生 hard 重开的 sid 补写；优先收割已完成的后台压缩
        if pending_async:
            async_group_id, dropped_map, head_map = pending_async
            if self.enable_summary_logging and self.logger:
                self.logger.info("[摘要调试] async 模式实际重开的 sid: %s", list(reset_sids))
            if reset_sids:
                to_schedule: dict = {}
                for sid in reset_sids:
                    if sid not in dropped_map:
                        continue
                    pending = await self._harvest_continuous_compression(sid, timeout=0)
                    final = None
                    if isinstance(pending, dict):
                        final = self._preheat_valid_final(
                            sid, dropped_map[sid], head_map.get(sid, ""), async_group_id
                        )
                    if final:
                        self._write_summary_to_memory(sid, final, async_group_id)
                        self._preheat_pending.pop(sid, None)
                        if self.enable_summary_logging and self.logger:
                            self.logger.info("[摘要调试] async 收割并写入 %s 持续压缩摘要", sid)
                    else:
                        to_schedule[sid] = dropped_map[sid]
                if to_schedule:
                    if self.enable_summary_logging and self.logger:
                        self.logger.info("[摘要调试] 调度后台摘要任务: %s", list(to_schedule.keys()))
                    self._schedule_async_summaries(to_schedule, async_group_id, head_map)

        # sync 模式下 append 策略未就绪的成员，也走 async 补写，避免请求路径阻塞
        if sync_missing and self.continuous_merge_strategy in ("append_then_merge", "append_only"):
            if self.enable_summary_logging and self.logger:
                self.logger.info("[摘要调试] sync 模式 append 策略未就绪，调度 async 补写: %s", list(sync_missing.keys()))
            async_sids = set(sync_missing.keys()) & reset_sids
            if async_sids:
                async_map = {s: sync_missing[s] for s in async_sids}
                self._schedule_async_summaries(async_map, group_id, head_map)
        return ok
