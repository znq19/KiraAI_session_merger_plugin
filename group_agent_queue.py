from __future__ import annotations

"""
合并组级单一 Agent 队列（逻辑一个窗口）。

- 同一 merge group 同时只跑一个 agent（串行）
- busy 时后来的 batch 入队，等当前 sid 落盘后再调度
- 不改 core：通过 ON_IM_BATCH stop + 重发 batch + wrap update_memory 实现
"""

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Set


@dataclass
class PendingBatch:
    sid: str
    messages: list
    adapter: Any
    session: Any
    message_types: list
    enqueued_at: float = field(default_factory=time.time)


@dataclass
class GroupRunState:
    running: bool = False
    active_sid: str = ""
    active_event_id: str = ""
    started_at: float = 0.0
    queue: Deque[PendingBatch] = field(default_factory=deque)
    # 已在队列中的 sid，用于合并同 sid 的后续 batch
    queued_sids: Set[str] = field(default_factory=set)


class GroupAgentQueue:
    """
    按 group_id 串行化 agent。

    流程：
    1. try_begin(batch) → True 则本 batch 继续跑；False 则已入队，调用方应 event.stop()
    2. 当前 agent 的 update_memory(sid) 后 → on_memory_written(sid) → 调度队列
    3. 调度：可选 settle 等待 → 取出队首 → 重新 publish batch
    """

    def __init__(
        self,
        enabled: bool = True,
        lock_ttl_sec: float = 180.0,
        settle_sec: float = 0.4,
        max_queue_per_group: int = 32,
        logger=None,
    ):
        self.enabled = bool(enabled)
        self.lock_ttl_sec = max(10.0, float(lock_ttl_sec or 180.0))
        self.settle_sec = max(0.0, float(settle_sec or 0.0))
        self.max_queue_per_group = max(1, int(max_queue_per_group or 32))
        self.logger = logger
        self._states: Dict[str, GroupRunState] = {}
        self._lock = asyncio.Lock()
        self._schedule_tasks: Dict[str, asyncio.Task] = {}
        # 防止调度重发的 batch 再次被当成「冲突入队」时丢消息：标记 event_id 为已授权
        self._authorized_event_ids: Set[str] = set()

    def _log(self, msg: str, *args):
        if self.logger:
            try:
                self.logger.info(msg, *args)
            except Exception:
                pass

    def _state(self, group_id: str) -> GroupRunState:
        if group_id not in self._states:
            self._states[group_id] = GroupRunState()
        return self._states[group_id]

    def _expired(self, st: GroupRunState, now: float) -> bool:
        if not st.running:
            return False
        return (now - st.started_at) > self.lock_ttl_sec

    def authorize_event(self, event_id: str):
        if event_id:
            self._authorized_event_ids.add(str(event_id))

    def is_authorized(self, event_id: str) -> bool:
        return bool(event_id) and str(event_id) in self._authorized_event_ids

    def consume_authorization(self, event_id: str):
        if event_id:
            self._authorized_event_ids.discard(str(event_id))

    async def try_begin(
        self,
        group_id: str,
        sid: str,
        event,
    ) -> bool:
        """
        尝试开始本 batch 的 agent。
        返回 True：占用组锁，继续处理。
        返回 False：已入队或无法处理，调用方应 stop event（勿再跑 LLM）。
        """
        if not self.enabled or not group_id or not sid:
            return True

        event_id = str(getattr(event, "event_id", "") or "")
        now = time.time()

        async with self._lock:
            st = self._state(group_id)

            # 调度重发的 batch：已预授权
            if event_id and event_id in self._authorized_event_ids:
                self._authorized_event_ids.discard(event_id)
                if st.running and not self._expired(st, now):
                    # 异常：锁仍被占 → 重新入队
                    self._enqueue_locked(st, group_id, sid, event, now)
                    return False
                st.running = True
                st.active_sid = sid
                st.active_event_id = event_id
                st.started_at = now
                self._log(
                    "[MERGER queue] begin authorized sid=%s group=%s",
                    sid,
                    group_id,
                )
                return True

            if st.running and self._expired(st, now):
                self._log(
                    "[MERGER queue] lock TTL expired group=%s was sid=%s, force release",
                    group_id,
                    st.active_sid,
                )
                st.running = False
                st.active_sid = ""
                st.active_event_id = ""

            if st.running:
                self._enqueue_locked(st, group_id, sid, event, now)
                self._log(
                    "[MERGER queue] busy group=%s active=%s; enqueued sid=%s qlen=%d",
                    group_id,
                    st.active_sid,
                    sid,
                    len(st.queue),
                )
                return False

            st.running = True
            st.active_sid = sid
            st.active_event_id = event_id
            st.started_at = now
            self._log(
                "[MERGER queue] begin sid=%s group=%s",
                sid,
                group_id,
            )
            return True

    def _enqueue_locked(
        self,
        st: GroupRunState,
        group_id: str,
        sid: str,
        event,
        now: float,
    ):
        messages = list(getattr(event, "messages", None) or [])
        if not messages:
            return

        # 同 sid 已在队列：追加 messages，避免丢话
        if sid in st.queued_sids:
            for p in st.queue:
                if p.sid == sid:
                    p.messages.extend(messages)
                    p.enqueued_at = now
                    self._log(
                        "[MERGER queue] merge into queued sid=%s (+%d msgs) group=%s",
                        sid,
                        len(messages),
                        group_id,
                    )
                    return

        if len(st.queue) >= self.max_queue_per_group:
            # 丢最旧，保最新
            old = st.queue.popleft()
            st.queued_sids.discard(old.sid)
            self._log(
                "[MERGER queue] queue full, drop oldest sid=%s group=%s",
                old.sid,
                group_id,
            )

        pending = PendingBatch(
            sid=sid,
            messages=messages,
            adapter=getattr(event, "adapter", None),
            session=getattr(event, "session", None),
            message_types=list(getattr(event, "message_types", None) or []),
            enqueued_at=now,
        )
        st.queue.append(pending)
        st.queued_sids.add(sid)

    async def release_if_active(
        self,
        group_id: str,
        sid: str = "",
        event_id: str = "",
        reason: str = "",
        schedule_fn=None,
    ) -> bool:
        """
        若当前组锁由该 sid（及可选 event_id）持有则释放。
        幂等：已释放 / 非持有者 → False。
        主路径：update_memory；补充路径：event 提前 stop / 安全网。
        """
        if not self.enabled or not group_id:
            return False

        should_schedule = False
        async with self._lock:
            st = self._state(group_id)
            if not st.running:
                return False
            now = time.time()
            expired = self._expired(st, now)
            if sid and st.active_sid != sid and not expired:
                return False
            if (
                event_id
                and st.active_event_id
                and str(st.active_event_id) != str(event_id)
                and not expired
            ):
                return False
            self._log(
                "[MERGER queue] release group=%s sid=%s event=%s reason=%s qlen=%d",
                group_id,
                st.active_sid,
                st.active_event_id,
                reason or "unspecified",
                len(st.queue),
            )
            st.running = False
            st.active_sid = ""
            st.active_event_id = ""
            st.started_at = 0.0
            should_schedule = bool(st.queue)

        if should_schedule and schedule_fn:
            await schedule_fn(group_id)
        return True

    async def on_memory_written(self, sid: str, group_id: str, schedule_fn):
        """官方 update_memory 之后：主释放路径。"""
        if not sid or not group_id:
            return
        await self.release_if_active(
            group_id,
            sid=sid,
            reason="memory_written",
            schedule_fn=schedule_fn,
        )

    async def force_release(self, group_id: str, reason: str = "", schedule_fn=None):
        async with self._lock:
            st = self._state(group_id)
            if not st.running:
                return
            self._log(
                "[MERGER queue] force release group=%s sid=%s reason=%s",
                group_id,
                st.active_sid,
                reason,
            )
            st.running = False
            st.active_sid = ""
            st.active_event_id = ""
            st.started_at = 0.0
            should_schedule = bool(st.queue)
        if should_schedule and schedule_fn:
            await schedule_fn(group_id)

    async def pop_next_and_begin(self, group_id: str) -> Optional[PendingBatch]:
        """取出队首并立即占锁，避免 publish 前被其它 batch 抢锁。"""
        async with self._lock:
            st = self._state(group_id)
            if st.running:
                return None
            if not st.queue:
                return None
            pending = st.queue.popleft()
            st.queued_sids.discard(pending.sid)
            st.running = True
            st.active_sid = pending.sid
            st.active_event_id = ""
            st.started_at = time.time()
            return pending

    def queue_len(self, group_id: str) -> int:
        st = self._states.get(group_id)
        return len(st.queue) if st else 0

    def clear_all(self):
        self._states.clear()
        self._authorized_event_ids.clear()
        for t in list(self._schedule_tasks.values()):
            if t and not t.done():
                t.cancel()
        self._schedule_tasks.clear()
