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


# settle 的旧默认值（v2.8.1 起默认改为 0 = 落盘后立即调度）
SETTLE_OLD_DEFAULT = 0.4


def resolve_settle_sec(raw, default: float = 0.0) -> "tuple[float, bool]":
    """把配置里的 settle 原始值解析成 (生效值, 是否触发了 v2.8.1 默认迁移)。

    规则（纯函数，无框架依赖，便于测试）：
      · 未配置（None）→ (default, False)
      · 非法值       → (default, False)
      · 恰为旧默认 0.4 → (default, True)   ← 存量默认值迁移（不是用户特意改的值）
      · 其它值        → (max(0, v), False) ← 用户显式选择，原样尊重
    """
    if raw is None:
        return float(default), False
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return float(default), False
    if abs(v - SETTLE_OLD_DEFAULT) < 1e-9:
        return float(default), True
    return (max(0.0, v), False)


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
        # 重放批次监听器：{event_id: task}（见 schedule_replay_watch）
        self._watch_tasks: Dict[str, asyncio.Task] = {}

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
                # 区分「pop_next_and_begin 为本次重放预留的占位锁」
                # （running 但 active_event_id 为空且 sid 匹配）与「别人占着的锁」：
                # 占位锁应直接接管；误判为"锁仍被占"会重新入队 + stop，而
                # watch 在"授权已消费"分支退出不放锁 → 幻影锁挂到 TTL（重放死锁）
                placeholder = (
                    st.running
                    and not st.active_event_id
                    and (not st.active_sid or st.active_sid == sid)
                    and not self._expired(st, now)
                )
                if st.running and not placeholder and not self._expired(st, now):
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
            # 内存治理：组已空闲（未运行且队列空）→ 删除状态，避免 _states 无界增长
            if not should_schedule:
                self._states.pop(group_id, None)

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
            # 内存治理：组已空闲（未运行且队列空）→ 删除状态
            if not should_schedule:
                self._states.pop(group_id, None)
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
        for t in list(self._watch_tasks.values()):
            if t and not t.done():
                t.cancel()
        self._watch_tasks.clear()

    # ---------------- 重放批次监听（批次阶段被 stop 的快速放锁） ----------------

    def schedule_replay_watch(
        self,
        group_id: str,
        sid: str,
        event,
        published_at: float = 0.0,
        schedule_fn=None,
        ttl_sec: float = 0.0,
    ) -> None:
        """监听一次重放：若它在**到达本插件之前**就被别的插件 stop，提前释放组锁。

        只做监听，不改任何既有分支：正常批次会在 `try_begin` 里消费授权，
        监听器随即退出（对既有行为零影响）。ttl_sec=0 时取 self.lock_ttl_sec。
        """
        if not self.enabled or event is None:
            return
        eid = str(getattr(event, "event_id", "") or "")
        if not eid:
            return
        try:
            ttl = float(ttl_sec) if ttl_sec and ttl_sec > 0 else float(self.lock_ttl_sec)
            task = asyncio.create_task(
                self._watch_replayed_batch(
                    group_id, str(sid or ""), event, eid, float(published_at or 0.0),
                    schedule_fn, max(1.0, ttl),
                )
            )
            self._watch_tasks[eid] = task
        except Exception:
            pass  # 监听只是优化，失败绝不能影响主流程

    async def _watch_replayed_batch(
        self,
        group_id: str,
        sid: str,
        event,
        eid: str,
        published_at: float,
        schedule_fn,
        ttl: float,
    ) -> None:
        try:
            deadline = time.time() + ttl
            # 首检 50ms：其它插件 stop 本批次发生在"发布后毫秒级"，50ms 足够捕获且无感知；
            # 之后逐步退避到 2s。正常批次会在第一次醒来时发现"授权已消费"并立即退出，
            # 所以这段轮询的成本 ≈ 每次重放多醒一次（可忽略）。
            interval = 0.05
            while True:
                await asyncio.sleep(interval)
                interval = min(2.0, interval * 1.6)
                # ① 授权已被消费 = 本插件的 try_begin 跑过（正常接手/重新入队）
                #    → 后续由常规释放路径（update_memory / llm_request_stopped / TTL）负责
                if eid not in self._authorized_event_ids:
                    return
                # ② 我们发布的那个 event 被 stop（批次阶段被其它插件掐停）：
                #    框架的批次钩子循环已 return，本批次不会再有 LLM/记忆写入 → 安全放锁
                if bool(getattr(event, "is_stopped", False)):
                    released = False
                    should_schedule = False
                    async with self._lock:
                        self._authorized_event_ids.discard(eid)
                        st = self._state(group_id)
                        # 双重校验：锁仍属于本次 sid，且获取时间不晚于本次发布
                        # （防 TTL 过期后锁已被更晚的批次接手时误放）
                        if (
                            st.running
                            and (not st.active_sid or st.active_sid == sid)
                            and (not published_at or st.started_at <= published_at + 1.0)
                        ):
                            st.running = False
                            st.active_sid = ""
                            st.active_event_id = ""
                            st.started_at = 0.0
                            should_schedule = bool(st.queue)
                            released = True
                    if released:
                        self._log(
                            "[MERGER queue] replay batch stopped before handler; "
                            "release group lock early group=%s sid=%s event=%s",
                            group_id, sid, eid,
                        )
                        if should_schedule and schedule_fn:
                            try:
                                await schedule_fn(group_id)
                            except Exception:
                                pass
                    return
                if time.time() >= deadline:
                    # 兜底退出时一并丢弃授权，避免 _authorized_event_ids 无界增长
                    self._authorized_event_ids.discard(eid)
                    return  # 未消费也未停：交回原有 TTL 兜底
        except asyncio.CancelledError:
            return
        except Exception:
            return  # 监听器绝不向外抛
        finally:
            self._watch_tasks.pop(eid, None)
