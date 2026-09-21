# -*- coding: utf-8 -*-
"""组锁「重放批次在到达本插件前被 stop」的快速释放（v2.8.1）。

复现场景（真实链路）：
    _schedule_group_drain → pop_next_and_begin（占锁）→ authorize_event → bus.publish
    → 批次进入 ON_IM_BATCH_MESSAGE 管线 → **其它插件在更高优先级把它 stop**
      （聊天插件的队列合并 / 随时插话类插件）
    → 本插件 on_im_batch_group_queue（LOW）根本没执行 → 授权未被消费
    → 本轮永远不会 execute → update_memory 不会发生
    → 组锁只能等 TTL（默认 180s）→ 同组合并组其它会话批次白白排队。

v2.1.0 的"组锁补充释放"覆盖了 LLM 请求阶段 stop 与走到 update_memory 的路径，
**批次阶段被 stop** 这条缝由本次补齐：发布后挂一个监听器，满足
「授权未被消费 且 事件已停」时提前放锁。

用例：
  T1 正常路径：授权被消费（本插件已接手）→ 监听器退出，**不放锁**（行为零变化）
  T2 本案：未消费 + 已停 → ≤2s 内放锁，并调度下一批
  T3 未消费也未停（总线繁忙/仍在排队）→ 不动作，到 TTL 交回原兜底
  T4 锁已属别的 sid（TTL 后被更晚的批次接手）→ 不误放
  T5 锁的获取时间晚于本次发布 → 不误放
  T6 生命周期：监听任务在结束后从注册表移除（不泄漏）

Run: python3 tests/test_replay_stop_release.py
"""
import asyncio
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from group_agent_queue import GroupAgentQueue, resolve_settle_sec  # noqa: E402

results = []


def check(label, cond, detail=""):
    results.append((label, bool(cond)))
    # detail 默认只在失败时打印（成功路径不刷无关信息）
    print(("  PASS  " if cond else "  FAIL  ") + label + ((" — " + str(detail)) if (detail and not cond) else ""))


class FakeEvent:
    """只保留队列与监听器读取的字段：event_id / is_stopped / messages。"""

    def __init__(self, eid, messages=None):
        self.event_id = eid
        self.is_stopped = False
        self.messages = list(messages if messages is not None else [object()])
        self.adapter = None
        self.session = None
        self.message_types = []

    def stop(self):
        self.is_stopped = True


class FakePending:
    def __init__(self, sid, eid):
        self.sid = sid
        self.event_id = eid
        self.messages = []
        self.adapter = None
        self.session = None
        self.message_types = []


async def make_queue_with_locked_replay(sid="qq:gm:222", eid="ev-replay-1"):
    """构造"重放已占锁 + 该 event 已授权"的真实状态，返回 (queue, group_id, event)。"""
    q = GroupAgentQueue(enabled=True, lock_ttl_sec=180, settle_sec=0, logger=None)
    gid = "g1"
    # 先让 A 占锁，并把 B 排队 —— 然后模拟 drain：释放 A、取出 B 并占锁
    ev_a = FakeEvent("ev-a")
    ev_b = FakeEvent("ev-b")
    assert await q.try_begin(gid, "qq:gm:111", ev_a) is True      # A 占锁
    assert await q.try_begin(gid, "qq:gm:333", ev_b) is False     # B 入队
    await q.release_if_active(gid, sid="qq:gm:111", reason="test")  # A 释放（B 仍在队）
    pending = await q.pop_next_and_begin(gid)                     # 取出 B → 占锁
    assert pending is not None
    # 再排一个 C：用于验证"提前放锁后会调度下一批"
    ev_c = FakeEvent("ev-c")
    assert await q.try_begin(gid, "qq:gm:444", ev_c) is False
    # 重放：构造新 event 并授权（与 main.py 一致）
    ev = FakeEvent(eid)
    q.authorize_event(eid)
    return q, gid, ev, pending


async def t1_consumed_no_release():
    q, gid, ev, pending = await make_queue_with_locked_replay()
    q.consume_authorization(ev.event_id)          # 模拟 try_begin 已接手
    q.schedule_replay_watch(gid, pending.sid, ev, published_at=time.time(), ttl_sec=10)
    await asyncio.sleep(1.8)                      # 超过第一个轮询周期
    st = q._state(gid)
    check("T1 授权已消费 → 监听器退出且不放锁", st.running and st.active_sid == pending.sid)
    check("T1 监听任务注册表已清理", ev.event_id not in q._watch_tasks)
    q.clear_all()


async def t2_stopped_releases_early():
    q, gid, ev, pending = await make_queue_with_locked_replay()
    drained = {"n": 0}

    async def schedule_fn(_gid):
        drained["n"] += 1

    q.schedule_replay_watch(gid, pending.sid, ev, published_at=time.time(),
                                  schedule_fn=schedule_fn, ttl_sec=30)
    ev.stop()                                     # ★ 批次阶段被别的插件 stop
    t0 = time.time()
    released_at = None
    while time.time() - t0 < 3.0:
        if not q._state(gid).running:
            released_at = time.time() - t0
            break
        await asyncio.sleep(0.1)
    check("T2 批次阶段被 stop → 提前释放组锁", released_at is not None,
          f"3s 内未释放（TTL 兜底要等 180s）")
    check("T2 释放延迟 ≤0.3s（首检 50ms + 退避）", released_at is not None and released_at <= 0.3,
          f"{released_at}")
    await asyncio.sleep(0.1)
    check("T2 已调度下一批（drain）", drained["n"] >= 1, f"drained={drained['n']}")
    check("T2 授权痕迹已清理", ev.event_id not in q._authorized_event_ids)
    check("T2 监听任务注册表已清理", ev.event_id not in q._watch_tasks)
    q.clear_all()


async def t3_not_stopped_no_release():
    q, gid, ev, pending = await make_queue_with_locked_replay()
    q.schedule_replay_watch(gid, pending.sid, ev, published_at=time.time(), ttl_sec=1)
    await asyncio.sleep(2.2)                      # 短 TTL 到期退出
    st = q._state(gid)
    check("T3 未停也未消费 → 不放锁（交回 TTL 兜底）", st.running and st.active_sid == pending.sid)
    check("T3 监听任务已退出", ev.event_id not in q._watch_tasks)
    q.clear_all()


async def t4_other_sid_lock_not_released():
    q, gid, ev, pending = await make_queue_with_locked_replay()
    # 模拟 TTL 过期后锁被**别的会话**的更晚批次接手
    st = q._state(gid)
    st.active_sid = "qq:gm:999"
    st.started_at = time.time()
    q.schedule_replay_watch(gid, pending.sid, ev, published_at=time.time(), ttl_sec=10)
    ev.stop()
    await asyncio.sleep(1.8)
    check("T4 锁属于别的 sid → 不误放", q._state(gid).running and q._state(gid).active_sid == "qq:gm:999")
    q.clear_all()


async def t5_newer_lock_not_released():
    q, gid, ev, pending = await make_queue_with_locked_replay()
    published = time.time()
    await asyncio.sleep(0.05)
    st = q._state(gid)
    st.started_at = published + 5.0               # 锁的获取时间晚于本次发布
    q.schedule_replay_watch(gid, pending.sid, ev, published_at=published, ttl_sec=10)
    ev.stop()
    await asyncio.sleep(1.8)
    check("T5 锁的获取时间晚于发布 → 不误放", q._state(gid).running)
    q.clear_all()


async def t6_clear_all_cancels():
    q, gid, ev, pending = await make_queue_with_locked_replay()
    q.schedule_replay_watch(gid, pending.sid, ev, published_at=time.time(), ttl_sec=60)
    await asyncio.sleep(0.2)
    check("T6 监听任务已注册", ev.event_id in q._watch_tasks)
    q.clear_all()
    await asyncio.sleep(0.2)
    check("T6 clear_all 取消并清空监听任务", not q._watch_tasks)
    check("T6 clear_all 后授权/状态一并清空",
          not q._authorized_event_ids and not q._states)


async def t7_settle_migration():
    """v2.8.1：settle 默认 0.4 → 0；只迁移"旧默认值"，用户显式改过的值原样尊重。"""
    check("S1 未配置 → 0（新默认）", resolve_settle_sec(None) == (0.0, False))
    check("S2 旧默认 0.4 → 迁移为 0（并标记需写回）", resolve_settle_sec(0.4) == (0.0, True))
    check("S3 用户显式值 1.5 → 保留", resolve_settle_sec(1.5) == (1.5, False))
    check("S4 用户显式 0 → 0（不触发迁移）", resolve_settle_sec(0) == (0.0, False))
    check("S5 负数 → 归零", resolve_settle_sec(-3) == (0.0, False))
    check("S6 非法值 → 0（不炸）", resolve_settle_sec("abc") == (0.0, False))
    check("S7 字符串 0.4 也按旧默认迁移", resolve_settle_sec("0.4") == (0.0, True))


async def t8_replay_takes_placeholder_lock():
    """v2.8.3 回归：重放批次走**真实 try_begin** 必须能接管占位锁。

    现有 T1 用 consume_authorization() 手动模拟"已接手"，绕过了真实
    try_begin 路径，所以漏掉了这个 P0：pop_next_and_begin 占锁（running=True,
    active_event_id=""）→ drain 授权新 event → try_begin 授权分支误判"锁仍被占"
    → 重新入队 + stop → watch 在"授权已消费"分支退出不放锁 → 幻影锁挂到 TTL。
    """
    q = GroupAgentQueue(enabled=True, lock_ttl_sec=180, settle_sec=0, logger=None)
    gid = "g1"
    ev_a = FakeEvent("ev-a")
    ev_b = FakeEvent("ev-b")
    assert await q.try_begin(gid, "qq:gm:111", ev_a) is True       # A 占锁运行
    assert await q.try_begin(gid, "qq:gm:222", ev_b) is False      # B 入队
    await q.release_if_active(gid, sid="qq:gm:111", reason="test") # A 落盘后释放
    pending = await q.pop_next_and_begin(gid)                      # drain：占位锁
    assert pending is not None and pending.sid == "qq:gm:222"
    replay = FakeEvent("ev-replay-8")                              # 重放批次（新 event_id）
    q.authorize_event(replay.event_id)
    ok = await q.try_begin(gid, "qq:gm:222", replay)               # 真实 try_begin
    check("T8 重放批次接管占位锁 → try_begin=True（修复前 FAIL）", ok)
    st = q._state(gid)
    check("T8 锁归属重放 event", st.running and st.active_event_id == "ev-replay-8")
    check("T8 未发生重新入队（队列空）", q.queue_len(gid) == 0)
    q.clear_all()


async def t9_authorized_reenqueue_when_foreign_lock():
    """授权批次撞上**别人真实占用的锁**（带 active_event_id）仍应重新入队。"""
    q = GroupAgentQueue(enabled=True, lock_ttl_sec=180, settle_sec=0, logger=None)
    gid = "g1"
    ev_a = FakeEvent("ev-a")
    assert await q.try_begin(gid, "qq:gm:111", ev_a) is True       # 别的批次真实占锁
    replay = FakeEvent("ev-replay-9")
    q.authorize_event(replay.event_id)
    ok = await q.try_begin(gid, "qq:gm:222", replay)
    check("T9 锁被别的批次真实占用 → 授权批次重新入队", not ok)
    check("T9 已入队等待（不丢消息）", q.queue_len(gid) == 1)
    q.clear_all()


async def main():
    await t7_settle_migration()
    await t8_replay_takes_placeholder_lock()
    await t9_authorized_reenqueue_when_foreign_lock()
    await t1_consumed_no_release()
    await t2_stopped_releases_early()
    await t3_not_stopped_no_release()
    await t4_other_sid_lock_not_released()
    await t5_newer_lock_not_released()
    await t6_clear_all_cancels()
    print()
    passed = sum(1 for _, ok in results if ok)
    print("TOTAL %d/%d passed" % (passed, len(results)))
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    asyncio.run(main())
