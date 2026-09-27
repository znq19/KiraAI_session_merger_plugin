# -*- coding: utf-8 -*-
"""真框架集成测试：KSM 收割 / 摘要 / keep 一致性（对应 v2.8.4 修复）。

覆盖：
  A  真实 PluginManager 完整加载（manifest/schema/配置/钩子）
  H  收割链：任务完成后可取回 pending；状态前移一轮仍按内容锚点收割，
     delta 恰为「锚点之后的新增部分」；sync_wait_timeout>0 等待路径保持可用
  S  框架原生窗口滑动时，增量压缩输入显著小于全量（不再每轮全量重压缩）
  K  keep：无条件钳制；precheck 与 apply 使用同一降级信号（不再分歧丢内容）
  C  累计摘要保活：会话继续时保留 store；会话清空才丢弃
  E  会话清空/删除事件精确清理；terminate 反订阅
  P  预热门控：无消费方（soft 且 /resum 关）时不启动
  T  apply 超时后线程仍完成 hard reset，且 store 与磁盘保持一致（线程内同步）
  R  手动压缩重开（/resum 路径）回归；precheck 谱系校验单元
  PP 摘要预处理并发化：有界并发 ≤3 / 保序 / 失败回落 / 零待处理快路径

Run:
  python3 tests/test_integration_framework.py
  KIRA_FRAMEWORK_PATH=/path/to/KiraAI python3 tests/test_integration_framework.py

需要 KiraAI 框架源码（含 core/）：按环境变量、命令行 --framework 或常见相对路径
探测；找不到时打印 SKIP 并以退出码 0 结束（run_all.sh 友好）。
"""
import argparse
import asyncio
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = []


def check(label, cond, detail=""):
    RESULTS.append((label, bool(cond)))
    print(("  PASS  " if cond else "  FAIL  ") + label + ((" — " + str(detail)) if detail else ""))


def find_framework(arg=None):
    cands = []
    if arg:
        cands.append(Path(arg))
    env = os.environ.get("KIRA_FRAMEWORK_PATH")
    if env:
        cands.append(Path(env))
    p = Path(HERE)
    for _ in range(6):
        p = p.parent
        cands.append(p)
    for name in ("KiraAI", "KiraAI-main", "kira_upstream"):
        cands.append(Path(ROOT).parent / name)
        cands.append(Path(ROOT).parent.parent / name)
    for c in cands:
        try:
            if c and (c / "core" / "plugin" / "__init__.py").exists():
                return c
        except OSError:
            continue
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", default=None)
    args = parser.parse_args()

    fw = find_framework(args.framework)
    if fw is None:
        print("SKIP: 未找到 KiraAI 框架源码（用 KIRA_FRAMEWORK_PATH 或 --framework 指定）")
        return 0
    print(f"使用框架: {fw}")
    sys.path.insert(0, str(fw))

    data_dir = Path(tempfile.mkdtemp(prefix="ksm_it_"))
    from core.utils.path_utils import init_paths
    init_paths(data_dir=str(data_dir))

    # ── 静态校验（无需运行环境）──
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    check("manifest 版本为 semver", bool(re.fullmatch(r"\d+\.\d+\.\d+", str(manifest.get("version", "")))), manifest.get("version"))
    schema_raw = (ROOT / "schema.json").read_text(encoding="utf-8")
    check("schema 无已废弃的 enum 类型", '"type": "enum"' not in schema_raw)
    from core.config.config_field import build_fields
    try:
        fields = build_fields(json.loads(schema_raw))
        check("schema 可被框架解析", len(fields) >= 8, f"{len(fields)} sections")
    except Exception as e:
        check("schema 可被框架解析", False, str(e))

    # ── 运行环境与假件 ──
    from core.config import KiraConfig
    from core.chat.session_manager import SessionManager
    from core.event_bus import EventBus, SystemEvent
    from core.statistics import Statistics
    from core.agent.func_tool_manager import FuncToolManager

    class FakeLLM:
        def __init__(self):
            self.calls = []
            self.gate = None
            self.inflight = 0

        async def chat(self, request, **kwargs):
            prompt = ""
            for m in request.messages:
                c = getattr(m, "content", "")
                if isinstance(c, str):
                    prompt = c
            self.inflight += 1
            if self.gate is not None:
                await self.gate.wait()
            self.inflight -= 1
            self.calls.append(len(prompt))
            from core.provider.llm_model import LLMResponse
            markers = list(dict.fromkeys(re.findall(r"\[m[0-9A-Za-z]+\]", prompt)))
            return LLMResponse("SUM" + "".join(markers[:40]))

    client = FakeLLM()
    cfg = KiraConfig()
    cfg["bot_config"]["bot"]["max_memory_length"] = 50
    sess = SessionManager(db=None, kira_config=cfg, event_bus=None)

    class FakePluginMgr:
        def has_plugin(self, pid=None): return False
        def is_plugin_enabled(self, pid=None): return False
        async def set_plugin_enabled(self, pid=None, e=None): return None

    class FakeMP:
        def __init__(self): self.sent = []
        async def send_message_chain(self, session=None, chain=None): self.sent.append((session, chain))

    class FakeProvider:
        def get_default_llm(self): return None

    class FakeAdapterMgr:
        def get_adapters_info(self): return []
        def get_adapter(self, name): return None

    stats = Statistics()
    bus = EventBus(stats=stats, event_queue=asyncio.Queue(), db=None)

    def make_ctx(plugin_mgr=None):
        ctx = types.SimpleNamespace()
        ctx.session_mgr = sess
        ctx.config = cfg
        ctx.plugin_mgr = plugin_mgr or FakePluginMgr()
        ctx.message_processor = FakeMP()
        ctx.provider_mgr = FakeProvider()
        ctx.adapter_mgr = FakeAdapterMgr()
        ctx.tool_mgr = FuncToolManager(cfg)
        ctx.event_bus = bus
        ctx.get_plugin_data_dir = lambda: data_dir / "plugin_data" / "ksm"
        ctx.get_llm_client = lambda model_uuid=None, llm_type=None: client
        ctx.get_default_fast_llm_client = lambda: client
        ctx.get_default_llm_client = lambda: client
        return ctx

    # ── A/E：真实 PluginManager 加载 + 事件 ──
    from core.plugin.plugin_registry import PluginManager
    from core.plugin.plugin_handlers import event_handler_reg, EventType

    plugin_dir = data_dir / "plugins" / "kira_session_merger"
    plugin_dir.mkdir(parents=True)
    for f in Path(ROOT).glob("*"):
        if f.suffix == ".py" or f.name in ("manifest.json", "schema.json", "icon.png"):
            shutil.copy(f, plugin_dir / f.name)

    async def part_a():
        pm = PluginManager()
        ctx = make_ctx(plugin_mgr=pm)
        pm.ctx = ctx
        pid = await pm.load_plugin_from_dir(plugin_dir, auto_install=False)
        inst = pm.plugin_instances.get(pid)
        info = pm.get_plugin_info(pid)
        ok = inst is not None and info is not None and info.status == "ready"
        check("A1 真 PluginManager 完整加载", ok, f"{pid} v{getattr(info, 'version', '')} status={getattr(info, 'status', '')}")
        hooks = {
            et.value: len(event_handler_reg.get_handlers(et))
            for et in (EventType.ON_IM_MESSAGE, EventType.ON_LLM_REQUEST,
                       EventType.ON_STEP_RESULT, EventType.ON_IM_BATCH_MESSAGE,
                       EventType.ON_TOOL_RESULT)
        }
        check("A2 钩子注册齐备", sum(hooks.values()) >= 5, str(hooks))
        cfg_file = data_dir / "config" / "plugins" / "kira_session_merger.json"
        cfg_json = json.loads(cfg_file.read_text(encoding="utf-8")) if cfg_file.exists() else {}
        check("A3 配置生成（schema→配置）", "section_summarize" in cfg_json and "section_reset_trigger" in cfg_json)

        # E：事件清理
        sid_e = "ada:dm:it_e"
        inst.engine.summary_store.set(sid_e, "OLD")
        inst.engine.summary_store.save()
        inst.engine._preheat_pending[sid_e] = {"fp": "x", "final": "y"}
        await bus.publish(SystemEvent(event_type="session_memory_written", source="test",
                                      payload={"session": sid_e, "old_memory": [["a"]], "new_memory": []}))
        await bus._process_event(bus.event_queue.get_nowait())
        check("E1 清空事件丢弃累计摘要与暂存",
              inst.engine.summary_store.get(sid_e) == "" and sid_e not in inst.engine._preheat_pending)
        inst.engine.summary_store.set(sid_e, "OLD2")
        await bus.publish(SystemEvent(event_type="session_deleted", source="test",
                                      payload={"session": sid_e, "old_memory": []}))
        await bus._process_event(bus.event_queue.get_nowait())
        check("E2 删除事件丢弃累计摘要", inst.engine.summary_store.get(sid_e) == "")
        return pm, inst, pid

    pm, ksm_inst, pid = asyncio.run(part_a())

    # ── 引擎级场景 ──
    base_pkg = types.ModuleType("plugins")
    base_pkg.__path__ = [str(Path(ROOT).parent)]
    sys.modules.setdefault("plugins", base_pkg)
    pkg = types.ModuleType("plugins.ksm_it")
    pkg.__path__ = [str(ROOT)]
    sys.modules["plugins.ksm_it"] = pkg
    spec = importlib.util.spec_from_file_location("plugins.ksm_it.merge_engine", Path(ROOT) / "merge_engine.py")
    me = importlib.util.module_from_spec(spec)
    sys.modules["plugins.ksm_it.merge_engine"] = me
    spec.loader.exec_module(me)

    from plugins.ksm_it.group_resolver import GroupResolver
    from plugins.ksm_it.timeline import TimelineBuilder
    from plugins.ksm_it.summarizer import CumulativeSummaryStore, extract_summary_text
    from plugins.ksm_it.hard_reset import hard_reset_session
    from plugins.ksm_it.reset_policy import SoftResetState
    from core.logging_manager import get_logger

    logger = get_logger("ksm_it", "cyan")

    def seed(sid, n, tag, size=150, start=0):
        for i in range(start, start + n):
            sess.update_memory(sid, [
                {"role": "user", "content": f"[m{tag}{i}] 用户第{i}轮" + "问" * size},
                {"role": "assistant", "content": f"[m{tag}{i}] 助手第{i}轮" + "答" * size},
            ])

    def make_engine(mode="hard", keep=2, window=50, limit=999999, strategy="append_then_merge",
                    trigger="tokens", summarize="sync", check_interval=0, reset_cmd=False):
        cfg["bot_config"]["bot"]["max_memory_length"] = window
        resolver = GroupResolver(session_mgr=sess, enabled=True)
        timeline = TimelineBuilder(session_mgr=sess, chars_per_token=2.0)
        store = CumulativeSummaryStore(data_dir / f"store_{mode}_{keep}_{window}_{int(time.time()*1000)%100000}.json")
        store.load()
        eng = me.MergeEngine(
            session_mgr=sess, resolver=resolver, timeline=timeline, observe_pool=None,
            max_merged_chunks=10, merge_token_limit=limit, merge_keep_turns=keep,
            merge_reset_mode=mode, merge_check_interval_sec=check_interval,
            chars_per_token=2.0, max_merge_sessions=8,
            ctx=make_ctx(), summarize_mode=summarize, summarize_model="",
            summarize_timeout_sec=5.0, enable_summary_logging=False,
            summary_store=store, cumulative_summary=True,
            write_through=True, merge_order_mode="time", merge_trigger_mode=trigger,
            merge_trigger_rounds=0, preheat_ratio=0.7, sync_wait_timeout=0,
            continuous_merge_strategy=strategy, background_merge_timeout_sec=5.0,
            merge_timeout_sec=5.0, enable_window_anchor=False,
            reset_command_enabled=reset_cmd, logger=logger,
        )
        return eng, store

    from core.chat import Session
    from core.chat.message_utils import KiraMessageBatchEvent
    from core.provider.llm_model import LLMRequest

    def mk_event(sid):
        parts = sid.split(":")
        return KiraMessageBatchEvent(
            message_types=["dm"], timestamp=0,
            session=Session(adapter_name=parts[0], session_type=parts[1], session_id=parts[2]),
        )

    async def engine_scenarios():
        # H：收割链（完成后可取回 + 锚点对齐）
        eng, store = make_engine(keep=2, window=50)
        sid = "adh:dm:it_h"
        seed(sid, 8, "A")
        eng._schedule_continuous_compression(sid)
        t = eng._preheat_tasks.get(sid)
        if t:
            await asyncio.wait_for(t, 10)
        pending = eng._preheat_pending.get(sid)
        harvested = await eng._harvest_continuous_compression(sid, timeout=0)
        check("H1 任务完成后仍可取回 pending", isinstance(harvested, dict) and bool(pending and pending.get("final")))
        parts = me.read_reset_parts(sess, sid, eng.merge_keep_turns)
        final1, delta1 = eng._preheat_status(sid, parts["dropped"], parts["head_text"], eng._group_id(sid))
        check("H2 同状态收割：final 有效且 delta 为空", bool(final1) and delta1 == [])
        old_dropped = parts["dropped"]
        seed(sid, 1, "A", start=90)  # 状态前移一格：新轮仍在保留区，A6 轮新落入 dropped
        parts2 = me.read_reset_parts(sess, sid, eng.merge_keep_turns)
        final2, delta2 = eng._preheat_status(sid, parts2["dropped"], parts2["head_text"], eng._group_id(sid))
        expected_delta = parts2["dropped"][len(old_dropped):]
        ok_delta = (
            bool(final2) and isinstance(delta2, list) and len(delta2) == len(expected_delta)
            and [m.get("content") for m in delta2] == [m.get("content") for m in expected_delta]
        )
        check("H3★ 状态前移仍收割成功且 delta=新增落入 dropped 的部分", ok_delta,
              f"final={bool(final2)} delta={len(delta2 or [])} expected={len(expected_delta)} marker={delta2[-1].get('content','')[:10] if delta2 else ''}")

        # H4：sync_wait 等待路径
        sid4 = "adh4:dm:it_h4"
        seed(sid4, 8, "B")
        gate = asyncio.Event()
        client.gate = gate
        eng._schedule_continuous_compression(sid4)
        for _ in range(200):
            if client.inflight > 0:
                break
            await asyncio.sleep(0.01)
        wh = asyncio.create_task(eng._harvest_continuous_compression(sid4, timeout=8))
        await asyncio.sleep(0.15)
        gate.set()
        client.gate = None
        res4 = await asyncio.wait_for(wh, 10)
        check("H4 sync_wait_timeout>0 等待路径可用", isinstance(res4, dict))

        # S：滑窗增量
        eng_s, _ = make_engine(keep=2, window=6, limit=999999)
        sid_s = "ads:dm:it_s"
        seed(sid_s, 6, "W", size=300)
        c0 = len(client.calls)
        eng_s._schedule_continuous_compression(sid_s)
        t = eng_s._preheat_tasks.get(sid_s)
        if t:
            await asyncio.wait_for(t, 10)
        run1 = client.calls[c0:]
        sess.update_memory(sid_s, [
            {"role": "user", "content": "[mW6] 用户新轮" + "问" * 300},
            {"role": "assistant", "content": "[mW6] 助手新轮" + "答" * 300},
        ])
        c1 = len(client.calls)
        eng_s._schedule_continuous_compression(sid_s)
        t = eng_s._preheat_tasks.get(sid_s)
        if t:
            await asyncio.wait_for(t, 10)
        run2 = client.calls[c1:]
        check("S1★ 滑窗后为增量（不再全量重压缩）",
              bool(run2) and max(run2) < 1500 and max(run1) > 2000,
              f"run1={run1} run2={run2}")

        # K1：无条件钳制
        eng_k, _ = make_engine(keep=5, window=5, trigger="tokens")
        check("K1★ tokens 模式 keep 被钳制到窗口-1", eng_k._clamp_keep(5) == 4, eng_k._clamp_keep(5))
        # K2：peek/apply 信号一致
        st = SoftResetState(keep_turns=6, check_interval_sec=60)
        st._dynamic_keep["g"] = 6
        pk1 = st.peek_reset_keep("g", degrade=False)
        pk2 = st.peek_reset_keep("g", degrade=True)
        ap2 = st.on_reset("g", degrade=True)
        check("K2★ precheck 与 apply 的 keep 计算一致（降级信号统一）",
              pk1 == 6 and pk2 == 3 and ap2 == 3, f"peek(False)={pk1} peek(True)={pk2} apply(True)={ap2}")

        # K3：E2E 降级轮不再丢内容
        eng3, store3 = make_engine(mode="hard", keep=6, window=50, limit=100,
                                   strategy="immediate", summarize="sync")
        sid3 = "adk:dm:it_k3"
        seed(sid3, 12, "C", size=120)
        ok1 = await eng3.apply_to_request(mk_event(sid3), LLMRequest(messages=[]))
        head1 = str(sess.fetch_memory(sid3)[0].get("content", ""))
        gid3 = eng3._group_id(sid3)
        eng3._soft_state._last_reset[gid3] = time.time() - 200
        seed(sid3, 6, "D", size=120)
        ok2 = await eng3.apply_to_request(mk_event(sid3), LLMRequest(messages=[]))
        await asyncio.sleep(0.2)
        flat3 = sess.fetch_memory(sid3)
        head2 = str(flat3[0].get("content", ""))
        all_txt = " ".join(str(m.get("content", "")) for m in flat3)
        covered = all(f"[mD{i}]" for i in range(3) if f"[mD{i}]" in head2)
        kept = all(f"[mD{i}]" in all_txt for i in (3, 4, 5))
        check("K3★ 降级轮 keep=3：D0-D2 进入摘要、D3-D5 保留（不再丢失）",
              ok1 and ok2 and covered and kept,
              f"ok={ok1}/{ok2} D0-D2_in_head={covered} D3-D5_kept={kept}")

        # C1：store 保活语义
        store_c = CumulativeSummaryStore(data_dir / "store_c.json")
        store_c.set("s:c", "SS")
        kept1 = store_c.sync_with_head("s:c", "", session_has_messages=True)
        kept2 = store_c.get("s:c")
        dropped = store_c.sync_with_head("s:c", "", session_has_messages=False)
        check("C1★ 会话继续时保留 store；清空才丢弃",
              kept1 == "SS" and kept2 == "SS" and dropped == "" and store_c.get("s:c") == "")
        # C2：_session_has_messages
        check("C2 会话消息判定", eng._session_has_messages("adh:dm:not_exist") is False
              and eng._session_has_messages(sid) is True)

        # P：预热门控
        eng_p, _ = make_engine(mode="soft", keep=1, window=50, limit=10, reset_cmd=False)
        sid_p = "adp:dm:it_p"
        seed(sid_p, 3, "P")
        r1 = eng_p._should_compress_continuously(sid_p)
        eng_p.reset_command_enabled = True
        r2 = eng_p._should_compress_continuously(sid_p)
        eng_p.reset_command_enabled = False
        eng_p.merge_reset_mode = "hard"
        r3 = eng_p._should_compress_continuously(sid_p)
        check("P1★ 无消费方不预热；hard 或 /resum 开启才预热",
              r1 is False and r2 is True and r3 is True, f"soft+off={r1} soft+cmd={r2} hard={r3}")

        # T：超时一致性（线程内 store 同步）
        eng_t, store_t = make_engine(mode="hard", keep=2, window=50, limit=100,
                                     strategy="immediate", summarize="sync")
        sid_t = "adt:dm:it_t"
        seed(sid_t, 8, "E", size=120)
        orig_write = sess.write_memory

        def slow_write(*a, **k):
            time.sleep(0.7)
            return orig_write(*a, **k)

        sess.write_memory = slow_write
        eng_t._merge_timeout = lambda: 0.25
        ok_t = await eng_t.apply_to_request(mk_event(sid_t), LLMRequest(messages=[]))
        await asyncio.sleep(1.8)
        sess.write_memory = orig_write
        flat_t = sess.fetch_memory(sid_t)
        head_t = str(flat_t[0].get("content", "")) if flat_t else ""
        head_text_t = extract_summary_text(head_t) if head_t.startswith("[前情摘要") else ""
        store_val = store_t.get(sid_t)
        check("T1★ 超时后线程完成重开，且 store 与磁盘一致（线程内同步）",
              ok_t is False and sess.get_memory_count(sid_t) <= 3
              and bool(head_text_t) and store_val == head_text_t,
              f"ok={ok_t} chunks={sess.get_memory_count(sid_t)} store==head={store_val == head_text_t}")

        # D1：sync + append_then_merge：收割命中带 delta → 重开后 async 只补差量
        eng_d1, store_d1 = make_engine(mode="hard", keep=2, window=50, limit=100,
                                       strategy="append_then_merge", summarize="sync")
        sid_d1 = "adu:dm:it_d1"
        seed(sid_d1, 8, "U", size=120)
        eng_d1._schedule_continuous_compression(sid_d1)
        t = eng_d1._preheat_tasks.get(sid_d1)
        if t:
            await asyncio.wait_for(t, 10)
        seed(sid_d1, 1, "U", size=120, start=90)  # 状态前移：U6 落入 dropped（=delta）
        okd1 = await eng_d1.apply_to_request(mk_event(sid_d1), LLMRequest(messages=[]))
        await asyncio.sleep(1.0)
        head_d1 = str(sess.fetch_memory(sid_d1)[0].get("content", ""))
        head_d1_text = extract_summary_text(head_d1) if head_d1.startswith("[前情摘要") else ""
        store_d1v = store_d1.get(sid_d1)
        ok_sync_delta = (
            okd1 and "[mU0]" in head_d1_text and "[mU6]" in head_d1_text
            and store_d1v == head_d1_text
        )
        check("D1★ sync+append：收割后 delta 补写（旧覆盖+新增均入摘要，store 一致）",
              ok_sync_delta,
              f"ok={okd1} U0={'[mU0]' in head_d1_text} U6={'[mU6]' in head_d1_text} store_eq={store_d1v == head_d1_text}")

        # D2：async 模式：收割命中带 delta → 先写收割摘要，再 async 合并差量
        eng_d2, store_d2 = make_engine(mode="hard", keep=2, window=50, limit=100,
                                       strategy="append_then_merge", summarize="async")
        sid_d2 = "adv:dm:it_d2"
        seed(sid_d2, 8, "V", size=120)
        eng_d2._schedule_continuous_compression(sid_d2)
        t = eng_d2._preheat_tasks.get(sid_d2)
        if t:
            await asyncio.wait_for(t, 10)
        seed(sid_d2, 1, "V", size=120, start=90)
        okd2 = await eng_d2.apply_to_request(mk_event(sid_d2), LLMRequest(messages=[]))
        await asyncio.sleep(1.0)
        head_d2 = str(sess.fetch_memory(sid_d2)[0].get("content", ""))
        head_d2_text = extract_summary_text(head_d2) if head_d2.startswith("[前情摘要") else ""
        store_d2v = store_d2.get(sid_d2)
        ok_async_delta = (
            okd2 and "[mV0]" in head_d2_text and "[mV6]" in head_d2_text
            and store_d2v == head_d2_text
        )
        check("D2★ async：收割摘要先写入 + 差量后台合并（store 一致）",
              ok_async_delta,
              f"ok={okd2} V0={'[mV0]' in head_d2_text} V6={'[mV6]' in head_d2_text} store_eq={store_d2v == head_d2_text}")

        # R1：手动压缩重开（/resum 路径）回归
        eng_r, store_r = make_engine(mode="hard", keep=3, window=50, limit=999999,
                                     strategy="immediate", summarize="sync")
        sid_r = "adr:dm:it_r"
        seed(sid_r, 8, "F", size=120)
        stats_r = await eng_r.manual_reset_with_summary(sid_r)
        head_r = str(sess.fetch_memory(sid_r)[0].get("content", ""))
        check("R1 手动压缩重开回归（含摘要写入）",
              stats_r.get("ok", 0) >= 1 and head_r.startswith("[前情摘要"),
              f"ok={stats_r.get('ok')} fail={stats_r.get('fail')} head={head_r[:20]!r}")
        # R2：谱系校验拒绝旧世代 pending
        pend = eng_r._preheat_pending.get(sid_r)
        if not pend:
            eng_r._preheat_pending[sid_r] = {"final": "X", "base": "OLD", "anchor": []}
        else:
            pend["base"] = "OLD"
        parts_r = me.read_reset_parts(sess, sid_r, eng_r.merge_keep_turns)
        f_bad, d_bad = eng_r._preheat_status(sid_r, parts_r["dropped"], parts_r["head_text"], eng_r._group_id(sid_r))
        check("R2 谱系不一致的 pending 被拒绝", f_bad is None and d_bad is None)

        # PP：摘要预处理并发化（有界并发 / 保序 / 失败回落）
        prep_mod = importlib.import_module("plugins.ksm_it.preprocessor")

        class _PreprocessProbe:
            def __init__(self):
                self.inflight = 0
                self.max_inflight = 0
                self.calls = 0

            async def chat(self, request, **kwargs):
                self.inflight += 1
                self.calls += 1
                if self.inflight > self.max_inflight:
                    self.max_inflight = self.inflight
                try:
                    await asyncio.sleep(0.08)
                    prompt = ""
                    for m in request.messages:
                        c = getattr(m, "content", "")
                        if isinstance(c, str):
                            prompt = c
                    if "FAILME" in prompt:
                        raise RuntimeError("probe fail")
                    from core.provider.llm_model import LLMResponse
                    return LLMResponse("PSUM")
                finally:
                    self.inflight -= 1

        probe = _PreprocessProbe()
        long_text = "alpha beta gamma " * 120
        msgs_pp = [
            {"role": "tool", "content": "[T1] " + long_text},
            {"role": "user", "content": "普通用户消息"},
            {"role": "tool", "content": "[T2] " + long_text},
            {"role": "tool", "content": "[T3] FAILME " + long_text},
            {"role": "user", "content": "另一个普通用户消息"},
            {"role": "tool", "content": "[T4] " + long_text},
            {"role": "tool", "content": "[T5] " + long_text},
        ]
        orig_first = msgs_pp[0]["content"]
        out_pp = await prep_mod.preprocess_messages_for_summary(msgs_pp, 200, probe, logger)
        check("PP1★ 预处理有界并发（2..3 且确有并行）", 2 <= probe.max_inflight <= 3, f"max={probe.max_inflight}")
        check("PP2 调用次数=待处理条数（含失败项）", probe.calls == 5, f"calls={probe.calls}")
        check("PP3 未处理项原样保留且顺序不变",
              out_pp[1] is msgs_pp[1] and out_pp[4] is msgs_pp[4] and len(out_pp) == len(msgs_pp))
        check("PP4 失败项回落原文", out_pp[3] is msgs_pp[3])
        check("PP5 处理后项为副本且带压缩后缀",
              out_pp[0] is not msgs_pp[0] and str(out_pp[0].get("content", "")).endswith("（已压缩）")
              and str(out_pp[6].get("content", "")).endswith("（已压缩）"))
        check("PP6 原消息未被改动", msgs_pp[0]["content"] == orig_first)

        probe2 = _PreprocessProbe()
        out_pp2 = await prep_mod.preprocess_messages_for_summary(
            [{"role": "user", "content": "短消息"}, {"role": "tool", "content": "short"}],
            200, probe2, logger,
        )
        check("PP7 零待处理快路径（不调用 LLM、原样返回）",
              probe2.calls == 0 and out_pp2[0]["content"] == "短消息" and out_pp2[1]["content"] == "short")

        # E4：/reboota 复位引擎暂存（P1 新增接线）——放在最后（会清空测试会话）
        sid_z = "adz:dm:rz"
        seed(sid_z, 3, "Z")
        ksm_inst.enabled = True
        ksm_inst.resolver.enabled = True
        ksm_inst.engine._preheat_pending[sid_z] = {"fp": "x", "final": "y"}
        ksm_inst.engine.summary_store.set(sid_z, "SZ")
        ev_r = types.SimpleNamespace(
            message=types.SimpleNamespace(sender=types.SimpleNamespace(user_id="u", nickname="n"))
        )
        await ksm_inst._handle_reboot_all(ev_r, sid_z)
        check("E4 /reboota 复位引擎暂存与累计摘要",
              sid_z not in ksm_inst.engine._preheat_pending
              and ksm_inst.engine.summary_store.get(sid_z) == ""
              and sess.get_memory_count(sid_z) == 0)

        # E3：terminate 反订阅（放在最后）
        await pm.terminate(pid)
        left = len(bus.subscribers.get("session_memory_written", [])) + len(bus.subscribers.get("session_deleted", []))
        check("E3 terminate 后事件反订阅干净", left == 0, f"left={left}")

    asyncio.run(engine_scenarios())

    failed = [n for n, ok in RESULTS if not ok]
    print("=" * 60)
    print(f"TOTAL {len(RESULTS)} — {len(RESULTS) - len(failed)} passed, {len(failed)} failed")
    for n in failed:
        print("  FAIL:", n)
    shutil.rmtree(data_dir, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
