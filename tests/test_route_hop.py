# -*- coding: utf-8 -*-
"""v2.8.3 回归：跨会话 hop 上限（防乒乓）+ 观察池异步落盘。

覆盖：
  H1 路由正文带 hop 行，route_hop_of_text 正确解析
  H2 旧格式（无 hop 行）路由正文按 hop=1 兼容
  H3 非路由文本返回 0；[merge_cross_session_request] 标记识别不受影响
  H4 hop >= ROUTE_MAX_HOPS 时 route_cross_session_request 拒绝投递（不发布）
  H5 hop=1 正常投递且正文 hop 递增
  O1 观察池节流落盘在事件循环里异步执行（不阻塞 add，文件最终落盘）

Run: python3 tests/test_route_hop.py
"""
import asyncio
import importlib.util
import json
import os
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# --- minimal KiraAI stubs so cross_session.py imports standalone ---
for name in ("core", "core.chat", "core.chat.message_elements"):
    sys.modules[name] = types.ModuleType(name)


class _Text:
    def __init__(self, text):
        self.text = text


class _MessageChain:
    def __init__(self, elems):
        self.chain = list(elems)


sys.modules["core.chat"].MessageChain = _MessageChain
sys.modules["core.chat.message_elements"].Text = _Text

# 包式加载（模块内是相对导入）
_pkg = "ksm_hoppkg"
_p = types.ModuleType(_pkg)
_p.__path__ = [ROOT]
_p.__package__ = _pkg
sys.modules[_pkg] = _p


def _load(sub):
    spec = importlib.util.spec_from_file_location(
        f"{_pkg}.{sub}", os.path.join(ROOT, f"{sub}.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules[f"{_pkg}.{sub}"] = m
    spec.loader.exec_module(m)
    return m


cs = _load("cross_session")
op = _load("observe_pool")

results = []


def check(label, cond, detail=""):
    results.append((label, bool(cond)))
    print(("  PASS  " if cond else "  FAIL  ") + label + ((" — " + str(detail)) if (detail and not cond) else ""))


class FakeCtx:
    def __init__(self):
        self.adapter_mgr = None
        self.session_mgr = None
        self.published = []

    async def publish_notice(self, target, chain, is_mentioned=False):
        self.published.append((target, chain, is_mentioned))


async def main():
    # H1/H3 hop 解析
    text1 = cs.build_route_notice_text("qq:gm:1", "干活", hop=1)
    check("H1 路由正文含 hop 行且解析为 1", cs.route_hop_of_text(text1) == 1)
    check("H3 非路由文本 hop=0", cs.route_hop_of_text("普通消息") == 0)
    check("H3 ROUTE 标记识别不受影响",
          cs.is_merge_route_request_text(text1) and not cs.is_merge_route_request_text("普通消息"))

    # H2 旧格式（无 hop 行）按 1 兼容
    legacy = (
        f"{cs.ROUTE_MARKER}\n"
        "source_session: qq:gm:1\n"
        "\n补充说明：\n旧任务\n"
    )
    check("H2 旧格式路由正文按 hop=1 兼容", cs.route_hop_of_text(legacy) == 1)

    # H4 hop 超限拒绝投递（不发布 notice）
    ctx = FakeCtx()
    ok, msg = await cs.route_cross_session_request(
        ctx, source_sid="qq:gm:1", target="qq:dm:2", description="t", hop=cs.ROUTE_MAX_HOPS
    )
    check("H4 hop>=上限拒绝投递", not ok and not ctx.published, msg)
    check("H4 拒绝文案提示在当前会话作答", "hop limit" in msg)

    # H5 hop=1 正常投递且正文 hop 递增
    ctx2 = FakeCtx()
    ok, _ = await cs.route_cross_session_request(
        ctx2, source_sid="qq:gm:1", target="qq:dm:2", description="t", hop=1
    )
    check("H5 hop=1 正常投递", ok and len(ctx2.published) == 1)
    delivered = ctx2.published[0][1].chain[0].text
    check("H5 投递正文 hop=1", cs.route_hop_of_text(delivered) == 1)

    # O1 观察池节流落盘异步执行：add 不阻塞、文件最终落盘
    with tempfile.TemporaryDirectory() as d:
        pool = op.ObservePool(data_dir=d, flush_every=3, logger=None)
        for i in range(3):
            pool.add(sid="qq:gm:1", content=f"msg{i}", message_id=str(i))
        # 等异步落盘任务完成
        for _ in range(50):
            path = os.path.join(d, "observe", "observe.json")
            if os.path.exists(path):
                break
            await asyncio.sleep(0.05)
        check("O1 节流触发后文件异步落盘", os.path.exists(path))
        if os.path.exists(path):
            data = json.load(open(path, encoding="utf-8"))
            check("O1 落盘内容完整", len(data.get("qq:gm:1", [])) == 3)
        pool.flush(force=True)  # 同步路径仍可用（terminate/clear）

    print()
    passed = sum(1 for _, ok in results if ok)
    print("TOTAL %d/%d passed" % (passed, len(results)))
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    asyncio.run(main())
