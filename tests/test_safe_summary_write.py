# -*- coding: utf-8 -*-
"""2.9.3 回归：安全摘要写回（并发 append 不丢）+ 计数补齐。

独立套件，run_all.sh 会自动执行；找不到框架时 SKIP。
"""
from __future__ import annotations

import asyncio
import importlib.util
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = []


def check(label, cond, detail=""):
    RESULTS.append((label, bool(cond)))
    print(("  PASS  " if cond else "  FAIL  ") + label + ((" — " + str(detail)) if detail else ""))


def find_framework():
    for env in ("KIRA_FRAMEWORK_PATH", "KIRA_CORE_PATH"):
        v = os.environ.get(env)
        if v and (Path(v) / "core" / "plugin" / "__init__.py").exists():
            return Path(v).resolve()
    for name in ("KiraAI", "KiraAI-main", "kira_upstream"):
        p = ROOT.parent / name
        if (p / "core" / "plugin" / "__init__.py").exists():
            return p
    return None


FW = find_framework()
if FW is None:
    print("SKIP: 未找到 KiraAI 框架源码（设置 KIRA_FRAMEWORK_PATH）")
    sys.exit(0)

WORK = Path(tempfile.mkdtemp(prefix="ksm_safe_"))
for sub in ("data/config", "data/memory"):
    (WORK / sub).mkdir(parents=True, exist_ok=True)
(WORK / "data/log.log").touch()
os.chdir(WORK)
sys.path.insert(0, str(FW))

from core.config import KiraConfig
from core.db.db_mgr import DatabaseManager
from core.db.service import DatabaseService
from core.chat.session_manager import SessionManager
from core.agent.message import OpenAIMessage

# 以包形式加载 merge_engine / reset_policy
pkg = types.ModuleType("ksm_probe")
pkg.__path__ = [str(ROOT)]
sys.modules["ksm_probe"] = pkg


def load_mod(name):
    spec = importlib.util.spec_from_file_location(f"ksm_probe.{name}", ROOT / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"ksm_probe.{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


me = load_mod("merge_engine")
rp = load_mod("reset_policy")
MergeEngine = me.MergeEngine


async def main():
    cfg = KiraConfig()
    cfg["bot_config"]["bot"]["max_memory_length"] = 10
    dbm = DatabaseManager(f"sqlite+aiosqlite:///{WORK}/data/safe.db")
    await dbm.init()
    db = DatabaseService(dbm)
    await db.init_tables()
    sm = SessionManager(db, cfg)
    if hasattr(sm, "initialize"):
        await sm.initialize()

    sid = "safe:dm:1"
    if hasattr(sm, "get_session_info"):
        sm.get_session_info(sid)
    else:
        await sm.get_session(sid, ensure_session=True)

    summary = {
        "role": "user",
        "content": "[前情摘要|系统注入] 以下是更早对话被清理前的自动摘要，仅供延续上下文参考：\nOLD",
    }
    chunks = [[summary, {"role": "user", "content": "q0"},
               {"role": "assistant", "content": "a0"}]]
    for i in range(1, 4):
        chunks.append([{"role": "user", "content": f"q{i}"},
                       {"role": "assistant", "content": f"a{i}"}])
    if hasattr(sm, "write_memory"):
        sm.write_memory(sid, chunks)
    else:
        await sm.update_session(sid, content=chunks)

    class Dummy:
        pass

    d = Dummy()
    d.session_mgr = sm

    async def append_turn():
        r = sm.update_memory(sid, [{"role": "user", "content": "MEANWHILE"},
                                   {"role": "assistant", "content": "OK"}])
        if hasattr(r, "__await__"):
            await r

    # 先 append 提交，再写摘要：旧实现读旧快照会把它覆盖掉
    task = asyncio.create_task(append_turn())
    await asyncio.sleep(0.02)
    applied, head = await MergeEngine._write_summary_head_safe(d, sid, "OLD", "NEW")
    await task

    if hasattr(sm, "read_memory"):
        ch = [list(c) for c in (sm.read_memory(sid) or [])]
    else:
        s = await sm.get_session(sid, ensure_session=False)
        ch = [list(c) for c in (s.content if s else [])]
    flat = [m for c in ch for m in c]
    check("安全写回 applied", applied)
    check("摘要头已替换为 NEW", str(flat[0].get("content", "")).endswith("NEW"), flat[0].get("content", "")[:20])
    check("并发 append 未丢", any(m.get("content") == "MEANWHILE" for m in flat))
    check("chunk 数正确", len(ch) == 5, f"chunks={len(ch)}")

    # 头部不一致时不写
    applied2, head2 = await MergeEngine._write_summary_head_safe(d, sid, "WRONG", "X")
    check("头部不一致拒绝写", (not applied2) and head2 == "NEW")

    # 计数补齐
    tc = [{"id": "1", "type": "function",
           "function": {"name": "write_file", "arguments": '{"content":"' + "甲" * 400 + '"}'}}]
    msg = OpenAIMessage(role="assistant", content="", tool_calls=tc,
                        reasoning_content="思考" * 100)
    n = rp.count_messages_tokens([msg], 2.0)
    check("tool_calls + reasoning 计入", n > 300, f"tokens={n}")
    list_msg = OpenAIMessage(role="user", content=[{"type": "text", "text": "图" * 500},
                                                   {"type": "image_url", "image_url": {"url": "x"}}])
    n2 = rp.count_messages_tokens([list_msg], 2.0)
    check("list 只取 text", 245 <= n2 <= 260, f"tokens={n2}")


try:
    asyncio.run(main())
finally:
    shutil.rmtree(WORK, ignore_errors=True)

failed = [r for r in RESULTS if not r[1]]
print("=" * 60)
print(f"TOTAL {len(RESULTS) - len(failed)}/{len(RESULTS)} passed")
sys.exit(1 if failed else 0)
