# -*- coding: utf-8 -*-
"""Static consistency audit for the history locate feature.

Checks that would have caught the class of bug this feature is prone to:
  * a schema option that is never read (silently does nothing)
  * a config option read but missing from schema (unconfigurable)
  * a locate_cfg key the service never consumes (dead option)
  * hardcoded plugin ids / API paths drifting from the manifest
  * locate.py + onebot_compat.py identical between hp and KSM

Run: python3 tests/test_consistency.py
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
results = []


def check(label, cond, detail=""):
    results.append((label, bool(cond)))
    print(("  PASS  " if cond else "  FAIL  ") + label + ((" — " + str(detail)) if detail else ""))


# Which plugin are we auditing? Decide from the manifest, never from the
# directory name - a clone can live anywhere (and a fork is not named "smp").
_manifest = json.load(open(os.path.join(ROOT, "manifest.json"), encoding="utf-8"))
PLUGIN_ID = _manifest.get("plugin_id", "")
IS_KSM = PLUGIN_ID == "kira_session_merger"
MAIN = os.path.join(ROOT, "main.py")
TOOL = os.path.join(ROOT, "history_tool.py") if IS_KSM else MAIN
SCHEMA = os.path.join(ROOT, "schema.json")

print("\n[schema <-> code]")
schema = json.load(open(SCHEMA, encoding="utf-8"))
if IS_KSM:
    fields = schema["section_history_tool"]["fields"]
else:
    fields = schema

main_src = open(MAIN, encoding="utf-8").read()
tool_src = open(TOOL, encoding="utf-8").read()
combined = main_src + tool_src

schema_keys = set(fields.keys())
# Every option name must appear as a config lookup somewhere.
unread = [k for k in sorted(schema_keys)
          if f'"{k}"' not in combined and f"'{k}'" not in combined]
check("no schema option is unread by the code", not unread, unread)

# Every locate_cfg key the service consumes must be configurable.
# KSM: the service takes a locate_cfg dict. hp: the plugin class reads cfg
# directly. Either way, whatever the code reads must exist in schema.json.
if IS_KSM:
    cfg_block = re.search(r"locate_cfg = locate_cfg or \{\}(.*?)\n\n", tool_src, re.S)
    consumed = set(re.findall(r'locate_cfg\.get\("([a-z_]+)"', cfg_block.group(1))) \
        if cfg_block else set()
    check("locate_cfg consumption block found", bool(cfg_block))
else:
    # hp: collect every cfg.get("...") inside __init__'s locate section.
    block = re.search(r"# ---------- locate \(time / user / keyword\) ----------(.*?)\n\n",
                      main_src, re.S)
    consumed = set(re.findall(r'cfg\.get\("([a-z_]+)"', block.group(1))) \
        if block else set()
    check("locate config block found", bool(block))
missing = sorted(consumed - schema_keys)
check("every consumed locate option is in schema.json", not missing, missing)

# And the reverse: every locate_* option in schema must be consumed.
locate_schema = {k for k in schema_keys if k in (
    "enable_locate", "enable_keyword", "enable_time_range", "enable_user_filter",
    "default_scan_limit", "max_scan_limit", "scan_max_seconds",
    "max_fetch_per_request", "fetch_timeout_sec", "max_scanned_per_turn",
    "early_stop_on_enough", "detect_boundary",
    "keyword_case_sensitive", "max_keywords", "offset_max", "max_return_count",
    "locate_fallback_on_error", "locate_head_meta", "locate_cache_ttl_sec")}
dead = [k for k in sorted(locate_schema) if f'"{k}"' not in tool_src]
check("no dead locate option", not dead, dead)

print("\n[tool params <-> service signature]")
tool_params = set(re.findall(r'"([a-z_]+)":\s*\{\s*"type"', combined))
svc_params = set(re.findall(r"^\s{8}([a-z_]+):\s*(?:Optional\[str\]|str|int)\s*=",
                            tool_src, re.M))
declared = set(re.findall(r'"([a-z_]+)",', combined))
del declared
locate_params = {"since", "until", "user_id", "keyword", "offset", "scan_limit"}
check("all locate params declared in the tool schema", locate_params <= tool_params,
      sorted(locate_params - tool_params))
svc_sig = open(TOOL, encoding="utf-8").read()
for p in sorted(locate_params):
    check("service accepts %s" % p, re.search(r"\b%s[:\s=]" % p, svc_sig) is not None)

print("\n[prompt <-> behaviour claims]")
if IS_KSM:
    tool_desc = re.search(r'name="get_session_history",\s*description=\((.*?)\),\s*params=',
                          main_src, re.S)
else:
    tool_desc = re.search(r'"get_history",\s*"(.*?)",\s*\{', main_src, re.S)
desc = tool_desc.group(1) if tool_desc else ""
check("description found", bool(desc), desc[:60])
check("description mentions the rewind caveat",
      ("倒带" in desc) or ("rewind" in desc) or ("翻页" in desc), desc[:200])
check("description warns against claiming absence",
      ("绝不能说" in desc) or ("never claim" in desc), desc[:200])
check("description lists the locate params",
      ("keyword" in desc) and ("since" in desc), desc[:200])
if IS_KSM:
    check("turn-limit text matches MAX_CALLS_PER_TARGET_PER_EVENT",
          "每回合最多 2 次" in desc, desc[:400])

print("\n[shared engine identical]")
# The two plugins ship byte-identical copies of the engine files. Compare
# against a sibling checkout when one is present (dev tree); SKIP otherwise
# (CI clone of a single repo) - never fail just because it is absent.
peer_manifest_id = "history_plugin" if IS_KSM else "kira_session_merger"
peer = None
for cand in ("/tmp/hp", "/tmp/smp", "/tmp/vhp", "/tmp/vsm",
             os.path.join(os.path.dirname(ROOT), peer_manifest_id)):
    if os.path.abspath(cand) == os.path.abspath(ROOT):
        continue                      # never compare against ourselves
    if not os.path.isfile(os.path.join(cand, "locate.py")):
        continue
    try:
        cand_manifest = json.load(open(os.path.join(cand, "manifest.json"), encoding="utf-8"))
    except Exception:
        continue
    if cand_manifest.get("plugin_id") == peer_manifest_id:
        peer = cand
        break
if peer:
    for fname in ("locate.py", "onebot_compat.py"):
        a = os.path.join(ROOT, fname)
        b = os.path.join(peer, fname)
        same = open(a, encoding="utf-8").read() == open(b, encoding="utf-8").read()
        check("%s identical across plugins" % fname, same,
              "differs" if not same else "")
else:
    print("  SKIP  peer plugin checkout not present")

print("\n[numbers match the doc]")
const_tr = re.search(r"MAX_CALLS_PER_TARGET_PER_EVENT\s*=\s*(\d+)", tool_src) if IS_KSM else None
const_tot = re.search(r"MAX_CALLS_PER_EVENT\s*=\s*(\d+)", tool_src) if IS_KSM else None
if IS_KSM:
    check("per-target limit is 2", const_tr and const_tr.group(1) == "2", const_tr.group(1) if const_tr else None)
    check("per-turn limit is 3", const_tot and const_tot.group(1) == "3", const_tot.group(1) if const_tot else None)
default_scan = fields.get("default_scan_limit", {}).get("default")
check("default_scan_limit schema default is 300", default_scan == 300, default_scan)
check("default_scan_limit code default is 300",
      'default_scan_limit", 300' in combined, "check code")
max_scan = fields.get("max_scan_limit", {}).get("default")
check("max_scan_limit schema default is 0 (auto)", max_scan == 0, max_scan)
head_meta = fields.get("locate_head_meta", {}).get("default")
check("locate_head_meta defaults on", head_meta is True, head_meta)

print("\n[impl table sanity]")
sys.path.insert(0, ROOT)
import onebot_compat  # noqa: E402
check("NapCat cap 2000", onebot_compat.resolve_impl("NapCat.Onebot")["max_scan_limit"] == 2000)
check("LLOneBot cap 1200", onebot_compat.resolve_impl("LLOneBot")["max_scan_limit"] == 1200)
check("SnowLuma cap 800", onebot_compat.resolve_impl("SnowLuma")["max_scan_limit"] == 800)
check("LLOneBot page cap 30 (its own hard limit)",
      onebot_compat.resolve_impl("LLOneBot")["max_page"] == 30)

print("\n[self-import style: 自家模块必须相对导入]")
# ============================================================================
# v2.8.1 事故防复发：框架以「包」的形式加载插件（plugins.<dir>.main），
# 插件目录**不在 sys.path** 上，因此自家兄弟模块只能用相对导入
# （from .group_agent_queue import ...）。
# 一旦写成绝对导入（from group_agent_queue import ...），模块导入阶段或许看不出来，
# 但只要那段代码跑在 initialize()/_load_cfg() 里 → 插件直接初始化失败：
#     ERROR [plugin_manager] Failed to initialize plugin kira_session_merger:
#     No module named 'group_agent_queue'
# 这里做静态扫描，成本为零、且不需要框架环境。
# ============================================================================
import re as _re
_OWN_MODULES = {f[:-3] for f in os.listdir(ROOT)
                if f.endswith(".py")} - {"__init__", "main"}
_ABS_SELF_IMPORT = _re.compile(r"^\s*(?:from\s+([A-Za-z_]\w*)\s+import|import\s+([A-Za-z_]\w*))")
_offenders = []
for _fname in sorted(f for f in os.listdir(ROOT) if f.endswith(".py")):
    _py = os.path.join(ROOT, _fname)
    with open(_py, encoding="utf-8", errors="ignore") as _fh:
        _lines = _fh.read().splitlines()
    for _i, _line in enumerate(_lines, 1):
        _stripped = _line.strip()
        if _stripped.startswith("#"):
            continue
        _m = _ABS_SELF_IMPORT.match(_line)
        if not _m:
            continue
        _name = _m.group(1) or _m.group(2)
        if _name in _OWN_MODULES:
            _offenders.append(f"{_fname}:{_i}: {_stripped[:70]}")
check("自家模块无绝对导入（{} 个模块）".format(len(_OWN_MODULES)), not _offenders,
      "; ".join(_offenders[:3]))
if _offenders:
    for _o in _offenders:
        print("        ✗", _o)

print()
passed = sum(1 for _, ok in results if ok)
print("TOTAL %d/%d passed" % (passed, len(results)))
sys.exit(0 if passed == len(results) else 1)
