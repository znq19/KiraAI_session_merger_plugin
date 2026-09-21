from __future__ import annotations

"""
合并模式下的跨会话：路由到目标会话并激活 LLM（跨会话请求）。

不做「直达发送」——目标会话应带着合并上文自己思考与执行。
通过 publish_notice(is_mentioned=True) 切入目标会话正常 agent 流程。
"""

from typing import List, Optional, Tuple

from core.chat import MessageChain
from core.chat.message_elements import Text


# 目标会话识别标记（写入 notice 正文）
ROUTE_MARKER = "[merge_cross_session_request]"

# 跨会话跳数上限：路由正文带 hop 计数，hop ≥ 上限时拒绝投递（防 A→B→A 乒乓，
# 每跳都是一整轮 LLM 费用；去重 TTL 之外的最后防线）
ROUTE_MAX_HOPS = 2

# 源会话 tool_result / 交棒识别（勿改语义关键字，ON_STEP_RESULT 依赖）
ROUTE_OK_PREFIX = "cross-session request routed to "
ROUTE_DEDUP_PREFIX = "cross-session request already routed to this target recently"


def build_route_notice_text(source_sid: str, description: str, hop: int = 1) -> str:
    """构造投递到目标会话的跨会话请求正文（非对用户最终话术模板）。"""
    desc = (description or "").strip()
    if len(desc) > 1500:
        desc = desc[:1500] + "…"
    # description 仅作补充；主依据是合并上文
    extra = f"\n补充说明：\n{desc}\n" if desc else "\n"
    return (
        f"{ROUTE_MARKER}\n"
        f"source_session: {source_sid}\n"
        f"hop: {max(1, int(hop or 1))}\n"
        f"{extra}"
        "你已切换到本会话。请结合合并后的对话上文，在本会话继续执行上文的任务。\n"
        "要求：\n"
        "1. 当前所在就是本会话：请在本会话发言并调用所需工具。\n"
        "2. 不要再次 session_send 回源会话，除非用户明确要求。\n"
        "3. 输出给本会话用户的 xml 消息。\n"
    )


def is_merge_route_request_text(text: str) -> bool:
    return bool(text) and ROUTE_MARKER in text


def route_hop_of_text(text: str) -> int:
    """从路由正文解析 hop 计数；旧版无 hop 行的正文按 1 处理（向后兼容）。"""
    if not text or ROUTE_MARKER not in text:
        return 0
    for line in str(text).splitlines()[1:6]:
        line = line.strip()
        if line.startswith("hop:"):
            try:
                return max(1, int(line[4:].strip()))
            except (TypeError, ValueError):
                return 1
    return 1


def is_route_handoff_result(text: str) -> bool:
    """源会话 session_send 成功路由或去重命中后的 tool_result。"""
    if not text:
        return False
    t = str(text)
    return ROUTE_OK_PREFIX in t or ROUTE_DEDUP_PREFIX in t


def mark_event_handoff(event, target: str = "") -> None:
    """标记本轮 agent 已交棒到目标会话（仅当前 event，不影响同会话其他请求）。"""
    if event is None:
        return
    try:
        extra = getattr(event, "extra", None)
        if extra is None or not isinstance(extra, dict):
            extra = {}
            try:
                event.extra = extra
            except Exception:
                return
        extra["merger_handoff"] = True
        if target:
            extra["merger_handoff_target"] = str(target)
    except Exception:
        pass


def event_has_handoff(event) -> bool:
    try:
        extra = getattr(event, "extra", None) or {}
        if isinstance(extra, dict) and extra.get("merger_handoff"):
            return True
    except Exception:
        pass
    return False


def build_route_ok_message(target: str) -> str:
    return (
        f"{ROUTE_OK_PREFIX}{target}; "
        "TASK HANDED OFF to the TARGET session (it continues with merged context). "
        "Do NOT call any more tools in the CURRENT session for that task "
        "(no search / music card / get_session_history / session_send again). "
        "A short xml in the CURRENT session is optional if useful; silence is also fine. "
        "Work continues only in the target session."
    )


def build_route_dedup_message(target: str = "") -> str:
    _ = target
    return (
        f"{ROUTE_DEDUP_PREFIX}; "
        "TASK already handed off. Do NOT call session_send or other tools for that task; "
        "work continues in the TARGET session. "
        "A short xml in the CURRENT session is optional if useful; silence is also fine."
    )


def list_enabled_adapters(ctx) -> List[Tuple[str, str]]:
    """枚举启用中且已加载的 adapter：[(name, platform)]。失败返回 []。"""
    mgr = getattr(ctx, "adapter_mgr", None)
    if mgr is None:
        return []
    try:
        infos = mgr.get_adapters_info()
    except Exception:
        infos = []
    out: List[Tuple[str, str]] = []
    for info in infos or []:
        try:
            if not getattr(info, "enabled", False):
                continue
            name = str(
                getattr(info, "name", "") or getattr(info, "adapter_id", "") or ""
            )
            if not name:
                continue
            if mgr.get_adapter(name) is None:
                continue
            out.append((name, str(getattr(info, "platform", "") or "")))
        except Exception:
            continue
    return out


def resolve_target_sid(ctx, target: str) -> Tuple[Optional[str], Optional[str]]:
    """
    校正 target 的 adapter 前缀（LLM 可能照工具描述示例脑补前缀）。
    返回 (sid, None) 或 (None, 可读错误)。
    规则（全部确定性，无猜测）：
      1. adapter 存在 → 原样通过；
      2. 会话列表按 *:type:id 反查，唯一命中 → 用真实 sid；
      3. 启用中的 adapter 仅 1 个 → 直接替换前缀；
      4. 否则拒绝并返回可用 adapter 候选，让 LLM 重试。
    """
    parts = (target or "").split(":")
    if len(parts) != 3 or not all(parts):
        return None, "failed: invalid target sid (expect adapter:type:id)"
    ada_name, st, sid = parts
    mgr = getattr(ctx, "adapter_mgr", None)
    if mgr is None:
        return target, None  # 无法校验，按原样投递
    try:
        if mgr.get_adapter(ada_name) is not None:
            return target, None
    except Exception:
        return target, None

    adapters = list_enabled_adapters(ctx)

    # 2) 会话列表反查 *:type:id 唯一命中
    found = set()
    sm = getattr(ctx, "session_mgr", None)
    if sm is not None:
        try:
            valid = {n for n, _ in adapters}
            for info in sm.get_session_info() or []:
                s = str(getattr(info, "sid", "") or "")
                p = s.split(":")
                if (
                    len(p) == 3
                    and p[1] == st
                    and p[2] == sid
                    and p[0] != ada_name
                    and p[0] in valid
                ):
                    found.add(s)
        except Exception:
            pass
    if len(found) == 1:
        return found.pop(), None

    # 3) 启用 adapter 唯一 → 直改前缀（覆盖首次触达、会话列表无记录的场景）
    if len(adapters) == 1:
        return f"{adapters[0][0]}:{st}:{sid}", None

    # 4) 歧义或无候选 → 可读错误，引导 LLM 用正确前缀重试
    if not adapters:
        return None, f"failed: no enabled adapter; cannot route to {target}"
    cand = ", ".join(f"{n}({pf})" if pf else n for n, pf in adapters)
    return None, (
        f"failed: adapter '{ada_name}' not found. "
        f"Available adapters: {cand}. "
        f"Retry with target like '{adapters[0][0]}:{st}:{sid}', "
        "copied verbatim from the session list."
    )


def check_session_send_permission(ctx, target: str) -> Optional[str]:
    """
    跨会话目标白/黑名单检查（与官方 builtin session_tools 的 session_send 一致）：
    - adapter.permission_mode == allow_list：目标必须在其 user_list/group_list 中；
    - adapter.permission_mode == deny_list：目标必须不在列表中。
    兼容策略（避免破坏老部署）：
    - adapter 不存在或旧框架无 permission_mode 属性 → 不拦截，交给 resolve 处理；
    - allow_list 且对应列表为空 → 视为未配置白名单，不拦截
      （官方语义下空 allow_list 会拒绝全部跨会话，这里保持宽容）；
    - 其它未知模式 → 不拦截。
    返回 None 表示允许，否则返回 Permission denied 错误消息。
    """
    parts = (target or "").split(":")
    if len(parts) != 3 or not all(parts):
        return None  # 格式错误交给 resolve_target_sid 处理
    ada_name, session_type, session_id = parts
    mgr = getattr(ctx, "adapter_mgr", None)
    if mgr is None:
        return None
    try:
        adapter = mgr.get_adapter(ada_name)
    except Exception:
        adapter = None
    if adapter is None:
        return None  # 不存在交给 resolve 报错/前缀校正
    permission_mode = getattr(adapter, "permission_mode", None)
    if permission_mode not in ("allow_list", "deny_list"):
        return None
    try:
        target_list = getattr(
            adapter, "user_list" if session_type == "dm" else "group_list", None
        ) or []
        target_is_listed = session_id in {str(item) for item in target_list}
    except Exception:
        return None
    if permission_mode == "allow_list":
        if not target_list:
            return None  # 空白名单视为未配置，不拦截
        is_allowed = target_is_listed
    else:  # deny_list
        is_allowed = not target_is_listed
    if not is_allowed:
        return f"Permission denied: target session is not allowed by adapter {ada_name}"
    return None


async def route_cross_session_request(
    ctx,
    source_sid: str,
    target: str,
    description: str,
    logger=None,
    hop: int = 1,
) -> Tuple[bool, str]:
    """
    将跨会话请求路由到目标会话并激活 LLM。
    返回 (ok, tool_result_message)
    """
    if hop >= ROUTE_MAX_HOPS:
        if logger:
            logger.warning(
                "[MERGER] route rejected %s -> %s: hop=%d >= %d（跨会话乒乓保护）",
                source_sid, target, hop, ROUTE_MAX_HOPS,
            )
        return False, (
            "failed: cross-session hop limit reached; "
            "answer in the current session instead of routing further."
        )
    resolved, err = resolve_target_sid(ctx, target)
    if err:
        if logger:
            logger.warning(
                "[MERGER] route rejected %s -> %s: %s", source_sid, target, err
            )
        return False, err
    if resolved != target and logger:
        logger.info("[MERGER] route target corrected %s -> %s", target, resolved)
    target = resolved

    if source_sid and source_sid == target:
        return False, (
            "failed: target is current session; "
            "output xml directly here, do not use session_send."
        )

    perm_err = check_session_send_permission(ctx, target)
    if perm_err:
        if logger:
            logger.warning(
                "[MERGER] route rejected %s -> %s: %s", source_sid, target, perm_err
            )
        return False, perm_err

    try:
        notice = build_route_notice_text(source_sid or "unknown", description, hop=hop)
        await ctx.publish_notice(
            target,
            MessageChain([Text(notice)]),
            is_mentioned=True,
        )
        if logger:
            logger.info(
                "[MERGER] cross-session ROUTE %s -> %s (target LLM will run with merge)",
                source_sid,
                target,
            )
        return True, build_route_ok_message(target)
    except Exception as e:
        if logger:
            logger.exception("[MERGER] route failed %s -> %s", source_sid, target)
        return False, f"failed to route: {e}"
