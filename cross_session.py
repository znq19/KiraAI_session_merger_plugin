from __future__ import annotations

"""
合并模式下的跨会话：路由到目标会话并激活 LLM（跨会话请求）。

不做「直达发送」——目标会话应带着合并上文自己思考与执行。
通过 publish_notice(is_mentioned=True) 切入目标会话正常 agent 流程。
"""

from typing import Tuple

from core.chat import MessageChain
from core.chat.message_elements import Text


# 目标会话识别标记（写入 notice 正文）
ROUTE_MARKER = "[merge_cross_session_request]"

# 源会话 tool_result / 交棒识别（勿改语义关键字，ON_STEP_RESULT 依赖）
ROUTE_OK_PREFIX = "cross-session request routed to "
ROUTE_DEDUP_PREFIX = "cross-session request already routed to this target recently"


def build_route_notice_text(source_sid: str, description: str) -> str:
    """构造投递到目标会话的跨会话请求正文（非对用户最终话术模板）。"""
    desc = (description or "").strip()
    if len(desc) > 1500:
        desc = desc[:1500] + "…"
    # description 仅作补充；主依据是合并上文
    extra = f"\n补充说明：\n{desc}\n" if desc else "\n"
    return (
        f"{ROUTE_MARKER}\n"
        f"source_session: {source_sid}\n"
        f"{extra}"
        "你已切换到本会话。请结合合并后的对话上文，在本会话继续执行上文的任务。\n"
        "要求：\n"
        "1. 当前所在就是本会话：请在本会话发言并调用所需工具。\n"
        "2. 不要再次 session_send 回源会话，除非用户明确要求。\n"
        "3. 输出给本会话用户的 xml 消息。\n"
    )


def is_merge_route_request_text(text: str) -> bool:
    return bool(text) and ROUTE_MARKER in text


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


async def route_cross_session_request(
    ctx,
    source_sid: str,
    target: str,
    description: str,
    logger=None,
) -> Tuple[bool, str]:
    """
    将跨会话请求路由到目标会话并激活 LLM。
    返回 (ok, tool_result_message)
    """
    parts = (target or "").split(":")
    if len(parts) != 3:
        return False, "failed: invalid target sid (expect adapter:type:id)"

    if source_sid and source_sid == target:
        return False, (
            "failed: target is current session; "
            "output xml directly here, do not use session_send."
        )

    try:
        notice = build_route_notice_text(source_sid or "unknown", description)
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
