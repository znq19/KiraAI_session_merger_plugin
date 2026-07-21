from __future__ import annotations

"""
硬重开前摘要：对将被丢弃的历史消息生成简短背景摘要，注入重开后记忆首部。

摘要失败/超时一律返回 None，调用方按原逻辑直接重开，绝不阻断主流程。
与 ADS 插件的 summarizer.py 保持同构（独立副本，避免跨插件依赖）。
"""

import asyncio
from typing import List, Optional

from core.agent.message import OpenAIMessage
from core.provider import LLMRequest

DEFAULT_SUMMARIZE_PROMPT = (
    "你是聊天记忆压缩器。下面这段较早的聊天记录即将被清理，请把它压缩成一段简短摘要（200字内），"
    "用于衔接后续对话，让你能像没有失忆一样自然地继续聊下去。\n"
    "请优先保留：\n"
    "- 正在进行或未完成的事情（话题、任务、约定、待办）\n"
    "- 重要事实（人物、关系、时间、地点、已做的决定）\n"
    "- 对方的偏好、称呼、语气和你们之间的相处方式\n"
    "直接输出摘要正文，不要任何开场白、标题或解释。"
)

# 摘要 chunk 首条 user message 的标记前缀（幂等判断依赖它，勿轻易改动）
SUMMARY_MARKER = "[前情摘要|系统注入]"


def build_summary_chunk(summary_text: str) -> list:
    """把摘要文本包装成官方记忆 chunk（user 起头，兼容 _clean_and_chunk）。"""
    return [
        {
            "role": "user",
            "content": (
                f"{SUMMARY_MARKER} 以下是更早对话被清理前的自动摘要，"
                f"仅供延续上下文参考：\n{summary_text}"
            ),
        }
    ]


def is_summary_chunk(chunk) -> bool:
    """判断一个记忆 chunk 是否已是摘要 chunk（避免重复注入）。"""
    if not isinstance(chunk, list) or not chunk:
        return False
    first = chunk[0]
    if not isinstance(first, dict):
        return False
    content = first.get("content")
    return isinstance(content, str) and content.startswith(SUMMARY_MARKER)


def extract_dropped_text(dropped_flat: List[dict], max_input_chars: int) -> str:
    """
    将被丢弃的消息压成纯文本：
    - user/assistant 取文本 content
    - assistant 带 tool_calls 压成一行工具调用记录
    - tool 结果跳过（工具细节对背景摘要价值低且占字数）
    从尾部截断到 max_input_chars（越新的内容越重要）。
    """
    lines: List[str] = []
    for m in dropped_flat:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        text = content if isinstance(content, str) else ""
        if role == "user" and text.strip():
            lines.append(f"用户: {text.strip()}")
        elif role == "assistant":
            tcs = m.get("tool_calls") or []
            if text.strip():
                lines.append(f"助手: {text.strip()}")
            elif tcs:
                names = []
                for tc in tcs:
                    if isinstance(tc, dict):
                        n = (tc.get("function") or {}).get("name")
                        if n:
                            names.append(str(n))
                if names:
                    lines.append(f"助手: [调用工具 {', '.join(names)}]")
        # role == "tool" / system 等跳过

    blob = "\n".join(lines)
    # max_input_chars <= 0 表示无上限（全文送去压缩）
    if max_input_chars > 0 and len(blob) > max_input_chars:
        blob = blob[-max_input_chars:]
    return blob


def _resolve_client(ctx, model_id: str, logger=None):
    """model_id 优先；失败回退快速模型 → 默认模型；全失败 None。"""
    if model_id:
        try:
            client = ctx.get_llm_client(model_id)
            if client:
                return client
        except Exception as e:
            if logger:
                logger.warning(
                    "[summarizer] get_llm_client(%s) failed: %s, fallback", model_id, e
                )
    for getter in ("get_default_fast_llm_client", "get_default_llm_client"):
        try:
            fn = getattr(ctx, getter, None)
            if fn:
                client = fn()
                if client:
                    return client
        except Exception:
            continue
    return None


async def summarize_history(
    ctx,
    sid: str,
    dropped_flat: List[dict],
    model_id: str = "",
    prompt_template: str = "",
    timeout_sec: float = 30.0,
    max_input_chars: int = 6000,
    max_output_chars: int = 3000,
    logger=None,
    enable_detail_log: bool = False,
) -> Optional[str]:
    """
    对将被丢弃的历史生成摘要。任何失败（无模型/超时/空结果/异常）返回 None。
    """
    if not dropped_flat:
        if enable_detail_log and logger:
            logger.info("[摘要调试] 待删除历史为空，跳过摘要")
        return None

    text = extract_dropped_text(dropped_flat, max_input_chars)
    if not text.strip():
        if enable_detail_log and logger:
            logger.info("[摘要调试] 提取文本为空，跳过摘要")
        return None

    if enable_detail_log and logger:
        logger.info(f"[摘要调试] 提取的历史文本 ({len(text)} 字符):\n{text[:500]}...")

    client = _resolve_client(ctx, (model_id or "").strip(), logger=logger)
    if client is None:
        if logger:
            logger.warning("[summarizer] no LLM client available, skip summary for %s", sid)
        if enable_detail_log and logger:
            logger.info(f"[摘要调试] 无可用 LLM 客户端，model_id={model_id}")
        return None

    if enable_detail_log and logger:
        # 不同 provider client 结构不一：model_id 可能在 client 上或在 client.model 上
        _mid = getattr(client, "model_id", None) or getattr(
            getattr(client, "model", None), "model_id", None
        ) or (model_id or "unknown")
        logger.info(f"[摘要调试] 使用模型: {_mid}, 超时: {timeout_sec}s")

    prompt = (prompt_template or "").strip() or DEFAULT_SUMMARIZE_PROMPT
    req = LLMRequest(
        messages=[
            OpenAIMessage(role="user", content=f"{prompt}\n\n{text}"),
        ]
    )

    try:
        timeout = max(0.5, float(timeout_sec or 30.0))
        resp = await asyncio.wait_for(client.chat(req), timeout)
    except asyncio.TimeoutError:
        if logger:
            logger.warning(
                "[summarizer] summary timeout (%.1fs) for %s, skip", timeout_sec, sid
            )
        if enable_detail_log and logger:
            logger.info(f"[摘要调试] LLM 调用超时: {timeout_sec}s")
        return None
    except Exception as e:
        if logger:
            logger.warning("[summarizer] summary failed for %s: %s", sid, e)
        if enable_detail_log and logger:
            logger.info(f"[摘要调试] LLM 调用异常: {e}")
        return None

    summary = (getattr(resp, "text_response", None) or "").strip()
    if not summary:
        if enable_detail_log and logger:
            logger.info("[摘要调试] LLM 返回空结果")
        return None
    # 防御：极端长输出截断，避免摘要本身撑大重开后的记忆（<=0 表示不截断）
    if max_output_chars > 0 and len(summary) > max_output_chars:
        summary = summary[:max_output_chars] + "…"
    if logger:
        logger.info("[summarizer] summary ok for %s (%d chars)", sid, len(summary))
    if enable_detail_log and logger:
        logger.info(f"[摘要调试] 生成成功，摘要内容:\n{summary}")
    return summary
