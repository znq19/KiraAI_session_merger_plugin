from __future__ import annotations

"""
摘要输入预处理（移植自 ContextCondensation 的 preprocessor.py，按 ADS/KSM 需要裁剪）。

与 CCS 的区别：CCS 在影子缓存里预处理、原文不动；这里只对「送给摘要模型的
被丢弃历史」生成压缩副本，会话记忆与正式上下文完全不受影响。

- tool 结果：JSON 感知压缩。单对象 → 最长文本字段替换为摘要并打
  `_condensed` 标记；多对象串接（如搜索结果）→ 汇总为 summary + sources；
  非 JSON 长文本 → 纯文本摘要。任何失败返回原文，绝不阻断摘要流程。
- 用户消息里的 [Image: ...] / [图片描述: ...] 长描述：逐块摘要，后缀（已压缩）。
"""

import asyncio
import json
import re
from typing import Any, List, Optional

from core.provider import LLMRequest
from core.agent.message import OpenAIMessage

SUMMARIZE_PROMPT = """请简洁地总结以下内容。
保留所有关键事实、数字、名称和结论。
去除冗余格式、套话和无关内容。

只输出总结后的文本，不要加任何解释。

待总结的内容：
{content}
"""

# 匹配 [Image: ...]、[Image ...] 与 [图片描述: ...] 块
_IMAGE_DESC_PATTERN = re.compile(r"\[(?:Image|图片描述)[:：]?\s*(.+?)\]", re.DOTALL)

_CONDENSED_SUFFIX = "（已压缩）"
# 单次预处理 LLM 输入上限
_PREPROCESS_INPUT_MAX = 8000
_LLM_CALL_TIMEOUT = 120.0


def _msg_get(msg: Any, key: str, default: Any = None) -> Any:
    if isinstance(msg, dict):
        return msg.get(key, default)
    return getattr(msg, key, default)


def _msg_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


async def _summarize(content: str, llm, logger=None) -> Optional[str]:
    prompt = SUMMARIZE_PROMPT.format(content=content[:_PREPROCESS_INPUT_MAX])
    try:
        request = LLMRequest(messages=[OpenAIMessage(role="user", content=prompt)])
        response = await asyncio.wait_for(llm.chat(request), timeout=_LLM_CALL_TIMEOUT)
        summary = (response.text_response or "").strip()
        return summary or None
    except Exception as e:
        if logger:
            logger.warning("[preprocessor] 预处理摘要失败: %s", e)
        return None


def _parse_json_stream(content: str) -> Optional[List[dict]]:
    """宽容解析 tool 结果 JSON：先严格解析，失败再按 raw_decode 流式解析
    （真实 tool 结果常是多个 JSON 对象无分隔串接，如搜索命中）。"""
    try:
        data = json.loads(content)
        return [data]
    except (json.JSONDecodeError, TypeError):
        pass
    decoder = json.JSONDecoder()
    objects: List[dict] = []
    idx = 0
    length = len(content)
    while idx < length:
        while idx < length and content[idx] not in "{[":
            idx += 1
        if idx >= length:
            break
        try:
            obj, end = decoder.raw_decode(content, idx)
        except json.JSONDecodeError:
            return None
        objects.append(obj)
        idx = end
    return objects or None


async def preprocess_tool_result(
    content: str, max_chars: int, llm, logger=None
) -> str:
    """压缩超长 tool 结果；任何失败/无需压缩返回原文。"""
    if len(content) <= max_chars:
        return content

    objects = _parse_json_stream(content)

    # 单 JSON 对象：字段级替换
    if objects is not None and len(objects) == 1 and isinstance(objects[0], dict):
        data = objects[0]
        if data.get("_condensed"):
            return content
        str_fields = {k: v for k, v in data.items() if isinstance(v, str)}
        total = sum(len(v) for v in str_fields.values())
        if total <= max_chars:
            return content
        summary = await _summarize("\n".join(str_fields.values()), llm, logger)
        if summary and len(summary) < total:
            longest_key = max(str_fields, key=lambda k: len(str_fields[k]))
            data[longest_key] = summary
            data["_condensed"] = True
            return json.dumps(data, ensure_ascii=False)
        return content

    # 多对象串接（如搜索命中）：汇总为 summary + sources
    if objects is not None and len(objects) > 1:
        if all(isinstance(o, dict) and o.get("_condensed") for o in objects):
            return content
        texts: List[str] = []
        urls: List[str] = []
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            for k, v in obj.items():
                if isinstance(v, str):
                    if k == "url":
                        urls.append(v)
                    else:
                        texts.append(v)
        total = sum(len(t) for t in texts)
        if total <= max_chars:
            return content
        summary = await _summarize("\n".join(texts), llm, logger)
        if summary and len(summary) < total:
            compact = {"summary": summary, "_condensed": True}
            if urls:
                compact["sources"] = urls[:10]
            return json.dumps(compact, ensure_ascii=False)
        return content

    # 非 JSON 长文本：纯文本摘要
    summary = await _summarize(content, llm, logger)
    if summary and len(summary) < len(content):
        return summary + _CONDENSED_SUFFIX
    return content


async def _preprocess_image_descriptions(
    content: str, max_chars: int, llm, logger=None
) -> str:
    matches = list(_IMAGE_DESC_PATTERN.finditer(content))
    if not matches:
        return content
    result = content
    for match in matches:
        desc = match.group(1)
        if len(desc) <= max_chars or desc.endswith(_CONDENSED_SUFFIX):
            continue
        summary = await _summarize(desc, llm, logger)
        if summary and len(summary) < len(desc):
            new_block = match.group(0).replace(desc, summary + _CONDENSED_SUFFIX)
            result = result.replace(match.group(0), new_block, 1)
    return result


async def preprocess_messages_for_summary(
    messages: List[Any],
    tool_max_chars: int,
    llm,
    logger=None,
) -> List[Any]:
    """返回预处理后的消息副本（原消息不动），仅用作摘要模型输入。

    - role=tool 且超长：JSON 感知压缩
    - role=user 且含超长图片描述：逐块压缩
    任何单条失败都保留该条原文。
    """
    result: List[Any] = []
    for msg in messages:
        try:
            role = _msg_get(msg, "role", "")
            content = _msg_get(msg, "content", "")
            new_content = content

            if role == "tool" and isinstance(content, str) and len(content) > tool_max_chars:
                new_content = await preprocess_tool_result(
                    content, tool_max_chars, llm, logger
                )
            elif role == "user":
                text = _msg_text(content)
                if text and ("[Image" in text or "[图片描述" in text):
                    new_text = await _preprocess_image_descriptions(
                        text, tool_max_chars, llm, logger
                    )
                    if new_text != text and isinstance(content, str):
                        new_content = new_text

            if new_content is content:
                result.append(msg)
            elif isinstance(msg, dict):
                new_msg = dict(msg)
                new_msg["content"] = new_content
                result.append(new_msg)
            else:
                result.append({
                    "role": role,
                    "content": new_content,
                    **({"tool_calls": _msg_get(msg, "tool_calls")} if _msg_get(msg, "tool_calls") else {}),
                    **({"tool_call_id": _msg_get(msg, "tool_call_id")} if _msg_get(msg, "tool_call_id") else {}),
                    **({"name": _msg_get(msg, "name")} if _msg_get(msg, "name") else {}),
                })
        except Exception:
            result.append(msg)
    return result
