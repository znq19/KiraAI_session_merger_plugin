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
import weakref
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
# 预处理 LLM 调用的全局并发上限（同进程、同事件循环内生效）：
# 串行逐条处理在「超长工具结果很多」的长尾场景会把后台摘要拖很久；
# 有界并发把墙钟时间从 Σt 收敛到约 Σt/c，并防止瞬时并发过高触发限流。
# 调用次数与失败回落语义不变（每条独立调用、独立失败回落原文）。
_PREPROCESS_MAX_CONCURRENT = 3
# 按事件循环维护限流器：不同 loop 各自持有，避免跨 loop 复用报错；
# loop 结束后条目自动回收（WeakKeyDictionary）。
_PREPROCESS_SEMS: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _msg_get(msg: Any, key: str, default: Any = None) -> Any:
    if isinstance(msg, dict):
        return msg.get(key, default)
    return getattr(msg, key, default)


def _msg_text(content: Any) -> str:
    """提取消息文本, 含图片时追加附件标记供摘要模型参考."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        img_count = 0
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            elif isinstance(part, dict) and "kira_image" in (part.get("type", "") or ""):
                img_count += 1
        text = "".join(parts)
        if img_count:
            text += f"\n[图片: {img_count} 张附件]"
        return text
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


def _get_preprocess_sem() -> asyncio.Semaphore:
    """按事件循环维护预处理全局限流器（不同 loop 各自持有，loop 结束自动回收）。"""
    loop = asyncio.get_running_loop()
    sem = _PREPROCESS_SEMS.get(loop)
    if sem is None:
        sem = asyncio.Semaphore(_PREPROCESS_MAX_CONCURRENT)
        _PREPROCESS_SEMS[loop] = sem
    return sem


def _rebuild_message(msg: Any, role: str, new_content: Any) -> Any:
    """按原逻辑重建消息副本（dict 浅拷贝；其它类型构造 OpenAI 风格 dict）。"""
    if isinstance(msg, dict):
        new_msg = dict(msg)
        new_msg["content"] = new_content
        return new_msg
    return {
        "role": role,
        "content": new_content,
        **({"tool_calls": _msg_get(msg, "tool_calls")} if _msg_get(msg, "tool_calls") else {}),
        **({"tool_call_id": _msg_get(msg, "tool_call_id")} if _msg_get(msg, "tool_call_id") else {}),
        **({"name": _msg_get(msg, "name")} if _msg_get(msg, "name") else {}),
    }


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
    各条目在有界并发下处理（全局并发上限 _PREPROCESS_MAX_CONCURRENT），
    消息顺序、LLM 调用次数与失败回落语义均保持不变，仅缩短长尾总耗时。
    """
    result: List[Any] = list(messages)

    # 先本地筛出需要处理的条目（纯判断，不触发 LLM）
    jobs: List[tuple] = []
    for idx, msg in enumerate(messages):
        try:
            role = _msg_get(msg, "role", "")
            content = _msg_get(msg, "content", "")
            if role == "tool" and isinstance(content, str) and len(content) > tool_max_chars:
                jobs.append((idx, role, msg, content))
            elif role == "user":
                text = _msg_text(content)
                if text and ("[Image" in text or "[图片描述" in text):
                    jobs.append((idx, role, msg, text))
        except Exception:
            continue
    if not jobs:
        return result

    sem = _get_preprocess_sem()

    async def _one(job):
        idx, role, msg, payload = job
        try:
            async with sem:
                if role == "tool":
                    new_content = await preprocess_tool_result(
                        payload, tool_max_chars, llm, logger
                    )
                    if new_content is not payload:
                        return idx, _rebuild_message(msg, role, new_content)
                    return idx, None
                # user：图片描述逐块压缩（仅 str 内容可原地替换）
                content = _msg_get(msg, "content", "")
                new_text = await _preprocess_image_descriptions(
                    payload, tool_max_chars, llm, logger
                )
                if new_text != payload and isinstance(content, str):
                    return idx, _rebuild_message(msg, role, new_text)
                return idx, None
        except Exception:
            return idx, None

    gathered = await asyncio.gather(
        *[_one(job) for job in jobs], return_exceptions=True
    )
    for item in gathered:
        if isinstance(item, tuple) and len(item) == 2 and item[1] is not None:
            result[item[0]] = item[1]
    return result
