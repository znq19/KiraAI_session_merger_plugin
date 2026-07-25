from __future__ import annotations

"""
硬重开前摘要：对将被丢弃的历史消息生成简短背景摘要，注入重开后记忆首部。
与 ADS 插件的 summarizer.py 保持同构（独立副本，避免跨插件依赖）。

v2 新增（借鉴 ContextCondensation 设计）：
- 累积摘要存储 CumulativeSummaryStore：摘要 = 旧摘要 + 新增量合并，不再层层有损叠加；
- merge_summaries / self_compress_summary：合并与兜底自压缩（去第三人称，无缝衔接风格）；
- dropped_fingerprint：预热结果与被丢弃范围的匹配校验；
- 摘要失败/超时一律返回 None，调用方按原逻辑直接重开，绝不阻断主流程。
"""

import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

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

# 合并提示词：旧累计摘要 + 新增量摘要 → 新累计摘要（去第三人称，保持无缝衔接风格；
# 600 字预算对齐 CCS 的信息密度，减缓长期会话磨损）
DEFAULT_MERGE_PROMPT = (
    "你是聊天记忆压缩器。下面有两段材料：【旧摘要】是更早对话的已有摘要，【新增记录】是刚刚被清理的对话片段。"
    "请把它们合并成一段新的摘要（600字内），用于衔接后续对话，让你能像没有失忆一样自然地继续聊下去。\n"
    "请优先保留：\n"
    "- 正在进行或未完成的事情（话题、任务、约定、待办）\n"
    "- 重要事实（人物、关系、时间、地点、已做的决定）\n"
    "- 对方的偏好、称呼、语气和你们之间的相处方式\n"
    "若新旧信息冲突，以新增记录为准。直接输出合并后的摘要正文，不要任何开场白、标题或解释。\n\n"
    "【旧摘要】\n{old_summary}\n\n【新增记录】\n{new_summary}"
)

# 兜底自压缩：累计摘要超过输出上限时调用（保持无缝衔接风格）
DEFAULT_SELF_COMPRESS_PROMPT = (
    "你是聊天记忆压缩器。下面这段对话摘要过长，请在不丢失关键信息的前提下进一步压缩（600字内），"
    "用于衔接后续对话，让你能像没有失忆一样自然地继续聊下去。\n"
    "优先保留：未完成的事情、重要事实（人物/关系/时间/地点/已做的决定）、对方的偏好与称呼。\n"
    "直接输出压缩后的摘要正文，不要任何开场白或解释。\n\n"
    "{summary}"
)

def _safe_format(template: str, mapping: dict) -> str:
    """显式占位符替换，避免摘要内容中的 { } 被 str.format 误解析。"""
    text = template
    for key, value in mapping.items():
        text = text.replace("{" + key + "}", str(value))
    return text


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


def extract_summary_text(content: str) -> str:
    """从带标记的摘要消息 content 中提取摘要正文（去掉标记行与引导语）。"""
    if not isinstance(content, str) or not content.startswith(SUMMARY_MARKER):
        return ""
    lines = content.split("\n", 1)
    return lines[1].strip() if len(lines) > 1 else ""


def dropped_fingerprint(dropped_flat: List[dict]) -> str:
    """被丢弃范围的指纹：条数 + 最后一条文本消息的哈希。

    用于预热结果校验：重开时重新计算，与预热暂存一致才能复用。
    """
    if not dropped_flat:
        return ""
    last_text = ""
    for m in reversed(dropped_flat):
        if isinstance(m, dict):
            c = m.get("content")
            if isinstance(c, str) and c.strip():
                last_text = c.strip()
                break
    h = hashlib.sha256(last_text[:200].encode("utf-8")).hexdigest()[:12]
    return f"{len(dropped_flat)}:{h}"


class CumulativeSummaryStore:
    """每会话累计摘要的持久化存储（插件数据目录 JSON）。

    关键不变量：store 里的摘要必须与会话记忆头部摘要保持一致；
    sync_with_head 负责在校验失败时以记忆为准（防止旧摘要污染新会话）。
    """

    def __init__(self, path: Path):
        self._path = Path(path)
        self._data: Dict[str, dict] = {}
        self._loaded = False

    def load(self) -> None:
        self._loaded = True
        try:
            if self._path.exists():
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    self._data = {
                        str(k): v for k, v in raw.items() if isinstance(v, dict)
                    }
        except Exception:
            self._data = {}

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def get(self, sid: str) -> str:
        self._ensure_loaded()
        entry = self._data.get(sid) or {}
        return str(entry.get("summary") or "")

    def set(self, sid: str, summary: str) -> None:
        self._ensure_loaded()
        import time as _time

        self._data[sid] = {"summary": summary, "updated_at": int(_time.time())}

    def pop(self, sid: str) -> None:
        self._ensure_loaded()
        self._data.pop(sid, None)

    def sync_with_head(self, sid: str, head_summary: str) -> str:
        """以记忆头部摘要为准对账，返回权威累计摘要。

        - 头部摘要与 store 一致 → 用 store（正常路径）；
        - 头部存在但与 store 不一致（外部改动）→ 采用头部版本并回写 store；
        - 头部不存在（用户清了会话 / 摘要丢失）→ 清掉 store 条目，返回空。
        """
        self._ensure_loaded()
        stored = self.get(sid)
        head_summary = (head_summary or "").strip()
        if head_summary:
            if stored != head_summary:
                self.set(sid, head_summary)
            return head_summary
        if stored:
            self.pop(sid)
        return ""

    def save(self) -> None:
        self._ensure_loaded()
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(self._data, ensure_ascii=False), encoding="utf-8"
            )
            os.replace(tmp, self._path)
        except Exception:
            pass


def extract_dropped_text(dropped_flat: List[dict], max_input_chars: int) -> str:
    """
    将被丢弃的消息压成纯文本：
    - user/assistant 取文本 content
    - assistant 带 tool_calls 压成一行工具调用记录
    - tool 结果纳入（通常已被 preprocessor 压缩过；原文超长时由
      max_input_chars 从尾部截断兜底）
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
        elif role == "tool" and text.strip():
            lines.append(f"工具结果: {text.strip()}")
        # system 等跳过

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


async def _chat_text(
    client,
    sid: str,
    prompt: str,
    timeout_sec: float,
    logger=None,
    enable_detail_log: bool = False,
) -> Optional[str]:
    """单次 LLM 调用取文本；任何失败返回 None。client 由调用方预先解析。"""
    if client is None:
        return None
    req = LLMRequest(messages=[OpenAIMessage(role="user", content=prompt)])
    try:
        timeout = max(0.5, float(timeout_sec or 30.0))
        resp = await asyncio.wait_for(client.chat(req), timeout)
    except asyncio.TimeoutError:
        if logger:
            logger.warning("[summarizer] LLM timeout (%.1fs) for %s", timeout_sec, sid)
        return None
    except Exception as e:
        if logger:
            logger.warning("[summarizer] LLM call failed for %s: %s", sid, e)
        return None
    text = (getattr(resp, "text_response", None) or "").strip()
    if not text and enable_detail_log and logger:
        logger.info("[摘要调试] LLM 返回空结果")
    return text or None


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
    preprocess_tools: bool = True,
    tool_max_chars: int = 2000,
) -> Optional[str]:
    """
    对将被丢弃的历史生成摘要。任何失败（无模型/超时/空结果/异常）返回 None。

    preprocess_tools=True 时，先对超长 tool 结果/图片描述做 JSON 感知压缩
    （仅作用于摘要输入副本，不动会话记忆），让工具细节参与摘要而非整条丢弃。
    """
    if not dropped_flat:
        if enable_detail_log and logger:
            logger.info("[摘要调试] 待删除历史为空，跳过摘要")
        return None

    client = _resolve_client(ctx, (model_id or "").strip(), logger=logger)
    if client is None:
        if logger:
            logger.warning("[summarizer] no LLM client available, skip summary for %s", sid)
        return None

    if preprocess_tools:
        try:
            from .preprocessor import preprocess_messages_for_summary

            dropped_flat = await preprocess_messages_for_summary(
                dropped_flat, max(200, int(tool_max_chars or 2000)), client, logger
            )
        except Exception as e:
            if enable_detail_log and logger:
                logger.info(f"[摘要调试] 预处理失败（用原文继续）: {e}")

    text = extract_dropped_text(dropped_flat, max_input_chars)
    if not text.strip():
        if enable_detail_log and logger:
            logger.info("[摘要调试] 提取文本为空，跳过摘要")
        return None

    if enable_detail_log and logger:
        logger.info(f"[摘要调试] 提取的历史文本 ({len(text)} 字符):\n{text[:500]}...")

    prompt = (prompt_template or "").strip() or DEFAULT_SUMMARIZE_PROMPT
    if "{text}" in prompt:
        prompt_text = _safe_format(prompt, {"text": text})
    else:
        prompt_text = f"{prompt}\n\n{text}"
    summary = await _chat_text(
        client,
        sid,
        prompt_text,
        timeout_sec=timeout_sec,
        logger=logger,
        enable_detail_log=enable_detail_log,
    )
    if not summary:
        return None
    # 防御：极端长输出截断，避免摘要本身撑大新会话（<=0 表示不截断）
    if max_output_chars > 0 and len(summary) > max_output_chars:
        summary = summary[:max_output_chars] + "…"
    if logger:
        logger.info("[summarizer] summary ok for %s (%d chars)", sid, len(summary))
    if enable_detail_log and logger:
        logger.info(f"[摘要调试] 生成成功，摘要内容:\n{summary}")
    return summary


async def merge_summaries(
    ctx,
    sid: str,
    old_summary: str,
    new_summary: str,
    model_id: str = "",
    prompt_template: str = "",
    timeout_sec: float = 30.0,
    logger=None,
    enable_detail_log: bool = False,
) -> Optional[str]:
    """合并旧累计摘要与新增量摘要。一边为空直接用另一边；LLM 失败返回 None。"""
    old_summary = (old_summary or "").strip()
    new_summary = (new_summary or "").strip()
    if not old_summary:
        return new_summary or None
    if not new_summary:
        return old_summary or None
    client = _resolve_client(ctx, (model_id or "").strip(), logger=logger)
    prompt = (prompt_template or "").strip() or DEFAULT_MERGE_PROMPT
    prompt_text = _safe_format(
        prompt, {"old_summary": old_summary, "new_summary": new_summary}
    )
    merged = await _chat_text(
        client,
        sid,
        prompt_text,
        timeout_sec=timeout_sec,
        logger=logger,
        enable_detail_log=enable_detail_log,
    )
    if merged and enable_detail_log and logger:
        logger.info(f"[摘要调试] 合并成功，累计摘要:\n{merged}")
    return merged


async def self_compress_summary(
    ctx,
    sid: str,
    summary: str,
    model_id: str = "",
    prompt_template: str = "",
    timeout_sec: float = 30.0,
    logger=None,
    enable_detail_log: bool = False,
) -> Optional[str]:
    """累计摘要超长时的兜底自压缩；失败返回 None（调用方硬截断）。"""
    summary = (summary or "").strip()
    if not summary:
        return None
    client = _resolve_client(ctx, (model_id or "").strip(), logger=logger)
    prompt = (prompt_template or "").strip() or DEFAULT_SELF_COMPRESS_PROMPT
    prompt_text = _safe_format(prompt, {"summary": summary})
    return await _chat_text(
        client,
        sid,
        prompt_text,
        timeout_sec=timeout_sec,
        logger=logger,
        enable_detail_log=enable_detail_log,
    )
