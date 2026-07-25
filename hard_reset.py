from __future__ import annotations

"""
硬重开：对合并组内 sid 做「保留最近 keep 轮 + 摘要写回」。
仅在 merge_reset_mode=hard 时使用。

v2（借鉴 ContextCondensation）：
- 写穿模式（默认）：不 delete_session，直接 write_memory 覆写，
  保留会话 title/description 等元信息；
- 头部摘要剥离：旧摘要不参与 keep 轮数计算，也不重复进入 dropped；
- 融合布局：新摘要融合进第一个保留 chunk，不占窗口槽位、位置恒定，
  利于命中模型上下文缓存。
"""

from typing import List, Optional

from .summarizer import SUMMARY_MARKER, extract_summary_text


def _clean_and_chunk(flat: List[dict]) -> List[List[dict]]:
    start_idx = 0
    for i, msg in enumerate(flat):
        if (msg.get("role") if isinstance(msg, dict) else None) == "user":
            start_idx = i
            break
    else:
        return []
    cleaned = flat[start_idx:]
    chunks: List[List[dict]] = []
    cur: List[dict] = []
    for msg in cleaned:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "user":
            if cur:
                chunks.append(cur)
            cur = [msg]
        else:
            cur.append(msg)
    if cur:
        chunks.append(cur)
    return chunks


def _flatten(chunks: List[List[dict]]) -> List[dict]:
    out: List[dict] = []
    for c in chunks:
        out.extend(c)
    return out


def _split_head_summary(flat: List[dict]) -> tuple:
    """剥离扁平记忆首条的摘要消息，返回 (摘要正文, 剩余消息)。"""
    if flat and isinstance(flat[0], dict):
        first = flat[0]
        if first.get("role") == "user":
            content = str(first.get("content", "") or "")
            if content.startswith(SUMMARY_MARKER):
                return extract_summary_text(content), list(flat[1:])
    return "", list(flat)


def _norm_messages(flat) -> List[dict]:
    norm: List[dict] = []
    for m in flat or []:
        if isinstance(m, dict):
            norm.append(m)
        elif hasattr(m, "to_dict"):
            norm.append(m.to_dict())
    return norm


def build_fused_chunks(
    kept_chunks: List[List[dict]], summary_chunk: Optional[list]
) -> List[List[dict]]:
    """缓存友好布局：摘要融合进第一个保留 chunk（不占窗口槽位）。"""
    chunks = [list(c) for c in kept_chunks]
    if summary_chunk:
        summary_msg = list(summary_chunk)[0]
        if chunks:
            chunks[0] = [summary_msg] + chunks[0]
        else:
            chunks = [[summary_msg]]
    return chunks


def read_reset_parts(session_mgr, sid: str, keep_turns: int) -> dict:
    """
    只读计算重开分区：{"head_text", "chunks", "kept", "dropped"}。

    旧头部摘会被剥离：不计入 keep 轮数，也不进 dropped
    （累计模式下它由 CumulativeSummaryStore 接管）。
    """
    result = {"head_text": "", "chunks": [], "kept": [], "dropped": []}
    if not session_mgr or not sid:
        return result
    keep_turns = max(1, int(keep_turns or 1))
    try:
        # 只读内存，避免 fetch_memory 的 _ensure_session_data 写盘
        data = getattr(session_mgr, "chat_memory", None)
        if not isinstance(data, dict) or sid not in data:
            return result
        mem_list = (data.get(sid) or {}).get("memory") or []
        if not isinstance(mem_list, list):
            return result
        flat: List[dict] = []
        for chunk in mem_list:
            if not isinstance(chunk, list):
                continue
            for m in chunk:
                if isinstance(m, dict):
                    flat.append(m)
                elif hasattr(m, "to_dict"):
                    flat.append(m.to_dict())
        head_text, rest = _split_head_summary(flat)
        chunks = _clean_and_chunk(rest)
        kept = chunks[-keep_turns:] if len(chunks) > keep_turns else chunks[:]
        dropped = _flatten(chunks[:-keep_turns]) if len(chunks) > keep_turns else []
        result.update(
            {"head_text": head_text, "chunks": chunks, "kept": kept, "dropped": dropped}
        )
        return result
    except Exception:
        return result


def compute_dropped_flat(session_mgr, sid: str, keep_turns: int) -> List[dict]:
    """
    只读计算：硬重开将丢弃的消息（keep 轮之前的部分）。
    供 async 层在真正重开前生成摘要用；不写盘。
    """
    return read_reset_parts(session_mgr, sid, keep_turns)["dropped"]


def hard_reset_session(
    session_mgr,
    sid: str,
    keep_turns: int,
    logger=None,
    summary_chunk: Optional[list] = None,
    write_through: bool = True,
) -> int:
    """
    对单个 sid：读记忆 → 剥离旧摘要 → 保留最近 keep_turns 轮 → 融合摘要写回。

    write_through=True（默认）：直接 write_memory 覆写，保留 title/description；
    False：旧路径 delete + 重建。返回保留的 chunk 数。失败返回 -1。
    """
    if not session_mgr or not sid:
        return -1
    keep_turns = max(1, int(keep_turns or 1))
    try:
        old_flat = _norm_messages(session_mgr.fetch_memory(sid) or [])
        _head, rest = _split_head_summary(old_flat)
        chunks = _clean_and_chunk(rest)
        if not chunks:
            kept = 0
            kept_chunks: List[List[dict]] = []
        else:
            kept_chunks = chunks[-keep_turns:] if len(chunks) > keep_turns else chunks[:]
            kept = len(kept_chunks)

        to_write = build_fused_chunks(kept_chunks, summary_chunk)

        if write_through:
            data = getattr(session_mgr, "chat_memory", None)
            if not isinstance(data, dict) or sid not in data:
                # write_memory 是下标赋值，session 必须先存在
                try:
                    session_mgr.get_session_info(sid)
                except Exception:
                    pass
            session_mgr.write_memory(sid, to_write)
        else:
            session_mgr.delete_session(sid)
            try:
                session_mgr.get_session_info(sid)
            except Exception:
                pass
            session_mgr.write_memory(sid, to_write)

        if logger:
            logger.warning(
                "[MERGER hard] reset sid=%s keep_chunks=%d summary=%s mode=%s",
                sid,
                kept,
                "yes" if summary_chunk else "no",
                "write-through" if write_through else "delete",
            )
        return kept
    except Exception as e:
        if logger:
            logger.error("[MERGER hard] reset failed sid=%s: %s", sid, e)
        return -1


def hard_reset_members(
    session_mgr,
    sids: List[str],
    keep_turns: int,
    logger=None,
    summary_chunks: Optional[dict] = None,
    write_through: bool = True,
) -> dict:
    """summary_chunks: sid -> summary_chunk 映射（可选，重开前摘要）。"""
    results = {}
    for sid in sids:
        sc = (summary_chunks or {}).get(sid)
        results[sid] = hard_reset_session(
            session_mgr, sid, keep_turns, logger, summary_chunk=sc,
            write_through=write_through,
        )
    return results
