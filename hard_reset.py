from __future__ import annotations

"""
硬重开：对合并组内 sid 做 ADS 式 delete + 写回最近 keep 轮。
仅在 merge_reset_mode=hard 时使用。
"""

from typing import List, Optional


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


def compute_dropped_flat(session_mgr, sid: str, keep_turns: int) -> List[dict]:
    """
    只读计算：硬重开将丢弃的消息（keep 轮之前的部分）。
    供 async 层在真正重开前生成摘要用；不写盘。
    """
    if not session_mgr or not sid:
        return []
    keep_turns = max(1, int(keep_turns or 1))
    try:
        # 只读内存，避免 fetch_memory 的 _ensure_session_data 写盘
        data = getattr(session_mgr, "chat_memory", None)
        if not isinstance(data, dict) or sid not in data:
            return []
        mem_list = (data.get(sid) or {}).get("memory") or []
        if not isinstance(mem_list, list):
            return []
        flat: List[dict] = []
        for chunk in mem_list:
            if not isinstance(chunk, list):
                continue
            for m in chunk:
                if isinstance(m, dict):
                    flat.append(m)
                elif hasattr(m, "to_dict"):
                    flat.append(m.to_dict())
        chunks = _clean_and_chunk(flat)
        if not chunks or len(chunks) <= keep_turns:
            return []
        return _flatten(chunks[:-keep_turns])
    except Exception:
        return []


def hard_reset_session(
    session_mgr,
    sid: str,
    keep_turns: int,
    logger=None,
    summary_chunk: Optional[list] = None,
) -> int:
    """
    对单个 sid：读记忆 → 保留最近 keep_turns 轮 → delete → write。
    summary_chunk 非空时插入写回 chunks 最前（重开前摘要）。
    返回保留的 chunk 数。失败返回 -1。
    """
    if not session_mgr or not sid:
        return -1
    keep_turns = max(1, int(keep_turns or 1))
    try:
        # 硬重开必须用官方 API 改磁盘
        old_flat = session_mgr.fetch_memory(sid) or []
        # normalize
        norm = []
        for m in old_flat:
            if isinstance(m, dict):
                norm.append(m)
            elif hasattr(m, "to_dict"):
                norm.append(m.to_dict())
        chunks = _clean_and_chunk(norm)
        if not chunks:
            new_flat: List[dict] = []
            kept = 0
        else:
            kept_chunks = chunks[-keep_turns:] if len(chunks) > keep_turns else chunks[:]
            new_flat = _flatten(kept_chunks)
            kept = len(kept_chunks)

        session_mgr.delete_session(sid)
        # 重建空会话元数据
        try:
            session_mgr.get_session_info(sid)
        except Exception:
            pass

        to_write: List[List[dict]] = []
        if new_flat:
            to_write = _clean_and_chunk(new_flat) or []
        if summary_chunk:
            to_write = [list(summary_chunk)] + to_write
        session_mgr.write_memory(sid, to_write)

        if logger:
            logger.warning(
                "[MERGER hard] reset sid=%s keep_chunks=%d summary=%s",
                sid,
                kept,
                "yes" if summary_chunk else "no",
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
) -> dict:
    """summary_chunks: sid -> summary_chunk 映射（可选，重开前摘要）。"""
    results = {}
    for sid in sids:
        sc = (summary_chunks or {}).get(sid)
        results[sid] = hard_reset_session(
            session_mgr, sid, keep_turns, logger, summary_chunk=sc
        )
    return results
