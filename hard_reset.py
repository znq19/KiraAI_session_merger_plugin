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


def hard_reset_session(session_mgr, sid: str, keep_turns: int, logger=None) -> int:
    """
    对单个 sid：读记忆 → 保留最近 keep_turns 轮 → delete → write。
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

        if new_flat:
            to_write = _clean_and_chunk(new_flat)
            session_mgr.write_memory(sid, to_write if to_write else [])
        else:
            session_mgr.write_memory(sid, [])

        if logger:
            logger.warning(
                "[MERGER hard] reset sid=%s keep_chunks=%d", sid, kept
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
) -> dict:
    results = {}
    for sid in sids:
        results[sid] = hard_reset_session(session_mgr, sid, keep_turns, logger)
    return results
