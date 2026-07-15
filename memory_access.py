from __future__ import annotations

"""
安全读取官方会话记忆，避免触发 SessionManager._ensure_session_data 写盘。
"""

import time
from typing import Any, Dict, List, Optional, Tuple

from .timeout_utils import is_expired, session_type_of


def list_existing_session_ids(session_mgr) -> List[str]:
    if not session_mgr:
        return []
    data = getattr(session_mgr, "chat_memory", None)
    if not isinstance(data, dict):
        return []
    return list(data.keys())


def safe_session_meta(session_mgr, sid: str) -> Dict[str, Any]:
    empty = {"title": "", "timestamp": 0, "description": "", "chunk_count": 0}
    if not session_mgr or not sid:
        return empty
    data = getattr(session_mgr, "chat_memory", None)
    if not isinstance(data, dict) or sid not in data:
        return empty
    sd = data.get(sid) or {}
    ts = sd.get("timestamp") or 0
    try:
        ts = int(ts)
    except Exception:
        ts = 0
    mem = sd.get("memory") or []
    chunk_count = len(mem) if isinstance(mem, list) else 0
    return {
        "title": str(sd.get("title") or "").strip(),
        "timestamp": ts,
        "description": str(sd.get("description") or "").strip(),
        "chunk_count": chunk_count,
    }


def safe_fetch_memory(
    session_mgr,
    sid: str,
    max_chunks: Optional[int] = None,
) -> List[dict]:
    if not session_mgr or not sid:
        return []
    data = getattr(session_mgr, "chat_memory", None)
    if not isinstance(data, dict) or sid not in data:
        return []
    session_data = data.get(sid) or {}
    mem_list = session_data.get("memory", []) or []
    if not isinstance(mem_list, list):
        return []

    if max_chunks is not None and max_chunks > 0 and len(mem_list) > max_chunks:
        mem_list = mem_list[-max_chunks:]

    messages: List[dict] = []
    for chunk in mem_list:
        if not isinstance(chunk, list):
            continue
        for message in chunk:
            if isinstance(message, dict):
                messages.append(message)
            elif hasattr(message, "to_dict"):
                try:
                    messages.append(message.to_dict())
                except Exception:
                    pass
    return messages


def pick_member_sids(
    session_mgr,
    candidates: List[str],
    current_sid: str,
    max_sessions: int = 8,
    other_session_timeout: int = 20,
    other_session_timeout_private: int = 0,
    other_session_timeout_group: int = 0,
) -> List[str]:
    """
    当前会话必留 + 其他会话：先超时过滤，再按活跃度取 max_sessions-1。
    """
    max_sessions = max(1, int(max_sessions or 1))
    if not candidates:
        return [current_sid] if current_sid else []

    now = time.time()
    scored: List[Tuple[int, int, str]] = []
    for sid in candidates:
        if not sid or sid == current_sid:
            continue
        meta = safe_session_meta(session_mgr, sid)
        if meta.get("chunk_count", 0) <= 0:
            continue
        ts = int(meta.get("timestamp") or 0)
        st = session_type_of(sid)
        if is_expired(
            ts,
            st,
            other_session_timeout,
            other_session_timeout_private,
            other_session_timeout_group,
            now=now,
        ):
            continue
        scored.append((ts, int(meta.get("chunk_count") or 0), sid))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    limit = max(0, max_sessions - 1)
    chosen = [sid for _, _, sid in scored[:limit]]

    out: List[str] = []
    if current_sid:
        out.append(current_sid)
    for sid in chosen:
        if sid not in out:
            out.append(sid)
    return out
