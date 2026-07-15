from __future__ import annotations

import time
from typing import Optional


def resolve_timeout_minutes(
    session_type: str,
    other_session_timeout: int = 20,
    other_session_timeout_private: int = 0,
    other_session_timeout_group: int = 0,
) -> int:
    """
    返回该会话类型应使用的超时分钟数。
    0 表示不因超时剔除。
    """
    st = (session_type or "").lower()
    if st == "dm" and other_session_timeout_private and other_session_timeout_private > 0:
        return int(other_session_timeout_private)
    if st == "gm" and other_session_timeout_group and other_session_timeout_group > 0:
        return int(other_session_timeout_group)
    return max(0, int(other_session_timeout or 0))


def is_expired(
    last_active_ts: int,
    session_type: str,
    other_session_timeout: int = 20,
    other_session_timeout_private: int = 0,
    other_session_timeout_group: int = 0,
    now: Optional[float] = None,
) -> bool:
    """last_active_ts 为 unix 秒；超时返回 True。"""
    minutes = resolve_timeout_minutes(
        session_type,
        other_session_timeout,
        other_session_timeout_private,
        other_session_timeout_group,
    )
    if minutes <= 0:
        return False
    if not last_active_ts or last_active_ts <= 0:
        return True
    now_ts = int(now if now is not None else time.time())
    return (now_ts - int(last_active_ts)) > minutes * 60


def session_type_of(sid: str) -> str:
    parts = (sid or "").split(":", 2)
    return parts[1] if len(parts) >= 2 else ""
