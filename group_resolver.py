from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from .memory_access import list_existing_session_ids


class GroupResolver:
    """根据配置解析 sid 所属合并组及成员列表（读时合并，无私有真相源）。"""

    GROUP_ALL = "all_merged"
    GROUP_GROUPS = "all_groups"
    GROUP_DMS = "all_dms"

    def __init__(
        self,
        session_mgr,
        enabled: bool = False,
        allowed_adapters: Optional[List[str]] = None,
        merge_all_groups: bool = True,
        merge_all_dms: bool = True,
        merge_groups_with_dms: bool = True,
        extra_sessions: Optional[List[str]] = None,
        exclude_sessions: Optional[List[str]] = None,
    ):
        self.session_mgr = session_mgr
        self.enabled = enabled
        self.allowed_adapters: Optional[Set[str]] = (
            set(a.strip() for a in (allowed_adapters or []) if str(a).strip()) or None
        )
        self.merge_all_groups = merge_all_groups
        self.merge_all_dms = merge_all_dms
        self.merge_groups_with_dms = merge_groups_with_dms
        self.extra_sessions = [s.strip() for s in (extra_sessions or []) if str(s).strip()]
        self.exclude_sessions = set(
            s.strip() for s in (exclude_sessions or []) if str(s).strip()
        )

    @staticmethod
    def parse_sid(sid: str) -> Dict[str, str]:
        parts = (sid or "").split(":", 2)
        return {
            "adapter": parts[0] if len(parts) >= 1 else "",
            "session_type": parts[1] if len(parts) >= 2 else "",
            "session_id": parts[2] if len(parts) >= 3 else sid or "",
        }

    def _adapter_allowed(self, adapter: str) -> bool:
        if self.allowed_adapters is None:
            return True
        return adapter in self.allowed_adapters

    def list_known_sessions(self) -> List[str]:
        try:
            return list_existing_session_ids(self.session_mgr)
        except Exception:
            return []

    def _is_member_of_type(self, sid: str, st: str) -> bool:
        """是否应进入合并池：全量开关 或 额外会话列表。"""
        if sid in self.extra_sessions:
            return True
        if st == "gm":
            return bool(self.merge_all_groups)
        if st == "dm":
            return bool(self.merge_all_dms)
        return False

    def is_session_eligible(self, sid: str) -> bool:
        if not self.enabled or not sid:
            return False
        if sid in self.exclude_sessions:
            return False
        meta = self.parse_sid(sid)
        if not self._adapter_allowed(meta["adapter"]):
            return False
        st = meta["session_type"]
        # 注意：merge_groups_with_dms 只决定「群与私是否同一时间线」，
        # 不表示「自动包含全部群/私」。范围由 merge_all_* + extra_sessions 决定。
        return self._is_member_of_type(sid, st)

    def resolve_group_id(self, sid: str) -> Optional[str]:
        if not self.is_session_eligible(sid):
            return None
        meta = self.parse_sid(sid)
        adapter = meta["adapter"] or "unknown"
        st = meta["session_type"]

        # 群私互通：同一条人生时间线
        if self.merge_groups_with_dms:
            return f"{adapter}:{self.GROUP_ALL}"

        # 不互通：群一组、私一组（额外会话按类型归入对应组）
        if st == "gm":
            return f"{adapter}:{self.GROUP_GROUPS}"
        if st == "dm":
            return f"{adapter}:{self.GROUP_DMS}"
        return f"{adapter}:{self.GROUP_ALL}"

    def list_group_members(self, group_id: str) -> List[str]:
        if not group_id:
            return []
        parts = group_id.split(":", 1)
        if len(parts) != 2:
            return []
        adapter, gtype = parts[0], parts[1]

        known = set(self.list_known_sessions())
        known.update(self.extra_sessions)

        members: List[str] = []
        for sid in known:
            if sid in self.exclude_sessions:
                continue
            meta = self.parse_sid(sid)
            if meta["adapter"] != adapter:
                continue
            if not self._adapter_allowed(meta["adapter"]):
                continue
            st = meta["session_type"]
            if not self._is_member_of_type(sid, st):
                continue

            if gtype == self.GROUP_ALL:
                if st in ("gm", "dm"):
                    members.append(sid)
            elif gtype == self.GROUP_GROUPS:
                if st == "gm":
                    members.append(sid)
            elif gtype == self.GROUP_DMS:
                if st == "dm":
                    members.append(sid)

        seen: Set[str] = set()
        ordered: List[str] = []
        for sid in sorted(members):
            if sid not in seen:
                seen.add(sid)
                ordered.append(sid)
        return ordered

    def members_for_session(self, sid: str) -> List[str]:
        gid = self.resolve_group_id(sid)
        if not gid:
            return [sid] if sid else []
        members = self.list_group_members(gid)
        if sid not in members:
            members = list(members) + [sid]
        return members

    def summarize(self) -> List[Dict[str, Any]]:
        groups: Dict[str, List[str]] = {}
        for sid in self.list_known_sessions() + self.extra_sessions:
            gid = self.resolve_group_id(sid)
            if not gid:
                continue
            groups.setdefault(gid, [])
        result = []
        for gid in sorted(groups.keys()):
            members = self.list_group_members(gid)
            result.append({"group_id": gid, "members": members, "count": len(members)})
        return result
