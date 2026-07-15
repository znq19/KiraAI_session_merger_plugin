from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from core.plugin import BasePlugin, logger, on, Priority, register
from core.provider import LLMRequest
from core.chat.message_utils import KiraMessageBatchEvent, KiraMessageEvent, KiraIMSentResult
from core.chat import MessageChain
from core.chat.message_elements import Text


from .anchor import DEFAULT_WINDOW_ANCHOR_PROMPT
from .compat import (
    log_compat_status,
    maybe_disable_ads,
    maybe_disable_history_plugin,
)
from .cross_session import (
    build_route_dedup_message,
    event_has_handoff,
    is_merge_route_request_text,
    is_route_handoff_result,
    mark_event_handoff,
    route_cross_session_request,
)
from .group_agent_queue import GroupAgentQueue
from .group_resolver import GroupResolver
from .history_tool import HistoryToolService
from .merge_engine import MergeEngine
from .observe_pool import ObservePool
from .timeline import TimelineBuilder




class SessionMergerPlugin(BasePlugin):
    """会话合并 v2：读时合并官方记忆 + 未唤醒观察池偷看。"""

    def __init__(self, ctx, cfg: dict):
        super().__init__(ctx, cfg)
        self.data_dir: Optional[Path] = None
        self.resolver: Optional[GroupResolver] = None
        self.timeline: Optional[TimelineBuilder] = None
        self.engine: Optional[MergeEngine] = None
        self.history_svc: Optional[HistoryToolService] = None
        self.observe_pool: Optional[ObservePool] = None

        # mode
        self.enabled = False
        self.allowed_adapters: List[str] = ["qq"]
        self.merge_all_groups = True
        self.merge_all_dms = True
        self.merge_groups_with_dms = True
        self.max_merge_sessions = 8
        self.extra_sessions: List[str] = []
        self.exclude_sessions: List[str] = []

        # context / reset
        self.max_merged_chunks = 10
        self.merge_token_limit = 15000
        self.merge_keep_turns = 6
        self.merge_reset_mode = "soft"
        self.merge_check_interval_sec = 60
        self.chars_per_token = 2.0
        self.other_session_timeout = 20
        self.other_session_timeout_private = 0
        self.other_session_timeout_group = 0
        self.unmentioned_probability = 0.01
        self.max_observe_per_session = 20
        self.peek_max_messages = 3
        self.include_tool_traces = True
        self.drop_unpaired_tools = True
        self.source_tag_mode = "prefix"
        self.cross_session_time_order = True
        self.merge_build_timeout_sec = 5.0


        # anchor
        self.enable_window_anchor = True
        self.window_anchor_prompt = DEFAULT_WINDOW_ANCHOR_PROMPT

        # history tool
        self.enable_history_tool = True
        self.disable_history_plugin = True
        self.http_host = "localhost"
        self.http_port = 3000
        self.access_token = ""
        self.master_id = ""
        self.allowed_users: List[str] = []
        self.restricted_groups: List[str] = []
        self.cache_ttl_sec = 120

        # commands
        self.enable_status_command = False
        self.status_commands: List[str] = ["/merge s", "/合并状态"]
        self.enable_preview_command = False
        self.preview_commands: List[str] = ["/merge p", "/合并预览"]
        self.command_allowed_users: List[str] = []

        # reboot all (参考 reboot_plugin)
        self.enable_reboot_all_command = True
        self.reboot_all_commands: List[str] = ["/reboota"]
        self.reboot_all_enable_permission = False
        self.reboot_all_allowed_users: List[str] = []
        self.reboot_all_success_message = (
            "✅ 已清除全部合并会话的上下文记忆（共 {count} 个），我们可以重新开始对话了！"
        )
        self.reboot_all_permission_denied_message = (
            "❌ 权限不足：您没有全部重启合并会话的权限"
        )
        self.reboot_all_error_message = "❌ 全部重启失败: {error}"

        # 合并组单一 agent 队列（逻辑一个窗口）
        self.enable_group_agent_queue = True
        self.group_agent_lock_ttl_sec = 180
        self.group_agent_settle_sec = 0.4
        self.group_agent_max_queue = 32
        self.group_queue: Optional[GroupAgentQueue] = None
        self._memory_write_wrapped = False
        self._orig_update_memory = None
        self._drain_tasks: Dict[str, asyncio.Task] = {}

        # debug
        self.enable_debug_log = False
        self.log_merged_message_preview = False
        self.enable_send_debug_log = False

        # compat
        self.auto_disable_ads = False

        # 跨会话 session_send 去重：(source_sid, target) -> last_ts
        self._session_send_dedup: Dict[Tuple[str, str], float] = {}
        self.session_send_dedup_sec = 25
        self._session_send_wrapped = False
        # 包装版本：升级后强制重装，避免热重载残留「直达」旧包装
        # 9 = 同会话拒绝文案 + description/tip 收紧（当前会话直接 xml）
        self._SESSION_SEND_WRAP_VERSION = 9





    def _load_cfg(self):

        cfg = self.plugin_cfg or {}

        mode = cfg.get("section_mode", {})
        self.enabled = bool(mode.get("enabled", False))
        adapters = mode.get("allowed_adapters", ["qq"])
        if isinstance(adapters, str):
            adapters = [adapters]
        self.allowed_adapters = [str(a).strip() for a in adapters if str(a).strip()]
        self.merge_all_groups = bool(mode.get("merge_all_groups", True))
        self.merge_all_dms = bool(mode.get("merge_all_dms", True))
        self.merge_groups_with_dms = bool(mode.get("merge_groups_with_dms", True))
        self.max_merge_sessions = int(mode.get("max_merge_sessions", 8) or 8)

        extra = mode.get("extra_sessions", None)
        if extra is None:
            extra = mode.get("pool_map", [])
        if isinstance(extra, str):
            extra = [extra]
        self.extra_sessions = [str(s).strip() for s in (extra or []) if str(s).strip()]

        exclude = mode.get("exclude_sessions", [])
        if isinstance(exclude, str):
            exclude = [exclude]
        self.exclude_sessions = [str(s).strip() for s in (exclude or []) if str(s).strip()]

        ctx_sec = cfg.get("section_context", {})
        self.max_merged_chunks = int(
            ctx_sec.get(
                "max_merged_chunks",
                ctx_sec.get("max_pool_turns", 10),
            )
            or 10
        )
        # 兼容旧 key max_merged_tokens
        self.merge_token_limit = int(
            ctx_sec.get(
                "merge_token_limit",
                ctx_sec.get("max_merged_tokens", ctx_sec.get("max_pool_tokens", 30000)),
            )
            or 30000
        )
        self.merge_keep_turns = int(ctx_sec.get("merge_keep_turns", 6) or 6)
        self.merge_reset_mode = str(ctx_sec.get("merge_reset_mode", "soft") or "soft").lower()
        if self.merge_reset_mode not in ("soft", "hard"):
            self.merge_reset_mode = "soft"
        self.merge_check_interval_sec = int(ctx_sec.get("merge_check_interval_sec", 60) or 0)
        self.chars_per_token = float(ctx_sec.get("chars_per_token", 2.0))
        self.other_session_timeout = int(ctx_sec.get("other_session_timeout", 20) or 0)
        self.other_session_timeout_private = int(
            ctx_sec.get("other_session_timeout_private", 0) or 0
        )
        self.other_session_timeout_group = int(
            ctx_sec.get("other_session_timeout_group", 0) or 0
        )
        self.unmentioned_probability = float(
            ctx_sec.get("unmentioned_probability", 0.01) or 0
        )
        self.max_observe_per_session = int(
            ctx_sec.get("max_observe_per_session", 20) or 20
        )
        self.peek_max_messages = int(ctx_sec.get("peek_max_messages", 3) or 3)
        self.include_tool_traces = bool(ctx_sec.get("include_tool_traces", True))
        self.drop_unpaired_tools = bool(ctx_sec.get("drop_unpaired_tools", True))
        self.source_tag_mode = str(ctx_sec.get("source_tag_mode", "prefix") or "prefix")
        self.cross_session_time_order = bool(ctx_sec.get("cross_session_time_order", True))
        self.merge_build_timeout_sec = float(
            ctx_sec.get("merge_build_timeout_sec", 5.0) or 5.0
        )
        if self.merge_build_timeout_sec < 1.0:
            self.merge_build_timeout_sec = 1.0
        # 优先 section_mode；兼容旧配置写在 section_context
        mode_for_dedup = cfg.get("section_mode") or {}
        if not isinstance(mode_for_dedup, dict):
            mode_for_dedup = {}
        if "session_send_dedup_sec" in mode_for_dedup:
            self.session_send_dedup_sec = int(
                mode_for_dedup.get("session_send_dedup_sec", 25) or 0
            )
        else:
            self.session_send_dedup_sec = int(
                ctx_sec.get("session_send_dedup_sec", 25) or 0
            )

        if self.session_send_dedup_sec < 0:
            self.session_send_dedup_sec = 0



        anchor = cfg.get("section_anchor", {})

        self.enable_window_anchor = bool(anchor.get("enable_window_anchor", True))
        self.window_anchor_prompt = (
            anchor.get("window_anchor_prompt", DEFAULT_WINDOW_ANCHOR_PROMPT)
            or DEFAULT_WINDOW_ANCHOR_PROMPT
        )

        hist = cfg.get("section_history_tool", {})
        http_legacy = cfg.get("section_http", {})
        perm_legacy = cfg.get("section_history_permission", {})
        self.enable_history_tool = bool(hist.get("enable_history_tool", True))
        self.disable_history_plugin = bool(hist.get("disable_history_plugin", True))
        self.http_host = hist.get("http_host", http_legacy.get("onebot_http_host", "localhost"))
        self.http_port = int(hist.get("http_port", http_legacy.get("onebot_http_port", 3000)))
        self.access_token = hist.get(
            "access_token", http_legacy.get("onebot_access_token", "")
        ) or ""
        self.master_id = str(hist.get("master_id", perm_legacy.get("master_id", "")) or "")
        au = hist.get("allowed_users", perm_legacy.get("allowed_users", []))
        if isinstance(au, str):
            au = [x.strip() for x in au.split(",") if x.strip()]
        self.allowed_users = [str(u).strip() for u in (au or []) if str(u).strip()]
        rg = hist.get("restricted_groups", perm_legacy.get("restricted_groups", []))
        if isinstance(rg, str):
            rg = [x.strip() for x in rg.split(",") if x.strip()]
        self.restricted_groups = [str(g).strip() for g in (rg or []) if str(g).strip()]
        self.cache_ttl_sec = int(hist.get("cache_ttl_sec", 30))

        cmd = cfg.get("section_command", {})
        self.enable_status_command = bool(cmd.get("enable_status_command", False))
        sc = cmd.get("status_commands", cmd.get("status_command", ["/merge s", "/合并状态"]))
        if isinstance(sc, str):
            sc = [sc]
        self.status_commands = [str(x).strip() for x in (sc or []) if str(x).strip()]
        self.enable_preview_command = bool(cmd.get("enable_preview_command", False))
        pc = cmd.get("preview_commands", cmd.get("preview_command", ["/merge p", "/合并预览"]))
        if isinstance(pc, str):
            pc = [pc]
        self.preview_commands = [str(x).strip() for x in (pc or []) if str(x).strip()]
        cau = cmd.get("command_allowed_users", [])
        if isinstance(cau, str):
            cau = [cau]
        self.command_allowed_users = [str(u).strip() for u in (cau or []) if str(u).strip()]

        self.enable_reboot_all_command = bool(cmd.get("enable_reboot_all_command", True))
        rac = cmd.get("reboot_all_commands", cmd.get("reboot_all_command", ["/reboota"]))
        if isinstance(rac, str):
            rac = [rac]
        self.reboot_all_commands = [str(x).strip() for x in (rac or []) if str(x).strip()]
        if not self.reboot_all_commands:
            self.reboot_all_commands = ["/reboota"]
        self.reboot_all_enable_permission = bool(
            cmd.get("reboot_all_enable_permission", False)
        )
        rau = cmd.get("reboot_all_allowed_users", [])
        if isinstance(rau, str):
            rau = [x.strip() for x in rau.split(",") if x.strip()]
        self.reboot_all_allowed_users = [
            str(u).strip() for u in (rau or []) if str(u).strip()
        ]
        self.reboot_all_success_message = str(
            cmd.get(
                "reboot_all_success_message",
                "✅ 已清除全部合并会话的上下文记忆（共 {count} 个），我们可以重新开始对话了！",
            )
            or "✅ 已清除全部合并会话的上下文记忆（共 {count} 个），我们可以重新开始对话了！"
        )
        self.reboot_all_permission_denied_message = str(
            cmd.get(
                "reboot_all_permission_denied_message",
                "❌ 权限不足：您没有全部重启合并会话的权限",
            )
            or "❌ 权限不足：您没有全部重启合并会话的权限"
        )
        self.reboot_all_error_message = str(
            cmd.get("reboot_all_error_message", "❌ 全部重启失败: {error}")
            or "❌ 全部重启失败: {error}"
        )

        # 组级 agent 队列（独立分组；兼容旧配置写在 section_context 的情况）
        q_sec = cfg.get("section_group_agent_queue") or {}
        ctx_legacy = cfg.get("section_context") or {}
        if not isinstance(q_sec, dict):
            q_sec = {}
        if not isinstance(ctx_legacy, dict):
            ctx_legacy = {}

        def _q_get(key, default):
            if key in q_sec:
                return q_sec.get(key)
            return ctx_legacy.get(key, default)

        self.enable_group_agent_queue = bool(_q_get("enable_group_agent_queue", True))
        self.group_agent_lock_ttl_sec = int(
            _q_get("group_agent_lock_ttl_sec", 180) or 180
        )
        self.group_agent_settle_sec = float(
            _q_get("group_agent_settle_sec", 0.4) or 0.0
        )
        self.group_agent_max_queue = int(_q_get("group_agent_max_queue", 32) or 32)

        debug = cfg.get("section_debug", {})
        self.enable_debug_log = bool(debug.get("enable_debug_log", False))
        self.log_merged_message_preview = bool(debug.get("log_merged_message_preview", False))
        self.enable_send_debug_log = bool(debug.get("enable_send_debug_log", False))

        compat = cfg.get("section_compat", {})
        self.auto_disable_ads = bool(compat.get("auto_disable_ads", False))

    def _build_components(self):

        self.resolver = GroupResolver(
            session_mgr=self.ctx.session_mgr,
            enabled=self.enabled,
            allowed_adapters=self.allowed_adapters,
            merge_all_groups=self.merge_all_groups,
            merge_all_dms=self.merge_all_dms,
            merge_groups_with_dms=self.merge_groups_with_dms,
            extra_sessions=self.extra_sessions,
            exclude_sessions=self.exclude_sessions,
        )
        self.timeline = TimelineBuilder(
            session_mgr=self.ctx.session_mgr,
            chars_per_token=self.chars_per_token,
            include_tool_traces=self.include_tool_traces,
            drop_unpaired_tools=self.drop_unpaired_tools,
            cross_session_time_order=self.cross_session_time_order,
            debug=self.enable_debug_log,
            logger=logger,
        )
        self.observe_pool = ObservePool(
            data_dir=self.data_dir,
            max_per_session=self.max_observe_per_session,
            other_session_timeout=self.other_session_timeout,
            other_session_timeout_private=self.other_session_timeout_private,
            other_session_timeout_group=self.other_session_timeout_group,
            logger=logger,
        )
        self.engine = MergeEngine(
            session_mgr=self.ctx.session_mgr,
            resolver=self.resolver,
            timeline=self.timeline,
            observe_pool=self.observe_pool,
            max_merged_chunks=self.max_merged_chunks,
            merge_token_limit=self.merge_token_limit,
            merge_keep_turns=self.merge_keep_turns,
            merge_reset_mode=self.merge_reset_mode,
            merge_check_interval_sec=self.merge_check_interval_sec,
            chars_per_token=self.chars_per_token,
            max_merge_sessions=self.max_merge_sessions,
            other_session_timeout=self.other_session_timeout,
            other_session_timeout_private=self.other_session_timeout_private,
            other_session_timeout_group=self.other_session_timeout_group,
            unmentioned_probability=self.unmentioned_probability,
            peek_max_messages=self.peek_max_messages,
            source_tag_mode=self.source_tag_mode,
            enable_window_anchor=self.enable_window_anchor,
            window_anchor_prompt=self.window_anchor_prompt,
            merge_build_timeout_sec=self.merge_build_timeout_sec,
            debug=self.enable_debug_log,
            log_preview=self.log_merged_message_preview,
            logger=logger,
        )

        self.history_svc = HistoryToolService(
            http_host=self.http_host,
            http_port=self.http_port,
            access_token=self.access_token,
            master_id=self.master_id,
            allowed_users=self.allowed_users,
            restricted_groups=self.restricted_groups,
            cache_ttl_sec=self.cache_ttl_sec,
            logger=logger,
        )

    async def initialize(self):
        self.data_dir = self.ctx.get_plugin_data_dir()
        self._load_cfg()
        self._build_components()

        await log_compat_status(self.ctx.plugin_mgr, logger)

        if self.disable_history_plugin and self.enable_history_tool:
            await maybe_disable_history_plugin(self.ctx.plugin_mgr, True, logger)

        # hard 模式与 ADS 互斥；soft 仅在用户显式 auto_disable_ads 时关 ADS
        if self.merge_reset_mode == "hard" or self.auto_disable_ads:
            await maybe_disable_ads(self.ctx.plugin_mgr, True, logger)
            if self.merge_reset_mode == "hard":
                logger.warning(
                    "合并超限模式为 hard：已自动禁用 ADS，避免与组级硬重开冲突"
                )

        logger.info(
            "会话合并初始化 enabled=%s mode=%s max_sessions=%d chunks=%d "
            "token_limit=%d keep=%d timeout=%dm peek=%.3f group_queue=%s",
            self.enabled,
            self.merge_reset_mode,
            self.max_merge_sessions,
            self.max_merged_chunks,
            self.merge_token_limit,
            self.merge_keep_turns,
            self.other_session_timeout,
            self.unmentioned_probability,
            self.enable_group_agent_queue,
        )

    @on.loaded()
    async def on_loaded(self, *_):
        """全部插件加载完后接管 session_send。"""
        self._ensure_session_send_wrap()

    def _ensure_session_send_wrap(self):
        """可重复调用：合并启用时强制安装/升级 ROUTE 包装。"""
        if not self.enabled:
            return
        llm_api = getattr(self.ctx, "llm_api", None)
        fn = getattr(llm_api, "tools_functions", {}).get("session_send") if llm_api else None
        ver = getattr(fn, "_merger_session_send_version", 0) if fn else 0
        if (
            fn
            and getattr(fn, "_merger_session_send_wrapped", False)
            and ver >= self._SESSION_SEND_WRAP_VERSION
        ):
            self._session_send_wrapped = True
            return
        self._session_send_wrapped = False
        self._wrap_session_send_tool()

    def _bind_session_send_to_request(self, req: LLMRequest) -> bool:
        """
        关键：core 在 ON_LLM_REQUEST 之前就 build_tool_set()，
        会把当时的函数引用冻进 tool_set。必须把当前 req 里的
        session_send 工具也改成 ROUTE 函数，否则本轮仍跑旧直达逻辑。
        """
        self._ensure_session_send_wrap()
        llm_api = getattr(self.ctx, "llm_api", None)
        if not llm_api:
            return False
        route_fn = llm_api.tools_functions.get("session_send")
        if not route_fn or not getattr(route_fn, "_merger_session_send_wrapped", False):
            return False
        if getattr(route_fn, "_merger_session_send_version", 0) < self._SESSION_SEND_WRAP_VERSION:
            return False

        ts = getattr(req, "tool_set", None)
        if ts is None:
            return False
        tool = ts.get("session_send") if hasattr(ts, "get") else None
        if tool is None:
            return False

        # _LegacyFuncTool 使用 _func
        if hasattr(tool, "_func"):
            tool._func = route_fn
            return True
        # 其它 BaseTool：替换 execute
        async def _exec(event, *args, **kwargs):
            return await route_fn(event, *args, **kwargs)

        try:
            tool.execute = _exec  # type: ignore
            return True
        except Exception:
            return False


    def _wrap_session_send_tool(self):
        """
        合并模式接管 session_send = 跨会话请求路由：
        切换到目标会话并激活 LLM，目标侧带着合并上文继续执行。
        """
        llm_api = getattr(self.ctx, "llm_api", None)
        if not llm_api or not hasattr(llm_api, "tools_functions"):
            logger.warning("[MERGER] cannot wrap session_send: llm_api missing")
            return

        current = llm_api.tools_functions.get("session_send")
        if current is None:
            logger.warning(
                "[MERGER] session_send not registered yet; will retry on next llm_request"
            )
            return

        # 若已是最新 ROUTE 包装则跳过
        if (
            getattr(current, "_merger_session_send_wrapped", False)
            and getattr(current, "_merger_session_send_version", 0)
            >= self._SESSION_SEND_WRAP_VERSION
        ):
            self._session_send_wrapped = True
            return

        # 若当前是旧包装，尽量解包拿到更底层 original（没有则用当前函数）
        original = getattr(current, "_merger_session_send_original", None) or current

        plugin = self

        async def wrapped_session_send(event, target: str = "", description: str = "", **kwargs):
            target = (target or kwargs.get("target") or "").strip()
            description = description or kwargs.get("description") or ""
            source = ""
            try:
                source = getattr(event, "sid", "") or ""
            except Exception:
                source = ""

            if not target:
                return "failed: missing target"

            if source and source == target:
                return (
                    "Do not session_send to the current session; output xml directly here. "
                    "If a tool already returned an attachment path, send it with "
                    '<file type="record|image|file">path</file> in <msg>; '
                    "do not call session_send again for this session."
                )

            ttl = float(getattr(plugin, "session_send_dedup_sec", 0) or 0)
            now = time.time()
            key = (source, target)
            if ttl > 0:
                expired = [
                    k for k, ts in plugin._session_send_dedup.items()
                    if now - ts > ttl
                ]
                for k in expired:
                    plugin._session_send_dedup.pop(k, None)
                last = plugin._session_send_dedup.get(key, 0)
                if last and (now - last) < ttl:
                    logger.info(
                        "[MERGER] dedupe session_send route %s -> %s (%.0fs left)",
                        source,
                        target,
                        ttl - (now - last),
                    )
                    mark_event_handoff(event, target)
                    return build_route_dedup_message(target)

            ok, msg = await route_cross_session_request(
                plugin.ctx,
                source_sid=source,
                target=target,
                description=description,
                logger=logger,
            )
            if ok:
                mark_event_handoff(event, target)
                if ttl > 0:
                    plugin._session_send_dedup[key] = now
            else:
                logger.error("[MERGER] route failed: %s", msg)
            return msg

        wrapped_session_send._merger_session_send_wrapped = True  # type: ignore
        wrapped_session_send._merger_session_send_version = (  # type: ignore
            self._SESSION_SEND_WRAP_VERSION
        )
        wrapped_session_send._merger_session_send_original = original  # type: ignore
        llm_api.tools_functions["session_send"] = wrapped_session_send
        try:
            for td in getattr(llm_api, "tools_definitions", []) or []:
                fn = td.get("function") or {}
                if fn.get("name") == "session_send":
                    fn["description"] = (
                        "【合并模式·跨会话请求】切换到目标会话，带着合并上文继续执行。"
                        "仅当 target ≠ 当前会话时使用；当前会话请直接输出 xml。"
                        "用户说「去群里/去私聊/到某会话聊」或任务应在另一会话完成时，应主动调用。"
                        "target 如 qq:dm:123 / qq:gm:456（可从会话列表复制）。"
                        "成功后任务交给目标会话；不要在源会话替目标执行业务工具；"
                        "同一目标短时间勿重复调用。"
                    )
                    props = (fn.get("parameters") or {}).get("properties") or {}
                    if "description" in props:
                        props["description"]["description"] = (
                            "跨会话请求说明：将在目标会话执行的任务（写清做什么即可）。"
                        )

                    break
        except Exception:
            pass

        self._session_send_wrapped = True
        logger.info(
            "[MERGER] session_send ROUTE v%s installed "
            "(route + handoff stop this event only; tool_set rebound each request)",
            self._SESSION_SEND_WRAP_VERSION,
        )






    async def terminate(self):
        if self.observe_pool:
            try:
                self.observe_pool.flush(force=True)
            except Exception:
                pass
        self._unwrap_update_memory()
        if self.group_queue:
            try:
                self.group_queue.clear_all()
            except Exception:
                pass
        for t in list(self._drain_tasks.values()):
            if t and not t.done():
                t.cancel()
        self._drain_tasks.clear()
        # 尽量恢复 session_send（若仍指向我们的包装则无法还原原函数，热重载会重建）
        self.engine = None
        self.timeline = None
        self.resolver = None
        self.history_svc = None
        self.observe_pool = None
        self.group_queue = None
        self._session_send_wrapped = False
        logger.info("会话合并已终止")

    # ── 组级 agent 队列：wrap 落盘 + 串行调度 ─────────────────

    def _wrap_update_memory(self):
        """在官方 update_memory 之后释放组锁并调度队列（等落盘）。"""
        sm = getattr(self.ctx, "session_mgr", None)
        if not sm or not hasattr(sm, "update_memory"):
            return
        if getattr(sm.update_memory, "_merger_memory_wrapped", False):
            self._memory_write_wrapped = True
            return
        if self._orig_update_memory is None:
            self._orig_update_memory = sm.update_memory

        plugin = self
        orig = self._orig_update_memory

        def wrapped_update_memory(session, new_chunk, *args, **kwargs):
            result = orig(session, new_chunk, *args, **kwargs)
            try:
                if plugin.enabled and plugin.group_queue and plugin.resolver:
                    sid = str(session or "")
                    gid = plugin.resolver.resolve_group_id(sid)
                    if gid:
                        # 同步钩子里调度异步 drain
                        try:
                            loop = asyncio.get_running_loop()
                            loop.create_task(
                                plugin._on_group_memory_written(sid, gid)
                            )
                        except RuntimeError:
                            pass
            except Exception:
                logger.exception("[MERGER queue] post update_memory hook failed")
            return result

        wrapped_update_memory._merger_memory_wrapped = True  # type: ignore
        sm.update_memory = wrapped_update_memory
        self._memory_write_wrapped = True
        logger.info("[MERGER queue] session_mgr.update_memory wrapped (release after disk write)")

    def _unwrap_update_memory(self):
        sm = getattr(self.ctx, "session_mgr", None)
        if not sm or not self._orig_update_memory:
            return
        try:
            if getattr(sm.update_memory, "_merger_memory_wrapped", False):
                sm.update_memory = self._orig_update_memory
        except Exception:
            pass
        self._memory_write_wrapped = False

    async def _on_group_memory_written(self, sid: str, group_id: str):
        if not self.group_queue:
            return
        await self.group_queue.on_memory_written(
            sid, group_id, schedule_fn=self._schedule_group_drain
        )

    async def _schedule_group_drain(self, group_id: str):
        """落盘后 settle，再取出队首重放 batch。"""
        old = self._drain_tasks.get(group_id)
        if old and not old.done():
            return

        async def _run():
            try:
                settle = float(self.group_agent_settle_sec or 0)
                if settle > 0:
                    await asyncio.sleep(settle)
                await self._drain_group_queue(group_id)
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("[MERGER queue] drain failed group=%s", group_id)
            finally:
                self._drain_tasks.pop(group_id, None)

        self._drain_tasks[group_id] = asyncio.create_task(_run())

    async def _drain_group_queue(self, group_id: str):
        if not self.group_queue or not self.enabled:
            return
        pending = await self.group_queue.pop_next_and_begin(group_id)
        if not pending:
            return

        try:
            batch = KiraMessageBatchEvent(
                message_types=pending.message_types or [],
                timestamp=int(time.time()),
                adapter=pending.adapter,
                session=pending.session,
                messages=list(pending.messages),
            )
            self.group_queue.authorize_event(batch.event_id)
            bus = getattr(self.ctx, "event_bus", None)
            if bus is None:
                logger.error(
                    "[MERGER queue] no event_bus; cannot replay sid=%s",
                    pending.sid,
                )
                await self.group_queue.force_release(
                    group_id, "no_event_bus", schedule_fn=self._schedule_group_drain
                )
                return

            logger.info(
                "[MERGER queue] replay batch sid=%s group=%s msgs=%d event=%s",
                pending.sid,
                group_id,
                len(pending.messages),
                batch.event_id,
            )
            await bus.publish(batch)
        except Exception:
            logger.exception(
                "[MERGER queue] replay failed sid=%s group=%s",
                pending.sid,
                group_id,
            )
            await self.group_queue.force_release(
                group_id, "replay_error", schedule_fn=self._schedule_group_drain
            )

    async def _release_group_for_event(self, event, reason: str) -> bool:
        """
        补充释放：仅当本 event 持有组锁且 agent 不会再跑到 update_memory 时调用。
        正常跑完仍以 update_memory 为主路径（等落盘后再调度）。
        """
        if not self.enabled or not self.group_queue or not self.resolver:
            return False
        if not self.enable_group_agent_queue:
            return False
        try:
            sid = getattr(event, "sid", None) or ""
            eid = str(getattr(event, "event_id", "") or "")
            if not sid:
                return False
            gid = self.resolver.resolve_group_id(sid)
            if not gid:
                return False
            return await self.group_queue.release_if_active(
                gid,
                sid=sid,
                event_id=eid,
                reason=reason,
                schedule_fn=self._schedule_group_drain,
            )
        except Exception:
            logger.exception("[MERGER queue] release_for_event failed reason=%s", reason)
            return False

    @on.im_batch_message(priority=Priority.LOW)
    async def on_im_batch_group_queue(self, event: KiraMessageBatchEvent):
        """
        合并组串行：组内已有 agent 时，本 batch 入队并 stop，
        等当前会话 update_memory 后再重放（Q1，接近官方单窗口）。

        使用 LOW：core 在 is_stopped 后不再跑后续 handler。
        若用 HIGH 占锁后被其它插件 stop，会既不进 agent 也不 update_memory，
        只能干等 TTL。LOW 保证「只有 batch 仍将继续时才占锁」。
        """
        if not self.enabled or not self.group_queue or not self.resolver:
            return
        if not self.enable_group_agent_queue:
            return
        if event.is_stopped:
            return
        try:
            sid = getattr(event, "sid", None) or ""
            if not sid:
                return
            # 跨会话 ROUTE 注入的 batch 也走同一组锁，保证一条线
            gid = self.resolver.resolve_group_id(sid)
            if not gid:
                return
            ok = await self.group_queue.try_begin(gid, sid, event)
            if not ok:
                event.stop()
                logger.info(
                    "[MERGER queue] deferred batch sid=%s group=%s event=%s",
                    sid,
                    gid,
                    getattr(event, "event_id", ""),
                )
        except Exception:
            logger.exception("[MERGER queue] on_im_batch failed")




    # ── helpers ──────────────────────────────────────────────

    def _extract_text(self, event: KiraMessageEvent) -> str:
        return "".join(
            elem.text for elem in event.message.chain if isinstance(elem, Text)
        ).strip()

    def _command_user_allowed(self, event: KiraMessageEvent) -> bool:
        if not self.command_allowed_users:
            return True
        try:
            uid = str(event.message.sender.user_id)
            return uid in self.command_allowed_users
        except Exception:
            return True

    def _get_event_user_id(self, event: KiraMessageEvent) -> str:
        try:
            return str(event.message.sender.user_id)
        except Exception:
            return "unknown"

    def _get_event_user_name(self, event: KiraMessageEvent) -> str:
        try:
            return event.message.sender.nickname or "未知"
        except Exception:
            return "未知"

    def _reboot_all_user_allowed(self, event: KiraMessageEvent) -> bool:
        """参考 reboot_plugin：未开权限 / 白名单为空 → 放行。"""
        if not self.reboot_all_enable_permission:
            return True
        if not self.reboot_all_allowed_users:
            return True
        return self._get_event_user_id(event) in self.reboot_all_allowed_users

    def _match_command(self, text: str, commands: List[str]) -> bool:
        if not text or not commands:
            return False
        t = text.strip().lower()
        for c in commands:
            if t == str(c).strip().lower():
                return True
        return False

    def _collect_all_merge_sids(self) -> List[str]:
        """收集所有参与合并的会话 sid（去重）。"""
        if not self.resolver:
            return []
        seen = set()
        ordered: List[str] = []
        try:
            for g in self.resolver.summarize() or []:
                for sid in g.get("members") or []:
                    s = str(sid).strip()
                    if s and s not in seen:
                        seen.add(s)
                        ordered.append(s)
        except Exception:
            logger.exception("[MERGER] collect merge sids failed")
        return ordered

    def _clear_session_context(self, sid: str) -> bool:
        """清除单个会话上下文（对齐 reboot_plugin：delete_session）。"""
        sm = getattr(self.ctx, "session_mgr", None)
        if sm is None or not sid:
            return False
        sm.delete_session(sid)
        return True

    async def _reply(self, sid: str, text: str):
        await self.ctx.message_processor.send_message_chain(
            session=sid,
            chain=MessageChain([Text(text)]),
        )

    def _message_content_for_observe(self, event: KiraMessageEvent) -> str:
        msg = event.message
        s = getattr(msg, "message_str", None)
        if s:
            return str(s).strip()
        # fallback chain text
        parts = []
        try:
            for elem in msg.chain:
                if isinstance(elem, Text) and elem.text:
                    parts.append(elem.text)
                elif hasattr(elem, "repr"):
                    parts.append(str(elem.repr))
        except Exception:
            pass
        return " ".join(parts).strip()

    @staticmethod
    def _batch_text_blob(event: KiraMessageBatchEvent) -> str:
        parts = []
        for msg in getattr(event, "messages", None) or []:
            try:
                parts.append(getattr(msg, "message_str", None) or "")
            except Exception:
                pass
        return "\n".join(parts)

    @classmethod
    def _is_merge_route_request(cls, event: KiraMessageBatchEvent) -> bool:
        """本插件路由的跨会话请求：目标会话应合并上文并执行。"""
        return is_merge_route_request_text(cls._batch_text_blob(event))

    @classmethod
    def _is_legacy_official_cross_delivery(cls, event: KiraMessageBatchEvent) -> bool:
        """官方旧版 notice 模板（弱上下文二次生成）。"""
        text = cls._batch_text_blob(event)
        if not text:
            return False
        return (
            "跨会话消息" in text
            and "不需要再次调用跨会话工具" in text
            and not is_merge_route_request_text(text)
        )


    # ── observe unmentioned (do NOT change chat strategy) ────

    @on.im_message(priority=Priority.MEDIUM)
    async def on_im_message_observe(self, event: KiraMessageEvent):
        if not self.enabled or not self.observe_pool or not self.resolver:
            return
        try:
            # 不改 buffer/flush/discard
            if event.is_mentioned:
                return
            # 不把跨会话路由 / 系统 notice 记入观察池
            try:
                if event.is_notice:
                    return
                mid = str(getattr(event.message, "message_id", "") or "")
                if mid in ("system_message", "system"):
                    return
            except Exception:
                pass
            sid = event.session.sid
            if not self.resolver.is_session_eligible(sid):
                return
            try:
                self_id = str(event.message.self_id or "")
                sender_id = str(event.message.sender.user_id) if event.message.sender else ""
                if self_id and sender_id and self_id == sender_id:
                    return
            except Exception:
                pass

            content = self._message_content_for_observe(event)
            if not content:
                return
            if is_merge_route_request_text(content):
                return
            if "跨会话消息" in content and "不需要再次调用跨会话工具" in content:
                return



            mid = str(getattr(event.message, "message_id", "") or "")
            ts = int(getattr(event.message, "timestamp", 0) or getattr(event, "timestamp", 0) or time.time())
            sender_id = ""
            sender_name = ""
            group_name = ""
            try:
                if event.message.sender:
                    sender_id = str(event.message.sender.user_id or "")
                    sender_name = str(event.message.sender.nickname or "")
                if event.message.group:
                    group_name = str(getattr(event.message.group, "group_name", "") or "")
            except Exception:
                pass

            self.observe_pool.add(
                sid=sid,
                content=content,
                message_id=mid,
                timestamp=ts,
                sender_id=sender_id,
                sender_name=sender_name,
                group_name=group_name,
            )
        except Exception:
            if self.enable_debug_log:
                logger.exception("[observe] record failed")

    # ── commands ─────────────────────────────────────────────

    @on.im_message(priority=Priority.HIGH)
    async def on_im_message_commands(self, event: KiraMessageEvent):
        if not self.enabled:
            return

        text = self._extract_text(event)
        if not text:
            return

        sid = event.session.sid

        if self.enable_status_command and self._match_command(text, self.status_commands):
            if not self._command_user_allowed(event):
                await self._reply(sid, "❌ 权限不足")
                event.discard(force=True)
                event.stop()
                return
            await self._handle_status(sid)
            event.discard(force=True)
            event.stop()
            return

        if self.enable_preview_command and self._match_command(text, self.preview_commands):
            if not self._command_user_allowed(event):
                await self._reply(sid, "❌ 权限不足")
                event.discard(force=True)
                event.stop()
                return
            await self._handle_preview(sid)
            event.discard(force=True)
            event.stop()
            return

        if self.enable_reboot_all_command and self._match_command(
            text, self.reboot_all_commands
        ):
            await self._handle_reboot_all(event, sid)
            event.discard(force=True)
            event.stop()
            return

    async def _handle_reboot_all(self, event: KiraMessageEvent, sid: str):
        """清除所有参与合并的会话上下文（参考 reboot_plugin）。"""
        user_id = self._get_event_user_id(event)
        user_name = self._get_event_user_name(event)

        if self.enable_debug_log:
            logger.info(
                "[MERGER] reboot_all cmd | user=%s(%s) | from=%s",
                user_name,
                user_id,
                sid,
            )

        if not self._reboot_all_user_allowed(event):
            if self.enable_debug_log:
                logger.info("[MERGER] reboot_all denied user=%s", user_id)
            await self._reply(sid, self.reboot_all_permission_denied_message)
            return

        try:
            if getattr(self.ctx, "session_mgr", None) is None:
                await self._reply(
                    sid,
                    self.reboot_all_error_message.format(error="会话管理器不可用"),
                )
                return

            targets = self._collect_all_merge_sids()
            if not targets:
                await self._reply(
                    sid,
                    self.reboot_all_success_message.format(count=0),
                )
                return

            ok_count = 0
            fail_count = 0
            for target_sid in targets:
                try:
                    if self._clear_session_context(target_sid):
                        ok_count += 1
                    else:
                        fail_count += 1
                except Exception as e:
                    fail_count += 1
                    logger.error(
                        "[MERGER] reboot_all delete failed sid=%s: %s",
                        target_sid,
                        e,
                    )

            # 软重启动态 keep / 跨会话去重一并清掉
            try:
                if self.engine and getattr(self.engine, "_soft_state", None):
                    self.engine._soft_state._dynamic_keep.clear()
                    self.engine._soft_state._last_reset.clear()
                    self.engine._soft_state._last_check.clear()
            except Exception:
                pass
            try:
                self._session_send_dedup.clear()
            except Exception:
                pass

            # 观察池一并清空（未点名缓存）
            try:
                if self.observe_pool and hasattr(self.observe_pool, "clear"):
                    self.observe_pool.clear()
                elif self.observe_pool and hasattr(self.observe_pool, "_data"):
                    self.observe_pool._data.clear()
            except Exception:
                pass
            try:
                if self.group_queue:
                    self.group_queue.clear_all()
            except Exception:
                pass

            msg = self.reboot_all_success_message.format(count=ok_count)
            if fail_count:
                msg = f"{msg}\n（失败 {fail_count} 个）"
            await self._reply(sid, msg)
            logger.warning(
                "[MERGER] reboot_all done by %s(%s): ok=%d fail=%d total=%d",
                user_name,
                user_id,
                ok_count,
                fail_count,
                len(targets),
            )
        except Exception as e:
            logger.error("[MERGER] reboot_all failed: %s", e)
            await self._reply(
                sid,
                self.reboot_all_error_message.format(error=str(e)),
            )

    async def _handle_status(self, sid: str):
        if not self.resolver or not self.engine:
            await self._reply(sid, "合并引擎未初始化")
            return
        gid = self.resolver.resolve_group_id(sid)
        if not gid:
            await self._reply(sid, f"当前会话 `{sid}` 未参与合并（enabled={self.enabled}）。")
            return
        members = self.engine._member_sids(sid)
        obs = self.observe_pool.stats() if self.observe_pool else {}
        lines = [
            "✅ 会话合并状态",
            f"- 当前会话: `{sid}`",
            f"- 合并组: `{gid}`",
            f"- 超限模式: {self.merge_reset_mode}",
            f"- 本次合并会话数: {len(members)} / max={self.max_merge_sessions}",
            f"- 最大合并轮数: {self.max_merged_chunks}",
            f"- token 上限: {self.merge_token_limit} / 保留轮数: {self.merge_keep_turns}",
            f"- 其他会话超时: {self.other_session_timeout}m (dm={self.other_session_timeout_private}, gm={self.other_session_timeout_group})",
            f"- 偷看概率: {self.unmentioned_probability}",
            f"- 观察池: sessions={obs.get('sessions', 0)} msgs={obs.get('messages', 0)}",
            "成员(超时过滤后):",
        ]

        for m in members[:40]:
            lines.append(f"  · {m}")
        if len(members) > 40:
            lines.append(f"  ... 共 {len(members)} 个")
        await self._reply(sid, "\n".join(lines))

    async def _handle_preview(self, sid: str):
        if not self.engine:
            await self._reply(sid, "合并引擎未初始化")
            return
        if not self.engine.should_merge(sid):
            await self._reply(sid, f"当前会话 `{sid}` 未参与合并。")
            return
        try:
            text = self.engine.preview(sid, limit=25)
            if len(text) > 3500:
                text = text[:3500] + "\n...(truncated)"
            await self._reply(sid, f"合并历史预览:\n{text}")
        except Exception as e:
            await self._reply(sid, f"预览失败: {e}")

    # ── core merge ───────────────────────────────────────────

    @on.llm_request(priority=Priority.LOW)
    async def on_llm_request(self, event: KiraMessageBatchEvent, req: LLMRequest, *_):
        if not self.enabled or not self.engine:
            return

        try:
            bound = self._bind_session_send_to_request(req)
            if not bound and self.enable_debug_log:
                logger.warning("[MERGER] session_send not bound to current tool_set")
        except Exception:
            logger.exception("[MERGER] bind session_send to request failed")

        from core.prompt_manager import Prompt


        # 本插件路由的跨会话请求：合并上文 + 在目标会话执行
        if self._is_merge_route_request(event):
            try:
                if req.tool_set is not None:
                    req.tool_set.remove("session_send")
                tip = (
                    "\n[Session Merger] 跨会话请求：你已切换到本会话。"
                    "请结合合并后的对话上文，在本会话继续执行上文的任务。"
                    "不要 session_send 回源会话；不要反复 get_session_history 确认。"
                )

                for p in req.system_prompt or []:
                    if getattr(p, "name", None) in ("chat_env", "attention", "tools", "output"):
                        p.content = (getattr(p, "content", None) or "") + tip
                        break
                else:
                    req.system_prompt.append(
                        Prompt(tip, name="session_merger_route", source="kira_session_merger")
                    )
                ok = await self.engine.apply_to_request(event, req)
                logger.info(
                    "[MERGER] cross-session ROUTE execute sid=%s merged=%s",
                    getattr(event, "sid", ""),
                    ok,
                )
            except Exception:
                logger.exception("[MERGER] route execute failed")
            return

        # 官方旧 notice：弱上下文，跳过合并并禁工具
        if self._is_legacy_official_cross_delivery(event):
            try:
                if req.tool_set is not None:
                    req.tool_set.remove("session_send", "get_session_history")
                try:
                    req.tool_choice = "none"
                except Exception:
                    pass
                tip = (
                    "\n[Session Merger] 旧版跨会话 notice：请仅根据本轮说明输出最终 xml；"
                    "禁止调用 session_send / get_session_history。"
                )
                for p in req.system_prompt or []:
                    if getattr(p, "name", None) in ("chat_env", "attention", "output"):
                        p.content = (getattr(p, "content", None) or "") + tip
                        break
                logger.info(
                    "[MERGER] legacy official cross notice sid=%s (no merge)",
                    getattr(event, "sid", ""),
                )
            except Exception:
                logger.exception("[MERGER] legacy cross guard failed")
            return

        try:
            tip2 = (
                "\n[Session Merger] session_send = 跨会话请求（合并模式）："
                "切换到目标会话，目标带着合并上文继续执行。\n"
                "### 何时应 session_send\n"
                "- 用户要求换会话，或任务须在另一会话完成\n"
                "### 行为\n"
                "- session_send(target, description=要在目标完成的事)；"
                "成功后勿在源会话替目标调业务工具；可简短确认也可不发；同一 target 一次即可\n"
                "### 不要 session_send\n"
                "- 目标就是当前会话 → 直接输出 xml / 在本会话调工具\n"
                "### get_session_history\n"
                "- 近期合并上下文已注入；需要更早平台真历史时再查；"
                "同一目标每回合最多 1 次；Rejected 后停止再查\n"
            )
            for p in req.system_prompt or []:
                if getattr(p, "name", None) == "tools":
                    if "跨会话请求（合并模式）" not in (p.content or ""):
                        # 覆盖旧 tip 片段，避免重复堆叠
                        content = p.content or ""
                        if "[Session Merger] session_send" in content:
                            idx = content.find("\n[Session Merger] session_send")
                            if idx >= 0:
                                content = content[:idx]
                        p.content = content + tip2
                    break
        except Exception:
            pass

        try:
            ok = await self.engine.apply_to_request(event, req)
            if self.enable_debug_log:
                logger.info(
                    "[MERGER] on_llm_request sid=%s merged=%s",
                    getattr(event, "sid", ""),
                    ok,
                )
        except Exception:
            logger.exception("[session_merger] apply_to_request failed (swallowed)")

        # 本阶段已被 stop、不会进 agent/update_memory 时补放组锁
        if event.is_stopped:
            await self._release_group_for_event(event, "llm_request_stopped_no_agent")

    # ── 源会话交棒：仅停止「当前这条」agent 回合 ─────────────
    # event.stop() 作用在 KiraMessageBatchEvent 实例上，不锁会话、
    # 不影响同会话其他用户稍后的新消息 / 新 batch。

    @on.tool_result(priority=Priority.HIGH)
    async def on_tool_result_handoff(self, event: KiraMessageBatchEvent, tool_result, *_):
        """
        session_send 路由成功/去重后：只标记交棒，不在此处 stop。

        原因：core execute_tool 在 ON_TOOL_RESULT 里若 is_stopped 会立刻 return，
        且发生在把 tool_result 写入 resp 之前，会导致 session_send 结果丢失、
        历史里 tool_calls 对不齐。交棒结束放到 step_result（xml 已发出之后）。
        """
        if not self.enabled:
            return
        try:
            if tool_result is None:
                return
            text = getattr(tool_result, "text", None) or getattr(
                tool_result, "result_str", None
            ) or ""
            if not text and not isinstance(tool_result, str):
                text = str(tool_result)
            elif isinstance(tool_result, str):
                text = tool_result
            if not is_route_handoff_result(text):
                return
            mark_event_handoff(event)
            # 尽早停私聊输入状态（仅调增强实例方法，不改其源码）
            self._stop_qq_typing_if_any(event)
            logger.info(
                "[MERGER] handoff marked after session_send sid=%s "
                "(will stop after this step; this event only)",
                getattr(event, "sid", ""),
            )
        except Exception:
            if self.enable_debug_log:
                logger.exception("[MERGER] handoff tool_result failed")

    @on.step_result(priority=Priority.HIGH)
    async def on_step_result_handoff(self, event: KiraMessageBatchEvent, *_):
        """
        交棒后结束本轮 agent：本 step 若已有 xml 会先发出；
        不强制补话。不改 core / 其它插件源码。

        作用范围：仅当前 KiraMessageBatchEvent。
        """
        if not self.enabled:
            return
        try:
            if not event_has_handoff(event):
                return

            # 兼容当前版 QQ 增强：handoff 时 agent 仍带 tool_calls 结束，
            # 增强插件只在「无 tool 的最终 llm_response」停输入状态。
            # 这里只调用其已有 _stop_typing_loop，不修改增强插件文件。
            self._stop_qq_typing_if_any(event)

            if not event.is_stopped:
                event.stop()
                logger.info(
                    "[MERGER] handoff stop on step_result sid=%s "
                    "(no further agent steps; other users/new messages unaffected)",
                    getattr(event, "sid", ""),
                )
        except Exception:
            if self.enable_debug_log:
                logger.exception("[MERGER] handoff step_result failed")

    def _stop_qq_typing_if_any(self, event: KiraMessageBatchEvent) -> None:
        """
        兼容当前版 QQ 增强私聊「正在输入」：
        仅通过 plugin_mgr 拿到实例并调用其已有 _stop_typing_loop。
        不修改 QQ 增强 / KiraAI 本体源码。
        """
        try:
            sess = getattr(event, "session", None)
            if sess is None:
                return
            # 群聊增强本身不发 typing
            if getattr(sess, "session_type", "") == "gm":
                return
            # 源会话若是群，本轮 handoff 也无需停私聊 typing；
            # 目标私聊由目标会话自己的 llm 流程管理。
            # 若源是私聊且本轮 handoff 结束，则停源会话 typing。
            pm = getattr(self.ctx, "plugin_mgr", None)
            if not pm:
                return
            candidate_ids = (
                "kira-ai-plugin-qq-enhance",
                "kira-ai-plugin-qq-enhance-main",
                "qq-enhance",
            )
            inst = None
            for pid in candidate_ids:
                try:
                    if hasattr(pm, "get_plugin_inst"):
                        inst = pm.get_plugin_inst(pid)
                    if inst is None and hasattr(self.ctx, "get_plugin_inst"):
                        inst = self.ctx.get_plugin_inst(pid)
                    if inst is not None:
                        break
                except Exception:
                    continue
            if inst is None:
                instances = getattr(pm, "plugin_instances", None) or {}
                if isinstance(instances, dict):
                    for pid, p in instances.items():
                        pl = str(pid).lower()
                        if "qq-enhance" in pl or "qq_enhance" in pl:
                            inst = p
                            break
            if inst is None:
                return
            stop_fn = getattr(inst, "_stop_typing_loop", None)
            if callable(stop_fn):
                stop_fn(sess)
                logger.info(
                    "[MERGER] stopped QQ typing for sid=%s (compat, no enhance source change)",
                    getattr(event, "sid", ""),
                )
        except Exception:
            if self.enable_debug_log:
                logger.exception("[MERGER] stop QQ typing failed")

    # ── history tool ─────────────────────────────────────────

    @register.tool(
        name="get_session_history",
        description=(
            "通过 OneBot HTTP 拉取群/私聊平台真历史（含更早记录、msg_id、图片 URL）。"
            "合并模式下近期上下文已注入：仅当需要更早/平台侧记录时再查；"
            "同一目标每回合最多成功查 1 次，返回 Rejected/Error 后禁止再调。"
            "session_id：qq:gm:群号 / qq:dm:QQ号。"
        ),
        params={
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "会话 ID：qq:gm:123456 或 qq:dm:789 或纯数字",
                },
                "count": {
                    "type": "integer",
                    "default": 20,
                    "description": "消息数量，建议 20-50，最少 5，最多 80",
                },
                "session_type": {
                    "type": "string",
                    "description": "可选：group/private/gm/dm；session_id 为纯数字时使用",
                },
            },
            "required": ["session_id"],
        },
    )
    async def get_session_history(
        self,
        event: KiraMessageBatchEvent,
        session_id: str,
        count: int = 20,
        session_type: str = "",
        **_,
    ) -> str:
        if not self.enable_history_tool:
            return "历史查询工具已禁用。"
        if not self.history_svc:
            return "历史服务未初始化。"
        return await self.history_svc.get_session_history(
            event,
            session_id=session_id,
            count=count,
            session_type=session_type or None,
            merge_enabled=bool(self.enabled),
        )

    @on.after_xml_parse(priority=Priority.HIGH)
    async def debug_after_xml_parse(self, event: KiraMessageBatchEvent, message_chains: list):
        if not self.enable_send_debug_log:
            return
        logger.debug("[SendDebug] parsed chains=%d", len(message_chains))

    @on.message_sent(priority=Priority.HIGH)
    async def debug_message_sent(
        self, event: KiraMessageBatchEvent, chain: MessageChain, result: KiraIMSentResult
    ):
        if not self.enable_send_debug_log:
            return
        if result.ok:
            logger.debug("[SendDebug] ok message_id=%s", result.message_id)
        else:
            logger.warning("[SendDebug] fail err=%s", result.err)

    # ── API ──────────────────────────────────────────────────

    @register.api(method="GET", path="/groups", auth=True)
    async def api_get_groups(self) -> dict:
        if not self.resolver:
            return {"error": "not initialized"}
        return {
            "enabled": self.enabled,
            "max_merge_sessions": self.max_merge_sessions,
            "groups": self.resolver.summarize(),
            "observe": self.observe_pool.stats() if self.observe_pool else {},
        }

    @register.api(method="GET", path="/preview", auth=True)
    async def api_preview(self, sid: str = "") -> dict:
        if not self.engine:
            return {"error": "not initialized"}
        if not sid:
            return {"error": "missing sid"}
        try:
            text = self.engine.preview(sid, limit=50)
            return {"ok": True, "sid": sid, "preview": text}
        except Exception as e:
            return {"error": str(e)}

