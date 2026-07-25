from __future__ import annotations

from typing import Optional

HISTORY_PLUGIN_ID = "history_plugin"
ADS_PLUGIN_ID = "auto_delete_session"

# ContextCondensation（CCS）候选 id（目录名可能带 -main 等后缀）
CCS_CANDIDATE_IDS = (
    "KiraAI-ContextCondensation",
    "KiraAI-ContextCondensation-main",
    "context_condensation",
    "context-condensation",
    "ContextCondensation",
)


def find_ccs_plugin_id(plugin_mgr) -> Optional[str]:
    """候选 id 精确匹配 + 模糊扫描。"""
    if not plugin_mgr:
        return None
    try:
        for pid in CCS_CANDIDATE_IDS:
            if plugin_mgr.has_plugin(pid):
                return pid
        for attr in ("plugin_instances", "plugins", "_plugins"):
            d = getattr(plugin_mgr, attr, None)
            if isinstance(d, dict):
                for key in d:
                    norm = str(key).lower().replace("-", "").replace("_", "")
                    if "contextcondensation" in norm:
                        return str(key)
    except Exception:
        pass
    return None


async def handle_ccs_conflict(plugin_mgr, policy: str, logger) -> str:
    """
    与 CCS 的互斥处理。返回 "none" | "ccs_disabled" | "self_disabled"。

    为什么 soft 模式也要互斥：KSM 在 Priority.LOW 改写 req.messages 为合并视图，
    CCS 的钩子在 LOW-1（更后）会把跨会话混合内容镜像进自己的缓存，
    周期触发时再通过 write_memory 写回本 sid 磁盘 → 跨会话污染落盘。
    """
    policy = (policy or "disable_ccs").strip().lower()
    pid = find_ccs_plugin_id(plugin_mgr)
    if not pid:
        return "none"
    try:
        if not plugin_mgr.is_plugin_enabled(pid):
            return "none"
    except Exception:
        return "none"

    if policy == "ignore":
        if logger:
            logger.warning(
                "[session_merger] 检测到 %s 已启用（ccs_conflict_policy=ignore）："
                "CCS 会把合并视图写入会话磁盘，可能跨会话污染，建议只保留一个",
                pid,
            )
        return "none"

    if policy == "disable_self":
        if logger:
            logger.warning(
                "[session_merger] 检测到 %s 已启用，按策略 disable_self：合并功能本次不生效",
                pid,
            )
        return "self_disabled"

    try:
        await plugin_mgr.set_plugin_enabled(pid, False)
        if logger:
            logger.warning(
                "[session_merger] 已自动禁用 %s（ccs_conflict_policy=disable_ccs）："
                "KSM 2.0 已吸收其累积压缩设计，请不要再同时启用",
                pid,
            )
        return "ccs_disabled"
    except Exception as e:
        if logger:
            logger.warning("[session_merger] 禁用 %s 失败: %s；两插件并行可能冲突", pid, e)
        return "none"



async def log_compat_status(plugin_mgr, logger) -> None:
    """启动时简要打印相关插件开关状态。"""
    if not plugin_mgr or not logger:
        return
    try:
        ads = plugin_mgr.has_plugin(ADS_PLUGIN_ID) and plugin_mgr.is_plugin_enabled(ADS_PLUGIN_ID)
        hist = plugin_mgr.has_plugin(HISTORY_PLUGIN_ID) and plugin_mgr.is_plugin_enabled(HISTORY_PLUGIN_ID)
        logger.info(
            "[session_merger] compat: ADS=%s history_plugin=%s",
            "on" if ads else "off",
            "on" if hist else "off",
        )
    except Exception as e:
        logger.warning("[session_merger] compat log failed: %s", e)



async def maybe_disable_history_plugin(plugin_mgr, enabled: bool, logger=None) -> None:
    if not enabled or not plugin_mgr:
        return
    try:
        if plugin_mgr.has_plugin(HISTORY_PLUGIN_ID) and plugin_mgr.is_plugin_enabled(HISTORY_PLUGIN_ID):
            await plugin_mgr.set_plugin_enabled(HISTORY_PLUGIN_ID, False)
            if logger:
                logger.info(
                    "[session_merger] disabled history_plugin (tool overlap with get_session_history)"
                )
    except Exception as e:
        if logger:
            logger.warning("[session_merger] failed to disable history_plugin: %s", e)


async def maybe_disable_ads(plugin_mgr, enabled: bool, logger=None) -> None:
    """仅当用户显式开启 auto_disable_ads 时调用。默认不应调用。"""
    if not enabled or not plugin_mgr:
        return
    try:
        if plugin_mgr.has_plugin(ADS_PLUGIN_ID) and plugin_mgr.is_plugin_enabled(ADS_PLUGIN_ID):
            await plugin_mgr.set_plugin_enabled(ADS_PLUGIN_ID, False)
            if logger:
                logger.warning(
                    "[session_merger] auto_disable_ads=true → disabled auto_delete_session "
                    "(not recommended; prefer coexist)"
                )
    except Exception as e:
        if logger:
            logger.warning("[session_merger] failed to disable ADS: %s", e)

