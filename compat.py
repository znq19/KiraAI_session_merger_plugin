from __future__ import annotations

HISTORY_PLUGIN_ID = "history_plugin"
ADS_PLUGIN_ID = "auto_delete_session"


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

