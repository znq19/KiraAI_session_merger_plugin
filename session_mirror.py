"""
from __future__ import annotations

双世代会话访问层（session merger 专用）。

世代形态（特征识别，不绑版本号）：
- 2.x：SessionManager 持有常驻 ``chat_memory`` dict，同步读写；
- 3.0 rc.1+：会话 SQLite 持久化，接口全异步（``get_sessions``/``get_session``/
  ``update_session``/``delete_session``），无常驻缓存。

设计：
- **读**：插件大量同步内部函数（合并引擎/时间线/组解析）保持同步签名不变，
  2.x 直读 live dict；3.0 读 ``SessionMirror`` —— 由异步入口（消息钩子、命令、
  工具、API）先 ``refresh_mirror`` 预热（TTL 1.5s 内不重复拉取）。
- **写**：异步区（merge_engine 的摘要写回等）直接 ``sm_write_chunks``；
  同步区/线程区（``asyncio.to_thread`` 里的 ``_apply_sync``）在 3.0 不能 await，
  写操作进**延迟队列**（defer list），由异步调用方在 to_thread 返回后冲刷
  （``flush_deferred``）；超时僵尸线程的迟到写入由 done-callback 兜底冲刷。
"""

import asyncio
import inspect
import time
from typing import Any, Dict, List, Optional

_MIRROR_ATTR = "_ksm_session_mirror"
_MIRROR_TTL = 1.5  # 秒；入口级刷新的节流窗口


class SessionMirror:
    """3.0 的 chat_memory 等价镜像：{sid: {title, description, timestamp, memory}}。"""

    def __init__(self) -> None:
        self.data: Dict[str, Dict[str, Any]] = {}
        self.ts: float = 0.0

    async def refresh(self, sm, force: bool = False) -> None:
        now = time.monotonic()
        if not force and self.ts and (now - self.ts) < _MIRROR_TTL:
            return
        sessions = await sm.get_sessions()  # 带 content 的快照
        data: Dict[str, Dict[str, Any]] = {}
        for s in sessions or []:
            sid = getattr(s, "sid", None)
            if not sid:
                continue
            ts = getattr(s, "updated_at", 0) or 0
            try:
                ts = int(ts)
            except Exception:
                ts = 0
            data[str(sid)] = {
                "title": str(getattr(s, "session_title", "") or ""),
                "description": str(getattr(s, "session_description", "") or ""),
                "timestamp": ts,
                "memory": [list(c) for c in (getattr(s, "content", None) or [])],
            }
        self.data = data
        self.ts = now


def is_legacy_sync_sm(sm) -> bool:
    """2.x 形态判定：有常驻 chat_memory dict。"""
    return hasattr(sm, "chat_memory")


def get_mirror(sm) -> SessionMirror:
    mirror = getattr(sm, _MIRROR_ATTR, None)
    if mirror is None:
        mirror = SessionMirror()
        setattr(sm, _MIRROR_ATTR, mirror)
    return mirror


def peek_mirror(sm) -> Dict[str, Dict[str, Any]]:
    """同步读镜像（未预热时返回空——异步入口负责 refresh）。"""
    return get_mirror(sm).data


async def refresh_mirror(sm, force: bool = False) -> None:
    """异步入口调用：2.x 直通（live dict 天然新鲜）；3.0 预热镜像。"""
    if sm is None or is_legacy_sync_sm(sm):
        return
    await get_mirror(sm).refresh(sm, force=force)


async def refresh_mirror_sid(sm, sid: str) -> bool:
    """只刷新单个会话的镜像条目（3.0）。

    请求级对账发现不一致时用：只拉当前会话（一条 SQL），避免 get_sessions()
    把所有会话内容都读一遍。返回是否执行了更新。
    """
    if sm is None or is_legacy_sync_sm(sm) or not sid:
        return False
    s = await sm.get_session(sid, ensure_session=False)
    data = get_mirror(sm).data
    if s is None:
        data.pop(str(sid), None)
        return True
    try:
        ts = int(getattr(s, "updated_at", 0) or 0)
    except Exception:
        ts = 0
    data[str(sid)] = {
        "title": str(getattr(s, "session_title", "") or ""),
        "description": str(getattr(s, "session_description", "") or ""),
        "timestamp": ts,
        "memory": [list(c) for c in (getattr(s, "content", None) or [])],
    }
    return True


def invalidate_mirror(sm, sid: Optional[str] = None) -> None:
    """写入/外部变更后调用：sid 级失效或整体打旧（下次入口强制刷新）。"""
    if sm is None or is_legacy_sync_sm(sm):
        return
    mirror = get_mirror(sm)
    if sid is None:
        mirror.ts = 0.0
    else:
        mirror.data.pop(str(sid), None)


# ── 异步区双世代读写助手 ────────────────────────────────────────────────────

async def sm_read_chunks(sm, sid: str) -> List[List[dict]]:
    if hasattr(sm, "read_memory"):
        return [list(c) for c in (sm.read_memory(sid) or [])]
    s = await sm.get_session(sid, ensure_session=False)
    return [list(c) for c in (s.content if s else [])]


async def sm_read_flat(sm, sid: str) -> List[dict]:
    if hasattr(sm, "fetch_memory"):
        return list(sm.fetch_memory(sid) or [])
    s = await sm.get_session(sid, ensure_session=False)
    return (s.flatten_content() if s else []) or []


async def wait_framework_write_lock(sm) -> None:
    """等框架当前正在进行的会话写入结束。

    3.0 的 update_session 是整段替换；KSM 的读-改-写不在框架 _write_lock 内，
    可能读到 update_memory 提交前的旧快照。写前先等锁释放，把窗口缩到最小。
    """
    lock = getattr(sm, "_write_lock", None)
    if lock is not None:
        try:
            async with lock:
                pass
        except Exception:
            pass


async def sm_write_chunks(sm, sid: str, chunks, *, wait: bool = True) -> None:
    """整体覆写会话内容（保留标题/描述/能力设置等元数据）。

    2.x 保持原有的「同步全量 JSON 落盘挪到线程」语义；3.0 走原生异步 upsert。
    写完同步失效镜像。
    """
    if wait:
        await wait_framework_write_lock(sm)
    if hasattr(sm, "write_memory"):
        await asyncio.to_thread(sm.write_memory, sid, chunks)
        return
    await sm.update_session(sid, content=chunks)
    invalidate_mirror(sm, sid)


async def sm_clear(sm, sid: str) -> None:
    """清空会话上下文（两代都保元数据：标题/能力设置不丢）。

    2.x：write_memory(sid, [])；3.0：update_session(sid, content=[])。
    """
    await wait_framework_write_lock(sm)
    if hasattr(sm, "write_memory"):
        sm.write_memory(sid, [])
        return
    await sm.update_session(sid, content=[])
    invalidate_mirror(sm, sid)


# ── 同步/线程区的延迟写（3.0 专用；2.x 仍直写）───────────────────────────────

def defer_or_write(sm, sid: str, chunks, defer: Optional[list], logger=None) -> None:
    """整体覆写。2.x 直写（保持线程区原语义）；3.0 必须入 defer 队列等异步冲刷。"""
    if hasattr(sm, "write_memory"):
        sm.write_memory(sid, chunks)
        return
    if defer is None:
        # 没有队列的 3.0 同步区 —— 不该发生（调用方都要传 defer），宁可告警也不静默丢写
        if logger:
            logger.warning("[MERGER] 新版异步接口下同步区写入未带 defer 队列，本笔写入被跳过 sid=%s", sid)
        return
    defer.append(("write", sid, chunks))


def defer_or_recreate(sm, sid: str, chunks, defer: Optional[list], logger=None) -> None:
    """delete+重建。2.x 直做；3.0 入队（删除→upsert 重建）。"""
    if hasattr(sm, "write_memory"):
        sm.delete_session(sid)
        sm.get_session_info(sid)  # recreate empty shell
        sm.write_memory(sid, chunks)
        return
    if defer is None:
        if logger:
            logger.warning("[MERGER] 新版异步接口下同步区删除重建未带 defer 队列，跳过 sid=%s", sid)
        return
    defer.append(("recreate", sid, chunks))


async def flush_deferred(sm, defer: Optional[list], logger=None) -> int:
    """冲刷延迟写队列（异步区调用）。返回冲刷笔数。"""
    if not defer:
        return 0
    n = 0
    for op, sid, chunks in list(defer):
        try:
            if op == "write":
                await sm_write_chunks(sm, sid, chunks)
            elif op == "recreate":
                res = sm.delete_session(sid)
                if inspect.isawaitable(res):
                    await res
                await sm.update_session(sid, content=chunks)
                invalidate_mirror(sm, sid)
            n += 1
        except Exception as e:  # noqa: BLE001
            if logger:
                logger.exception("[MERGER] 延迟写冲刷失败 sid=%s: %s", sid, e)
    defer.clear()
    return n
