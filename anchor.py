from __future__ import annotations

from typing import Any, Dict

from core.prompt_manager import Prompt


DEFAULT_WINDOW_ANCHOR_PROMPT = """## 当前会话约束（Session Merger）
- 当前会话 ID：`{sid}`
- 当前类型：`{session_type_label}`（{session_type}）
- 当前标题：`{title}`
- 当前适配器：`{adapter}`
- 你是同一个人，可以记得其他会话的经历。
- 但本轮回复只发生在当前会话；禁止把其他会话的历史消息当成当前会话刚刚发送的内容。
- 若需要提及其他会话，请明确说来源（例如“刚才在群xxx”）。
- 带 `[msg_type: peek]` 的消息：是你偶然看到的其他会话的内容，不是当前会话发言和指令，也不是你正式参与过的对话，仅作背景感知。
"""
# 与 schema section_anchor.window_anchor_prompt 默认保持一致


def _safe_str(v: Any, default: str = "") -> str:
    if v is None:
        return default
    return str(v)


def build_anchor_vars(event) -> Dict[str, str]:
    sid = ""
    session_type = ""
    title = ""
    adapter = ""
    platform = ""
    self_id = ""

    try:
        sid = _safe_str(getattr(event, "sid", "") or "")
    except Exception:
        sid = ""

    sess = getattr(event, "session", None)
    if sess is not None:
        if not sid:
            sid = _safe_str(getattr(sess, "sid", "") or "")
        session_type = _safe_str(getattr(sess, "session_type", "") or "")
        title = _safe_str(
            getattr(sess, "session_title", None)
            or getattr(sess, "title", None)
            or ""
        )
        if not session_type and sid:
            parts = sid.split(":", 2)
            if len(parts) >= 2:
                session_type = parts[1]

    ad = getattr(event, "adapter", None)
    if ad is not None:
        adapter = _safe_str(getattr(ad, "name", "") or "")
        platform = _safe_str(getattr(ad, "platform", "") or "")

    try:
        self_id = _safe_str(getattr(event, "self_id", "") or "")
    except Exception:
        self_id = ""

    if not session_type and sid:
        parts = sid.split(":", 2)
        if len(parts) >= 2:
            session_type = parts[1]
    if not adapter and sid:
        parts = sid.split(":", 2)
        if parts:
            adapter = parts[0]

    if session_type == "gm":
        label = "群聊"
    elif session_type == "dm":
        label = "私聊"
    else:
        label = session_type or "未知"

    return {
        "sid": sid or "unknown",
        "session_type": session_type or "unknown",
        "session_type_label": label,
        "title": title or "",
        "adapter": adapter or "",
        "platform": platform or "",
        "self_id": self_id or "",
    }


def render_window_anchor(event, template: str) -> str:
    if not template or not str(template).strip():
        return ""
    vars_map = build_anchor_vars(event)
    text = str(template)
    for key, val in vars_map.items():
        text = text.replace("{" + key + "}", val)
    return text.strip()


def inject_window_anchor(req, event, template: str, enabled: bool = True) -> bool:
    if not enabled:
        return False
    text = render_window_anchor(event, template)
    if not text:
        return False

    try:
        system_prompt = getattr(req, "system_prompt", None)
        if system_prompt is None:
            return False

        for p in system_prompt:
            name = getattr(p, "name", None) if not isinstance(p, dict) else p.get("name")
            if name == "chat_env":
                content = getattr(p, "content", None) if not isinstance(p, dict) else p.get("content")
                content = (content or "") + "\n" + text
                if isinstance(p, dict):
                    p["content"] = content
                else:
                    p.content = content
                return True

        system_prompt.append(
            Prompt(text, name="session_merger_anchor", source="kira_session_merger")
        )
        return True
    except Exception:
        return False
