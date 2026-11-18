"""Clean chat-style Streamlit UI for Musical Story Video."""

from __future__ import annotations

import html
import importlib
import os
import sys
import time
from pathlib import Path

import streamlit as st

from pipeline.config import load_env_file
from pipeline.progress import ProgressEvent
from pipeline.sadtalker import sadtalker_ready
from pipeline.tasks import list_tasks


def _load_run_pipeline():
    """Import run_pipeline, refreshing stale Streamlit module cache safely."""
    # Reload leaves in dependency order so `from pipeline.config import ContentMode` works.
    for name in (
        "pipeline.config",
        "pipeline.progress",
        "pipeline.face_detect",
        "pipeline.diagrams",
        "pipeline.story",
        "pipeline.tts",
        "pipeline.images",
        "pipeline.music",
        "pipeline.assemble",
        "pipeline.sadtalker",
        "pipeline.tasks",
        "pipeline.run",
    ):
        mod = sys.modules.get(name)
        if mod is not None:
            importlib.reload(mod)
    import pipeline.run as run_mod

    return run_mod.run_pipeline


run_pipeline = _load_run_pipeline()

st.set_page_config(
    page_title="Musical Story Video",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_env_file()

st.markdown(
    """
<style>
[data-testid="stSidebar"] {
  background: #0f1115;
  border-right: 1px solid #2a2f3a;
}
.block-container {
  padding-top: 1rem;
  padding-bottom: 0.6rem;
  max-width: 1080px;
}

.sidebar-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  margin-bottom: 10px;
}
.sidebar-head h3 {
  margin: 0;
  font-size: 1.05rem;
  color: #e8eaed;
}

.tasks-scroll {
  max-height: 72vh;
  overflow-y: auto;
  padding-right: 2px;
}

div[data-testid="stSidebar"] button[kind="secondary"] {
  text-align: left;
  justify-content: flex-start;
  border: 1px solid #2a2f3a;
  background: #161a22;
  border-radius: 10px;
  padding-top: 0.55rem;
  padding-bottom: 0.55rem;
  margin-bottom: 0.35rem;
}
div[data-testid="stSidebar"] button[kind="secondary"]:hover {
  border-color: #4c8bf5;
  background: #1a2332;
}

.stage-grid {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 8px;
  margin: 8px 0 12px 0;
}
.stage-box {
  border: 1px solid #2a2f3a;
  border-radius: 12px;
  padding: 10px;
  background: #12151c;
  min-height: 74px;
}
.stage-box.done { border-color: #3d8b5c; background: #122018; }
.stage-box.active { border-color: #4c8bf5; background: #141c2b; }
.stage-box.pending { opacity: 0.5; }
.stage-box .name {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: #9aa0a6;
  margin-bottom: 6px;
}
.stage-box .detail { font-size: 12px; color: #e8eaed; line-height: 1.35; }
.stage-box .count { margin-top: 6px; font-size: 11px; color: #8ab4f8; }

.bubble-user, .bubble-bot {
  border-radius: 16px;
  padding: 12px 14px;
  margin: 0 0 12px 0;
  line-height: 1.45;
  font-size: 14.5px;
}
.bubble-user {
  background: #1b2330;
  border: 1px solid #2d3648;
  margin-left: 14%;
}
.bubble-bot {
  background: #14171e;
  border: 1px solid #2a2f3a;
  margin-right: 8%;
}
.bubble-role {
  font-size: 11px;
  color: #9aa0a6;
  margin-bottom: 6px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

.terminal {
  max-height: 160px;
  overflow-y: auto;
  background: #0b0d10;
  color: #b9f6c8;
  border: 1px solid #243028;
  border-radius: 12px;
  padding: 10px 12px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 12px;
  line-height: 1.4;
  white-space: pre-wrap;
}
.terminal-title {
  font-size: 12px;
  color: #9aa0a6;
  margin: 0 0 6px 0;
}

.settings-inline {
  border: 1px solid #2a2f3a;
  background: #151922;
  border-radius: 14px;
  padding: 10px 12px 4px 12px;
  margin-bottom: 8px;
}
</style>
""",
    unsafe_allow_html=True,
)

STAGES = ["story", "voice", "images", "music", "assemble"]

defaults = {
    "logs": [],
    "progress_fraction": 0.0,
    "progress_label": "Idle",
    "selected_task_id": None,
    "result": None,
    "current_task": None,
    "chat_messages": [],
    "stage_status": {
        s: {"state": "pending", "detail": "Waiting", "done": None, "total": None} for s in STAGES
    },
    "running": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v if not isinstance(v, dict) else dict(v)


def _reset_stages() -> None:
    st.session_state.stage_status = {
        s: {"state": "pending", "detail": "Waiting", "done": None, "total": None} for s in STAGES
    }


def _stage_boxes_html() -> str:
    parts = ['<div class="stage-grid">']
    for s in STAGES:
        info = st.session_state.stage_status.get(s, {})
        state = info.get("state", "pending")
        detail = html.escape(str(info.get("detail") or ""))
        count = ""
        if info.get("done") is not None and info.get("total"):
            rem = max(0, int(info["total"]) - int(info["done"]))
            count = (
                f'<div class="count">{int(info["done"])}/{int(info["total"])} done'
                f" · {rem} left</div>"
            )
        parts.append(
            f'<div class="stage-box {state}">'
            f'<div class="name">{s}</div>'
            f'<div class="detail">{detail}</div>'
            f"{count}</div>"
        )
    parts.append("</div>")
    return "".join(parts)


def _append_log(line: str) -> None:
    st.session_state.logs.append(line)
    st.session_state.logs = st.session_state.logs[-250:]


def _open_task(task: dict) -> None:
    """Show a task in the main pane without duplicating chat spam."""
    rid = task.get("run_id")
    st.session_state.selected_task_id = rid
    st.session_state.result = task
    # Keep chat clean: only replace/set a single "loaded task" result view
    st.session_state.chat_messages = [
        {
            "role": "assistant",
            "kind": "result",
            "content": (
                f"{task.get('title') or 'Untitled'} · "
                f"{task.get('scene_count', '?')} scenes · "
                f"{task.get('animation_mode', '')} · "
                f"{task.get('language', '')}"
            ),
            "meta": task,
        }
    ]


api_present = bool((os.environ.get("OPENAI_API_KEY") or "").strip())
tasks = list_tasks()

# ===================== LEFT: tasks only =====================
with st.sidebar:
    head_l, head_r = st.columns([1.4, 1])
    with head_l:
        st.markdown("### Tasks")
    with head_r:
        if st.button("Refresh", use_container_width=True, key="refresh_tasks_top"):
            st.rerun()

    if st.button("New chat", use_container_width=True, key="new_chat_btn"):
        st.session_state.chat_messages = []
        st.session_state.result = None
        st.session_state.logs = []
        st.session_state.current_task = None
        st.session_state.selected_task_id = None
        st.session_state.progress_fraction = 0.0
        st.session_state.progress_label = "Idle"
        _reset_stages()
        st.rerun()

    st.markdown('<div class="tasks-scroll">', unsafe_allow_html=True)

    if st.session_state.current_task:
        ct = st.session_state.current_task
        st.button(
            f"● In progress\n{ct.get('title', 'Working...')}",
            use_container_width=True,
            disabled=True,
            key="task_in_progress",
        )

    if not tasks and not st.session_state.current_task:
        st.caption("No tasks yet")

    for t in tasks:
        rid = t.get("run_id", "")
        title = (t.get("title") or "Untitled")[:48]
        meta = f"{t.get('created_at') or rid} · {t.get('animation_mode') or ''} · {t.get('language') or ''}"
        label = f"{'▸ ' if rid == st.session_state.selected_task_id else ''}{title}\n{meta}"
        if st.button(label, key=f"task_{rid}", use_container_width=True):
            _open_task(t)
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# ===================== MAIN =====================
st.markdown("## Musical Story Video")

# Chat messages
if st.session_state.chat_messages:
    for i, msg in enumerate(st.session_state.chat_messages):
        if msg.get("role") == "user":
            st.markdown(
                f'<div class="bubble-user"><div class="bubble-role">You</div>'
                f'{html.escape(msg.get("content", ""))}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="bubble-bot"><div class="bubble-role">Studio</div>'
                f'{html.escape(msg.get("content", ""))}</div>',
                unsafe_allow_html=True,
            )
            meta = msg.get("meta")
            if msg.get("kind") == "result" and meta:
                vp = Path(meta.get("final_video") or "")
                if vp.is_file():
                    st.video(str(vp))
                    st.download_button(
                        "Download MP4",
                        data=vp.read_bytes(),
                        file_name=f"{meta.get('run_id', 'story')}_story.mp4",
                        mime="video/mp4",
                        key=f"dl_{meta.get('run_id')}_{i}",
                    )

# Stage progress (Gemini-style boxes)
if st.session_state.running or any(
    st.session_state.stage_status[s]["state"] != "pending" for s in STAGES
):
    st.markdown(_stage_boxes_html(), unsafe_allow_html=True)
    st.progress(
        float(st.session_state.progress_fraction),
        text=st.session_state.progress_label,
    )

# Terminal
st.markdown('<p class="terminal-title">Terminal</p>', unsafe_allow_html=True)
term_body = (
    "\n".join(st.session_state.logs[-200:])
    if st.session_state.logs
    else "$ ready"
)
st.markdown(f'<div class="terminal">{html.escape(term_body)}</div>', unsafe_allow_html=True)

# ===================== Composer: settings beside chat bar =====================
with st.expander("Settings", expanded=False):
    s1, s2, s3, s4 = st.columns([1.25, 1.25, 1.0, 1.1])
    with s1:
        language_label = st.radio(
            "Language",
            ["Hindi (Indian · female)", "English (British · female)"],
            horizontal=True,
            key="ui_language",
        )
        content_label = st.radio(
            "Content",
            ["Kids story", "System design explainer"],
            horizontal=True,
            key="ui_content",
        )
        if content_label.startswith("System"):
            st.caption(
                "System design = ByteByteGo-style Mermaid explainers "
                "(flowchart + sequence diagrams, overlays, live highlight). "
                "No GPT/SD images, no characters."
            )
        else:
            st.caption(
                "Kids animated rhyme = OpenAI native motion GIFs (play-once) → "
                "merge + female VO + background music. Same size for every GIF & final video."
            )
    with s2:
        if content_label.startswith("System"):
            style_label = st.radio(
                "Video style",
                ["Static / diagram pan", "Dynamic (live build)"],
                horizontal=True,
                key="ui_style_sd",
            )
            st.caption(
                "Dynamic = Mermaid live-build (nodes appear as you explain). "
                "Supports large multi-layer architectures."
            )
        else:
            style_label = st.radio(
                "Cartoon video type",
                ["Animated GIFs (OpenAI · recommended)", "Legacy stills / face motion"],
                horizontal=True,
                key="ui_style_story",
            )
            st.caption(
                "Animated GIFs = ChatGPT-style character motion, each beat plays once. "
                "Includes rhyme voice-over + BGM."
            )
        size_label = st.radio(
            "Video size",
            ["YouTube / fullscreen (16:9)", "Reel / Shorts (9:16)"],
            horizontal=True,
            key="ui_video_size",
        )
        st.caption(
            "1280×720 landscape" if size_label.startswith("YouTube") else "720×1280 vertical — all GIFs match."
        )
    with s3:
        duration_sec = st.select_slider(
            "Duration",
            options=[1, 5, 10, 15, 30, 45, 60, 90, 120, 150, 180, 240, 300],
            value=60,
            format_func=lambda s: (
                f"{s}s" if s < 60 else (f"{s // 60}m" if s % 60 == 0 else f"{s // 60}m {s % 60}s")
            ),
            key="ui_duration_sec",
        )
        st.caption("From 1 second to 5 minutes.")
    with s4:
        title_hint = st.text_input("Title (optional)", key="ui_title")
        prefer_musicgen = st.checkbox(
            "MusicGen BGM",
            value=True,
            key="ui_musicgen",
            help="Kids rhyme videos always mix background music under the voice-over.",
        )
        st.caption(
            ("API key OK · " if api_present else "API key missing · ")
            + (
                "kids uses OpenAI animated GIFs + VO + music"
                if not content_label.startswith("System")
                else "system design = Mermaid only"
            )
        )

language = "hindi" if language_label.startswith("Hindi") else "english"
content_mode = "system_design" if content_label.startswith("System") else "story"
video_format = "reel" if size_label.startswith("Reel") else "youtube"
if content_mode == "story":
    # Default: OpenAI native animated GIFs (POC path) + VO + music.
    use_openai_gifs = not style_label.startswith("Legacy")
    animation_mode = "dynamic" if style_label.startswith("Legacy") else "static"
else:
    use_openai_gifs = False
    animation_mode = "dynamic" if style_label.startswith("Dynamic") else "static"
title_hint = (title_hint or "").strip()

prompt = st.chat_input(
    "Kids rhyme story, or system design (e.g. Low-latency API for 5M users)…",
    disabled=not api_present or st.session_state.running,
)
if prompt and api_present and not st.session_state.running:
    user_text = prompt.strip()
    if user_text:
        st.session_state.chat_messages.append({"role": "user", "content": user_text})
        st.session_state.chat_messages.append(
            {
                "role": "assistant",
                "content": (
                    "Generating Mermaid explainer…"
                    if content_mode == "system_design"
                    else (
                        f"Creating OpenAI motion GIFs ({video_format}) + rhyme VO + music…"
                    )
                ),
                "kind": "status",
            }
        )
        st.session_state.logs = []
        st.session_state.result = None
        st.session_state.progress_fraction = 0.0
        st.session_state.progress_label = "Starting…"
        st.session_state.running = True
        st.session_state.selected_task_id = None
        _reset_stages()
        st.session_state.current_task = {
            "title": (title_hint or user_text)[:48],
            "animation_mode": animation_mode,
            "language": language,
            "content_mode": content_mode,
            "duration_sec": duration_sec,
            "video_format": video_format,
        }

        progress_bar_slot = st.empty()
        stage_slot = st.empty()
        term_slot = st.empty()

        def _render_live() -> None:
            stage_slot.markdown(_stage_boxes_html(), unsafe_allow_html=True)
            progress_bar_slot.progress(
                float(st.session_state.progress_fraction),
                text=st.session_state.progress_label,
            )
            body_live = "\n".join(st.session_state.logs[-200:])
            term_slot.markdown(
                f'<p class="terminal-title">Terminal · live</p>'
                f'<div class="terminal">{html.escape(body_live)}</div>',
                unsafe_allow_html=True,
            )

        def _on_progress(event: ProgressEvent) -> None:
            _append_log(f"[{event.stage}] {event.message}")
            frac = float(event.fraction if event.fraction is not None else 0.0)
            st.session_state.progress_fraction = max(0.0, min(1.0, frac))

            stage = event.stage if event.stage in STAGES else None
            if event.stage == "done":
                for s in STAGES:
                    st.session_state.stage_status[s]["state"] = "done"
                    if st.session_state.stage_status[s]["detail"] == "Waiting":
                        st.session_state.stage_status[s]["detail"] = "Complete"
                st.session_state.progress_label = "DONE — 100%"
            elif stage:
                idx = STAGES.index(stage)
                for s in STAGES[:idx]:
                    if st.session_state.stage_status[s]["state"] != "done":
                        st.session_state.stage_status[s]["state"] = "done"
                        st.session_state.stage_status[s]["detail"] = "Complete"
                st.session_state.stage_status[stage]["state"] = "active"
                st.session_state.stage_status[stage]["detail"] = event.message[:80]
                if event.done is not None and event.total:
                    st.session_state.stage_status[stage]["done"] = event.done
                    st.session_state.stage_status[stage]["total"] = event.total
                    rem = max(0, event.total - event.done)
                    st.session_state.progress_label = (
                        f"{stage.upper()} — {event.done}/{event.total} done, {rem} left"
                    )
                    if event.done >= event.total:
                        st.session_state.stage_status[stage]["state"] = "done"
                        st.session_state.stage_status[stage]["detail"] = "Complete"
                else:
                    st.session_state.progress_label = (
                        f"{stage.upper()} — {event.message[:60]}"
                    )
            _render_live()

        _render_live()
        try:
            _run = _load_run_pipeline()
            meta = _run(
                prompt=user_text,
                language=language,
                duration_sec=int(duration_sec),
                title_hint=title_hint or None,
                prefer_musicgen=prefer_musicgen,
                animation_mode=animation_mode,
                content_mode=content_mode,
                silent=False,
                use_openai_images=use_openai_gifs if content_mode == "story" else False,
                video_format=video_format,
                progress=_on_progress,
            )
            st.session_state.result = meta
            st.session_state.selected_task_id = meta["run_id"]
            st.session_state.current_task = None
            st.session_state.progress_fraction = 1.0
            st.session_state.progress_label = "DONE — 100%"
            for s in STAGES:
                st.session_state.stage_status[s]["state"] = "done"
                st.session_state.stage_status[s]["detail"] = "Complete"
            _append_log(f"[done] {meta.get('final_video')}")
            # Drop the temporary status bubble, keep user + final result
            st.session_state.chat_messages = [
                m
                for m in st.session_state.chat_messages
                if m.get("kind") != "status"
            ]
            st.session_state.chat_messages.append(
                {
                    "role": "assistant",
                    "kind": "result",
                    "content": (
                        f"Ready: {meta.get('title')} · {meta.get('scene_count')} scenes · "
                        f"{meta.get('video_format') or meta.get('animation_mode')} · "
                        f"{meta.get('language')}"
                        + (
                            f" · {meta.get('frame_w')}x{meta.get('frame_h')}"
                            if meta.get("frame_w")
                            else ""
                        )
                    ),
                    "meta": meta,
                }
            )
        except Exception as exc:
            st.session_state.current_task = None
            _append_log(f"[error] {exc}")
            st.session_state.chat_messages.append(
                {
                    "role": "assistant",
                    "kind": "error",
                    "content": f"Pipeline failed: {exc}",
                }
            )
        finally:
            st.session_state.running = False
            time.sleep(0.05)
            st.rerun()
