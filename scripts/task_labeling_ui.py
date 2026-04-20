#!/usr/bin/env python3
"""
Task Labeling UI — Streamlit app for human experts to annotate all
possible tasks in a scene image.

Run:
    pip install streamlit
    streamlit run scripts/task_labeling_ui.py

Input:  data/scenes/  (place .png / .jpg images here)
Output: data/human_tasks.json
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
_SCENES_DIR = _ROOT / "data" / "scenes"
_TASKS_FILE = _ROOT / "data" / "human_tasks.json"


def _ensure_paths() -> None:
    _SCENES_DIR.mkdir(parents=True, exist_ok=True)
    if not _TASKS_FILE.exists():
        _TASKS_FILE.write_text("{}", encoding="utf-8")


def _load_tasks() -> dict:
    return json.loads(_TASKS_FILE.read_text(encoding="utf-8"))


def _save_tasks(data: dict) -> None:
    _TASKS_FILE.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8",
    )


def _list_images() -> list[str]:
    exts = {".png", ".jpg", ".jpeg"}
    return sorted(
        p.name for p in _SCENES_DIR.iterdir() if p.suffix.lower() in exts
    )


def main() -> None:
    _ensure_paths()

    st.set_page_config(page_title="Task Labeling", layout="wide")
    st.title("Scene Task Labeling")

    st.info(
        "**Your goal: list every physically feasible task a robot could "
        "perform in this scene.**\n\n"
        "Look carefully at every object, surface, and container. "
        "Think beyond the obvious — consider:\n"
        "- **Single-object actions** (pick up, push, flip, rotate)\n"
        "- **Object-object interactions** (stack, insert, pour into, "
        "place on top of)\n"
        "- **State changes** (open/close, fill/empty, cover/uncover)\n"
        "- **Multi-step compound tasks** (clear the table, sort by color, "
        "rearrange)\n"
        "- **Edge cases** (what if something is upside-down? behind another "
        "object?)\n\n"
        "The more exhaustive your annotations, the better we can evaluate "
        "our abstract graph coverage. **Aim for at least 10 tasks per scene** "
        "— there are usually more than you first think!",
        icon="\U0001f4cb",
    )

    images = _list_images()
    if not images:
        st.warning(
            f"No images found in `{_SCENES_DIR}`.  "
            "Place `.png` / `.jpg` files there and refresh."
        )
        return

    # ── Sidebar: image selector ──
    selected = st.sidebar.selectbox("Select scene image", images)

    # ── Load persisted tasks into session state ──
    if "all_tasks" not in st.session_state:
        st.session_state.all_tasks = _load_tasks()

    tasks: list[dict] = st.session_state.all_tasks.setdefault(selected, [])

    # ── Two-column layout ──
    col_img, col_anno = st.columns([1, 1], gap="large")

    with col_img:
        st.image(str(_SCENES_DIR / selected), use_container_width=True)

    with col_anno:
        st.subheader(f"Tasks for `{selected}`")
        st.caption(f"Total tasks annotated for this scene: **{len(tasks)}**")

        # ── Existing tasks ──
        to_delete: int | None = None
        for idx, t in enumerate(tasks):
            c1, c2 = st.columns([5, 1])
            with c1:
                st.markdown(f"**{idx + 1}. {t['task_name']}**")
                if t.get("steps"):
                    st.text(t["steps"])
            with c2:
                if st.button("Delete", key=f"del_{selected}_{idx}"):
                    to_delete = idx

        if to_delete is not None:
            tasks.pop(to_delete)
            _save_tasks(st.session_state.all_tasks)
            st.rerun()

        # ── Add-task form ──
        st.divider()
        with st.form("add_task", clear_on_submit=True):
            name = st.text_input("Task Name / Goal")
            steps = st.text_area("Detailed Steps (Optional)")
            submitted = st.form_submit_button("Add Task")

        if submitted and name.strip():
            tasks.append({
                "task_name": name.strip(),
                "steps": steps.strip(),
            })
            _save_tasks(st.session_state.all_tasks)
            st.rerun()


if __name__ == "__main__":
    main()
