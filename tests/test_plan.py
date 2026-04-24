"""Tests for lra/plan.py: the Plan/Task/Revision model, persistence, mutations, guard."""
from __future__ import annotations

import pytest


@pytest.fixture
def plan_env(tmp_path, monkeypatch):
    from lra import config, plan
    monkeypatch.setattr(config, "RESEARCH_DIR", tmp_path)
    monkeypatch.setattr(config, "PLAN_PATH", tmp_path / "plan.md")
    monkeypatch.setattr(plan, "RESEARCH_DIR", tmp_path)
    monkeypatch.setattr(plan, "PLAN_PATH", tmp_path / "plan.md")
    monkeypatch.setattr(plan, "PLAN_JSON_PATH", tmp_path / "plan.json")
    return plan, tmp_path


# ── reset / persistence ────────────────────────────────────────────────
def test_reset_creates_seed_tasks_and_sets_focus(plan_env):
    plan, tmp = plan_env
    p = plan.reset("some topic")
    assert p.root_goal == "some topic"
    assert len(p.tasks) == 5
    assert all(t.status == "open" or t.status == "in_progress" for t in p.tasks)
    assert p.current_focus_id is not None
    assert p.focus_task().status == "in_progress"
    # both files written
    assert (tmp / "plan.json").exists()
    assert (tmp / "plan.md").exists()
    md = (tmp / "plan.md").read_text(encoding="utf-8")
    assert "[FOCUS]" in md
    assert "## [TODO]" in md


def test_load_returns_none_when_absent(plan_env):
    plan, _ = plan_env
    assert plan.load() is None


def test_save_and_load_roundtrip(plan_env):
    plan, _ = plan_env
    p = plan.reset("x")
    p.add_task("new subtopic", origin="emerged", why="test")
    plan.save(p)
    reloaded = plan.load()
    assert reloaded is not None
    assert reloaded.root_goal == "x"
    assert len(reloaded.tasks) == 6
    # revisions are also preserved
    assert any(r.action == "add" for r in reloaded.revisions)


# ── mutations ──────────────────────────────────────────────────────────
def test_add_task_respects_max_open_limit(plan_env):
    plan, _ = plan_env
    p = plan.reset("x")
    # add up to the limit
    while len(p.open_tasks()) < plan.MAX_OPEN_TASKS:
        p.add_task("filler", origin="emerged")
    with pytest.raises(ValueError, match="MAX_OPEN_TASKS"):
        p.add_task("overflow", origin="emerged")


def test_add_child_generates_dotted_id(plan_env):
    plan, _ = plan_env
    p = plan.reset("x")
    parent = p.tasks[0]
    child = p.add_task("child", parent=parent.id, origin="emerged")
    assert child.id.startswith(parent.id + ".")
    assert child.parent == parent.id


def test_close_task_sets_done_and_clears_focus(plan_env):
    plan, _ = plan_env
    p = plan.reset("x")
    focus_id = p.current_focus_id
    p.close_task(focus_id, evidence=["kb:paper1"], why="done")
    assert p.get(focus_id).status == "done"
    assert "kb:paper1" in p.get(focus_id).evidence_refs
    assert p.current_focus_id is None


def test_split_task_creates_children_and_drops_parent(plan_env):
    plan, _ = plan_env
    p = plan.reset("x")
    tid = p.tasks[1].id  # do not touch focus
    children = p.split_task(tid, ["sub A", "sub B", "sub C"], why="too broad")
    assert len(children) == 3
    assert p.get(tid).status == "dropped"
    assert all(c.parent == tid for c in children)


def test_increment_attempts_and_block(plan_env):
    plan, _ = plan_env
    p = plan.reset("x")
    tid = p.current_focus_id
    for _ in range(plan.MAX_ATTEMPTS_PER_TASK):
        p.increment_attempts(tid, iter_=1, why="empty")
    p.block_task(tid, iter_=1, why="exhausted")
    assert p.get(tid).status == "blocked"
    assert p.get(tid).attempts == plan.MAX_ATTEMPTS_PER_TASK


# ── render_md ──────────────────────────────────────────────────────────
def test_render_md_shows_plan_complete_when_no_open(plan_env):
    plan, tmp = plan_env
    p = plan.reset("x")
    for t in list(p.tasks):
        if t.status in ("open", "in_progress"):
            p.close_task(t.id, why="test")
    plan.save(p)
    md = (tmp / "plan.md").read_text(encoding="utf-8")
    assert "PLAN_COMPLETE" in md
    assert "## [DONE]" in md


# ── guard ──────────────────────────────────────────────────────────────
def test_guard_increments_attempts_on_empty_iteration(plan_env):
    plan, _ = plan_env
    p = plan.reset("x")
    focus_id = p.current_focus_id
    rep = plan.guard(p, iter_=1, notes_grew=0, new_ids=0)
    reloaded = plan.load()
    assert reloaded.get(focus_id).attempts == 1
    assert not rep.halt


def test_guard_blocks_after_max_attempts(plan_env):
    plan, _ = plan_env
    p = plan.reset("x")
    focus_id = p.current_focus_id
    for i in range(plan.MAX_ATTEMPTS_PER_TASK):
        rep = plan.guard(p, iter_=i + 1, notes_grew=0, new_ids=0)
        p = plan.load()
    # task must be blocked, focus — rotated to the next open
    assert p.get(focus_id).status == "blocked"
    assert p.current_focus_id is not None and p.current_focus_id != focus_id
    assert rep.rotated_focus or len(rep.blocked_ids) > 0


def test_guard_halts_when_all_blocked_or_done(plan_env):
    plan, _ = plan_env
    p = plan.reset("x")
    for t in list(p.tasks):
        if t.status in ("open", "in_progress"):
            p.drop_task(t.id, why="test")
    plan.save(p)
    rep = plan.guard(p, iter_=1, notes_grew=0, new_ids=0)
    assert rep.halt
    assert rep.halt_reason == "ALL_DONE_OR_BLOCKED"


def test_guard_global_empty_streak_halts(plan_env):
    """4+ iterations in a row with no notes/ids growth → global halt
    even with open tasks present (protection from infinite auto-rotation with no results).
    """
    plan, _ = plan_env
    p = plan.reset("x")
    plan.save(p)
    # streak=3 — NOT halt yet (open tasks exist)
    rep = plan.guard(p, iter_=1, notes_grew=500, new_ids=2, empty_iter_streak=3)
    assert not rep.halt
    # streak=4 — halt triggers
    rep = plan.guard(p, iter_=2, notes_grew=500, new_ids=2, empty_iter_streak=4)
    assert rep.halt
    assert "GLOBAL_EMPTY_STREAK" in rep.halt_reason
    assert "(4)" in rep.halt_reason


def test_guard_auto_rotates_focus_after_block(plan_env):
    plan, _ = plan_env
    p = plan.reset("x")
    # force-block the current focus
    p.block_task(p.current_focus_id, iter_=1, why="manual")
    p.current_focus_id = None
    plan.save(p)
    rep = plan.guard(p, iter_=2, notes_grew=500, new_ids=2)
    assert rep.rotated_focus
    reloaded = plan.load()
    assert reloaded.current_focus_id is not None


# ── sync_focus_from_md (backward compatibility with write_plan) ──────────
def test_sync_focus_from_md_creates_corrective_task(plan_env):
    plan, _ = plan_env
    p = plan.reset("x")
    md = "# Plan\n\n[FOCUS] radically new direction\n\n## [TODO]\n"
    assert plan.sync_focus_from_md(p, md, iter_=1) is True
    assert p.focus_title() == "radically new direction"
    assert any(t.origin == "corrective" for t in p.tasks)


def test_sync_focus_from_md_plan_complete_closes_all_open(plan_env):
    plan, _ = plan_env
    p = plan.reset("x")
    md = "[FOCUS] PLAN_COMPLETE\n## [TODO]\n"
    assert plan.sync_focus_from_md(p, md, iter_=1) is True
    # all tasks must be done; no corrective plan named "PLAN_COMPLETE" must exist
    assert all(t.status == "done" for t in p.tasks)
    assert not any(t.title == "PLAN_COMPLETE" for t in p.tasks)


def test_sync_focus_from_md_no_change_when_focus_matches(plan_env):
    plan, _ = plan_env
    p = plan.reset("x")
    current = p.focus_title()
    md = f"[FOCUS] {current}\n"
    assert plan.sync_focus_from_md(p, md, iter_=1) is False
