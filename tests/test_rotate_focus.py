"""Tests for programmatic focus rotation via plan.json —
safety net against replanner looping."""
from __future__ import annotations


def _setup(tmp_path, monkeypatch):
    """Patches all modules that bind PLAN_*/RESEARCH_DIR at import time."""
    from lra import config, memory, pipeline, plan
    monkeypatch.setattr(config, "PLAN_PATH", tmp_path / "plan.md")
    monkeypatch.setattr(config, "RESEARCH_DIR", tmp_path)
    monkeypatch.setattr(memory, "PLAN_PATH", tmp_path / "plan.md")
    monkeypatch.setattr(pipeline, "PLAN_PATH", tmp_path / "plan.md")
    monkeypatch.setattr(plan, "PLAN_PATH", tmp_path / "plan.md")
    monkeypatch.setattr(plan, "PLAN_JSON_PATH", tmp_path / "plan.json")
    monkeypatch.setattr(plan, "RESEARCH_DIR", tmp_path)
    return pipeline, plan


def test_rotate_focus_uses_next_open_task(tmp_path, monkeypatch):
    pipeline, plan = _setup(tmp_path, monkeypatch)
    p = plan.reset("research topic")
    original_focus_id = p.current_focus_id
    assert original_focus_id is not None

    assert pipeline._rotate_focus_fallback("topic") is True
    reloaded = plan.load()
    assert reloaded is not None
    # the old focus must be blocked (replanner failed)
    old = reloaded.get(original_focus_id)
    assert old is not None and old.status == "blocked"
    # new focus — the first remaining open task
    assert reloaded.current_focus_id is not None
    assert reloaded.current_focus_id != original_focus_id
    new_focus = reloaded.get(reloaded.current_focus_id)
    assert new_focus is not None and new_focus.status == "in_progress"


def test_rotate_focus_returns_false_when_no_open(tmp_path, monkeypatch):
    pipeline, plan = _setup(tmp_path, monkeypatch)
    p = plan.reset("topic")
    for t in list(p.tasks):
        if t.status in ("open", "in_progress"):
            p.close_task(t.id, why="test cleanup")
    plan.save(p)
    assert pipeline._rotate_focus_fallback("topic") is False


def test_rotate_focus_no_plan_returns_false(tmp_path, monkeypatch):
    pipeline, _ = _setup(tmp_path, monkeypatch)
    # plan.json does not exist
    assert pipeline._rotate_focus_fallback("x") is False
