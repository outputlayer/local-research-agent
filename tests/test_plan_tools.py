"""Tests for plan_add_task / plan_close_task / plan_split_task —
the only legitimate way for the model to mutate the plan."""
from __future__ import annotations

import pytest


@pytest.fixture
def plan_tools_env(tmp_path, monkeypatch):
    from lra import config, plan, tools
    monkeypatch.setattr(config, "RESEARCH_DIR", tmp_path)
    monkeypatch.setattr(config, "PLAN_PATH", tmp_path / "plan.md")
    monkeypatch.setattr(plan, "RESEARCH_DIR", tmp_path)
    monkeypatch.setattr(plan, "PLAN_PATH", tmp_path / "plan.md")
    monkeypatch.setattr(plan, "PLAN_JSON_PATH", tmp_path / "plan.json")
    # tools.plan_mod is the same object; patches apply automatically via attr-access
    return plan, tools, tmp_path


def test_plan_add_task_without_plan_returns_error(plan_tools_env):
    _, tools, _ = plan_tools_env
    out = tools.PlanAddTask().call({"title": "no plan"})
    assert "error" in out.lower()


def test_plan_add_task_adds_emerged_child(plan_tools_env):
    plan, tools, _ = plan_tools_env
    p = plan.reset("root topic")
    parent_id = p.tasks[0].id
    out = tools.PlanAddTask().call({
        "title": "emerged subtopic",
        "parent": parent_id,
        "why": "seen in the abstract",
    })
    assert "added" in out
    reloaded = plan.load()
    assert reloaded is not None
    child = next((t for t in reloaded.tasks if t.title == "emerged subtopic"), None)
    assert child is not None
    assert child.parent == parent_id
    assert child.origin == "emerged"


def test_plan_close_task_marks_done_with_evidence(plan_tools_env):
    plan, tools, _ = plan_tools_env
    p = plan.reset("x")
    tid = p.tasks[1].id
    out = tools.PlanCloseTask().call({
        "id": tid,
        "evidence": "2401.00001, kb:Paper_Alpha",
        "why": "3 facts in notes",
    })
    assert "closed" in out
    reloaded = plan.load()
    closed = reloaded.get(tid)
    assert closed.status == "done"
    assert "2401.00001" in closed.evidence_refs
    assert "kb:Paper_Alpha" in closed.evidence_refs


def test_plan_split_task_creates_children(plan_tools_env):
    plan, tools, _ = plan_tools_env
    p = plan.reset("x")
    tid = p.tasks[2].id
    out = tools.PlanSplitTask().call({
        "id": tid,
        "subtitles": "sub A | sub B | sub C",
        "why": "too broad",
    })
    assert "split" in out
    reloaded = plan.load()
    assert reloaded.get(tid).status == "dropped"
    children = [t for t in reloaded.tasks if t.parent == tid]
    assert len(children) == 3


def test_plan_split_task_rejects_single_subtitle(plan_tools_env):
    plan, tools, _ = plan_tools_env
    plan.reset("x")
    out = tools.PlanSplitTask().call({
        "id": "T1",
        "subtitles": "only one subtask",
    })
    assert "error" in out.lower()


def test_plan_add_task_respects_max_open_limit(plan_tools_env):
    plan, tools, _ = plan_tools_env
    p = plan.reset("x")
    # fill up to the limit
    while len(p.open_tasks()) < plan.MAX_OPEN_TASKS:
        p.add_task("filler", origin="emerged")
    plan.save(p)
    out = tools.PlanAddTask().call({"title": "overflow attempt"})
    assert "error" in out.lower()
    assert "MAX_OPEN_TASKS" in out


def test_plan_close_task_unknown_id_returns_error(plan_tools_env):
    plan, tools, _ = plan_tools_env
    plan.reset("x")
    out = tools.PlanCloseTask().call({"id": "T99"})
    assert "error" in out.lower()
