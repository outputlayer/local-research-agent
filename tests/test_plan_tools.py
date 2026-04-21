"""Тесты plan_add_task / plan_close_task / plan_split_task —
единственный легитимный способ модели мутировать план."""
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
    # tools.plan_mod — тот же объект, патчи применяются автоматически через attr-access
    return plan, tools, tmp_path


def test_plan_add_task_without_plan_returns_error(plan_tools_env):
    _, tools, _ = plan_tools_env
    out = tools.PlanAddTask().call({"title": "без плана"})
    assert "ошибка" in out.lower()


def test_plan_add_task_adds_emerged_child(plan_tools_env):
    plan, tools, _ = plan_tools_env
    p = plan.reset("root topic")
    parent_id = p.tasks[0].id
    out = tools.PlanAddTask().call({
        "title": "всплывшая подтема",
        "parent": parent_id,
        "why": "встретили в абстракте",
    })
    assert "добавлено" in out
    reloaded = plan.load()
    assert reloaded is not None
    child = next((t for t in reloaded.tasks if t.title == "всплывшая подтема"), None)
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
        "why": "3 факта в notes",
    })
    assert "закрыто" in out
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
        "subtitles": "под A | под B | под C",
        "why": "слишком широко",
    })
    assert "разбито" in out
    reloaded = plan.load()
    assert reloaded.get(tid).status == "dropped"
    children = [t for t in reloaded.tasks if t.parent == tid]
    assert len(children) == 3


def test_plan_split_task_rejects_single_subtitle(plan_tools_env):
    plan, tools, _ = plan_tools_env
    plan.reset("x")
    out = tools.PlanSplitTask().call({
        "id": "T1",
        "subtitles": "только одна подзадача",
    })
    assert "ошибка" in out.lower()


def test_plan_add_task_respects_max_open_limit(plan_tools_env):
    plan, tools, _ = plan_tools_env
    p = plan.reset("x")
    # заполняем до лимита
    while len(p.open_tasks()) < plan.MAX_OPEN_TASKS:
        p.add_task("filler", origin="emerged")
    plan.save(p)
    out = tools.PlanAddTask().call({"title": "overflow attempt"})
    assert "ошибка" in out.lower()
    assert "MAX_OPEN_TASKS" in out


def test_plan_close_task_unknown_id_returns_error(plan_tools_env):
    plan, tools, _ = plan_tools_env
    plan.reset("x")
    out = tools.PlanCloseTask().call({"id": "T99"})
    assert "ошибка" in out.lower()
