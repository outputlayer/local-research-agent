"""Тесты программной ротации [FOCUS] — страховка от зацикливания replanner'а."""
from __future__ import annotations

from pathlib import Path


def _setup(tmp_path, monkeypatch, plan_text: str):
    from lra import config, memory, pipeline
    monkeypatch.setattr(config, "PLAN_PATH", tmp_path / "plan.md")
    monkeypatch.setattr(memory, "PLAN_PATH", tmp_path / "plan.md")
    monkeypatch.setattr(pipeline, "PLAN_PATH", tmp_path / "plan.md")
    (tmp_path / "plan.md").write_text(plan_text, encoding="utf-8")
    return pipeline


def test_rotate_focus_moves_first_todo_into_focus(tmp_path, monkeypatch):
    plan = (
        "# Plan: тема\n\n"
        "[FOCUS] старый фокус\n\n"
        "## [TODO]\n"
        "- новый приоритет\n"
        "- вторичная подтема\n\n"
        "## [DONE]\n"
    )
    pipeline = _setup(tmp_path, monkeypatch, plan)
    assert pipeline._rotate_focus_fallback("тема") is True
    updated = Path(tmp_path / "plan.md").read_text(encoding="utf-8")
    assert "[FOCUS] новый приоритет" in updated
    # старый фокус должен уехать в DONE
    assert "- старый фокус" in updated
    # выбранный TODO должен быть удалён
    focus_section = updated.split("## [TODO]", 1)[1].split("## [DONE]", 1)[0]
    assert "новый приоритет" not in focus_section


def test_rotate_focus_no_todo_returns_false(tmp_path, monkeypatch):
    plan = (
        "# Plan: тема\n\n"
        "[FOCUS] единственная цель\n\n"
        "## [TODO]\n\n"
        "## [DONE]\n"
    )
    pipeline = _setup(tmp_path, monkeypatch, plan)
    assert pipeline._rotate_focus_fallback("тема") is False


def test_rotate_focus_no_plan_returns_false(tmp_path, monkeypatch):
    from lra import config, memory, pipeline
    monkeypatch.setattr(config, "PLAN_PATH", tmp_path / "nope.md")
    monkeypatch.setattr(memory, "PLAN_PATH", tmp_path / "nope.md")
    monkeypatch.setattr(pipeline, "PLAN_PATH", tmp_path / "nope.md")
    assert pipeline._rotate_focus_fallback("x") is False
