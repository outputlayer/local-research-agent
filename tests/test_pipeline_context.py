"""Статусный контекст пайплайна: покрытие плана и rejected evidence summary."""

import json


def test_build_status_context_includes_plan_coverage(tmp_path, monkeypatch):
    from lra import config, pipeline
    from lra import plan as plan_mod

    monkeypatch.setattr(config, "RESEARCH_DIR", tmp_path)
    monkeypatch.setattr(config, "PLAN_PATH", tmp_path / "plan.md")
    monkeypatch.setattr(config, "REJECTED_PATH", tmp_path / "rejected.jsonl")
    monkeypatch.setattr(plan_mod, "RESEARCH_DIR", tmp_path)
    monkeypatch.setattr(plan_mod, "PLAN_PATH", tmp_path / "plan.md")
    monkeypatch.setattr(plan_mod, "PLAN_JSON_PATH", tmp_path / "plan.json")
    monkeypatch.setattr(pipeline, "PLAN_PATH", tmp_path / "plan.md")
    monkeypatch.setattr(pipeline, "REJECTED_PATH", tmp_path / "rejected.jsonl")

    plan = plan_mod.reset("test topic")
    current = plan.focus_task()
    current.evidence_refs.append("kb:2401.00001")
    plan.close_task("T2", iter_=1, evidence=["kb:2402.00002"], why="done")
    plan.block_task("T3", iter_=1, why="stuck")
    plan_mod.save(plan)

    ctx = pipeline._build_status_context("test topic", current.title)

    assert "Research status:" in ctx
    assert "plan_progress:" in ctx
    assert "done=1/5" in ctx
    assert "blocked=1" in ctx
    assert f"[{current.id}] {current.title}" in ctx
    assert "undercovered_tasks:" in ctx


def test_build_status_context_includes_rejected_summary(tmp_path, monkeypatch):
    from lra import config, pipeline

    monkeypatch.setattr(config, "REJECTED_PATH", tmp_path / "rejected.jsonl")
    monkeypatch.setattr(pipeline, "REJECTED_PATH", tmp_path / "rejected.jsonl")

    rows = [
        {"reason": "no_core_hit"},
        {"reason": "weak_overlap"},
        {"reason": "no_core_hit"},
    ]
    (tmp_path / "rejected.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )

    ctx = pipeline._build_status_context("x")

    assert "rejected_evidence: 3" in ctx
    assert "no_core_hit=2" in ctx
    assert "weak_overlap=1" in ctx

