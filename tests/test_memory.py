"""Memory tests: seen_queries, archive, reset. Use path monkeypatch."""
import importlib

import pytest


@pytest.fixture
def mem(tmp_path, monkeypatch):
    """Redirects all paths to tmp_path and reloads the memory module."""
    from lra import config, memory, plan, research_memory
    monkeypatch.setattr(config, "RESEARCH_DIR", tmp_path)
    monkeypatch.setattr(config, "ARCHIVE_DIR", tmp_path / "archive")
    monkeypatch.setattr(config, "DRAFT_PATH", tmp_path / "draft.md")
    monkeypatch.setattr(config, "NOTES_PATH", tmp_path / "notes.md")
    monkeypatch.setattr(config, "PLAN_PATH", tmp_path / "plan.md")
    monkeypatch.setattr(config, "SYNTHESIS_PATH", tmp_path / "synthesis.md")
    monkeypatch.setattr(config, "LESSONS_PATH", tmp_path / "lessons.md")
    monkeypatch.setattr(config, "QUERYLOG_PATH", tmp_path / "querylog.md")
    monkeypatch.setattr(config, "RESEARCH_MEMORY_DIR", tmp_path / "memory")
    # plan.py binds PLAN_PATH/PLAN_JSON_PATH/RESEARCH_DIR at import time — patch explicitly
    monkeypatch.setattr(plan, "PLAN_PATH", tmp_path / "plan.md")
    monkeypatch.setattr(plan, "PLAN_JSON_PATH", tmp_path / "plan.json")
    monkeypatch.setattr(plan, "RESEARCH_DIR", tmp_path)
    monkeypatch.setattr(research_memory, "RESEARCH_MEMORY_DIR", tmp_path / "memory")
    # memory imported paths BY VALUE at load time — reload
    importlib.reload(memory)
    return memory, tmp_path


def test_seen_queries_empty(mem):
    memory, _ = mem
    assert memory.seen_queries() == set()


def test_log_and_seen(mem):
    memory, tmp = mem
    memory.log_query("Transformer scaling laws")
    memory.log_query("Mixture of experts routing")
    seen = memory.seen_queries()
    assert "transformer scaling laws" in seen
    assert "mixture of experts routing" in seen


def test_seen_is_normalized(mem):
    memory, _ = mem
    memory.log_query("Neural   Networks")
    assert "neural networks" in memory.seen_queries()


def test_seen_ignores_headers(mem):
    memory, tmp = mem
    (tmp / "querylog.md").write_text(
        "# Query log\n## Session 2024-01-01: topic\n- real query\n"
    )
    seen = memory.seen_queries()
    assert "real query" in seen
    assert all(not q.startswith("#") for q in seen)


def test_archive_previous_noop_when_empty(mem):
    memory, _ = mem
    assert memory.archive_previous("test") is None


def test_archive_previous_copies_files(mem):
    memory, tmp = mem
    (tmp / "draft.md").write_text("# draft")
    (tmp / "notes.md").write_text("# notes")
    dest = memory.archive_previous("MyTopic 2024")
    assert dest is not None
    assert (dest / "draft.md").read_text() == "# draft"
    assert (dest / "notes.md").read_text() == "# notes"
    # Safe slug
    assert "MyTopic" in dest.name or "mytopic" in dest.name.lower()


def test_reset_research_preserves_lessons_and_querylog(mem):
    memory, tmp = mem
    # First run
    memory.reset_research("topic A")
    memory.log_query("query one")
    # Second run — lessons and querylog must remain + receive a session marker
    memory.reset_research("topic B")
    ql = (tmp / "querylog.md").read_text()
    assert "query one" in ql
    assert "Session" in ql
    assert "topic A" in ql and "topic B" in ql


def test_reset_research_wipes_workspace_files(mem):
    memory, tmp = mem
    (tmp / "draft.md").write_text("stale draft")
    (tmp / "synthesis.md").write_text("stale synth")
    memory.reset_research("new topic")
    assert not (tmp / "draft.md").exists()
    assert not (tmp / "synthesis.md").exists()
    assert (tmp / "notes.md").exists()
    assert (tmp / "plan.md").exists()
    assert "new topic" in (tmp / "plan.md").read_text()
