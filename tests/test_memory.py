"""Тесты памяти: seen_queries, archive, reset. Используют monkeypatch путей."""
import importlib

import pytest


@pytest.fixture
def mem(tmp_path, monkeypatch):
    """Перенаправляет все пути в tmp_path и перезагружает модуль memory."""
    from lra import config, memory, plan
    monkeypatch.setattr(config, "RESEARCH_DIR", tmp_path)
    monkeypatch.setattr(config, "ARCHIVE_DIR", tmp_path / "archive")
    monkeypatch.setattr(config, "DRAFT_PATH", tmp_path / "draft.md")
    monkeypatch.setattr(config, "NOTES_PATH", tmp_path / "notes.md")
    monkeypatch.setattr(config, "PLAN_PATH", tmp_path / "plan.md")
    monkeypatch.setattr(config, "SYNTHESIS_PATH", tmp_path / "synthesis.md")
    monkeypatch.setattr(config, "LESSONS_PATH", tmp_path / "lessons.md")
    monkeypatch.setattr(config, "QUERYLOG_PATH", tmp_path / "querylog.md")
    # plan.py биндит PLAN_PATH/PLAN_JSON_PATH/RESEARCH_DIR на import-time — патчим явно
    monkeypatch.setattr(plan, "PLAN_PATH", tmp_path / "plan.md")
    monkeypatch.setattr(plan, "PLAN_JSON_PATH", tmp_path / "plan.json")
    monkeypatch.setattr(plan, "RESEARCH_DIR", tmp_path)
    # memory импортировал пути ПО ЗНАЧЕНИЮ на момент загрузки — перезагружаем
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
    # Slug безопасный
    assert "MyTopic" in dest.name or "mytopic" in dest.name.lower()


def test_reset_research_preserves_lessons_and_querylog(mem):
    memory, tmp = mem
    # Первый прогон
    memory.reset_research("topic A")
    memory.log_query("query one")
    # Второй прогон — lessons и querylog должны остаться + получить маркер сессии
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
