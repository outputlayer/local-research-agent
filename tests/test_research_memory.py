"""Тесты typed research memory: сохранение, поиск релевантного контекста, run-summary."""

from lra import research_memory


def test_save_and_load_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(research_memory, "RESEARCH_MEMORY_DIR", tmp_path / "memory")

    path = research_memory.save_memory(
        memory_type="domain-heuristic",
        title="EW query reformulations",
        description="Use radar, jamming and ELINT synonyms before generic agentic terms",
        body="Prefer concrete spectrum and radar terms before broad AI terms.",
        topic="electronic warfare research",
        tags=["research", "ew", "queries"],
    )

    assert path.exists()
    entries = research_memory.load_memories()
    assert len(entries) == 1
    assert entries[0].memory_type == "domain-heuristic"
    assert "radar terms" in entries[0].body


def test_select_relevant_memories_prefers_topic_overlap(tmp_path, monkeypatch):
    monkeypatch.setattr(research_memory, "RESEARCH_MEMORY_DIR", tmp_path / "memory")

    research_memory.save_memory(
        memory_type="run-summary",
        title="EW run summary",
        description="Strong overlap on radar and jamming papers",
        body="Keep radar, EW and jamming terminology in the first explorer query.",
        topic="electronic warfare radar jamming",
        tags=["research", "ew"],
    )
    research_memory.save_memory(
        memory_type="run-summary",
        title="LLM agents summary",
        description="General agent orchestration notes",
        body="Useful for coding agents, not for radar.",
        topic="agent orchestration",
        tags=["agents"],
    )

    selected = research_memory.select_relevant_memories(
        "modern radar jamming methods in electronic warfare", k=1
    )

    assert len(selected) == 1
    assert selected[0].title == "EW run summary"


def test_record_run_memory_includes_lessons_tail(tmp_path, monkeypatch):
    monkeypatch.setattr(research_memory, "RESEARCH_MEMORY_DIR", tmp_path / "memory")

    path = research_memory.record_run_memory(
        query="electronic warfare",
        stopped_reason="LOW_GAIN",
        valid_ids=4,
        invalid_ids=["2501.12345"],
        suspicious_citations=["2502.54321"],
        lessons_tail="[iter 2] worked: radar terms; failed: broad agentic query",
    )

    text = path.read_text(encoding="utf-8")
    assert "valid_ids=4" in text
    assert "stop=LOW_GAIN" in text
    assert "worked: radar terms" in text
