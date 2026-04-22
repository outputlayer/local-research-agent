"""Позволяет запускать тесты без установки пакета + глобальная изоляция research/.

Мотивация: run.log показал тестовые данные (ComVo, dialog agents), утекавшие
в prod notes.md. Индивидуальные тесты монкипатчили RESEARCH_DIR, но не все.
Здесь — глобальный autouse: каждому тесту своя tmp research-папка.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def _isolate_research_dir(tmp_path, monkeypatch):
    """Каждому тесту — свежая tmp research/ папка. Патчит ВСЕ модули, захватившие
    пути через `from ... import RESEARCH_DIR` (setattr на config недостаточно —
    другие модули уже держат свою копию).

    Также сбрасывает loop-tracker между тестами (ин-мемори state).
    """
    from lra import config, kb, memory, metrics, pipeline, research_memory, tools, validator
    from lra import plan as plan_mod

    research = tmp_path / "research"
    research.mkdir(parents=True, exist_ok=True)
    archive = research / "archive"
    archive.mkdir(parents=True, exist_ok=True)

    paths = {
        "RESEARCH_DIR": research,
        "ARCHIVE_DIR": archive,
        "DRAFT_PATH": research / "draft.md",
        "NOTES_PATH": research / "notes.md",
        "PLAN_PATH": research / "plan.md",
        "PLAN_JSON_PATH": research / "plan.json",
        "LESSONS_PATH": research / "lessons.md",
        "QUERYLOG_PATH": research / "querylog.md",
        "REJECTED_PATH": research / "rejected.jsonl",
        "SYNTHESIS_PATH": research / "synthesis.md",
        "KB_PATH": research / "kb.jsonl",
        "RESEARCH_MEMORY_DIR": research / "memory",
    }
    for mod in (config, kb, memory, metrics, pipeline, research_memory, tools, validator, plan_mod):
        for attr, new in paths.items():
            if hasattr(mod, attr):
                monkeypatch.setattr(mod, attr, new, raising=False)

    try:
        from lra import tool_tracker
        tool_tracker.reset_tracker()
    except Exception:
        pass

    yield research
