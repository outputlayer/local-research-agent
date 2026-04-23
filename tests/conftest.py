"""Позволяет запускать тесты без установки пакета + глобальная изоляция research/.

Мотивация: run.log показал тестовые данные (ComVo, dialog agents), утекавшие
в prod notes.md. Индивидуальные тесты монкипатчили RESEARCH_DIR, но не все.
Здесь — глобальный autouse: каждому тесту своя tmp research-папка.
"""
import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Изоляция логгера на уровне модуля ────────────────────────────────────
# Критично: делается ДО первого импорта любого lra.* модуля (кроме этого файла).
# lra.logger.get_logger() при первом вызове цепляет FileHandler на реальный
# research/run.log и ставит _CONFIGURED=True — после этого перенастроить
# нельзя. Если этого не сделать, test tool-calls (append_notes, github_search
# с тестовыми payload'ами вроде "[2401.99999] hallucination") будут писаться
# в prod run.log при каждом `pytest`, создавая иллюзию что "тесты запускаются
# в каждом прогоне агента".
#
# Решение: заранее помечаем lra-logger как сконфигурированный с пустым
# набором хендлеров. get_logger() увидит _CONFIGURED=True и не будет
# цеплять FileHandler. Тестам логи не нужны — они проверяют return-values,
# не stderr/log output.
def _silence_lra_logger() -> None:
    from lra import logger as _logger_mod
    root = logging.getLogger("lra")
    for h in list(root.handlers):
        root.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass
    root.addHandler(logging.NullHandler())
    root.propagate = False
    _logger_mod._CONFIGURED = True


_silence_lra_logger()


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
        "RUN_LOG_PATH": research / "run.log",
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
