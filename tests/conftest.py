"""Allows running the tests without installing the package + global isolation of research/.

Motivation: run.log showed test data (ComVo, dialog agents) leaking
into prod notes.md. Individual tests monkey-patched RESEARCH_DIR, but not all of them.
This is a global autouse: each test gets its own tmp research folder.
"""
import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Module-level logger isolation ────────────────────────────────────
# Critical: done BEFORE the first import of any lra.* module (except this file).
# lra.logger.get_logger() on first call attaches a FileHandler to the real
# research/run.log and sets _CONFIGURED=True — after that reconfiguring is
# impossible. Without this, test tool-calls (append_notes, github_search
# with test payloads like "[2401.99999] hallucination") would be written
# into the prod run.log on every `pytest`, creating an illusion that "tests are run
# on every agent run".
#
# Fix: mark lra-logger as configured up-front with an empty
# handler set. get_logger() will see _CONFIGURED=True and will not
# attach a FileHandler. Tests do not need logs — they check return values,
# not stderr/log output.
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
    """Each test gets a fresh tmp research/ folder. Patches ALL modules that captured
    paths via `from ... import RESEARCH_DIR` (setattr on config alone is not enough —
    other modules already hold their own copy).

    Also resets the loop-tracker between tests (in-memory state).
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
