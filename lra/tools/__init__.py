"""Публичный API пакета `lra.tools`.

Раньше был один монолит `lra/tools.py` (~1070 LOC). Разбит на:
- `_helpers.py` — верификаторы, domain gate, HTTP-утилиты, wrap_logger
- `_search.py`  — hf_papers, arxiv_search, github_search (~400 LOC)
- `_artifacts.py` — notes/draft/plan/synthesis/lessons/kb/plan-mutation тулы (~330 LOC)

Все исторические символы (`lra.tools.NOTES_PATH`, `lra.tools.HfPapers`,
`lra.tools.verify_ids_against_kb` и т.д.) оставлены как re-exports ради
существующих тестовых monkeypatch'ей. Новый код должен импортировать
напрямую из submodule при желании, но публичный API — это только этот файл.
"""
from __future__ import annotations

# Submodule aliases historically exposed on the `tools` namespace —
# tests do `monkeypatch.setattr(tools.cli_run, "run", ...)` and patch
# `lra.tools.cli_run.run` directly. Must remain accessible as attribute.
from .. import cli as cli_run  # noqa: F401
from .. import plan as plan_mod  # noqa: F401

# Paths re-exported for backwards-compatible monkeypatch surface
# (tests do `monkeypatch.setattr(tools, "NOTES_PATH", ...)`).
# Implementation code читает эти константы через `from .. import config as _cfg`,
# поэтому сами патчи на tools.X — no-op но сохранены для совместимости тестов,
# которые парно патчат и config, и tools.
from ..config import (
    ARXIV_RECENT_DAYS,
    CFG,
    DRAFT_PATH,
    GITHUB_RECENT_DAYS,
    LESSONS_PATH,
    MAX_GITHUB_QUERY_WORDS,
    NOTES_PATH,
    PLAN_PATH,
    QUERYLOG_PATH,
    REJECTED_PATH,
    SYNTHESIS_PATH,
)

# Artifact tools
from ._artifacts import (
    AppendDraft,
    AppendLessons,
    AppendNotes,
    CompactNotes,
    KbAdd,
    KbSearch,
    PlanAddTask,
    PlanCloseTask,
    PlanSplitTask,
    ReadDraft,
    ReadLessons,
    ReadNotes,
    ReadNotesFocused,
    ReadPlan,
    ReadQueryLog,
    ReadSynthesis,
    WriteDraft,
    WritePlan,
    WriteSynthesis,
    _require_plan,
)

# Helpers + logger
from ._helpers import (
    _fetch_text,
    _log_rejected,
    _parse_arxiv_feed,
    _wrap_with_logging,
    domain_gate,
    gate_paper_for_kb,
    log,
    verify_ids_against_kb,
)

# Search tools
from ._search import ArxivSearch, GithubSearch, HfPapers

__all__ = [
    # paths
    "NOTES_PATH", "PLAN_PATH", "DRAFT_PATH", "LESSONS_PATH", "REJECTED_PATH",
    "SYNTHESIS_PATH", "QUERYLOG_PATH", "CFG",
    "ARXIV_RECENT_DAYS", "GITHUB_RECENT_DAYS", "MAX_GITHUB_QUERY_WORDS",
    # helpers
    "verify_ids_against_kb", "domain_gate", "gate_paper_for_kb",
    "_log_rejected", "_fetch_text", "_parse_arxiv_feed", "log",
    "_wrap_with_logging",
    # search
    "HfPapers", "ArxivSearch", "GithubSearch",
    # artifacts
    "CompactNotes", "WriteDraft", "AppendDraft", "ReadDraft",
    "AppendNotes", "ReadNotes", "ReadNotesFocused",
    "WritePlan", "ReadPlan", "WriteSynthesis", "ReadSynthesis",
    "AppendLessons", "ReadLessons", "ReadQueryLog",
    "KbAdd", "KbSearch",
    "PlanAddTask", "PlanCloseTask", "PlanSplitTask", "_require_plan",
]
