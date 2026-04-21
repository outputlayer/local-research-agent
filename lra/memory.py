"""Управление Reflexion-памятью: lessons, querylog, архивы."""
from __future__ import annotations

import re
from datetime import datetime

from .config import (
    ARCHIVE_DIR,
    DRAFT_PATH,
    LESSONS_PATH,
    NOTES_PATH,
    PLAN_PATH,
    QUERYLOG_PATH,
    RESEARCH_DIR,
    SYNTHESIS_PATH,
)
from .utils import normalize_query


def ensure_dir():
    RESEARCH_DIR.mkdir(exist_ok=True)
    ARCHIVE_DIR.mkdir(exist_ok=True)


def seen_queries() -> set[str]:
    if not QUERYLOG_PATH.exists():
        return set()
    return {normalize_query(ln.lstrip("- ").strip())
            for ln in QUERYLOG_PATH.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")}


def log_query(query: str):
    """Регистрируем выполненные hf_papers запросы (Reflexion episodic memory)."""
    ensure_dir()
    with QUERYLOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"- {query}\n")


def archive_previous(query_hint: str = ""):
    """Сохраняет предыдущий draft+notes+plan+synthesis в archive/<timestamp>_<slug>/."""
    if not DRAFT_PATH.exists() and not NOTES_PATH.exists():
        return None
    slug = re.sub(r"[^a-zA-Z0-9а-яА-Я]+", "-", query_hint)[:40].strip("-") or "run"
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    dest = ARCHIVE_DIR / f"{stamp}_{slug}"
    dest.mkdir(parents=True, exist_ok=True)
    for p in (DRAFT_PATH, NOTES_PATH, PLAN_PATH, SYNTHESIS_PATH):
        if p.exists():
            (dest / p.name).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
    return dest


def reset_research(query: str):
    """Готовит рабочую папку к новому запуску.
    draft/notes/plan/synthesis — архивируются и очищаются.
    lessons/querylog — СОХРАНЯЮТСЯ (кросс-сессионная Reflexion-память).
    """
    ensure_dir()
    archived = archive_previous(query)
    if archived:
        print(f"📦 Прошлый прогон сохранён: {archived.relative_to(RESEARCH_DIR.parent)}")
    for p in (DRAFT_PATH, NOTES_PATH, PLAN_PATH, SYNTHESIS_PATH):
        p.unlink(missing_ok=True)
    NOTES_PATH.write_text(f"# Notes: {query}\n", encoding="utf-8")
    if not LESSONS_PATH.exists():
        LESSONS_PATH.write_text("# Lessons (global, across sessions)\n", encoding="utf-8")
    if not QUERYLOG_PATH.exists():
        QUERYLOG_PATH.write_text("# Query log (global, across sessions)\n", encoding="utf-8")
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    with LESSONS_PATH.open("a", encoding="utf-8") as f:
        f.write(f"\n## Session {stamp}: {query}\n")
    with QUERYLOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"\n## Session {stamp}: {query}\n")
    PLAN_PATH.write_text(
        f"# Plan: {query}\n\n"
        f"[FOCUS] {query} — обзор архитектур и ключевых работ\n\n"
        "## Digest\n(пока пусто — первая итерация)\n\n"
        "## Direction check\n(будет обновлено replanner'ом)\n\n"
        "## [TODO]\n"
        f"- {query}: обзорные статьи\n"
        f"- {query}: ключевые методы\n"
        f"- {query}: ограничения и открытые вопросы\n\n"
        "## [DONE]\n",
        encoding="utf-8",
    )
