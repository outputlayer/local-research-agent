"""Управление Reflexion-памятью: lessons, querylog, архивы."""
from __future__ import annotations

import re
from datetime import datetime

from . import plan as plan_mod
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
from .kb import KB_PATH
from .utils import jaccard, keyword_set, normalize_query


def ensure_dir():
    RESEARCH_DIR.mkdir(exist_ok=True)
    ARCHIVE_DIR.mkdir(exist_ok=True)


def seen_queries() -> set[str]:
    if not QUERYLOG_PATH.exists():
        return set()
    return {normalize_query(ln.lstrip("- ").strip())
            for ln in QUERYLOG_PATH.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")}


# Порог, выше которого два запроса считаются семантически эквивалентными.
# Эмпирически: 0.75 ловит перестановки слов типа "X Y Z" vs "X Y Z python stars pushedAt"
# не трогая разные темы. См. research/querylog.md от 2026-04-21 — массовое дублирование
# запросов-перефразировок к WebWeaver/AgentCPM обходило точный dedup.
FUZZY_DUP_THRESHOLD = 0.75


def is_similar_to_seen(query: str) -> str | None:
    """Fuzzy-поиск ближайшего уже виденного запроса по jaccard на keyword-set.

    Возвращает найденный дубликат (исходный текст) если похожесть ≥ FUZZY_DUP_THRESHOLD,
    иначе None. Учитывает последние 30 запросов — ограничение для скорости O(30·log).
    """
    if not QUERYLOG_PATH.exists():
        return None
    q_kw = keyword_set(query)
    if len(q_kw) < 3:  # слишком короткие запросы не фильтруем — могут быть разными
        return None
    recent: list[str] = []
    for ln in QUERYLOG_PATH.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if not s or s.startswith("#") or s.startswith("##"):
            continue
        recent.append(s.lstrip("- ").strip())
    for past in reversed(recent[-30:]):
        if jaccard(q_kw, keyword_set(past)) >= FUZZY_DUP_THRESHOLD:
            return past
    return None


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
    for p in (DRAFT_PATH, NOTES_PATH, PLAN_PATH, SYNTHESIS_PATH, KB_PATH):
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
    for p in (DRAFT_PATH, NOTES_PATH, PLAN_PATH, SYNTHESIS_PATH, KB_PATH):
        p.unlink(missing_ok=True)
    # plan.json — источник истины для плана; plan.md рендерится автоматически
    plan_mod.PLAN_JSON_PATH.unlink(missing_ok=True)
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
    # Инициализируем структурированный план (plan.json) + рендер plan.md через plan.reset()
    plan_mod.reset(query)
