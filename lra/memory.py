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
# История: 0.75 ловил перестановки слов, но с добавлением годов в keyword_set
# (fix 2026-04-23) запрос "X 2024" vs "X 2023" даёт jaccard=0.75 — ровно на пороге.
# Повышаем до 0.80: year-diff (7 слов, 1 год разный) → 0.75 (пропускает),
# настоящий дубликат (перестановка/+1 слово) → ≥0.86 (блокирует).
FUZZY_DUP_THRESHOLD = 0.80


def is_similar_to_seen(query: str) -> str | None:
    """Fuzzy-поиск ближайшего уже виденного запроса по jaccard на keyword-set.

    Возвращает найденный дубликат (исходный текст) если похожесть ≥ FUZZY_DUP_THRESHOLD,
    иначе None. Учитывает последние 30 запросов — ограничение для скорости O(30·log).

    Дополнительно: если новый запрос является НАДМНОЖЕСТВОМ старого (все слова
    старого есть в новом + лишние шумовые), тоже считается дубликатом — это ловит
    случаи добавления `pushedAt`, `framework`, `stars` к уже выполненному запросу.
    НО: разные годы (2023 vs 2024) — НЕ дубликаты даже если всё остальное совпадает,
    т.к. год — значимый фильтр для поиска новых статей.
    """
    if not QUERYLOG_PATH.exists():
        return None
    q_kw = keyword_set(query)
    if len(q_kw) < 3:  # слишком короткие запросы не фильтруем — могут быть разными
        return None
    q_years = {w for w in q_kw if re.fullmatch(r"20\d{2}", w)}
    recent: list[str] = []
    for ln in QUERYLOG_PATH.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if not s or s.startswith("#") or s.startswith("##"):
            continue
        recent.append(s.lstrip("- ").strip())
    for past in reversed(recent[-30:]):
        p_kw = keyword_set(past)
        # Разные годы — явно разные запросы, не блокируем.
        p_years = {w for w in p_kw if re.fullmatch(r"20\d{2}", w)}
        if q_years and p_years and q_years != p_years:
            continue
        score = jaccard(q_kw, p_kw)
        if score >= FUZZY_DUP_THRESHOLD:
            return past
        # Containment: новый = надмножество старого (добавлены шумовые слова).
        # Исключаем годы из containment-check — год отличает запросы.
        q_no_year = q_kw - q_years
        p_no_year = p_kw - p_years
        if p_no_year and p_no_year <= q_no_year and len(p_no_year) >= 3:
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
