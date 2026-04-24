"""Reflexion memory management: lessons, querylog, archives."""
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


# Threshold above which two queries are considered semantically equivalent.
# History: 0.75 caught word permutations, but after adding years to keyword_set
# (fix 2026-04-23) the query "X 2024" vs "X 2023" gives jaccard=0.75 — exactly on the threshold.
# Raised to 0.80: year-diff (7 words, 1 year different) → 0.75 (let through),
# a true duplicate (permutation/+1 word) → ≥0.86 (blocks).
FUZZY_DUP_THRESHOLD = 0.80


def is_similar_to_seen(query: str) -> str | None:
    """Fuzzy lookup of the nearest already-seen query by jaccard over keyword-set.

    Returns the matching duplicate (original text) if similarity ≥ FUZZY_DUP_THRESHOLD,
    else None. Considers the last 30 queries — limited for speed O(30·log).

    Additionally: if the new query is a SUPERSET of an old one (all old words are
    in the new one + extra noise), it is also treated as a duplicate — catches
    cases of adding `pushedAt`, `framework`, `stars` to an already executed query.
    BUT: different years (2023 vs 2024) are NOT duplicates even if everything else
    matches, because the year is a meaningful filter for finding new papers.
    """
    if not QUERYLOG_PATH.exists():
        return None
    q_kw = keyword_set(query)
    if len(q_kw) < 3:  # very short queries are not filtered — they may be different
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
        # Different years — clearly different queries, do not block.
        p_years = {w for w in p_kw if re.fullmatch(r"20\d{2}", w)}
        if q_years and p_years and q_years != p_years:
            continue
        score = jaccard(q_kw, p_kw)
        if score >= FUZZY_DUP_THRESHOLD:
            return past
        # Containment: new = superset of old (noise words added).
        # Exclude years from the containment check — year distinguishes queries.
        q_no_year = q_kw - q_years
        p_no_year = p_kw - p_years
        if p_no_year and p_no_year <= q_no_year and len(p_no_year) >= 3:
            return past
    return None


def log_query(query: str):
    """Log the executed hf_papers queries (Reflexion episodic memory)."""
    ensure_dir()
    with QUERYLOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"- {query}\n")


def archive_previous(query_hint: str = ""):
    """Saves the previous draft+notes+plan+synthesis into archive/<timestamp>_<slug>/."""
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
    """Prepares the working directory for a new run.
    draft/notes/plan/synthesis — archived and cleared.
    lessons/querylog — PRESERVED (cross-session Reflexion memory).
    """
    ensure_dir()
    archived = archive_previous(query)
    if archived:
        print(f"📦 Previous run archived: {archived.relative_to(RESEARCH_DIR.parent)}")
    for p in (DRAFT_PATH, NOTES_PATH, PLAN_PATH, SYNTHESIS_PATH, KB_PATH):
        p.unlink(missing_ok=True)
    # plan.json — source of truth for the plan; plan.md is rendered automatically
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
    # Initialize the structured plan (plan.json) + render plan.md via plan.reset()
    plan_mod.reset(query)
