"""Shared helpers for tools: verifier, domain gate, arxiv feed parsing, wrap-logger.

Extracted from the monolithic `lra/tools.py` to shrink the main module and
pave the way for further splits into topical subpackages (search, notes,
draft, plan, kb, etc.). All @register_tool classes import verifiers/gate and
helpers for arXiv HTTP fetching from here.
"""
from __future__ import annotations

import json
import re
from datetime import UTC
from urllib.request import urlopen
from xml.etree import ElementTree as ET

from qwen_agent.tools.base import BaseTool

from .. import config as _cfg
from .. import kb as kb_mod
from ..logger import get_logger
from ..memory import ensure_dir
from ..utils import (
    extract_ids,
    extract_topic_keywords_tiered,
    has_anti_keyword,
    keyword_set,
)

log = get_logger("tools")

_ARXIV_FEED_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _has_core_vocabulary(plan_text: str) -> bool:
    """Checks whether a `**Core vocabulary:** ...` line is present in plan.md.

    It signals that the bootstrap planner succeeded (the LLM generated 8-15
    specific domain terms). If the line is missing — bootstrap failed (mlx
    crashed / invalid JSON / fallback to static plan) and the domain gate
    operates only on the 4 words from the topic header. That let through
    mines and an audio vocoder disguised as EW.

    With strict_domain_gate=True and vocab_required=True (default) this mode
    is fail-closed: the gate returns False with reason='no_vocabulary' until
    the user explicitly allows work without a vocabulary via
    `CFG['allow_no_vocab'] = True`.
    """
    if not plan_text:
        return False
    for ln in plan_text.splitlines()[:10]:
        if ln.lstrip().lower().startswith("**core vocabulary:"):
            # Check that after the colon there are some terms, not just an
            # empty line from broken bootstrap.
            payload = ln.split(":", 1)[1] if ":" in ln else ""
            payload = payload.strip().strip("*").strip()
            return len(payload) >= 5
    return False


def verify_ids_against_kb(content: str) -> tuple[set[str], set[str]]:
    """Returns (known_ids, unknown_ids) — which arxiv-ids from content are in kb.jsonl.

    Used by the pre-append verifier in AppendNotes: the explorer must not add facts
    with ids that neither hf_papers nor kb_search ever returned in this session.
    This cuts hallucinations at write time rather than at final draft.md validation.
    """
    ids = extract_ids(content)
    if not ids:
        return set(), set()
    known_in_kb = {a.get("id", "") for a in kb_mod.load() if a.get("kind") == "paper"}
    return ids & known_in_kb, ids - known_in_kb


def domain_gate(content: str) -> tuple[bool, str, set[str], set[str]]:
    """Two-tier domain gate for AppendNotes and hf_papers kb auto-save.

    Rule: a paper passes ⇔ ≥2 matches with the HEADER of plan.md.
    HEADER = the first line ('# Plan: ...') — it is the user's original
    topic and the most stable carrier of core terms. Seeds from [Tn] tasks
    drift and carry noise ('support' inside 'electronic support measures'
    yields a false positive on emotional-support-conversations papers) —
    so they are used only for diagnosing the rejection reason in
    rejected.jsonl, not for letting a paper through.

    Slow-start bypass: if the header has <2 candidates — the plan is still
    generic and the gate is blind.

    Returns (passed, reason, overlap_header, overlap_seeds). reason ∈
    {"no_plan", "slow_start", "no_core_hit", "weak_overlap", "passed"}.

    Adaptive threshold: with a poor header (≤ 4 core-kws) one overlap is
    enough, otherwise ≥2 is required. This catches narrow topics like
    `# Plan: electronic warfare (EW) and ELINT` (4 words: electronic/warfare/
    elint/intelligence), where a paper 'cognitive radar jamming' produces
    only 1 hit (`warfare` → 0, `electronic` → 1) and would otherwise be cut.
    """
    if not _cfg.PLAN_PATH.exists():
        return True, "no_plan", set(), set()
    plan_text = _cfg.PLAN_PATH.read_text(encoding="utf-8")
    # Fail-closed: if bootstrap did not emit **Core vocabulary:**, the gate
    # operates on only the 4 header words → lets mines and audio vocoders
    # through. Block all appends until the user explicitly allows via
    # CFG['allow_no_vocab']=True.
    if not _has_core_vocabulary(plan_text) and not _cfg.CFG.get("allow_no_vocab", False):
        return False, "no_vocabulary", set(), set()
    header_kws, seed_kws = extract_topic_keywords_tiered(plan_text)
    if len(header_kws) < 2:
        return True, "slow_start", header_kws, set()
    # Explicit anti-keywords: audio vocoder, jailbreak, automotive lidar —
    # block BEFORE positive-check to not waste overlap slots.
    anti = has_anti_keyword(content)
    if anti:
        return False, f"anti_keyword:{anti}", set(), set()
    # Gather material from kb for ids mentioned in content.
    ids_in_content = extract_ids(content)
    abstracts: list[str] = [content]
    if ids_in_content:
        kb_by_id = {a.get("id"): a for a in kb_mod.load() if a.get("kind") == "paper"}
        for aid in ids_in_content:
            atom = kb_by_id.get(aid)
            if atom:
                abstracts.append(f"{atom.get('title', '')} {atom.get('claim', '')}")
    content_kws = keyword_set(" ".join(abstracts))
    o_header = content_kws & header_kws
    o_seeds = content_kws & seed_kws
    if not o_header:
        return False, "no_core_hit", set(), o_seeds
    # Adaptive threshold: poor header ⇒ 1 hit is enough, rich ⇒ ≥2.
    min_hits = 1 if len(header_kws) <= 4 else 2
    if len(o_header) < min_hits:
        return False, "weak_overlap", o_header, o_seeds
    return True, "passed", o_header, o_seeds


def gate_paper_for_kb(paper_id: str, title: str, abstract: str) -> tuple[bool, str, set[str], set[str]]:
    """Lightweight gate for hf_papers/kb_add BEFORE writing to kb.jsonl.

    Otherwise the explorer fills KB with off-topic papers (ComVo with
    auto-save remains in KB even if AppendNotes rejects it, since KB writes
    happen EARLIER in hf_papers). Works on the raw abstract (kb does not
    yet know this id).

    Returns (passed, reason, o_header, header_kws) — the caller needs
    o_header/header_kws to record diagnostics in rejected.jsonl
    (see `_log_kb_rejected`). Adaptive threshold identical to `domain_gate`:
    ≤4 core-kws → 1 hit, otherwise ≥2.
    """
    if not _cfg.CFG.get("strict_domain_gate", True) or not _cfg.PLAN_PATH.exists():
        return True, "bypass", set(), set()
    plan_text = _cfg.PLAN_PATH.read_text(encoding="utf-8")
    if not _has_core_vocabulary(plan_text) and not _cfg.CFG.get("allow_no_vocab", False):
        return False, "no_vocabulary", set(), set()
    header_kws, _ = extract_topic_keywords_tiered(plan_text)
    if len(header_kws) < 2:
        return True, "slow_start", set(), header_kws
    anti = has_anti_keyword(f"{title} {abstract}")
    if anti:
        return False, f"anti_keyword:{anti}", set(), header_kws
    kws = keyword_set(f"{title} {abstract}")
    o_h = kws & header_kws
    if not o_h:
        return False, "no_core_hit", o_h, header_kws
    min_hits = 1 if len(header_kws) <= 4 else 2
    if len(o_h) < min_hits:
        return False, "weak_overlap", o_h, header_kws
    return True, "passed", o_h, header_kws


def gate_repo_for_kb(repo_name: str, description: str) -> tuple[bool, str]:
    """Gate for a GitHub repo before auto-saving to kb.jsonl.

    ComVo (hs-oh-prml/ComVo) — audio vocoder, desc: 'neural vocoder for audio synthesis'.
    Without this gate it is saved to KB because desc was not inspected.

    Returns (passed, reason). reason ∈ {"bypass", "slow_start", "anti_keyword",
    "no_core_hit", "weak_overlap", "passed"}.
    """
    if not _cfg.CFG.get("strict_domain_gate", True) or not _cfg.PLAN_PATH.exists():
        return True, "bypass"
    plan_text = _cfg.PLAN_PATH.read_text(encoding="utf-8")
    if not _has_core_vocabulary(plan_text) and not _cfg.CFG.get("allow_no_vocab", False):
        return False, "no_vocabulary"
    anti = has_anti_keyword(f"{repo_name} {description}")
    if anti:
        return False, f"anti_keyword:{anti}"
    header_kws, _ = extract_topic_keywords_tiered(plan_text)
    if len(header_kws) < 2:
        return True, "slow_start"
    kws = keyword_set(f"{repo_name} {description}")
    o_h = kws & header_kws
    if not o_h:
        return False, "no_core_hit"
    min_hits = 1 if len(header_kws) <= 4 else 2
    if len(o_h) < min_hits:
        return False, "weak_overlap"
    return True, "passed"


def _log_kb_rejected(paper_id: str, title: str, reason: str,
                     o_header: set[str], header_kws: set[str],
                     source: str) -> None:
    """Logs a paper skipped by `gate_paper_for_kb` to rejected.jsonl.

    Without this the skip lives only in log.debug and the reason is
    invisible to the user — see session 2026-04-22: 17 searches, 1 paper
    in KB, 0 reject entries from gate_paper_for_kb.
    """
    from datetime import datetime
    ensure_dir()
    entry = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "reason": f"kb_autosave:{reason}",
        "source": source,  # hf_papers / arxiv_search
        "paper_id": paper_id,
        "title": (title or "")[:200],
        "overlap_header": sorted(o_header),
        "header_keywords": sorted(header_kws)[:20],
    }
    with _cfg.REJECTED_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _log_rejected(content: str, ids: set[str], reason: str,
                  header_kws: set[str], seed_kws: set[str],
                  o_header: set[str], o_seeds: set[str]) -> None:
    """Writes a rejected note to research/rejected.jsonl for later analysis."""
    from datetime import datetime
    ensure_dir()
    entry = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "reason": reason,
        "ids": sorted(ids),
        "overlap_header": sorted(o_header),
        "overlap_seeds": sorted(o_seeds),
        "header_keywords": sorted(header_kws)[:15],
        "seed_keywords_sample": sorted(seed_kws)[:15],
        "content_preview": content[:300],
    }
    with _cfg.REJECTED_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _fetch_text(url: str, timeout: int = 20) -> str:
    with urlopen(url, timeout=timeout) as resp:  # noqa: S310 - controlled arXiv endpoint
        return resp.read().decode("utf-8", errors="replace")


def _parse_arxiv_feed(xml_text: str) -> list[dict[str, str]]:
    root = ET.fromstring(xml_text)
    entries: list[dict[str, str]] = []
    for entry in root.findall("atom:entry", _ARXIV_FEED_NS):
        id_text = (entry.findtext("atom:id", default="", namespaces=_ARXIV_FEED_NS) or "").strip()
        m = re.search(r"(\d{4}\.\d{4,5})(?:v\d+)?", id_text)
        if not m:
            continue
        authors = [
            (node.findtext("atom:name", default="", namespaces=_ARXIV_FEED_NS) or "").strip()
            for node in entry.findall("atom:author", _ARXIV_FEED_NS)
        ]
        entries.append({
            "id": m.group(1),
            "title": " ".join((entry.findtext("atom:title", default="", namespaces=_ARXIV_FEED_NS) or "").split()),
            "summary": " ".join((entry.findtext("atom:summary", default="", namespaces=_ARXIV_FEED_NS) or "").split()),
            "published_at": (entry.findtext("atom:published", default="", namespaces=_ARXIV_FEED_NS) or "").strip(),
            "authors": ", ".join(a for a in authors[:4] if a),
        })
    return entries


def _wrap_with_logging(cls):
    """Wraps `cls.call` so every call lands in the log uniformly.

    Idempotent (a second call is a no-op via the `_tool_logged` flag).
    Loop-detection via lra.tool_tracker — blocks the Nth consecutive
    identical call.
    """
    orig = cls.call
    if getattr(orig, "_tool_logged", False):
        return cls
    tool_name = getattr(cls, "name", cls.__name__)

    def call(self, params: str = "", **kwargs):  # type: ignore[override]
        try:
            preview_src = params if isinstance(params, str) else json.dumps(params, ensure_ascii=False)
        except Exception:
            preview_src = str(params)
        preview = (preview_src or "").replace("\n", " ")[:160]
        log.info("[TOOL_CALL] %s(%s)", tool_name, preview)
        # Loop detection: block the Nth identical consecutive call.
        # See lra/tool_tracker.py for motivation (run.log compact_notes x16).
        try:
            from lra import tool_tracker
            allowed, n = tool_tracker.check_call(tool_name, params)
            if not allowed:
                budget = tool_tracker._TRACKER._max_per_tool.get(tool_name)
                total = tool_tracker._TRACKER._call_totals.get(tool_name, n)
                if budget is not None and total > budget:
                    log.warning("[TOOL_BUDGET] %s blocked: budget exceeded %d/%d calls this session",
                                tool_name, total, budget)
                    return (
                        f"error: budget exceeded — {tool_name} has already been called {total} times "
                        f"(limit {budget} per session). Move to the next plan step. "
                        f"Do not call {tool_name} again in this run."
                    )
                log.warning("[TOOL_LOOP] %s blocked: %d-th consecutive identical call", tool_name, n)
                return (
                    f"error: loop detected — {tool_name} was called {n} times in a row "
                    f"with the same params. Change strategy: try another tool, "
                    f"change params or move to the next plan step."
                )
        except Exception as _loop_err:
            # a tracker bug must not crash tool execution
            log.debug("tool_tracker error: %s", _loop_err)
        try:
            return orig(self, params, **kwargs)
        except Exception as e:
            log.warning("[TOOL_ERR]  %s: %s", tool_name, e)
            raise

    call._tool_logged = True  # type: ignore[attr-defined]
    cls.call = call
    return cls


def _wrap_module_tools(module_globals: dict, module_name: str) -> None:
    """Wraps every BaseTool subclass of the module with wrap_with_logging.

    Called at the end of each tools/ submodule so that only its own
    @register_tool classes get the logger (imported ones from foreign
    modules are left alone).
    """
    for _obj in list(module_globals.values()):
        if isinstance(_obj, type) and issubclass(_obj, BaseTool) and _obj is not BaseTool:
            if _obj.__module__ == module_name:
                _wrap_with_logging(_obj)
