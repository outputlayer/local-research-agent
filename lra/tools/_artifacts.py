"""Artifact tools: notes/draft/plan/synthesis/lessons/querylog + kb + plan mutations.

Small tools that read/write artifact files under research/. They import paths via
`from .. import config as _cfg` (not direct `from ..config import X`) so that
tests can monkeypatch `config.NOTES_PATH` etc. — runtime lookup works.
"""
from __future__ import annotations

import re

from qwen_agent.tools.base import BaseTool, register_tool

from .. import config as _cfg
from .. import kb as kb_mod
from .. import plan as plan_mod
from ..memory import ensure_dir
from ..utils import extract_ids, get_content, parse_args
from ._helpers import (
    _log_rejected,
    _wrap_module_tools,
    domain_gate,
    verify_ids_against_kb,
)


@register_tool("compact_notes")
class CompactNotes(BaseTool):
    description = "Overwrites research/notes.md with compressed content."
    parameters = [{"name": "content", "type": "string", "description": "Compressed markdown", "required": True}]

    def call(self, params: str, **kwargs) -> str:
        ensure_dir()
        content = get_content(params)
        old = _cfg.NOTES_PATH.stat().st_size if _cfg.NOTES_PATH.exists() else 0
        _cfg.NOTES_PATH.write_text(content, encoding='utf-8')
        return f"notes.md: {old} → {len(content)} chars (compressed by {max(1, old // max(1, len(content)))}x)"


@register_tool("write_draft")
class WriteDraft(BaseTool):
    description = "Overwrites the draft research/draft.md from scratch."
    parameters = [{"name": "content", "type": "string", "description": "Markdown", "required": True}]

    def call(self, params: str, **kwargs) -> str:
        ensure_dir()
        content = get_content(params)
        _cfg.DRAFT_PATH.write_text(content, encoding='utf-8')
        return f"draft.md saved ({len(content)} chars)"


@register_tool("append_draft")
class AppendDraft(BaseTool):
    description = ("Appends a section at the end of research/draft.md. "
                   "Prefer write_draft + multiple append_draft over one huge write_draft.")
    parameters = [{"name": "content", "type": "string", "description": "Markdown section", "required": True}]

    def call(self, params: str, **kwargs) -> str:
        ensure_dir()
        content = get_content(params)
        with _cfg.DRAFT_PATH.open('a', encoding='utf-8') as f:
            f.write("\n\n" + content.strip() + "\n")
        return f"draft.md +{len(content)} chars (total {_cfg.DRAFT_PATH.stat().st_size})"


@register_tool("read_draft")
class ReadDraft(BaseTool):
    description = "Reads research/draft.md."
    parameters = []

    def call(self, params: str, **kwargs) -> str:
        return _cfg.DRAFT_PATH.read_text(encoding='utf-8') if _cfg.DRAFT_PATH.exists() else "(empty)"


@register_tool("append_notes")
class AppendNotes(BaseTool):
    description = ("Appends a new knowledge fragment to research/notes.md "
                   "(sub-topic, arXiv id, key facts).")
    parameters = [{"name": "content", "type": "string", "description": "Markdown fragment", "required": True}]

    def call(self, params: str, **kwargs) -> str:
        ensure_dir()
        content = get_content(params)
        if _cfg.CFG.get("notes_strict", True):
            known, unknown = verify_ids_against_kb(content)
            if unknown:
                return (
                    f"REJECTED: the note mentions arxiv-ids that are NOT in kb.jsonl: "
                    f"{sorted(unknown)}. First add them via kb_add (for each id: "
                    "kb_add(id=..., kind='paper', title=..., claim=..., url=...)) "
                    "or via hf_papers/kb_search to get a verified record, "
                    "then IMMEDIATELY repeat append_notes with the same content. "
                    "Do not write intermediate replies — only the kb_add tool-call, then the append_notes tool-call. "
                    f"Already known: {sorted(known) or '(empty)'}.")
        ids = extract_ids(content)
        if _cfg.CFG.get("strict_domain_gate", True) and ids:
            passed, reason, o_h, o_s = domain_gate(content)
            if not passed:
                from ..utils import extract_topic_keywords_tiered as _kw
                h_kws, s_kws = _kw(_cfg.PLAN_PATH.read_text(encoding="utf-8"))
                _log_rejected(content, ids, reason, h_kws, s_kws, o_h, o_s)
                hint = {
                    "no_core_hit": "no core term from the topic header",
                    "weak_overlap": f"only 1 match (need ≥2, at least 1 core). "
                                    f"Header hits: {sorted(o_h) or '∅'}, "
                                    f"seeds hits: {sorted(o_s) or '∅'}",
                }.get(reason, reason)
                return (f"REJECTED (domain gate, {reason}): note with ids {sorted(ids)} "
                        f"does not relate to the plan topic — {hint}. This is a paper from an "
                        "adjacent domain, do not push it into KB/notes. Logged to rejected.jsonl.")
        with _cfg.NOTES_PATH.open('a', encoding='utf-8') as f:
            f.write("\n\n" + content.strip() + "\n")
        return f"notes.md +{len(content)} chars (total {_cfg.NOTES_PATH.stat().st_size} chars)"


@register_tool("read_notes")
class ReadNotes(BaseTool):
    description = "Reads accumulated notes from research/notes.md."
    parameters = []

    def call(self, params: str, **kwargs) -> str:
        if not _cfg.NOTES_PATH.exists():
            return "(no notes)"
        text = _cfg.NOTES_PATH.read_text(encoding='utf-8')
        return text[-20000:] if len(text) > 20000 else text


@register_tool("read_notes_focused")
class ReadNotesFocused(BaseTool):
    """Anti-drift filter: returns only blocks of notes.md relevant to the focus query."""
    description = ("Reads notes.md with an anti-drift filter by focus query. "
                   "Returns only blocks relevant to focus (jaccard keyword overlap).")
    parameters = [
        {"name": "focus", "type": "string",
         "description": "Focus phrase (e.g. the [FOCUS] from plan.md)", "required": True},
        {"name": "max_chars", "type": "integer",
         "description": "Output char cap (default 4000)", "required": False},
        {"name": "min_jaccard", "type": "number",
         "description": "Relevance threshold 0..1 (default 0.08)", "required": False},
    ]

    def call(self, params: str, **kwargs) -> str:
        from ..utils import jaccard as _jaccard
        from ..utils import keyword_set as _kws
        args = parse_args(params)
        focus = (args.get("focus") or "").strip()
        if not focus:
            return "(focus is required)"
        max_chars = int(args.get("max_chars") or 4000)
        min_jac = float(args.get("min_jaccard") or 0.08)

        if not _cfg.NOTES_PATH.exists():
            return "(no notes)"
        text = _cfg.NOTES_PATH.read_text(encoding='utf-8')
        if not text.strip():
            return "(notes are empty)"

        focus_kws = _kws(focus)
        if not focus_kws:
            return text[-max_chars:]

        blocks = [b for b in re.split(r"\n\s*\n", text) if b.strip()]
        scored = [(b, _jaccard(_kws(b), focus_kws)) for b in blocks]
        relevant = [(b, s) for b, s in scored if s >= min_jac]
        relevant.sort(key=lambda x: x[1], reverse=True)

        if not relevant:
            return (f"(0/{len(blocks)} blocks passed jaccard>={min_jac} filter "
                    f"for focus='{focus[:60]}'; try a lower threshold or use read_notes)")

        placeholder_header_len = 80
        budget = max_chars - placeholder_header_len
        included: list[str] = []
        total = 0
        for block, score in relevant:
            chunk = f"<!-- score={score:.3f} -->\n{block}\n"
            if total + len(chunk) > budget:
                break
            included.append(chunk)
            total += len(chunk)
        truncated = len(included) < len(relevant)
        header = (f"[focus-filter: {len(included)}/{len(blocks)} blocks, "
                  f"jaccard>={min_jac}"
                  f"{f', truncated by max_chars={max_chars}' if truncated else ''}]\n\n")
        return header + "\n".join(included)


@register_tool("write_plan")
class WritePlan(BaseTool):
    description = "Overwrites the research plan in research/plan.md."
    parameters = [{"name": "content", "type": "string", "description": "Markdown plan", "required": True}]

    def call(self, params: str, **kwargs) -> str:
        ensure_dir()
        content = get_content(params)
        _cfg.PLAN_PATH.write_text(content, encoding='utf-8')
        todos = content.count("[TODO]")
        done = content.count("[DONE]")
        return f"plan.md: {todos} TODO, {done} DONE"


@register_tool("read_plan")
class ReadPlan(BaseTool):
    description = "Reads the current plan from research/plan.md."
    parameters = []

    def call(self, params: str, **kwargs) -> str:
        return _cfg.PLAN_PATH.read_text(encoding='utf-8') if _cfg.PLAN_PATH.exists() else "(no plan)"


@register_tool("write_synthesis")
class WriteSynthesis(BaseTool):
    description = ("Writes research/synthesis.md — NEW thoughts: "
                   "bridges, contradictions, gaps, extrapolations, testable hypotheses.")
    parameters = [{"name": "content", "type": "string", "description": "Markdown", "required": True}]

    def call(self, params: str, **kwargs) -> str:
        ensure_dir()
        content = get_content(params)
        _cfg.SYNTHESIS_PATH.write_text(content, encoding='utf-8')
        tags = sum(content.count(t) for t in ("[BRIDGE]", "[CONTRADICTION]", "[GAP]", "[EXTRAPOLATION]", "[REUSE]", "[TESTABLE]"))
        return f"synthesis.md ({len(content)} chars, {tags} insights)"


@register_tool("read_synthesis")
class ReadSynthesis(BaseTool):
    description = "Reads research/synthesis.md — new insights after analyzing notes."
    parameters = []

    def call(self, params: str, **kwargs) -> str:
        return _cfg.SYNTHESIS_PATH.read_text(encoding='utf-8') if _cfg.SYNTHESIS_PATH.exists() else "(no synthesis)"


@register_tool("append_lessons")
class AppendLessons(BaseTool):
    description = ("Writes an iteration lesson to research/lessons.md (Reflexion memory). "
                   "Format: what worked, what did not, what to avoid.")
    parameters = [{"name": "content", "type": "string", "description": "Markdown", "required": True}]

    def call(self, params: str, **kwargs) -> str:
        ensure_dir()
        content = get_content(params)
        with _cfg.LESSONS_PATH.open('a', encoding='utf-8') as f:
            f.write("\n" + content.strip() + "\n")
        return f"lessons.md +{len(content)} chars"


@register_tool("read_lessons")
class ReadLessons(BaseTool):
    description = "Reads lessons.md — lessons from past iterations, what not to repeat."
    parameters = []

    def call(self, params: str, **kwargs) -> str:
        return _cfg.LESSONS_PATH.read_text(encoding='utf-8') if _cfg.LESSONS_PATH.exists() else "(no lessons yet)"


@register_tool("read_querylog")
class ReadQueryLog(BaseTool):
    description = "Reads the list of already-executed hf_papers queries — do NOT repeat them."
    parameters = []

    def call(self, params: str, **kwargs) -> str:
        if not _cfg.QUERYLOG_PATH.exists():
            return "(no queries yet)"
        text = _cfg.QUERYLOG_PATH.read_text(encoding='utf-8')
        lines, seen = [], set()
        for ln in reversed(text.splitlines()):
            if ln.strip() and ln not in seen:
                seen.add(ln)
                lines.append(ln)
            if len(lines) >= 30:
                break
        return "\n".join(reversed(lines))


@register_tool("kb_add")
class KbAdd(BaseTool):
    description = (
        "Adds a knowledge atom to the structured base research/kb.jsonl. "
        "Call every time you learn a NEW fact/repository/authors. "
        "Runs in parallel with append_notes but enables programmatic search. "
        "kind: 'paper' (for arxiv-id) or 'repo' (for owner/name). "
        "claim: 1-3 sentences on what exactly you learned."
    )
    parameters = [
        {"name": "id", "type": "string", "description": "arxiv-id or owner/name", "required": True},
        {"name": "kind", "type": "string", "description": "'paper' or 'repo'", "required": True},
        {"name": "claim", "type": "string", "description": "1-3 sentences on the essence", "required": True},
        {"name": "title", "type": "string", "description": "title", "required": False},
        {"name": "authors", "type": "string", "description": "authors (for paper)", "required": False},
        {"name": "url", "type": "string", "description": "link", "required": False},
        {"name": "stars", "type": "integer", "description": "stars (for repo)", "required": False},
        {"name": "lang", "type": "string", "description": "language (for repo)", "required": False},
        {"name": "topic", "type": "string", "description": "current FOCUS", "required": False},
    ]

    def call(self, params: str, **kwargs) -> str:
        ensure_dir()
        p = parse_args(params)
        kind = (p.get("kind") or "").strip().lower()
        if kind not in ("paper", "repo"):
            return "error: kind must be 'paper' or 'repo'"
        if not p.get("id") or not p.get("claim"):
            return "error: id and claim are required"
        atom = kb_mod.Atom(
            id=p["id"].strip(),
            kind=kind,
            topic=(p.get("topic") or "").strip(),
            claim=p["claim"].strip(),
            title=(p.get("title") or "").strip(),
            authors=(p.get("authors") or "").strip(),
            url=(p.get("url") or "").strip(),
            stars=int(p.get("stars", 0) or 0),
            lang=(p.get("lang") or "").strip(),
        )
        kb_mod.add(atom)
        return f"kb += {kind}:{atom.id}"


@register_tool("kb_search")
class KbSearch(BaseTool):
    description = (
        "Searches research/kb.jsonl for already-known facts/repos by query (BM25). "
        "Use BEFORE issuing a new hf_papers/github_search — maybe we already know it. "
        "Returns top-k atoms with id, title and claim."
    )
    parameters = [
        {"name": "query", "type": "string", "description": "natural-language query", "required": True},
        {"name": "k", "type": "integer", "description": "how many results (1-10)", "required": False},
    ]

    def call(self, params: str, **kwargs) -> str:
        p = parse_args(params)
        query = (p.get("query") or "").strip()
        if not query:
            return "error: query is required"
        k = max(1, min(int(p.get("k", 5)), 10))
        hits = kb_mod.search(query, k=k)
        if not hits:
            return "(nothing relevant in kb yet)"
        return kb_mod.format_atoms(hits)


# ── Structured plan: explicit mutations via tools ──────────────────────────
def _require_plan() -> plan_mod.Plan:
    p = plan_mod.load()
    if not p:
        raise RuntimeError("plan.json missing — start a new research first")
    return p


@register_tool("plan_add_task")
class PlanAddTask(BaseTool):
    description = (
        "Add a new sub-task to the plan. Use when an IMPORTANT sub-topic emerged during "
        "research that was not in the initial plan (origin=emerged). parent is optional — "
        "if set to an existing task id, the new task becomes its child (2nd tree level). "
        "Limit: MAX_OPEN_TASKS=8 open tasks, overflow → error."
    )
    parameters = [
        {"name": "title", "type": "string", "description": "Short task title", "required": True},
        {"name": "parent", "type": "string", "description": "parent id (T1/T2.3/...) or empty", "required": False},
        {"name": "why", "type": "string", "description": "Why this task is added (for audit)", "required": False},
    ]

    def call(self, params: str, **kwargs) -> str:
        p = parse_args(params)
        title = (p.get("title") or "").strip()
        if not title:
            return "error: title is required"
        parent = (p.get("parent") or "").strip() or None
        why = (p.get("why") or "").strip()
        try:
            plan = _require_plan()
            task = plan.add_task(title, parent=parent, origin="emerged",
                                 iter_=0, why=why or "model decided to extend the plan")
            plan_mod.save(plan)
            return (f"added: [{task.id}] {task.title}"
                    f"{' (parent=' + parent + ')' if parent else ''} — {len(plan.open_tasks())}/"
                    f"{plan_mod.MAX_OPEN_TASKS} open")
        except Exception as e:
            return f"error: {e}"


@register_tool("plan_close_task")
class PlanCloseTask(BaseTool):
    description = (
        "Mark a task as done (status=done). Use when enough facts on the sub-topic have "
        "been collected in notes/kb. Provide evidence — a list of kb keys or arxiv-ids "
        "from notes that support closing."
    )
    parameters = [
        {"name": "id", "type": "string", "description": "task id (e.g. T2 or T2.1)", "required": True},
        {"name": "evidence", "type": "string",
         "description": "Comma-separated: kb keys (kb:PaperId) or arxiv-ids (2309.12345)", "required": False},
        {"name": "why", "type": "string", "description": "Short summary of what was found", "required": False},
    ]

    def call(self, params: str, **kwargs) -> str:
        p = parse_args(params)
        tid = (p.get("id") or "").strip()
        if not tid:
            return "error: id is required"
        evidence_raw = (p.get("evidence") or "").strip()
        evidence = [e.strip() for e in evidence_raw.split(",") if e.strip()] if evidence_raw else []
        why = (p.get("why") or "").strip()
        try:
            plan = _require_plan()
            t = plan.close_task(tid, iter_=0, evidence=evidence, why=why or "closed by model")
            plan_mod.save(plan)
            return f"closed: [{t.id}] {t.title}  evidence={len(t.evidence_refs)}"
        except Exception as e:
            return f"error: {e}"


@register_tool("plan_split_task")
class PlanSplitTask(BaseTool):
    description = (
        "Decompose a task into 2-4 sub-tasks. Use when a task is too broad to close in "
        "one iteration. The original task becomes a container (status=dropped + "
        "split-container), sub-tasks become its children in the tree."
    )
    parameters = [
        {"name": "id", "type": "string", "description": "task id to decompose", "required": True},
        {"name": "subtitles", "type": "string",
         "description": "Sub-tasks separated by ' | ' (minimum 2)", "required": True},
        {"name": "why", "type": "string", "description": "Why split", "required": False},
    ]

    def call(self, params: str, **kwargs) -> str:
        p = parse_args(params)
        tid = (p.get("id") or "").strip()
        raw = (p.get("subtitles") or "").strip()
        if not tid or not raw:
            return "error: id and subtitles are required"
        subs = [s.strip() for s in raw.split("|") if s.strip()]
        if len(subs) < 2:
            return "error: need at least 2 sub-tasks (separator ' | ')"
        why = (p.get("why") or "").strip()
        try:
            plan = _require_plan()
            children = plan.split_task(tid, subs, iter_=0, why=why or "decomposition by model")
            plan_mod.save(plan)
            return f"split [{tid}] → " + ", ".join(f"[{c.id}] {c.title[:40]}" for c in children)
        except Exception as e:
            return f"error: {e}"


_wrap_module_tools(globals(), __name__)
