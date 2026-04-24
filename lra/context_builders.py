"""Pure functions that build textual context blocks from research/ artifacts.

Extracted from `lra.pipeline` to shrink the orchestrator and improve cohesion:
only read-only builders live here, assembling text blocks for user-messages of
agent roles (explorer / synthesizer / writer) and the fallback path.

Paths to research artifacts are read via `config` as a namespace so tests can
monkeypatch `config.REJECTED_PATH` (etc.) transparently.
"""
from __future__ import annotations

import json

from . import config as _config
from . import kb as kb_mod
from . import plan as plan_mod
from . import research_memory as research_memory_mod


def _build_kb_context(query: str) -> str:
    """Builds the authoritative block from kb.jsonl for writer and fallback."""
    kb_all = kb_mod.load()
    repos = sorted([a for a in kb_all if a.get("kind") == "repo"],
                   key=lambda a: a.get("stars", 0), reverse=True)[:8]
    papers = kb_mod.search(query, k=12, atoms=kb_all) or \
        [a for a in kb_all if a.get("kind") == "paper"][:12]
    blocks: list[str] = []
    if repos:
        blocks.append("Repositories (for '## Implementations' section):")
        for r in repos:
            blocks.append(
                f"- [repo: {r.get('id','?')} ★{r.get('stars',0)} {r.get('lang','')}] "
                f"{r.get('url','')} — {r.get('claim','')[:180]}")
    else:
        blocks.append(
            "Repositories: KB has no repos with ★≥10 (use the exact placeholder "
            "string from the prompt).")
    if papers:
        blocks.append(
            "\nPapers (for '## Approaches' / '## Benchmarks and Metrics' sections):")
        for p in papers:
            blocks.append(f"- [{p.get('id','?')}] {p.get('title','')[:90]} — {p.get('claim','')[:180]}")
    return "\n".join(blocks)


def _build_memory_context(*parts: str, k: int = 3) -> str:
    """Top-k relevant cross-session memories for the current query/focus."""
    query = " ".join(part.strip() for part in parts if part and part.strip())
    if not query:
        return ""
    entries = research_memory_mod.select_relevant_memories(query, k=k)
    return research_memory_mod.format_memory_context(entries)


def _latest_lessons_tail(max_lines: int = 8, max_chars: int = 1200) -> str:
    """Tail of lessons.md, recorded into the run-summary memory."""
    lessons_path = _config.LESSONS_PATH
    if not lessons_path.exists():
        return ""
    lines = [ln.rstrip() for ln in lessons_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        return ""
    tail = "\n".join(lines[-max_lines:])
    return tail[-max_chars:]


def _build_status_context(query: str, focus: str = "") -> str:
    """Compact research status: plan coverage, problematic branches, rejected evidence."""
    plan = plan_mod.load()
    lines: list[str] = [f"- query: {query}"]
    if focus:
        lines.append(f"- requested_focus: {focus}")

    if plan:
        done = [t for t in plan.tasks if t.status == "done"]
        open_tasks = [t for t in plan.tasks if t.status == "open"]
        in_progress = [t for t in plan.tasks if t.status == "in_progress"]
        blocked = [t for t in plan.tasks if t.status == "blocked"]
        lines.append(
            f"- plan_progress: done={len(done)}/{len(plan.tasks)} open={len(open_tasks)} "
            f"in_progress={len(in_progress)} blocked={len(blocked)}"
        )
        focus_task = plan.focus_task()
        if focus_task:
            lines.append(
                f"- focus_task: [{focus_task.id}] {focus_task.title} "
                f"(attempts={focus_task.attempts}, evidence={len(focus_task.evidence_refs)})"
            )
        undercovered = [t for t in plan.tasks if t.status in ("open", "in_progress") and not t.evidence_refs][:3]
        if undercovered:
            lines.append("- undercovered_tasks:")
            lines.extend(f"  - [{t.id}] {t.title}" for t in undercovered)
        if blocked:
            lines.append("- blocked_tasks:")
            lines.extend(f"  - [{t.id}] {t.title} (attempts={t.attempts})" for t in blocked[:3])

    rejected_path = _config.REJECTED_PATH
    if rejected_path.exists():
        rejected_rows = [ln for ln in rejected_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if rejected_rows:
            reasons: dict[str, int] = {}
            for line in rejected_rows:
                try:
                    reason = json.loads(line).get("reason", "unknown")
                except json.JSONDecodeError:
                    reason = "invalid_json"
                reasons[reason] = reasons.get(reason, 0) + 1
            reason_str = ", ".join(f"{key}={value}" for key, value in sorted(reasons.items()))
            lines.append(f"- rejected_evidence: {len(rejected_rows)} ({reason_str})")

    return "Research status:\n" + "\n".join(lines)


def _fallback_draft_from_kb(query: str) -> None:
    """Programmatic fallback when the writer fails twice in a row: assemble a
    minimal draft.md straight from kb.jsonl + synthesis.md. The user then gets
    something rather than nothing. All claims are verbatim excerpts from KB
    claims so the validator accepts the [id] citations."""
    kb_all = kb_mod.load()
    papers = [a for a in kb_all if a.get("kind") == "paper"][:15]
    repos = sorted([a for a in kb_all if a.get("kind") == "repo"],
                   key=lambda a: a.get("stars", 0), reverse=True)[:5]
    synthesis_path = _config.SYNTHESIS_PATH
    draft_path = _config.DRAFT_PATH
    synth = synthesis_path.read_text(encoding="utf-8") if synthesis_path.exists() else ""

    lines = [f"# {query}", "",
             "> Fallback draft: assembled programmatically from kb.jsonl and "
             "synthesis.md (the writer did not call write_draft after retry).", ""]
    lines.append("## TL;DR")
    for p in papers[:6]:
        claim_short = (p.get('claim', '') or '').replace('\n', ' ')[:180]
        lines.append(f"- [{p.get('id','?')}] {p.get('title','')[:70]}: {claim_short}")
    lines.append("")
    lines.append("## Approaches")
    for p in papers[:8]:
        lines.append(f"\n### {p.get('title','?')[:80]} [{p.get('id','?')}]")
        claim = (p.get('claim', '') or '').strip()
        lines.append(claim[:500] if claim else "_(no claim in kb)_")
    lines.append("")
    lines.append("## Implementations")
    if repos:
        for i, r in enumerate(repos, 1):
            lines.append(f"{i}. [{r.get('id','?')} ★{r.get('stars',0)}] ({r.get('lang','')}) — "
                         f"{(r.get('claim','') or '')[:150]}")
    else:
        lines.append("No public implementations with ★≥10 found in the gathered sample.")
    lines.append("")
    lines.append("## Key Insights")
    lines.append(synth or "_(synthesis.md missing)_")
    lines.append("")
    lines.append("## Sources")
    lines.append("### Papers")
    for i, p in enumerate(papers, 1):
        lines.append(f"{i}. [{p.get('id','?')}] {p.get('title','')[:100]}")
    if repos:
        lines.append("\n### Repositories")
        for i, r in enumerate(repos, 1):
            lines.append(f"{i}. {r.get('id','?')} ({r.get('url','')})")
    draft_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"   🛟 fallback draft assembled programmatically: {draft_path} "
          f"({draft_path.stat().st_size} chars)")
