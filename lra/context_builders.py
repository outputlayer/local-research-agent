"""Pure-функции построения контекста из файловых артефактов research/.

Выделены из `lra.pipeline` чтобы снизить размер оркестратора и повысить cohesion:
здесь — только read-only билдеры, которые собирают текстовые блоки
для user-сообщений агентов (explorer / synthesizer / writer) и fallback-пути.

Пути к research-артефактам читаются через `config` как namespace,
чтобы monkeypatch на `config.REJECTED_PATH` (и т.п.) в тестах работал прозрачно.
"""
from __future__ import annotations

import json

from . import config as _config
from . import kb as kb_mod
from . import plan as plan_mod
from . import research_memory as research_memory_mod


def _build_kb_context(query: str) -> str:
    """Собирает authoritative-блок из kb.jsonl для writer'а и fallback'а."""
    kb_all = kb_mod.load()
    repos = sorted([a for a in kb_all if a.get("kind") == "repo"],
                   key=lambda a: a.get("stars", 0), reverse=True)[:8]
    papers = kb_mod.search(query, k=12, atoms=kb_all) or \
        [a for a in kb_all if a.get("kind") == "paper"][:12]
    blocks: list[str] = []
    if repos:
        blocks.append("Репозитории (для секции '## Реализации'):")
        for r in repos:
            blocks.append(
                f"- [repo: {r.get('id','?')} ★{r.get('stars',0)} {r.get('lang','')}] "
                f"{r.get('url','')} — {r.get('claim','')[:180]}")
    else:
        blocks.append("Репозитории: в KB нет репо с ★≥10 (используй точную строку-заглушку из промпта).")
    if papers:
        blocks.append("\nPapers (для секций '## Подходы' / '## Бенчмарки и метрики'):")
        for p in papers:
            blocks.append(f"- [{p.get('id','?')}] {p.get('title','')[:90]} — {p.get('claim','')[:180]}")
    return "\n".join(blocks)


def _build_memory_context(*parts: str, k: int = 3) -> str:
    """Top-k релевантных cross-session memories для текущего запроса/фокуса."""
    query = " ".join(part.strip() for part in parts if part and part.strip())
    if not query:
        return ""
    entries = research_memory_mod.select_relevant_memories(query, k=k)
    return research_memory_mod.format_memory_context(entries)


def _latest_lessons_tail(max_lines: int = 8, max_chars: int = 1200) -> str:
    """Хвост lessons.md для записи в run-summary memory."""
    lessons_path = _config.LESSONS_PATH
    if not lessons_path.exists():
        return ""
    lines = [ln.rstrip() for ln in lessons_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        return ""
    tail = "\n".join(lines[-max_lines:])
    return tail[-max_chars:]


def _build_status_context(query: str, focus: str = "") -> str:
    """Сжатый статус исследования: покрытие плана, проблемные ветки, rejected evidence."""
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
    """Программный fallback если writer не смог 2 раза подряд: собираем минимальный
    draft.md прямо из kb.jsonl + synthesis.md. Пользователь получит хоть что-то
    вместо пустоты. Все утверждения — прямые выдержки из claim'ов KB, так что
    validator пропустит [id] без проблем.
    """
    kb_all = kb_mod.load()
    papers = [a for a in kb_all if a.get("kind") == "paper"][:15]
    repos = sorted([a for a in kb_all if a.get("kind") == "repo"],
                   key=lambda a: a.get("stars", 0), reverse=True)[:5]
    synthesis_path = _config.SYNTHESIS_PATH
    draft_path = _config.DRAFT_PATH
    synth = synthesis_path.read_text(encoding="utf-8") if synthesis_path.exists() else ""

    lines = [f"# {query}", "",
             "> Fallback-черновик: собран программно из kb.jsonl и synthesis.md "
             "(writer не вызвал write_draft после retry).", ""]
    lines.append("## Краткий ответ")
    for p in papers[:6]:
        claim_short = (p.get('claim', '') or '').replace('\n', ' ')[:180]
        lines.append(f"- [{p.get('id','?')}] {p.get('title','')[:70]}: {claim_short}")
    lines.append("")
    lines.append("## Подходы")
    for p in papers[:8]:
        lines.append(f"\n### {p.get('title','?')[:80]} [{p.get('id','?')}]")
        claim = (p.get('claim', '') or '').strip()
        lines.append(claim[:500] if claim else "_(claim отсутствует в kb)_")
    lines.append("")
    lines.append("## Реализации")
    if repos:
        for i, r in enumerate(repos, 1):
            lines.append(f"{i}. [{r.get('id','?')} ★{r.get('stars',0)}] ({r.get('lang','')}) — "
                         f"{(r.get('claim','') or '')[:150]}")
    else:
        lines.append("Публичных реализаций с ★≥10 в собранной выборке не обнаружено.")
    lines.append("")
    lines.append("## Ключевые инсайты")
    lines.append(synth or "_(synthesis.md отсутствует)_")
    lines.append("")
    lines.append("## Источники")
    lines.append("### Papers")
    for i, p in enumerate(papers, 1):
        lines.append(f"{i}. [{p.get('id','?')}] {p.get('title','')[:100]}")
    if repos:
        lines.append("\n### Repositories")
        for i, r in enumerate(repos, 1):
            lines.append(f"{i}. {r.get('id','?')} ({r.get('url','')})")
    draft_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"   🛟 fallback-draft собран программно: {draft_path} "
          f"({draft_path.stat().st_size} симв)")
