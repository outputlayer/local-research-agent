"""Artifact tools: notes/draft/plan/synthesis/lessons/querylog + kb + plan mutations.

Мелкие тулы, пишущие/читающие файлы-артефакты под research/. Импортируют пути
через `from .. import config as _cfg` (не прямым `from ..config import X`), чтобы
тесты могли monkeypatch-ить `config.NOTES_PATH` и т.д. — runtime lookup работает.
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
    description = "Перезаписывает research/notes.md сжатым содержанием."
    parameters = [{"name": "content", "type": "string", "description": "Сжатый markdown", "required": True}]

    def call(self, params: str, **kwargs) -> str:
        ensure_dir()
        content = get_content(params)
        old = _cfg.NOTES_PATH.stat().st_size if _cfg.NOTES_PATH.exists() else 0
        _cfg.NOTES_PATH.write_text(content, encoding='utf-8')
        return f"notes.md: {old} → {len(content)} симв (сжато в {max(1, old // max(1, len(content)))}x)"


@register_tool("write_draft")
class WriteDraft(BaseTool):
    description = "Перезаписывает черновик research/draft.md с нуля."
    parameters = [{"name": "content", "type": "string", "description": "Markdown", "required": True}]

    def call(self, params: str, **kwargs) -> str:
        ensure_dir()
        content = get_content(params)
        _cfg.DRAFT_PATH.write_text(content, encoding='utf-8')
        return f"draft.md сохранён ({len(content)} симв)"


@register_tool("append_draft")
class AppendDraft(BaseTool):
    description = ("Добавляет секцию в конец research/draft.md. "
                   "Предпочтительно write_draft+много append_draft, чем один огромный write_draft.")
    parameters = [{"name": "content", "type": "string", "description": "Markdown-секция", "required": True}]

    def call(self, params: str, **kwargs) -> str:
        ensure_dir()
        content = get_content(params)
        with _cfg.DRAFT_PATH.open('a', encoding='utf-8') as f:
            f.write("\n\n" + content.strip() + "\n")
        return f"draft.md +{len(content)} симв (всего {_cfg.DRAFT_PATH.stat().st_size})"


@register_tool("read_draft")
class ReadDraft(BaseTool):
    description = "Читает research/draft.md."
    parameters = []

    def call(self, params: str, **kwargs) -> str:
        return _cfg.DRAFT_PATH.read_text(encoding='utf-8') if _cfg.DRAFT_PATH.exists() else "(пусто)"


@register_tool("append_notes")
class AppendNotes(BaseTool):
    description = ("Добавляет новый фрагмент знаний в research/notes.md "
                   "(подтема, arXiv id, ключевые факты).")
    parameters = [{"name": "content", "type": "string", "description": "Markdown-фрагмент", "required": True}]

    def call(self, params: str, **kwargs) -> str:
        ensure_dir()
        content = get_content(params)
        if _cfg.CFG.get("notes_strict", True):
            known, unknown = verify_ids_against_kb(content)
            if unknown:
                return (f"ОТКАЗ: в заметке упомянуты arxiv-id, которых НЕТ в kb.jsonl: "
                        f"{sorted(unknown)}. Это признак галлюцинации. "
                        "Сначала вызови hf_papers или kb_search по нужной теме, чтобы получить "
                        "ВЕРИФИЦИРОВАННЫЙ [arxiv-id], затем повторно append_notes. "
                        "Либо перепиши без этих id, цитируя только известные: "
                        f"{sorted(known) or '(пока нет)'}.")
        ids = extract_ids(content)
        if _cfg.CFG.get("strict_domain_gate", True) and ids:
            passed, reason, o_h, o_s = domain_gate(content)
            if not passed:
                from ..utils import extract_topic_keywords_tiered as _kw
                h_kws, s_kws = _kw(_cfg.PLAN_PATH.read_text(encoding="utf-8"))
                _log_rejected(content, ids, reason, h_kws, s_kws, o_h, o_s)
                hint = {
                    "no_core_hit": "ни одного core-термина из заголовка темы",
                    "weak_overlap": f"только 1 совпадение (нужно ≥2, из них ≥1 core). "
                                    f"Header hits: {sorted(o_h) or '∅'}, "
                                    f"seeds hits: {sorted(o_s) or '∅'}",
                }.get(reason, reason)
                return (f"ОТКАЗ (domain gate, {reason}): заметка с id {sorted(ids)} "
                        f"не относится к теме плана — {hint}. Это paper из смежного "
                        "домена, не лей в KB/notes. Записано в rejected.jsonl.")
        with _cfg.NOTES_PATH.open('a', encoding='utf-8') as f:
            f.write("\n\n" + content.strip() + "\n")
        return f"notes.md +{len(content)} симв (всего {_cfg.NOTES_PATH.stat().st_size} симв)"


@register_tool("read_notes")
class ReadNotes(BaseTool):
    description = "Читает накопленные заметки research/notes.md."
    parameters = []

    def call(self, params: str, **kwargs) -> str:
        if not _cfg.NOTES_PATH.exists():
            return "(заметок нет)"
        text = _cfg.NOTES_PATH.read_text(encoding='utf-8')
        return text[-20000:] if len(text) > 20000 else text


@register_tool("read_notes_focused")
class ReadNotesFocused(BaseTool):
    """Anti-drift фильтр: возвращает только блоки notes.md, релевантные focus-запросу."""
    description = ("Читает notes.md с anti-drift фильтром по focus-запросу. "
                   "Возвращает только блоки, релевантные focus (jaccard keyword overlap).")
    parameters = [
        {"name": "focus", "type": "string",
         "description": "Фокусная фраза (например, из [FOCUS] plan.md)", "required": True},
        {"name": "max_chars", "type": "integer",
         "description": "Лимит символов вывода (default 4000)", "required": False},
        {"name": "min_jaccard", "type": "number",
         "description": "Порог релевантности 0..1 (default 0.08)", "required": False},
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
            return "(заметок нет)"
        text = _cfg.NOTES_PATH.read_text(encoding='utf-8')
        if not text.strip():
            return "(заметки пустые)"

        focus_kws = _kws(focus)
        if not focus_kws:
            return text[-max_chars:]

        blocks = [b for b in re.split(r"\n\s*\n", text) if b.strip()]
        scored = [(b, _jaccard(_kws(b), focus_kws)) for b in blocks]
        relevant = [(b, s) for b, s in scored if s >= min_jac]
        relevant.sort(key=lambda x: x[1], reverse=True)

        if not relevant:
            return (f"(0/{len(blocks)} блоков прошли фильтр jaccard>={min_jac} "
                    f"для focus='{focus[:60]}'; рассмотри более широкий порог или read_notes)")

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
        header = (f"[focus-filter: {len(included)}/{len(blocks)} блоков, "
                  f"jaccard>={min_jac}"
                  f"{f', обрезано по max_chars={max_chars}' if truncated else ''}]\n\n")
        return header + "\n".join(included)


@register_tool("write_plan")
class WritePlan(BaseTool):
    description = "Перезаписывает план ресёрча в research/plan.md."
    parameters = [{"name": "content", "type": "string", "description": "Markdown план", "required": True}]

    def call(self, params: str, **kwargs) -> str:
        ensure_dir()
        content = get_content(params)
        _cfg.PLAN_PATH.write_text(content, encoding='utf-8')
        todos = content.count("[TODO]")
        done = content.count("[DONE]")
        return f"plan.md: {todos} TODO, {done} DONE"


@register_tool("read_plan")
class ReadPlan(BaseTool):
    description = "Читает текущий план research/plan.md."
    parameters = []

    def call(self, params: str, **kwargs) -> str:
        return _cfg.PLAN_PATH.read_text(encoding='utf-8') if _cfg.PLAN_PATH.exists() else "(плана нет)"


@register_tool("write_synthesis")
class WriteSynthesis(BaseTool):
    description = ("Записывает research/synthesis.md — НОВЫЕ мысли: "
                   "мосты, противоречия, пробелы, экстраполяции, testable гипотезы.")
    parameters = [{"name": "content", "type": "string", "description": "Markdown", "required": True}]

    def call(self, params: str, **kwargs) -> str:
        ensure_dir()
        content = get_content(params)
        _cfg.SYNTHESIS_PATH.write_text(content, encoding='utf-8')
        tags = sum(content.count(t) for t in ("[BRIDGE]", "[CONTRADICTION]", "[GAP]", "[EXTRAPOLATION]", "[REUSE]", "[TESTABLE]"))
        return f"synthesis.md ({len(content)} симв, {tags} инсайтов)"


@register_tool("read_synthesis")
class ReadSynthesis(BaseTool):
    description = "Читает research/synthesis.md — новые инсайты после анализа notes."
    parameters = []

    def call(self, params: str, **kwargs) -> str:
        return _cfg.SYNTHESIS_PATH.read_text(encoding='utf-8') if _cfg.SYNTHESIS_PATH.exists() else "(синтеза нет)"


@register_tool("append_lessons")
class AppendLessons(BaseTool):
    description = ("Записывает урок итерации в research/lessons.md (Reflexion-память). "
                   "Формат: что сработало, что нет, чего избегать.")
    parameters = [{"name": "content", "type": "string", "description": "Markdown", "required": True}]

    def call(self, params: str, **kwargs) -> str:
        ensure_dir()
        content = get_content(params)
        with _cfg.LESSONS_PATH.open('a', encoding='utf-8') as f:
            f.write("\n" + content.strip() + "\n")
        return f"lessons.md +{len(content)} симв"


@register_tool("read_lessons")
class ReadLessons(BaseTool):
    description = "Читает lessons.md — уроки прошлых итераций, что не повторять."
    parameters = []

    def call(self, params: str, **kwargs) -> str:
        return _cfg.LESSONS_PATH.read_text(encoding='utf-8') if _cfg.LESSONS_PATH.exists() else "(уроков ещё нет)"


@register_tool("read_querylog")
class ReadQueryLog(BaseTool):
    description = "Читает список уже выполненных hf_papers запросов — НЕ повторяй их."
    parameters = []

    def call(self, params: str, **kwargs) -> str:
        if not _cfg.QUERYLOG_PATH.exists():
            return "(запросов ещё не было)"
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
        "Добавляет атом знания в структурированную базу research/kb.jsonl. "
        "Вызывай каждый раз, когда узнал НОВЫЙ факт/репозиторий/авторов. "
        "Это параллельно append_notes, но для программного поиска. "
        "kind: 'paper' (для arxiv-id) или 'repo' (для owner/name). "
        "claim: 1-3 предложения что именно узнал."
    )
    parameters = [
        {"name": "id", "type": "string", "description": "arxiv-id или owner/name", "required": True},
        {"name": "kind", "type": "string", "description": "'paper' или 'repo'", "required": True},
        {"name": "claim", "type": "string", "description": "1-3 предложения о сути", "required": True},
        {"name": "title", "type": "string", "description": "заголовок/название", "required": False},
        {"name": "authors", "type": "string", "description": "авторы (для paper)", "required": False},
        {"name": "url", "type": "string", "description": "ссылка", "required": False},
        {"name": "stars", "type": "integer", "description": "звёзды (для repo)", "required": False},
        {"name": "lang", "type": "string", "description": "язык (для repo)", "required": False},
        {"name": "topic", "type": "string", "description": "текущий FOCUS", "required": False},
    ]

    def call(self, params: str, **kwargs) -> str:
        ensure_dir()
        p = parse_args(params)
        kind = (p.get("kind") or "").strip().lower()
        if kind not in ("paper", "repo"):
            return "ошибка: kind должен быть 'paper' или 'repo'"
        if not p.get("id") or not p.get("claim"):
            return "ошибка: id и claim обязательны"
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
        "Ищет в research/kb.jsonl уже известные факты/репо по запросу (BM25). "
        "Используй ПЕРЕД тем как делать новый hf_papers/github_search — вдруг мы уже это знаем. "
        "Возвращает top-k атомов с id, title и claim."
    )
    parameters = [
        {"name": "query", "type": "string", "description": "запрос на естественном языке", "required": True},
        {"name": "k", "type": "integer", "description": "сколько результатов (1-10)", "required": False},
    ]

    def call(self, params: str, **kwargs) -> str:
        p = parse_args(params)
        query = (p.get("query") or "").strip()
        if not query:
            return "ошибка: query обязателен"
        k = max(1, min(int(p.get("k", 5)), 10))
        hits = kb_mod.search(query, k=k)
        if not hits:
            return "(в kb пока ничего релевантного)"
        return kb_mod.format_atoms(hits)


# ── Структурированный план: явные мутации через тулы ───────────────────────
def _require_plan() -> plan_mod.Plan:
    p = plan_mod.load()
    if not p:
        raise RuntimeError("plan.json отсутствует — сначала запусти новый ресёрч")
    return p


@register_tool("plan_add_task")
class PlanAddTask(BaseTool):
    description = (
        "Добавить новую под-задачу в план. Используй когда в ходе ресёрча всплыла ВАЖНАЯ "
        "под-тема, которой нет в исходном плане (origin=emerged). parent опционален — "
        "если указан id существующей задачи, новая станет её потомком (2-й уровень дерева). "
        "Лимит: MAX_OPEN_TASKS=8 открытых задач, переполнение → ошибка."
    )
    parameters = [
        {"name": "title", "type": "string", "description": "Краткое название задачи", "required": True},
        {"name": "parent", "type": "string", "description": "id родителя (T1/T2.3/...) или пусто", "required": False},
        {"name": "why", "type": "string", "description": "Почему эту задачу добавляем (для аудита)", "required": False},
    ]

    def call(self, params: str, **kwargs) -> str:
        p = parse_args(params)
        title = (p.get("title") or "").strip()
        if not title:
            return "ошибка: title обязателен"
        parent = (p.get("parent") or "").strip() or None
        why = (p.get("why") or "").strip()
        try:
            plan = _require_plan()
            task = plan.add_task(title, parent=parent, origin="emerged",
                                 iter_=0, why=why or "модель решила расширить план")
            plan_mod.save(plan)
            return (f"добавлено: [{task.id}] {task.title}"
                    f"{' (parent=' + parent + ')' if parent else ''} — {len(plan.open_tasks())}/"
                    f"{plan_mod.MAX_OPEN_TASKS} open")
        except Exception as e:
            return f"ошибка: {e}"


@register_tool("plan_close_task")
class PlanCloseTask(BaseTool):
    description = (
        "Пометить задачу как выполненную (status=done). Используй когда по под-теме "
        "собраны достаточные факты в notes/kb. Укажи evidence — список ключей из kb "
        "или arxiv-id из notes, подтверждающих закрытие."
    )
    parameters = [
        {"name": "id", "type": "string", "description": "id задачи (например, T2 или T2.1)", "required": True},
        {"name": "evidence", "type": "string",
         "description": "Через запятую: ключи kb (kb:PaperId) или arxiv-id (2309.12345)", "required": False},
        {"name": "why", "type": "string", "description": "Краткое резюме что нашли", "required": False},
    ]

    def call(self, params: str, **kwargs) -> str:
        p = parse_args(params)
        tid = (p.get("id") or "").strip()
        if not tid:
            return "ошибка: id обязателен"
        evidence_raw = (p.get("evidence") or "").strip()
        evidence = [e.strip() for e in evidence_raw.split(",") if e.strip()] if evidence_raw else []
        why = (p.get("why") or "").strip()
        try:
            plan = _require_plan()
            t = plan.close_task(tid, iter_=0, evidence=evidence, why=why or "закрыта моделью")
            plan_mod.save(plan)
            return f"закрыто: [{t.id}] {t.title}  evidence={len(t.evidence_refs)}"
        except Exception as e:
            return f"ошибка: {e}"


@register_tool("plan_split_task")
class PlanSplitTask(BaseTool):
    description = (
        "Декомпозировать задачу на 2-4 подзадачи. Используй когда задача оказалась "
        "слишком широкой и её нельзя закрыть одной итерацией. Исходная задача "
        "становится контейнером (status=dropped + split-container), подзадачи — "
        "её детьми в дереве."
    )
    parameters = [
        {"name": "id", "type": "string", "description": "id задачи для декомпозиции", "required": True},
        {"name": "subtitles", "type": "string",
         "description": "Подзадачи через ' | ' (минимум 2)", "required": True},
        {"name": "why", "type": "string", "description": "Почему разделяем", "required": False},
    ]

    def call(self, params: str, **kwargs) -> str:
        p = parse_args(params)
        tid = (p.get("id") or "").strip()
        raw = (p.get("subtitles") or "").strip()
        if not tid or not raw:
            return "ошибка: id и subtitles обязательны"
        subs = [s.strip() for s in raw.split("|") if s.strip()]
        if len(subs) < 2:
            return "ошибка: нужно минимум 2 подзадачи (разделитель ' | ')"
        why = (p.get("why") or "").strip()
        try:
            plan = _require_plan()
            children = plan.split_task(tid, subs, iter_=0, why=why or "декомпозиция моделью")
            plan_mod.save(plan)
            return f"разбито [{tid}] → " + ", ".join(f"[{c.id}] {c.title[:40]}" for c in children)
        except Exception as e:
            return f"ошибка: {e}"


_wrap_module_tools(globals(), __name__)
