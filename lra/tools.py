"""Все tools для qwen-agent."""
from __future__ import annotations

import json
import re
from datetime import UTC

from qwen_agent.tools.base import BaseTool, register_tool

from . import cli as cli_run
from . import kb as kb_mod
from .config import (
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
from .logger import get_logger
from .memory import ensure_dir, is_similar_to_seen, log_query, seen_queries
from .utils import (
    extract_ids,
    extract_topic_keywords_tiered,
    get_content,
    keyword_set,
    normalize_query,
    parse_args,
)

log = get_logger("tools")


def verify_ids_against_kb(content: str) -> tuple[set[str], set[str]]:
    """Возвращает (known_ids, unknown_ids) — какие arxiv-id из content есть в kb.jsonl.

    Используется pre-append verifier'ом в AppendNotes: explorer не должен добавлять в notes
    факты с id, которых ни hf_papers, ни kb_search ни разу не возвращали в этой сессии.
    Это отсекает галлюцинации на этапе записи, а не на финальной валидации draft.md.
    """
    ids = extract_ids(content)
    if not ids:
        return set(), set()
    known_in_kb = {a.get("id", "") for a in kb_mod.load() if a.get("kind") == "paper"}
    return ids & known_in_kb, ids - known_in_kb


def domain_gate(content: str) -> tuple[bool, str, set[str], set[str]]:
    """Two-tier domain gate для AppendNotes и hf_papers kb auto-save.

    Правило: paper проходит ⇔ ≥2 совпадений с HEADER plan.md.
    HEADER = первая строка ('# Plan: ...') — это исходный topic пользователя
    и наиболее стабильный носитель core-терминов. Seeds из [Tn]-задач дрейфуют
    и содержат мусор ('support' в 'electronic support measures' даёт false
    positive на emotional-support-conversations paper) — используются только
    для диагностики причины отказа в rejected.jsonl, но не для прохода.

    Slow-start bypass: если в header <2 кандидатов — plan ещё generic, gate слеп.

    Returns (passed, reason, overlap_header, overlap_seeds). reason ∈
    {"no_plan", "slow_start", "no_core_hit", "weak_overlap", "passed"}.
    """
    if not PLAN_PATH.exists():
        return True, "no_plan", set(), set()
    header_kws, seed_kws = extract_topic_keywords_tiered(PLAN_PATH.read_text(encoding="utf-8"))
    if len(header_kws) < 2:
        return True, "slow_start", header_kws, set()
    # Собираем материал из kb для id, упомянутых в content.
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
    if len(o_header) < 2:
        return False, "weak_overlap", o_header, o_seeds
    return True, "passed", o_header, o_seeds


def gate_paper_for_kb(paper_id: str, title: str, abstract: str) -> tuple[bool, str]:
    """Облегчённый gate для hf_papers/kb_add ДО записи в kb.jsonl.

    Иначе explorer наполняет KB off-topic paper'ами (ComVo с auto-save остаётся
    в KB даже если AppendNotes его режет, т.к. KB-запись идёт РАНЬШЕ в hf_papers).
    Работает на сыром abstract (kb ещё не знает этот id).
    """
    if not CFG.get("strict_domain_gate", True) or not PLAN_PATH.exists():
        return True, "bypass"
    header_kws, _ = extract_topic_keywords_tiered(PLAN_PATH.read_text(encoding="utf-8"))
    if len(header_kws) < 2:
        return True, "slow_start"
    kws = keyword_set(f"{title} {abstract}")
    o_h = kws & header_kws
    if not o_h:
        return False, "no_core_hit"
    if len(o_h) < 2:
        return False, "weak_overlap"
    return True, "passed"


def _log_rejected(content: str, ids: set[str], reason: str,
                  header_kws: set[str], seed_kws: set[str],
                  o_header: set[str], o_seeds: set[str]) -> None:
    """Пишет отклонённую заметку в research/rejected.jsonl для анализа."""
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
    with REJECTED_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


@register_tool("hf_papers")
class HfPapers(BaseTool):
    description = ("Поиск научных статей на Hugging Face Papers через локальный `hf` CLI. "
                   "Возвращает id (arxiv), заголовок, авторов, дату и abstract.")
    parameters = [
        {"name": "query", "type": "string", "description": "Запрос", "required": True},
        {"name": "limit", "type": "integer", "description": "Сколько результатов (1-10)", "required": False},
    ]

    def call(self, params: str, **kwargs) -> str:
        p = parse_args(params)
        query = p["query"]
        limit = max(1, min(int(p.get("limit", 5)), 10))
        if normalize_query(query) in seen_queries():
            return (f"ОТКАЗ: запрос '{query}' уже выполнялся в этой сессии. "
                    "Переформулируй (другие ключевые слова, автор, год, техника) или читай read_notes.")
        # Fuzzy-блок: ловим семантические дубликаты через jaccard >= 0.75
        fuzzy = is_similar_to_seen(query)
        if fuzzy and fuzzy != query:
            return (f"ОТКАЗ: запрос '{query}' слишком похож на уже выполненный '{fuzzy}'. "
                    "Смени тему (другие термины, автор, год) или перейди к другому [TODO].")
        log_query(query)
        r = cli_run.run(
            ["hf", "papers", "search", query, "--limit", str(limit * 2), "--format", "json"],
            timeout=30,
        )
        if r.returncode == 127:
            return "ошибка: `hf` CLI не найден в PATH (pip install huggingface_hub[cli])"
        if r.returncode == 124:
            return "таймаут поиска hf_papers"
        if not r.ok:
            return f"ошибка: {r.stderr.strip()[:500]}"
        try:
            data = json.loads(r.stdout)
        except Exception:
            return f"не удалось распарсить JSON: {r.stdout[:300]}"
        if not data:
            return f"нет результатов: {query}"
        data.sort(key=lambda x: x.get("published_at", ""), reverse=True)
        # Freshness: оставляем только papers младше ARXIV_RECENT_DAYS. Если отсечение
        # выкидывает всё — откатываемся на полный список с пометкой в ответе (лучше
        # старое, чем ничего).
        from datetime import datetime, timedelta
        cutoff = (datetime.now(UTC) - timedelta(days=ARXIV_RECENT_DAYS)).date().isoformat()
        fresh = [x for x in data if (x.get("published_at") or "")[:10] >= cutoff]
        stale_note = ""
        if fresh:
            data = fresh[:limit]
        else:
            data = data[:limit]
            stale_note = (f"\n\n⚠️  все результаты старше {ARXIV_RECENT_DAYS // 365} лет "
                          f"(cutoff={cutoff}) — показаны как fallback")
        lines = []
        auto_saved = 0
        auto_filtered = 0
        for paper in data:
            authors = ", ".join(a["name"] for a in paper.get("authors", [])[:4])
            if len(paper.get("authors", [])) > 4:
                authors += " и др."
            title = " ".join(paper.get("title", "").split())
            summary = " ".join(paper.get("summary", "").split())[:800]
            date = paper.get("published_at", "")[:10]
            pid = paper.get("id", "")
            lines.append(f"[{pid}] {title}\n  {authors} · {date}\n  https://hf.co/papers/{pid}\n  {summary}")
            # Автосейв скелетного атома в KB — модель всё равно забывает kb_add вручную.
            # claim=summary (первые 400 симв) даст BM25-поиску на что опереться на следующих итерациях.
            # Domain gate до auto-save: off-topic paper'ы не должны осесть в kb.jsonl
            # даже если explorer их больше не упомянет. Это закрывает дыру, через
            # которую ComVo попадал в KB, несмотря на AppendNotes gate.
            if pid:
                passed, reason = gate_paper_for_kb(pid, title, summary)
                if not passed:
                    auto_filtered += 1
                    log.debug("kb auto-save skipped %s (%s)", pid, reason)
                    continue
                try:
                    kb_mod.add(kb_mod.Atom(
                        id=pid, kind="paper", topic=query,
                        title=title[:200], authors=authors[:200],
                        url=f"https://hf.co/papers/{pid}",
                        claim=summary[:400],
                    ))
                    auto_saved += 1
                except Exception as e:
                    log.debug("kb auto-save paper failed %s: %s", pid, e)
        footer_parts = []
        if auto_saved:
            footer_parts.append(f"📥 авто-сохранено в kb: {auto_saved}")
        if auto_filtered:
            footer_parts.append(f"🚫 отфильтровано domain gate: {auto_filtered}")
        footer = f"\n\n({', '.join(footer_parts)})" if footer_parts else ""
        return "\n\n".join(lines) + footer + stale_note


@register_tool("compact_notes")
class CompactNotes(BaseTool):
    description = "Перезаписывает research/notes.md сжатым содержанием."
    parameters = [{"name": "content", "type": "string", "description": "Сжатый markdown", "required": True}]

    def call(self, params: str, **kwargs) -> str:
        ensure_dir()
        content = get_content(params)
        old = NOTES_PATH.stat().st_size if NOTES_PATH.exists() else 0
        NOTES_PATH.write_text(content, encoding='utf-8')
        return f"notes.md: {old} → {len(content)} симв (сжато в {max(1, old // max(1, len(content)))}x)"


@register_tool("write_draft")
class WriteDraft(BaseTool):
    description = "Перезаписывает черновик research/draft.md с нуля."
    parameters = [{"name": "content", "type": "string", "description": "Markdown", "required": True}]

    def call(self, params: str, **kwargs) -> str:
        ensure_dir()
        content = get_content(params)
        DRAFT_PATH.write_text(content, encoding='utf-8')
        return f"draft.md сохранён ({len(content)} симв)"


@register_tool("append_draft")
class AppendDraft(BaseTool):
    description = ("Добавляет секцию в конец research/draft.md. "
                   "Предпочтительно write_draft+много append_draft, чем один огромный write_draft.")
    parameters = [{"name": "content", "type": "string", "description": "Markdown-секция", "required": True}]

    def call(self, params: str, **kwargs) -> str:
        ensure_dir()
        content = get_content(params)
        with DRAFT_PATH.open('a', encoding='utf-8') as f:
            f.write("\n\n" + content.strip() + "\n")
        return f"draft.md +{len(content)} симв (всего {DRAFT_PATH.stat().st_size})"


@register_tool("read_draft")
class ReadDraft(BaseTool):
    description = "Читает research/draft.md."
    parameters = []

    def call(self, params: str, **kwargs) -> str:
        return DRAFT_PATH.read_text(encoding='utf-8') if DRAFT_PATH.exists() else "(пусто)"


@register_tool("append_notes")
class AppendNotes(BaseTool):
    description = ("Добавляет новый фрагмент знаний в research/notes.md "
                   "(подтема, arXiv id, ключевые факты).")
    parameters = [{"name": "content", "type": "string", "description": "Markdown-фрагмент", "required": True}]

    def call(self, params: str, **kwargs) -> str:
        ensure_dir()
        content = get_content(params)
        # Pre-append verifier: блокируем запись, если в content есть arxiv-id,
        # которых нет в kb.jsonl (т.е. explorer их не находил через hf_papers/kb_search).
        # Это отсекает галлюцинации id на этапе записи.
        if CFG.get("notes_strict", True):
            known, unknown = verify_ids_against_kb(content)
            if unknown:
                return (f"ОТКАЗ: в заметке упомянуты arxiv-id, которых НЕТ в kb.jsonl: "
                        f"{sorted(unknown)}. Это признак галлюцинации. "
                        "Сначала вызови hf_papers или kb_search по нужной теме, чтобы получить "
                        "ВЕРИФИЦИРОВАННЫЙ [arxiv-id], затем повторно append_notes. "
                        "Либо перепиши без этих id, цитируя только известные: "
                        f"{sorted(known) or '(пока нет)'}.")
        # Domain gate: если заметка содержит arxiv-id, проверяем что paper из
        # нашего домена (plan.md topic). Reflection-заметки без id пропускаем.
        # Ловит ComVo-style citation laundering ДО попадания в notes.md.
        ids = extract_ids(content)
        if CFG.get("strict_domain_gate", True) and ids:
            passed, reason, o_h, o_s = domain_gate(content)
            if not passed:
                from .utils import extract_topic_keywords_tiered as _kw
                h_kws, s_kws = _kw(PLAN_PATH.read_text(encoding="utf-8"))
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
        with NOTES_PATH.open('a', encoding='utf-8') as f:
            f.write("\n\n" + content.strip() + "\n")
        return f"notes.md +{len(content)} симв (всего {NOTES_PATH.stat().st_size} симв)"


@register_tool("read_notes")
class ReadNotes(BaseTool):
    description = "Читает накопленные заметки research/notes.md."
    parameters = []

    def call(self, params: str, **kwargs) -> str:
        if not NOTES_PATH.exists():
            return "(заметок нет)"
        text = NOTES_PATH.read_text(encoding='utf-8')
        return text[-20000:] if len(text) > 20000 else text


@register_tool("read_notes_focused")
class ReadNotesFocused(BaseTool):
    """Anti-drift фильтр: возвращает только блоки notes.md, релевантные focus-запросу.

    Блоки делятся по двойному переводу строки. Для каждого блока считается jaccard между
    keyword_set(block) и keyword_set(focus); блоки < min_jaccard отбрасываются, остальные
    сортируются по релевантности и обрезаются по max_chars. Решает проблему semantic drift
    в workspace-памяти (накопление нерелевантных фактов по мере роста notes).
    """
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
        from .utils import jaccard as _jaccard
        from .utils import keyword_set as _kws
        args = parse_args(params)
        focus = (args.get("focus") or "").strip()
        if not focus:
            return "(focus is required)"
        max_chars = int(args.get("max_chars") or 4000)
        min_jac = float(args.get("min_jaccard") or 0.08)

        if not NOTES_PATH.exists():
            return "(заметок нет)"
        text = NOTES_PATH.read_text(encoding='utf-8')
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

        # Сначала собираем фактически вмещающиеся блоки, потом формируем header
        # (иначе header обещает N блоков, а в budget вмещается M<N — LLM думает
        # что получила все релевантные блоки)
        placeholder_header_len = 80  # консервативная оценка для budget
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
        PLAN_PATH.write_text(content, encoding='utf-8')
        todos = content.count("[TODO]")
        done = content.count("[DONE]")
        return f"plan.md: {todos} TODO, {done} DONE"


@register_tool("read_plan")
class ReadPlan(BaseTool):
    description = "Читает текущий план research/plan.md."
    parameters = []

    def call(self, params: str, **kwargs) -> str:
        return PLAN_PATH.read_text(encoding='utf-8') if PLAN_PATH.exists() else "(плана нет)"


@register_tool("write_synthesis")
class WriteSynthesis(BaseTool):
    description = ("Записывает research/synthesis.md — НОВЫЕ мысли: "
                   "мосты, противоречия, пробелы, экстраполяции, testable гипотезы.")
    parameters = [{"name": "content", "type": "string", "description": "Markdown", "required": True}]

    def call(self, params: str, **kwargs) -> str:
        ensure_dir()
        content = get_content(params)
        SYNTHESIS_PATH.write_text(content, encoding='utf-8')
        tags = sum(content.count(t) for t in ("[BRIDGE]", "[CONTRADICTION]", "[GAP]", "[EXTRAPOLATION]", "[REUSE]", "[TESTABLE]"))
        return f"synthesis.md ({len(content)} симв, {tags} инсайтов)"


@register_tool("read_synthesis")
class ReadSynthesis(BaseTool):
    description = "Читает research/synthesis.md — новые инсайты после анализа notes."
    parameters = []

    def call(self, params: str, **kwargs) -> str:
        return SYNTHESIS_PATH.read_text(encoding='utf-8') if SYNTHESIS_PATH.exists() else "(синтеза нет)"


@register_tool("append_lessons")
class AppendLessons(BaseTool):
    description = ("Записывает урок итерации в research/lessons.md (Reflexion-память). "
                   "Формат: что сработало, что нет, чего избегать.")
    parameters = [{"name": "content", "type": "string", "description": "Markdown", "required": True}]

    def call(self, params: str, **kwargs) -> str:
        ensure_dir()
        content = get_content(params)
        with LESSONS_PATH.open('a', encoding='utf-8') as f:
            f.write("\n" + content.strip() + "\n")
        return f"lessons.md +{len(content)} симв"


@register_tool("read_lessons")
class ReadLessons(BaseTool):
    description = "Читает lessons.md — уроки прошлых итераций, что не повторять."
    parameters = []

    def call(self, params: str, **kwargs) -> str:
        return LESSONS_PATH.read_text(encoding='utf-8') if LESSONS_PATH.exists() else "(уроков ещё нет)"


@register_tool("read_querylog")
class ReadQueryLog(BaseTool):
    description = "Читает список уже выполненных hf_papers запросов — НЕ повторяй их."
    parameters = []

    def call(self, params: str, **kwargs) -> str:
        if not QUERYLOG_PATH.exists():
            return "(запросов ещё не было)"
        text = QUERYLOG_PATH.read_text(encoding='utf-8')
        lines, seen = [], set()
        for ln in reversed(text.splitlines()):
            if ln.strip() and ln not in seen:
                seen.add(ln)
                lines.append(ln)
            if len(lines) >= 30:
                break
        return "\n".join(reversed(lines))


@register_tool("github_search")
class GithubSearch(BaseTool):
    description = (
        "Поиск по GitHub через официальный `gh` CLI. "
        "Используй для нахождения РЕАЛИЗАЦИЙ и ДАТАСЕТОВ к бумагам из hf_papers: "
        "нашёл метод в абстракте → ищи его репозиторий здесь. "
        "type='repos' — репозитории (по умолчанию), type='code' — поиск кода.\n"
        "ВАЖНО: query должен быть КОРОТКИМ — 2-4 ключевых слова. "
        "НЕ пиши внутрь query 'stars:>=10' или 'language:python' — для этого используй "
        "отдельные параметры min_stars и language."
    )
    parameters = [
        {"name": "query", "type": "string",
         "description": "2-4 ключевых слова БЕЗ qualifiers", "required": True},
        {"name": "type", "type": "string",
         "description": "'repos' (по умолчанию) или 'code'", "required": False},
        {"name": "limit", "type": "integer",
         "description": "Кол-во результатов 1-10 (по умолчанию 5)", "required": False},
        {"name": "min_stars", "type": "integer",
         "description": "минимум звёзд (для type=repos)", "required": False},
        {"name": "language", "type": "string",
         "description": "язык программирования (для type=repos)", "required": False},
    ]

    # Паттерн для чистки инлайновых GitHub-qualifiers в query — модель любит их лепить,
    # хотя gh CLI принимает их отдельными флагами.
    _QUALIFIER_RE = None  # lazy-инициализация в call()

    @staticmethod
    def _parse_qualifiers(raw_query: str) -> tuple[str, int | None, str | None]:
        """Достаёт `stars:>=N` и `language:X` из query, возвращает очищенный query и флаги.

        Извлечено из GithubSearch.call чтобы снизить CCN основной функции.
        Поведение идентично: regex тот же (lazy-init на классе), итерация та же,
        валидация та же.
        """
        import re
        if GithubSearch._QUALIFIER_RE is None:
            GithubSearch._QUALIFIER_RE = re.compile(
                r"\b(stars|language|lang|forks|size|pushed|created|user|org|topic|in|is):[\w:>=<.+/-]+",
                re.IGNORECASE,
            )
        extracted_min_stars: int | None = None
        extracted_language: str | None = None
        for m in GithubSearch._QUALIFIER_RE.finditer(raw_query):
            tok = m.group(0).lower()
            if tok.startswith("stars:"):
                val = tok.split(":", 1)[1].lstrip(">=<")
                try:
                    extracted_min_stars = int(val)
                except ValueError:
                    pass
            elif tok.startswith(("language:", "lang:")):
                extracted_language = tok.split(":", 1)[1]
        cleaned_query = GithubSearch._QUALIFIER_RE.sub("", raw_query).strip()
        # схлопываем двойные пробелы
        cleaned_query = " ".join(cleaned_query.split())
        return cleaned_query, extracted_min_stars, extracted_language

    @staticmethod
    def _format_repo_results(data: list, cleaned_query: str) -> tuple[list[str], int]:
        """Форматирует repos-ответ от `gh search repos` + автосейв в kb (только ≥10★).

        Извлечено из GithubSearch.call. Побочный эффект (kb_mod.add) сохранён —
        критерий автосейва (name!="?" и stars>=10) не меняется.
        """
        lines: list[str] = []
        auto_saved = 0
        for item in data:
            name = item.get("fullName", "?")
            url = item.get("url", "")
            desc = (item.get("description") or "").strip()[:120]
            stars = item.get("stargazersCount", 0)
            lang = item.get("language") or ""
            pushed = (item.get("pushedAt") or "")[:10]
            lines.append(
                f"★{stars:>6}  [{name}]({url})  {lang}  updated:{pushed}\n"
                f"         {desc}"
            )
            # Автосейв: только репо с ≥10 звёзд — отсекает мусор и форки без описания.
            if name != "?" and stars >= 10:
                try:
                    kb_mod.add(kb_mod.Atom(
                        id=name, kind="repo", topic=cleaned_query,
                        title=name, url=url,
                        stars=int(stars or 0), lang=lang,
                        claim=desc or f"{lang} репозиторий, {stars}★",
                    ))
                    auto_saved += 1
                except Exception as e:
                    log.debug("kb auto-save repo failed %s: %s", name, e)
        return lines, auto_saved

    @staticmethod
    def _format_code_results(data: list) -> list[str]:
        """Форматирует code-ответ от `gh search code`. Без побочных эффектов."""
        lines: list[str] = []
        for item in data:
            path = item.get("path", "")
            url = item.get("url", "")
            repo = (item.get("repository") or {}).get("fullName", "?")
            lines.append(f"[{repo}] {path}\n  {url}")
        return lines

    def call(self, params: str, **kwargs) -> str:
        p = parse_args(params)
        raw_query = p.get("query", "") or ""
        if not raw_query:
            return "ошибка: query обязателен"
        search_type = p.get("type", "repos").strip().lower()
        if search_type not in ("repos", "code"):
            search_type = "repos"
        limit = max(1, min(int(p.get("limit", 5)), 10))

        # Вынимаем qualifiers из query → отдельные флаги
        cleaned_query, extracted_min_stars, extracted_language = self._parse_qualifiers(raw_query)
        if not cleaned_query:
            return "ошибка: после удаления qualifiers query пустой — пиши 2-4 ключевых слова"

        # Жёсткий ранний reject длинных фраз — gh search почти всегда даёт 0
        # на 6+ слов, а модель упрямо повторяет их с вариациями.
        word_count = len([w for w in cleaned_query.split() if len(w) >= 2])
        if word_count > MAX_GITHUB_QUERY_WORDS:
            words = cleaned_query.split()
            short_hint = " ".join(words[:3])
            return (f"ОТКАЗ: query '{cleaned_query}' слишком длинный ({word_count} слов). "
                    f"github search плохо работает с длинными фразами — сократи до 2-3 ключевых "
                    f"терминов (попробуй: '{short_hint}') ИЛИ откажись от github_search для этой "
                    "подтемы, если она чисто теоретическая.")
        # Объединяем явные параметры и извлечённые из query (явные приоритетнее)
        min_stars = p.get("min_stars")
        if min_stars is None and extracted_min_stars is not None:
            min_stars = extracted_min_stars
        language = (p.get("language") or "").strip() or extracted_language or ""

        # Dedup — по ЧИСТОМУ query + type + language (без min_stars, чтобы не плодить лишние записи)
        dedup_key = f"gh-{search_type}: {cleaned_query}"
        if language:
            dedup_key += f" lang={language}"
        if normalize_query(dedup_key) in seen_queries():
            return (f"ОТКАЗ: GitHub-запрос '{cleaned_query}' (type={search_type}"
                    f"{', lang=' + language if language else ''}) уже делался. "
                    "Переформулируй или читай read_notes.")
        fuzzy = is_similar_to_seen(dedup_key)
        if fuzzy and fuzzy != dedup_key:
            # Конкретный совет вместо «смени тему». kb_search + read_notes почти всегда
            # лучше пятой переформулировки одного и того же запроса.
            return (f"ОТКАЗ: GitHub-запрос '{cleaned_query}' слишком похож на '{fuzzy}'. "
                    f"Действия по приоритету: (1) kb_search '{cleaned_query}' — возможно мы уже "
                    f"это искали и атом лежит в kb; (2) read_notes — поищи что уже записано; "
                    f"(3) если всё-таки нужен github — выбери другой аспект [FOCUS] "
                    f"(конкретный метод/датасет, а не саму тему); (4) plan_close_task текущий "
                    f"[FOCUS] с evidence из notes и переходи к следующему [TODO].")
        log_query(dedup_key)

        if search_type == "repos":
            fields = "fullName,url,description,stargazersCount,language,pushedAt"
        else:
            fields = "path,url,repository"

        from datetime import datetime, timedelta
        recent_cutoff = (datetime.now(UTC)
                         - timedelta(days=GITHUB_RECENT_DAYS)).date().isoformat()

        def _build_args(with_freshness: bool) -> list[str]:
            args = ["gh", "search", search_type, cleaned_query,
                    "--limit", str(limit), "--json", fields]
            if search_type == "repos":
                if min_stars is not None:
                    try:
                        args += ["--stars", f">={int(min_stars)}"]
                    except (TypeError, ValueError):
                        pass
                if language:
                    args += ["--language", language]
                if with_freshness:
                    # gh search repos --updated принимает GitHub qualifier syntax
                    args += ["--updated", f">={recent_cutoff}"]
            return args

        # Сначала ищем в свежем окне; если 0 — повторяем без фильтра с пометкой.
        stale_note = ""
        r = cli_run.run(_build_args(with_freshness=(search_type == "repos")), timeout=20)
        if r.returncode == 127:
            return "ошибка: `gh` CLI не найден в PATH (установи: brew install gh)"
        if r.returncode == 124:
            return "таймаут поиска GitHub"
        if not r.ok:
            err = r.stderr.strip()[:400]
            if "authentication" in err.lower() or "auth" in err.lower() or "login" in err.lower():
                return "ошибка gh: нужна авторизация. Выполни в терминале: `gh auth login`"
            return f"ошибка gh: {err}"
        try:
            data = json.loads(r.stdout)
        except Exception:
            return f"не удалось распарсить JSON: {r.stdout[:300]}"
        # Если freshness-фильтр выкинул всё — повторяем без него и помечаем.
        if not data and search_type == "repos":
            r2 = cli_run.run(_build_args(with_freshness=False), timeout=20)
            if r2.ok:
                try:
                    data2 = json.loads(r2.stdout)
                except Exception:
                    data2 = []
                if data2:
                    data = data2
                    stale_note = (f"\n\n⚠️  свежих репо (updated >= {recent_cutoff}) нет — "
                                  f"показаны более старые результаты как fallback")
        if not data:
            hint = ""
            if len(cleaned_query.split()) >= 4:
                hint = " — попробуй СОКРАТИТЬ запрос до 2-3 ключевых слов"
            elif min_stars and int(min_stars) >= 50:
                hint = f" — попробуй снизить min_stars (текущий={min_stars})"
            elif language:
                hint = f" — попробуй БЕЗ language='{language}'"
            return f"нет результатов на GitHub: '{cleaned_query}'{hint}"

        if search_type == "repos":
            lines, auto_saved = self._format_repo_results(data, cleaned_query)
        else:
            lines = self._format_code_results(data)
            auto_saved = 0

        footer = f"\n\n(📥 авто-сохранено в kb: {auto_saved})" if auto_saved else ""
        return "\n\n".join(lines) + footer + stale_note

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
# Единственный легитимный способ модели менять структуру plan.json. Каждый
# вызов пишется в revisions как аудит-лог.
from . import plan as plan_mod  # noqa: E402


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


# ── Унифицированный [TOOL_CALL] лог для всех тулов ─────────────────────────
def _wrap_with_logging(cls):
    """Оборачивает `cls.call`, чтобы каждый вызов попадал в лог единообразно."""
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
        # Loop detection: блокируем N-ный подряд идентичный вызов.
        # See lra/tool_tracker.py для мотивации (run.log compact_notes x16).
        try:
            from lra import tool_tracker
            allowed, n = tool_tracker.check_call(tool_name, params)
            if not allowed:
                log.warning("[TOOL_LOOP] %s заблокирован: %d-й идентичный вызов подряд", tool_name, n)
                return (
                    f"ошибка: loop detected — {tool_name} вызван {n} раз подряд "
                    f"с одинаковыми params. Смени стратегию: попробуй другой "
                    f"tool, измени params или перейди к следующему шагу плана."
                )
        except Exception as _loop_err:
            # tracker-баг не должен ронять tool execution
            log.debug("tool_tracker error: %s", _loop_err)
        try:
            return orig(self, params, **kwargs)
        except Exception as e:
            log.warning("[TOOL_ERR]  %s: %s", tool_name, e)
            raise

    call._tool_logged = True  # type: ignore[attr-defined]
    cls.call = call
    return cls


# Применяем ко всем тулам, определённым в этом модуле
for _name, _obj in list(globals().items()):
    if isinstance(_obj, type) and issubclass(_obj, BaseTool) and _obj is not BaseTool:
        if _obj.__module__ == __name__:
            _wrap_with_logging(_obj)

