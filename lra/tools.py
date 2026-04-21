"""Все tools для qwen-agent."""
from __future__ import annotations

import json
import subprocess
import sys

from qwen_agent.tools.base import BaseTool, register_tool

from . import cli as cli_run
from . import kb as kb_mod
from .config import DRAFT_PATH, LESSONS_PATH, NOTES_PATH, PLAN_PATH, QUERYLOG_PATH, SYNTHESIS_PATH
from .logger import get_logger
from .memory import ensure_dir, is_similar_to_seen, log_query, seen_queries
from .utils import normalize_query, parse_args

log = get_logger("tools")

_SANDBOX_PRELUDE = r"""
import resource, sys, builtins
# CPU 5s, address space 512MB, no fork
resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
try: resource.setrlimit(resource.RLIMIT_AS, (512*1024*1024, 512*1024*1024))
except Exception: pass
try: resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
except Exception: pass
# блок сети через monkey-patch socket
import socket
def _blocked(*a, **k): raise OSError('сеть запрещена в песочнице')
socket.socket = _blocked  # type: ignore
socket.create_connection = _blocked  # type: ignore
# запись только в /tmp
_orig_open = builtins.open
def _safe_open(f, mode='r', *a, **k):
    if any(m in mode for m in ('w','a','x','+')) and not str(f).startswith('/tmp/'):
        raise PermissionError('запись разрешена только в /tmp/')
    return _orig_open(f, mode, *a, **k)
builtins.open = _safe_open
CODE = sys.stdin.read()
exec(compile(CODE, '<tool>', 'exec'), {'__name__':'__main__'})
"""


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
        data = data[:limit]
        lines = []
        auto_saved = 0
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
            if pid:
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
        footer = f"\n\n(📥 авто-сохранено в kb: {auto_saved})" if auto_saved else ""
        return "\n\n".join(lines) + footer


@register_tool("run_python")
class RunPython(BaseTool):
    description = ("Выполняет Python-код в изолированной подпроцессной песочнице "
                   "(без сети, 5с CPU, 512MB RAM). Возвращает stdout+stderr.")
    parameters = [{"name": "code", "type": "string", "description": "Код", "required": True}]

    def call(self, params: str, **kwargs) -> str:
        code = parse_args(params)["code"]
        try:
            r = subprocess.run(
                [sys.executable, "-I", "-c", _SANDBOX_PRELUDE],
                input=code, capture_output=True, text=True, timeout=10,
            )
        except subprocess.TimeoutExpired:
            return "таймаут (>10с)"
        out = (r.stdout + (r.stderr and f"\n[stderr]\n{r.stderr}" or "")).strip()
        return (out or "(вывода нет)")[:5000]


@register_tool("compact_notes")
class CompactNotes(BaseTool):
    description = "Перезаписывает research/notes.md сжатым содержанием."
    parameters = [{"name": "content", "type": "string", "description": "Сжатый markdown", "required": True}]

    def call(self, params: str, **kwargs) -> str:
        ensure_dir()
        content = parse_args(params)["content"]
        old = NOTES_PATH.stat().st_size if NOTES_PATH.exists() else 0
        NOTES_PATH.write_text(content, encoding='utf-8')
        return f"notes.md: {old} → {len(content)} симв (сжато в {max(1, old // max(1, len(content)))}x)"


@register_tool("write_draft")
class WriteDraft(BaseTool):
    description = "Перезаписывает черновик research/draft.md с нуля."
    parameters = [{"name": "content", "type": "string", "description": "Markdown", "required": True}]

    def call(self, params: str, **kwargs) -> str:
        ensure_dir()
        content = parse_args(params)["content"]
        DRAFT_PATH.write_text(content, encoding='utf-8')
        return f"draft.md сохранён ({len(content)} симв)"


@register_tool("append_draft")
class AppendDraft(BaseTool):
    description = ("Добавляет секцию в конец research/draft.md. "
                   "Предпочтительно write_draft+много append_draft, чем один огромный write_draft.")
    parameters = [{"name": "content", "type": "string", "description": "Markdown-секция", "required": True}]

    def call(self, params: str, **kwargs) -> str:
        ensure_dir()
        content = parse_args(params)["content"]
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
        content = parse_args(params)["content"]
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


@register_tool("write_plan")
class WritePlan(BaseTool):
    description = "Перезаписывает план ресёрча в research/plan.md."
    parameters = [{"name": "content", "type": "string", "description": "Markdown план", "required": True}]

    def call(self, params: str, **kwargs) -> str:
        ensure_dir()
        content = parse_args(params)["content"]
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
        content = parse_args(params)["content"]
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
        content = parse_args(params)["content"]
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

    def call(self, params: str, **kwargs) -> str:
        import re
        if GithubSearch._QUALIFIER_RE is None:
            GithubSearch._QUALIFIER_RE = re.compile(
                r"\b(stars|language|lang|forks|size|pushed|created|user|org|topic|in|is):[\w:>=<.+/-]+",
                re.IGNORECASE,
            )
        p = parse_args(params)
        raw_query = p.get("query", "") or ""
        if not raw_query:
            return "ошибка: query обязателен"
        search_type = p.get("type", "repos").strip().lower()
        if search_type not in ("repos", "code"):
            search_type = "repos"
        limit = max(1, min(int(p.get("limit", 5)), 10))

        # Вынимаем qualifiers из query → отдельные флаги
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
        if not cleaned_query:
            return "ошибка: после удаления qualifiers query пустой — пиши 2-4 ключевых слова"

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
            return (f"ОТКАЗ: GitHub-запрос '{cleaned_query}' слишком похож на '{fuzzy}'. "
                    "Смени тему или переходи к следующему [TODO] — ты крутишься по одному и тому же.")
        log_query(dedup_key)

        if search_type == "repos":
            fields = "fullName,url,description,stargazersCount,language,pushedAt"
        else:
            fields = "path,url,repository"

        cli_args = ["gh", "search", search_type, cleaned_query,
                    "--limit", str(limit), "--json", fields]
        if search_type == "repos":
            if min_stars is not None:
                try:
                    cli_args += ["--stars", f">={int(min_stars)}"]
                except (TypeError, ValueError):
                    pass
            if language:
                cli_args += ["--language", language]

        r = cli_run.run(cli_args, timeout=20)
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
        if not data:
            hint = ""
            if len(cleaned_query.split()) >= 4:
                hint = " — попробуй СОКРАТИТЬ запрос до 2-3 ключевых слов"
            elif min_stars and int(min_stars) >= 50:
                hint = f" — попробуй снизить min_stars (текущий={min_stars})"
            elif language:
                hint = f" — попробуй БЕЗ language='{language}'"
            return f"нет результатов на GitHub: '{cleaned_query}'{hint}"

        lines = []
        auto_saved = 0
        if search_type == "repos":
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
        else:  # code
            for item in data:
                path = item.get("path", "")
                url = item.get("url", "")
                repo = (item.get("repository") or {}).get("fullName", "?")
                lines.append(f"[{repo}] {path}\n  {url}")

        footer = f"\n\n(📥 авто-сохранено в kb: {auto_saved})" if auto_saved else ""
        return "\n\n".join(lines) + footer


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

