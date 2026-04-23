"""Search-тулы: hf_papers, arxiv_search, github_search.

Делят общие helpers для HTTP-фетчинга arXiv (`_fetch_text`, `_parse_arxiv_feed`),
автосохраняют результаты в KB через `gate_paper_for_kb`, используют общий
querylog/seen_queries dedup.

ВАЖНО: helpers вызываются через qualified `_helpers.X`, чтобы тесты могли
monkeypatch-ить `lra.tools._helpers._fetch_text` (иначе import-time binding
заморозит оригинал в этом модуле).
"""
from __future__ import annotations

import json
import re
from datetime import UTC
from urllib.parse import urlencode

from qwen_agent.tools.base import BaseTool, register_tool

from .. import cli as cli_run
from .. import config as _cfg
from .. import kb as kb_mod
from ..memory import is_similar_to_seen, log_query, seen_queries
from ..utils import normalize_query, parse_args
from . import _helpers


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
        from datetime import datetime, timedelta
        cutoff = (datetime.now(UTC) - timedelta(days=_cfg.ARXIV_RECENT_DAYS)).date().isoformat()
        fresh = [x for x in data if (x.get("published_at") or "")[:10] >= cutoff]
        stale_note = ""
        if fresh:
            data = fresh[:limit]
        else:
            data = data[:limit]
            stale_note = (f"\n\n⚠️  все результаты старше {_cfg.ARXIV_RECENT_DAYS // 365} лет "
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
            if pid:
                passed, reason, o_h, header_kws = _helpers.gate_paper_for_kb(pid, title, summary)
                if not passed:
                    auto_filtered += 1
                    _helpers._log_kb_rejected(pid, title, reason, o_h, header_kws, source="hf_papers")
                    _helpers.log.debug("kb auto-save skipped %s (%s)", pid, reason)
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
                    _helpers.log.debug("kb auto-save paper failed %s: %s", pid, e)
        footer_parts = []
        if auto_saved:
            footer_parts.append(f"📥 авто-сохранено в kb: {auto_saved}")
        if auto_filtered:
            footer_parts.append(f"🚫 отфильтровано domain gate: {auto_filtered}")
        footer = f"\n\n({', '.join(footer_parts)})" if footer_parts else ""
        return "\n\n".join(lines) + footer + stale_note


@register_tool("arxiv_search")
class ArxivSearch(BaseTool):
    description = (
        "Fallback-поиск статей через arXiv API. Используй когда hf_papers пуст, "
        "устарел или не нашёл нужную подтему. Возвращает arxiv-id, заголовок, "
        "авторов, дату и abstract.\n"
        "ВАЖНО: для узких доменов (signal processing, info theory, crypto) указывай "
        "categories=['eess.SP','cs.IT','cs.CR'] — это сильно повышает релевантность "
        "и режет шум из NLP/CV. Список arxiv категорий: arxiv.org/category_taxonomy."
    )
    parameters = [
        {"name": "query", "type": "string", "description": "Запрос", "required": True},
        {"name": "limit", "type": "integer", "description": "Сколько результатов (1-10)", "required": False},
        {"name": "categories", "type": "array",
         "description": "Опц. arxiv-категории (eess.SP, cs.IT, cs.CR, cs.LG, ...). "
                        "Если задано — поиск ограничен этими категориями (OR).",
         "required": False},
    ]

    def call(self, params: str, **kwargs) -> str:
        p = parse_args(params)
        query = (p.get("query") or "").strip()
        if not query:
            return "ошибка: query обязателен"
        limit = max(1, min(int(p.get("limit", 5)), 10))
        # Категории: список или CSV-строка
        raw_cats = p.get("categories") or []
        if isinstance(raw_cats, str):
            raw_cats = [c.strip() for c in raw_cats.split(",") if c.strip()]
        # Sanitize: только токены формата letters[.letters], не длиннее 20 симв
        import re as _re
        cats = [c for c in raw_cats
                if isinstance(c, str) and _re.fullmatch(r"[a-z\-]+(?:\.[A-Za-z\-]+)?", c)
                and len(c) <= 25]
        cat_suffix = " [cats=" + ",".join(cats) + "]" if cats else ""
        dedup_key = f"arxiv: {query}{cat_suffix}"
        if normalize_query(dedup_key) in seen_queries():
            return (f"ОТКАЗ: запрос '{query}' уже выполнялся через arxiv_search в этой сессии. "
                    "Переформулируй или читай read_notes/kb_search.")
        fuzzy = is_similar_to_seen(dedup_key)
        if fuzzy and fuzzy != dedup_key:
            return (f"ОТКАЗ: arxiv_search '{query}' слишком похож на уже выполненный '{fuzzy}'. "
                    "Смени термины, автора или год.")
        log_query(dedup_key)

        if cats:
            cat_clause = "(" + " OR ".join(f"cat:{c}" for c in cats) + ")"
            search_query = f"{cat_clause} AND all:{query}"
        else:
            search_query = f"all:{query}"
        url = "http://export.arxiv.org/api/query?" + urlencode({
            "search_query": search_query,
            "start": 0,
            "max_results": limit * 2,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        })
        try:
            xml_text = _helpers._fetch_text(url, timeout=20)
        except TimeoutError:
            return "таймаут поиска arxiv_search"
        except Exception as exc:
            return f"ошибка arxiv_search: {str(exc)[:300]}"

        try:
            data = _helpers._parse_arxiv_feed(xml_text)
        except Exception:
            return f"не удалось распарсить arXiv feed: {xml_text[:300]}"
        if not data:
            return f"нет результатов arxiv_search: {query}"

        from datetime import datetime, timedelta
        cutoff = (datetime.now(UTC) - timedelta(days=_cfg.ARXIV_RECENT_DAYS)).date().isoformat()
        fresh = [x for x in data if (x.get("published_at") or "")[:10] >= cutoff]
        stale_note = ""
        if fresh:
            data = fresh[:limit]
        else:
            data = data[:limit]
            stale_note = (f"\n\n⚠️  все результаты старше {_cfg.ARXIV_RECENT_DAYS // 365} лет "
                          f"(cutoff={cutoff}) — показаны как fallback")

        lines = []
        auto_saved = 0
        auto_filtered = 0
        for paper in data:
            pid = paper.get("id", "")
            title = paper.get("title", "")
            summary = paper.get("summary", "")[:800]
            authors = paper.get("authors", "")
            date = (paper.get("published_at", "") or "")[:10]
            lines.append(
                f"[{pid}] {title}\n  {authors} · {date}\n  https://arxiv.org/abs/{pid}\n  {summary}"
            )
            if pid:
                passed, reason, o_h, header_kws = _helpers.gate_paper_for_kb(pid, title, summary)
                if not passed:
                    auto_filtered += 1
                    _helpers._log_kb_rejected(pid, title, reason, o_h, header_kws, source="arxiv_search")
                    _helpers.log.debug("kb auto-save skipped %s (%s)", pid, reason)
                    continue
                try:
                    kb_mod.add(kb_mod.Atom(
                        id=pid, kind="paper", topic=query,
                        title=title[:200], authors=authors[:200],
                        url=f"https://arxiv.org/abs/{pid}",
                        claim=summary[:400],
                    ))
                    auto_saved += 1
                except Exception as e:
                    _helpers.log.debug("kb auto-save arxiv paper failed %s: %s", pid, e)

        footer_parts = []
        if auto_saved:
            footer_parts.append(f"📥 авто-сохранено в kb: {auto_saved}")
        if auto_filtered:
            footer_parts.append(f"🚫 отфильтровано domain gate: {auto_filtered}")
        footer = f"\n\n({', '.join(footer_parts)})" if footer_parts else ""
        return "\n\n".join(lines) + footer + stale_note


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

    _QUALIFIER_RE = None  # lazy-инициализация в call()

    @staticmethod
    def _parse_qualifiers(raw_query: str) -> tuple[str, int | None, str | None]:
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
        cleaned_query = " ".join(cleaned_query.split())
        return cleaned_query, extracted_min_stars, extracted_language

    @staticmethod
    def _format_repo_results(data: list, cleaned_query: str) -> tuple[list[str], int]:
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
            if name != "?" and stars >= 10:
                passed, reason = _helpers.gate_repo_for_kb(name, desc)
                if not passed:
                    _helpers.log.debug("github repo gate skipped %s (%s)", name, reason)
                    lines[-1] += f"\n         ⊘ gate:{reason} — не сохранено в kb"
                    continue
                try:
                    kb_mod.add(kb_mod.Atom(
                        id=name, kind="repo", topic=cleaned_query,
                        title=name, url=url,
                        stars=int(stars or 0), lang=lang,
                        claim=desc or f"{lang} репозиторий, {stars}★",
                    ))
                    auto_saved += 1
                except Exception as e:
                    _helpers.log.debug("kb auto-save repo failed %s: %s", name, e)
        return lines, auto_saved

    @staticmethod
    def _format_code_results(data: list) -> list[str]:
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

        cleaned_query, extracted_min_stars, extracted_language = self._parse_qualifiers(raw_query)
        if not cleaned_query:
            return "ошибка: после удаления qualifiers query пустой — пиши 2-4 ключевых слова"

        word_count = len([w for w in cleaned_query.split() if len(w) >= 2])
        if word_count > _cfg.MAX_GITHUB_QUERY_WORDS:
            words = cleaned_query.split()
            short_hint = " ".join(words[:3])
            return (f"ОТКАЗ: query '{cleaned_query}' слишком длинный ({word_count} слов). "
                    f"github search плохо работает с длинными фразами — сократи до 2-3 ключевых "
                    f"терминов (попробуй: '{short_hint}') ИЛИ откажись от github_search для этой "
                    "подтемы, если она чисто теоретическая.")
        min_stars = p.get("min_stars")
        if min_stars is None and extracted_min_stars is not None:
            min_stars = extracted_min_stars
        language = (p.get("language") or "").strip() or extracted_language or ""

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
                         - timedelta(days=_cfg.GITHUB_RECENT_DAYS)).date().isoformat()

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
                    args += ["--updated", f">={recent_cutoff}"]
            return args

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


_helpers._wrap_module_tools(globals(), __name__)
