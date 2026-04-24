"""Search tools: hf_papers, arxiv_search, github_search.

They share helpers for arXiv HTTP fetching (`_fetch_text`, `_parse_arxiv_feed`),
auto-save results to KB via `gate_paper_for_kb`, and use a common
querylog/seen_queries dedup.

IMPORTANT: helpers are called via the qualified `_helpers.X` form so that
tests can monkeypatch `lra.tools._helpers._fetch_text` (otherwise import-time
binding would freeze the original in this module).
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
    description = ("Searches scientific papers on Hugging Face Papers via the local `hf` CLI. "
                   "Returns id (arxiv), title, authors, date and abstract.")
    parameters = [
        {"name": "query", "type": "string", "description": "Query", "required": True},
        {"name": "limit", "type": "integer", "description": "How many results (1-10)", "required": False},
    ]

    def call(self, params: str, **kwargs) -> str:
        p = parse_args(params)
        query = p["query"]
        limit = max(1, min(int(p.get("limit", 5)), 10))
        if normalize_query(query) in seen_queries():
            return (f"REJECTED: query '{query}' has already been executed in this session. "
                    "Rephrase (different keywords, author, year, technique) or read read_notes.")
        fuzzy = is_similar_to_seen(query)
        if fuzzy and fuzzy != query:
            return (f"REJECTED: query '{query}' is too similar to an already executed '{fuzzy}'. "
                    "Change the topic (different terms, author, year) or move to another [TODO].")
        log_query(query)
        r = cli_run.run(
            ["hf", "papers", "search", query, "--limit", str(limit * 2), "--format", "json"],
            timeout=30,
        )
        if r.returncode == 127:
            return "error: `hf` CLI not found in PATH (pip install huggingface_hub[cli])"
        if r.returncode == 124:
            return "hf_papers search timeout"
        if not r.ok:
            return f"error: {r.stderr.strip()[:500]}"
        try:
            data = json.loads(r.stdout)
        except Exception:
            return f"failed to parse JSON: {r.stdout[:300]}"
        if not data:
            return f"no results: {query}"
        data.sort(key=lambda x: x.get("published_at", ""), reverse=True)
        from datetime import datetime, timedelta
        cutoff = (datetime.now(UTC) - timedelta(days=_cfg.ARXIV_RECENT_DAYS)).date().isoformat()
        fresh = [x for x in data if (x.get("published_at") or "")[:10] >= cutoff]
        stale_note = ""
        if fresh:
            data = fresh[:limit]
        else:
            data = data[:limit]
            stale_note = (f"\n\n⚠️  all results older than {_cfg.ARXIV_RECENT_DAYS // 365} years "
                          f"(cutoff={cutoff}) — shown as fallback")
        lines = []
        auto_saved = 0
        auto_filtered = 0
        for paper in data:
            authors = ", ".join(a["name"] for a in paper.get("authors", [])[:4])
            if len(paper.get("authors", [])) > 4:
                authors += " et al."
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
            footer_parts.append(f"📥 auto-saved to kb: {auto_saved}")
        if auto_filtered:
            footer_parts.append(f"🚫 filtered by domain gate: {auto_filtered}")
        footer = f"\n\n({', '.join(footer_parts)})" if footer_parts else ""
        return "\n\n".join(lines) + footer + stale_note


@register_tool("arxiv_search")
class ArxivSearch(BaseTool):
    description = (
        "Fallback paper search via the arXiv API. Use when hf_papers is empty, "
        "stale or did not find the required sub-topic. Returns arxiv-id, title, "
        "authors, date and abstract.\n"
        "IMPORTANT: for narrow domains (signal processing, info theory, crypto) pass "
        "categories=['eess.SP','cs.IT','cs.CR'] — it sharply increases relevance "
        "and filters out noise from NLP/CV. Full category list: arxiv.org/category_taxonomy."
    )
    parameters = [
        {"name": "query", "type": "string", "description": "Query", "required": True},
        {"name": "limit", "type": "integer", "description": "How many results (1-10)", "required": False},
        {"name": "categories", "type": "array",
         "description": "Optional arxiv categories (eess.SP, cs.IT, cs.CR, cs.LG, ...). "
                        "If set — search is restricted to these categories (OR).",
         "required": False},
    ]

    def call(self, params: str, **kwargs) -> str:
        p = parse_args(params)
        query = (p.get("query") or "").strip()
        if not query:
            return "error: query is required"
        limit = max(1, min(int(p.get("limit", 5)), 10))
        # Categories: list or CSV string
        raw_cats = p.get("categories") or []
        if isinstance(raw_cats, str):
            raw_cats = [c.strip() for c in raw_cats.split(",") if c.strip()]
        # Sanitize: only tokens shaped like letters[.letters], length ≤ 25 chars
        import re as _re
        cats = [c for c in raw_cats
                if isinstance(c, str) and _re.fullmatch(r"[a-z\-]+(?:\.[A-Za-z\-]+)?", c)
                and len(c) <= 25]
        cat_suffix = " [cats=" + ",".join(cats) + "]" if cats else ""
        dedup_key = f"arxiv: {query}{cat_suffix}"
        if normalize_query(dedup_key) in seen_queries():
            return (f"REJECTED: query '{query}' has already been executed via arxiv_search in this session. "
                    "Rephrase or read read_notes/kb_search.")
        fuzzy = is_similar_to_seen(dedup_key)
        if fuzzy and fuzzy != dedup_key:
            return (f"REJECTED: arxiv_search '{query}' is too similar to an already executed '{fuzzy}'. "
                    "Change terms, author or year.")
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
            return "arxiv_search timeout"
        except Exception as exc:
            return f"arxiv_search error: {str(exc)[:300]}"

        try:
            data = _helpers._parse_arxiv_feed(xml_text)
        except Exception:
            return f"failed to parse arXiv feed: {xml_text[:300]}"
        if not data:
            return f"no arxiv_search results: {query}"

        from datetime import datetime, timedelta
        cutoff = (datetime.now(UTC) - timedelta(days=_cfg.ARXIV_RECENT_DAYS)).date().isoformat()
        fresh = [x for x in data if (x.get("published_at") or "")[:10] >= cutoff]
        stale_note = ""
        if fresh:
            data = fresh[:limit]
        else:
            data = data[:limit]
            stale_note = (f"\n\n⚠️  all results older than {_cfg.ARXIV_RECENT_DAYS // 365} years "
                          f"(cutoff={cutoff}) — shown as fallback")

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
            footer_parts.append(f"📥 auto-saved to kb: {auto_saved}")
        if auto_filtered:
            footer_parts.append(f"🚫 filtered by domain gate: {auto_filtered}")
        footer = f"\n\n({', '.join(footer_parts)})" if footer_parts else ""
        return "\n\n".join(lines) + footer + stale_note


@register_tool("semantic_scholar_search")
class SemanticScholarSearch(BaseTool):
    description = (
        "Search via the Semantic Scholar Graph API (api.semanticscholar.org). "
        "Alternative to arxiv_search for cross-disciplinary topics (when you need not "
        "only arxiv preprints but also journal publications, ACL/NeurIPS/ICML etc.). "
        "Returns paperId, title, year, authors, abstract; if the paper has an arxiv-id "
        "(via externalIds.ArXiv) it is used as the primary ID and lands in kb. Papers "
        "without an arxiv-id are shown but NOT auto-saved to kb (the verifier requires "
        "an arxiv-id for AppendNotes).\n"
        "Optional: year='2023-2025' (year range), fields_of_study=['Computer Science']."
    )
    parameters = [
        {"name": "query", "type": "string", "description": "Query (free text)", "required": True},
        {"name": "limit", "type": "integer", "description": "1-10 (default 5)", "required": False},
        {"name": "year", "type": "string",
         "description": "Optional year range '2023-2025' or single year '2024'", "required": False},
    ]

    def call(self, params: str, **kwargs) -> str:
        p = parse_args(params)
        query = (p.get("query") or "").strip()
        if not query:
            return "error: query is required"
        limit = max(1, min(int(p.get("limit", 5)), 10))
        year_raw = (p.get("year") or "").strip()
        # Sanitize year: digits and optionally one dash
        year = ""
        if year_raw and re.fullmatch(r"20\d{2}(?:-20\d{2})?", year_raw):
            year = year_raw

        dedup_key = f"s2: {query}" + (f" [year={year}]" if year else "")
        if normalize_query(dedup_key) in seen_queries():
            return (f"REJECTED: query '{query}' has already been executed via semantic_scholar_search. "
                    "Rephrase or read read_notes/kb_search.")
        fuzzy = is_similar_to_seen(dedup_key)
        if fuzzy and fuzzy != dedup_key:
            return (f"REJECTED: semantic_scholar_search '{query}' is similar to '{fuzzy}'. "
                    "Change terms.")
        log_query(dedup_key)

        qs: dict[str, str | int] = {
            "query": query,
            "limit": limit,
            "fields": "title,abstract,year,authors,externalIds",
        }
        if year:
            qs["year"] = year
        url = "https://api.semanticscholar.org/graph/v1/paper/search?" + urlencode(qs)
        try:
            raw = _helpers._fetch_text(url, timeout=20)
        except TimeoutError:
            return "semantic_scholar_search timeout"
        except Exception as exc:
            return f"semantic_scholar_search error: {str(exc)[:300]}"

        try:
            payload = json.loads(raw)
        except Exception:
            return f"invalid JSON from Semantic Scholar: {raw[:300]}"
        items = payload.get("data") or []
        if not items:
            return f"no semantic_scholar_search results: {query}"

        lines: list[str] = []
        auto_saved = 0
        auto_filtered = 0
        no_arxiv = 0
        for it in items[:limit]:
            ext = it.get("externalIds") or {}
            arx = (ext.get("ArXiv") or "").strip()
            paper_id = it.get("paperId") or ""
            title = (it.get("title") or "").strip()
            abstract = (it.get("abstract") or "").strip()
            year_p = it.get("year") or ""
            authors = ", ".join(
                (a.get("name") or "").strip() for a in (it.get("authors") or [])[:4] if a
            )
            display_id = arx or f"s2:{paper_id}"
            url_paper = f"https://arxiv.org/abs/{arx}" if arx else \
                        f"https://www.semanticscholar.org/paper/{paper_id}"
            lines.append(
                f"[{display_id}] {title}\n  {authors} · {year_p}\n  {url_paper}\n"
                f"  {abstract[:800]}"
            )
            if not arx:
                no_arxiv += 1
                continue
            passed, reason, o_h, header_kws = _helpers.gate_paper_for_kb(arx, title, abstract)
            if not passed:
                auto_filtered += 1
                _helpers._log_kb_rejected(arx, title, reason, o_h, header_kws,
                                          source="semantic_scholar_search")
                continue
            try:
                kb_mod.add(kb_mod.Atom(
                    id=arx, kind="paper", topic=query,
                    title=title[:200], authors=authors[:200],
                    url=f"https://arxiv.org/abs/{arx}",
                    claim=abstract[:400],
                ))
                auto_saved += 1
            except Exception as e:
                _helpers.log.debug("kb auto-save s2 paper failed %s: %s", arx, e)

        footer_parts = []
        if auto_saved:
            footer_parts.append(f"📥 auto-saved to kb: {auto_saved}")
        if auto_filtered:
            footer_parts.append(f"🚫 filtered by domain gate: {auto_filtered}")
        if no_arxiv:
            footer_parts.append(f"⚠️ without arxiv-id (not saved): {no_arxiv}")
        footer = f"\n\n({', '.join(footer_parts)})" if footer_parts else ""
        return "\n\n".join(lines) + footer


@register_tool("github_search")
class GithubSearch(BaseTool):
    description = (
        "Search GitHub via the official `gh` CLI. "
        "Use to find IMPLEMENTATIONS and DATASETS for papers from hf_papers: "
        "found a method in the abstract → look for its repository here. "
        "type='repos' — repositories (default), type='code' — code search.\n"
        "IMPORTANT: query must be SHORT — 2-4 keywords. "
        "Do NOT put 'stars:>=10' or 'language:python' inside the query — use the "
        "dedicated min_stars and language parameters instead."
    )
    parameters = [
        {"name": "query", "type": "string",
         "description": "2-4 keywords WITHOUT qualifiers", "required": True},
        {"name": "type", "type": "string",
         "description": "'repos' (default) or 'code'", "required": False},
        {"name": "limit", "type": "integer",
         "description": "Number of results 1-10 (default 5)", "required": False},
        {"name": "min_stars", "type": "integer",
         "description": "minimum stars (for type=repos)", "required": False},
        {"name": "language", "type": "string",
         "description": "programming language (for type=repos)", "required": False},
    ]

    _QUALIFIER_RE = None  # lazy init in call()

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
                    lines[-1] += f"\n         ⊘ gate:{reason} — not saved to kb"
                    continue
                try:
                    kb_mod.add(kb_mod.Atom(
                        id=name, kind="repo", topic=cleaned_query,
                        title=name, url=url,
                        stars=int(stars or 0), lang=lang,
                        claim=desc or f"{lang} repository, {stars}★",
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
            return "error: query is required"
        search_type = p.get("type", "repos").strip().lower()
        if search_type not in ("repos", "code"):
            search_type = "repos"
        limit = max(1, min(int(p.get("limit", 5)), 10))

        cleaned_query, extracted_min_stars, extracted_language = self._parse_qualifiers(raw_query)
        if not cleaned_query:
            return "error: after stripping qualifiers the query is empty — pass 2-4 keywords"

        word_count = len([w for w in cleaned_query.split() if len(w) >= 2])
        if word_count > _cfg.MAX_GITHUB_QUERY_WORDS:
            words = cleaned_query.split()
            short_hint = " ".join(words[:3])
            return (f"REJECTED: query '{cleaned_query}' is too long ({word_count} words). "
                    f"github search works poorly with long phrases — shorten to 2-3 key "
                    f"terms (try: '{short_hint}') OR skip github_search for this "
                    "sub-topic if it is purely theoretical.")
        min_stars = p.get("min_stars")
        if min_stars is None and extracted_min_stars is not None:
            min_stars = extracted_min_stars
        language = (p.get("language") or "").strip() or extracted_language or ""

        dedup_key = f"gh-{search_type}: {cleaned_query}"
        if language:
            dedup_key += f" lang={language}"
        if normalize_query(dedup_key) in seen_queries():
            return (f"REJECTED: GitHub query '{cleaned_query}' (type={search_type}"
                    f"{', lang=' + language if language else ''}) has already been issued. "
                    "Rephrase or read read_notes.")
        fuzzy = is_similar_to_seen(dedup_key)
        if fuzzy and fuzzy != dedup_key:
            return (f"REJECTED: GitHub query '{cleaned_query}' is too similar to '{fuzzy}'. "
                    f"Priority actions: (1) kb_search '{cleaned_query}' — maybe we already "
                    f"searched and an atom sits in kb; (2) read_notes — look for what is already written; "
                    f"(3) if github really is needed — pick another [FOCUS] angle "
                    f"(a specific method/dataset rather than the topic itself); (4) plan_close_task the "
                    f"current [FOCUS] with evidence from notes and move to the next [TODO].")
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
            return "error: `gh` CLI not found in PATH (install: brew install gh)"
        if r.returncode == 124:
            return "GitHub search timeout"
        if not r.ok:
            err = r.stderr.strip()[:400]
            if "authentication" in err.lower() or "auth" in err.lower() or "login" in err.lower():
                return "gh error: authentication required. Run in terminal: `gh auth login`"
            return f"gh error: {err}"
        try:
            data = json.loads(r.stdout)
        except Exception:
            return f"failed to parse JSON: {r.stdout[:300]}"
        if not data and search_type == "repos":
            r2 = cli_run.run(_build_args(with_freshness=False), timeout=20)
            if r2.ok:
                try:
                    data2 = json.loads(r2.stdout)
                except Exception:
                    data2 = []
                if data2:
                    data = data2
                    stale_note = (f"\n\n⚠️  no fresh repos (updated >= {recent_cutoff}) — "
                                  f"older results shown as fallback")
        if not data:
            hint = ""
            if len(cleaned_query.split()) >= 4:
                hint = " — try SHORTENING the query to 2-3 keywords"
            elif min_stars and int(min_stars) >= 50:
                hint = f" — try lowering min_stars (current={min_stars})"
            elif language:
                hint = f" — try WITHOUT language='{language}'"
            return f"no GitHub results: '{cleaned_query}'{hint}"

        if search_type == "repos":
            lines, auto_saved = self._format_repo_results(data, cleaned_query)
        else:
            lines = self._format_code_results(data)
            auto_saved = 0

        footer = f"\n\n(📥 auto-saved to kb: {auto_saved})" if auto_saved else ""
        return "\n\n".join(lines) + footer + stale_note


_helpers._wrap_module_tools(globals(), __name__)
