"""Чистые утилиты без побочных эффектов (легко тестируются)."""
from __future__ import annotations

import json
import re

import json5


def parse_args(params) -> dict:
    """Толерантный парсер tool-аргументов: чинит литеральные переносы строк и
    гарантирует dict на выходе (LLM иногда присылает просто строку)."""
    if isinstance(params, dict):
        return params
    if not isinstance(params, str):
        return {"content": str(params)}

    # Pre-clean: LLM часто лепит хвост tool-call wrapper'а внутрь arguments —
    # `{"content": "..."}</arguments`, `</tool_call>`, trailing ```, etc.
    # Без этого json5 парсер падает, мы валимся в финальный fallback и пишем
    # в файл сырой JSON-блоб вместо content (баг был виден в draft.md).
    s = params.strip()
    for tail in ("</arguments>", "</arguments", "</tool_call>", "</tool_call", "```"):
        if s.endswith(tail):
            s = s[: -len(tail)].rstrip()

    # Double-encoded JSON: LLM иногда оборачивает args в JSON string literal
    # вместо JSON object — `"{\"content\": \"markdown\"}"`. Если мы оставим как есть,
    # json5.loads распарсит это в обычную python-строку, _wrap обернёт → в файл
    # улетит сериализованный JSON-блоб вместо markdown (виден в draft.md этой сессии).
    # Разворачиваем до 3 уровней обёртки (больше не встречали).
    for _ in range(3):
        if len(s) >= 2 and s.startswith('"') and s.endswith('"'):
            try:
                inner = json.loads(s)
                if isinstance(inner, str) and inner.lstrip()[:1] in ('{', '"'):
                    s = inner.strip()
                    continue
            except Exception:
                pass
        break
    # balanced-brace trim: отсекаем любой хвост после последней `}` на верхнем уровне
    if s.startswith("{"):
        depth = 0
        last_close = -1
        in_str = False
        esc = False
        for i, ch in enumerate(s):
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    last_close = i
                    break
        if last_close > 0:
            s = s[: last_close + 1]

    def _wrap(obj):
        if isinstance(obj, dict):
            return obj
        return {"content": obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)}

    try:
        return _wrap(json5.loads(s))
    except Exception:
        pass
    try:
        fixed = re.sub(
            r'"((?:[^"\\]|\\.)*)"',
            lambda m: '"' + m.group(1).replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t') + '"',
            s,
            flags=re.DOTALL,
        )
        return _wrap(json5.loads(fixed))
    except Exception:
        pass
    # Последний шанс: ручное извлечение значения ключа с УЧЁТОМ escaped quotes.
    # `(?:\\.|[^"\\])*` ест любой экранированный символ или не-кавычку/не-бэкслэш.
    for key in ("content", "code", "query", "url"):
        m = re.search(rf'"{key}"\s*:\s*"((?:\\.|[^"\\])*)"', s, re.DOTALL)
        if m:
            raw = m.group(1)
            # unescape стандартный JSON-набор
            try:
                decoded = json.loads(f'"{raw}"')
            except Exception:
                decoded = raw.replace('\\n', '\n').replace('\\"', '"').replace("\\\\", "\\")
            return {key: decoded}
    # Совсем ничего не сматчилось. Если вход ПОХОЖ на JSON (начинается с `{` или
    # содержит `"content"`/`"code"`) — это сломанная структура, помечаем префиксом
    # чтобы проблема была видна в draft/notes. Иначе — обычная bare-string, которую
    # LLM прислал без объекта: отдаём как content без изменений.
    looks_like_json = s.startswith("{") or re.search(r'"(?:content|code|query|url)"', s) is not None
    if looks_like_json:
        return {"content": f"[parse_args: unrecognized tool args]\n{params}"}
    return {"content": params}


def get_content(params) -> str:
    """Достаёт content из tool-аргументов с мягким fallback. Используй в тулах,
    которые принимают markdown-текст (write_draft, append_notes и т.п.) —
    чтобы KeyError не ронял всю цепочку когда LLM прислал `{"text": ...}`,
    `{"markdown": ...}` или просто один ключ не того имени.
    """
    d = parse_args(params)
    if "content" in d:
        return d["content"] or ""
    # Common aliases LLM изобретает
    for alt in ("text", "markdown", "md", "body", "data", "value"):
        if alt in d:
            return d[alt] or ""
    # Иначе берём первое строковое значение — лучше странный draft чем крэш.
    for v in d.values():
        if isinstance(v, str):
            return v
    return ""


def normalize_query(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


# Arxiv-id: YYMM.NNNNN (YY=год, MM=01-12 месяц, N=4-5 цифр seq).
# Строгая версия (2026-04): negative lookahead/behind отсекают:
#   - десятичные числа в тексте ("effect size 2504.03 mm/s" — только 2 цифры, но 2504.03456 матчился бы)
#   - обрывки id ("2301.123456" → 6 цифр, не arxiv)
#   - id внутри длиннее числа ("A12301.12345", "2301.12345.67")
# MM=01-12 чтобы "2013.12345" (невалидный месяц 13) не матчился.
ARXIV_RE = re.compile(
    r"(?<![\d.])(\d{2}(?:0[1-9]|1[0-2])\.\d{4,5})(?![\d.])"
)

# Стоп-слова для keyword-overlap проверок (используются валидатором).
STOPWORDS = frozenset({"paper", "paperов", "статья", "работа", "авторы", "model", "method"})


def extract_ids(text: str) -> set[str]:
    """Единая точка правды для извлечения arxiv-id из любого текста."""
    return set(ARXIV_RE.findall(text or ""))


def keyword_set(s: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[A-Za-zА-Яа-я][A-Za-zА-Яа-я\-]{4,}", s)}


# Generic-слова из plan.md, которые НЕ должны служить domain-якорем:
# встречаются в любом научном abstract и дают false positive overlap.
_TOPIC_GENERIC = frozenset({
    "modern", "approach", "approaches", "survey", "review", "study", "studies",
    "analysis", "research", "technique", "techniques", "method", "methods",
    "system", "systems", "model", "models", "application", "applications",
    "based", "using", "toward", "towards", "paper", "papers",
    "plan", "focus", "todo", "done", "blocked", "progress", "digest",
    "iter", "attempts", "evidence", "revision", "revisions",
    "deep-dive", "trade-offs", "tradeoffs",
    # generic-соединители (ловились на "electronic" в 2508.12935 emotional support)
    "advanced", "novel", "complex", "between", "among", "within",
    "challenges", "challenge", "problem", "problems", "issue", "issues",
    "future", "recent", "current", "emerging", "integration", "integrations",
    # локальный шум из seeds типа "ELINT fingerprinting in urban canyons"
    "canyons", "canyon", "contested", "environments", "environment",
    "algorithm", "algorithms", "architecture", "architectures",
    "современ", "современные", "подход", "подходы", "обзор", "анализ",
    "метод", "методы", "система", "системы", "модель", "модели",
    "статья", "работа", "работы", "исследование", "исследования",
})


def _plan_sections(plan_text: str) -> tuple[str, str]:
    """Разбивает plan.md на (header, seeds).

    header = первая строка (# Plan: ...) + строка `**Core vocabulary:** ...`,
             если LLM-bootstrap её сгенерил. Это ядро темы.
    seeds  = все строки с [Tn] — часто drift'ят и содержат specific jargon,
             но сами по себе слабый якорь (см. canyons / challenges).
    """
    if not plan_text:
        return "", ""
    lines = plan_text.splitlines()
    header_parts: list[str] = []
    if lines:
        header_parts.append(lines[0])
    for ln in lines[1:10]:  # vocab-line рендерится сразу после header
        if ln.lstrip().lower().startswith("**core vocabulary:"):
            header_parts.append(ln)
            break
    header = "\n".join(header_parts)
    seeds = "\n".join(ln for ln in lines if re.search(r"\[T\d+\]", ln))
    return header, seeds


def extract_topic_keywords(plan_text: str) -> set[str]:
    """Все доменные ключевые слова из plan.md (header ∪ seeds). Backwards-compat."""
    header, seeds = _plan_sections(plan_text)
    kws = keyword_set(header + " " + seeds)
    return {w for w in kws if w not in STOPWORDS and w not in _TOPIC_GENERIC}


def extract_topic_keywords_tiered(plan_text: str) -> tuple[set[str], set[str]]:
    """Двухуровневое разделение для domain gate: (header_kws, seed_kws).

    header_kws — ядро домена из заголовка (# Plan: ...). Это то, что изначально
    спросил пользователь. ComVo (audio vocoder) не имеет ни одного overlap с
    {electronic, warfare, elint, intelligence} — gate его режет.
    seed_kws — специфика из [Tn] задач. Слабый якорь (drift), используется
    только как bonus для paper'ов, которые УЖЕ прошли по header.
    """
    header, seeds = _plan_sections(plan_text)
    h = {w for w in keyword_set(header)
         if w not in STOPWORDS and w not in _TOPIC_GENERIC}
    s = {w for w in keyword_set(seeds)
         if w not in STOPWORDS and w not in _TOPIC_GENERIC}
    return h, s - h  # seeds disjoint от header, чтобы не двойной учёт


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
