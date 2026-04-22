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


ARXIV_RE = re.compile(r"\b(\d{4}\.\d{4,5})\b")

# Стоп-слова для keyword-overlap проверок (используются валидатором).
STOPWORDS = frozenset({"paper", "paperов", "статья", "работа", "авторы", "model", "method"})


def extract_ids(text: str) -> set[str]:
    """Единая точка правды для извлечения arxiv-id из любого текста."""
    return set(ARXIV_RE.findall(text or ""))


def keyword_set(s: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[A-Za-zА-Яа-я][A-Za-zА-Яа-я\-]{4,}", s)}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
