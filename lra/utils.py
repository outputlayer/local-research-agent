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

    def _wrap(obj):
        if isinstance(obj, dict):
            return obj
        return {"content": obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)}

    try:
        return _wrap(json5.loads(params))
    except Exception:
        pass
    try:
        fixed = re.sub(
            r'"((?:[^"\\]|\\.)*)"',
            lambda m: '"' + m.group(1).replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t') + '"',
            params,
            flags=re.DOTALL,
        )
        return _wrap(json5.loads(fixed))
    except Exception:
        pass
    for key in ("content", "code", "query", "url"):
        m = re.search(rf'"{key}"\s*:\s*"(.*?)"\s*(?:,\s*"|\}})', params, re.DOTALL)
        if m:
            return {key: m.group(1).replace('\\n', '\n').replace('\\"', '"')}
    return {"content": params}


def normalize_query(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


ARXIV_RE = re.compile(r"\b(\d{4}\.\d{4,5})\b")


def count_arxiv_ids(text: str) -> set[str]:
    return set(ARXIV_RE.findall(text))


def keyword_set(s: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[A-Za-zА-Яа-я][A-Za-zА-Яа-я\-]{4,}", s)}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
