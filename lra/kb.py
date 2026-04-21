"""Structured knowledge base: append-only JSONL + lightweight BM25-ish search.

Каждый атом — одна строка JSON в research/kb.jsonl. Пишется параллельно с notes.md
(notes.md остаётся человекочитаемым), KB нужен для программного поиска релевантного
контекста по текущему [FOCUS] перед тем как уходить в новую итерацию.

Схема атома:
    id       — строка: arxiv-id для papers, 'owner/name' для repos
    kind     — 'paper' | 'repo'
    topic    — [FOCUS] на момент записи (строка)
    title    — заголовок paper'а или owner/name
    claim    — 1-3 предложения: что именно мы узнали и почему это важно
    authors  — только для paper (optional)
    url      — ссылка (optional)
    stars    — для repo (optional)
    lang     — для repo (optional)
    iteration — номер итерации explorer'а
    ts       — ISO timestamp
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime

from .config import RESEARCH_DIR
from .utils import keyword_set

KB_PATH = RESEARCH_DIR / "kb.jsonl"


@dataclass
class Atom:
    id: str
    kind: str
    topic: str
    claim: str
    title: str = ""
    authors: str = ""
    url: str = ""
    stars: int = 0
    lang: str = ""
    iteration: int = 0
    ts: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


def add(atom: Atom) -> None:
    """Добавить атом в KB. Дедуп по (kind, id) — последняя запись побеждает только
    в поиске (в файле остаются все для истории)."""
    RESEARCH_DIR.mkdir(exist_ok=True)
    with KB_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(atom), ensure_ascii=False) + "\n")


def load() -> list[dict]:
    """Читает все атомы. Дедуп по (kind, id) — оставляем последнюю версию."""
    if not KB_PATH.exists():
        return []
    seen: dict[tuple[str, str], dict] = {}
    for ln in KB_PATH.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            a = json.loads(ln)
        except json.JSONDecodeError:
            continue
        seen[(a.get("kind", ""), a.get("id", ""))] = a
    return list(seen.values())


_WORD_RE = re.compile(r"[A-Za-zА-Яа-я][A-Za-zА-Яа-я\-]{2,}")


def _tokens(s: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(s or "")]


def search(query: str, k: int = 5, atoms: list[dict] | None = None) -> list[dict]:
    """BM25-lite поиск по полям claim/title/topic. Без внешних зависимостей.

    Возвращает топ-k атомов, отсортированных по убыванию релевантности.
    Пустой query или пустая KB → [].
    """
    q_tokens = _tokens(query)
    if not q_tokens:
        return []
    pool = atoms if atoms is not None else load()
    if not pool:
        return []

    # docs как конкатенация ключевых полей с весами (title важнее)
    docs = []
    for a in pool:
        text = f"{a.get('title','')} {a.get('title','')} {a.get('claim','')} {a.get('topic','')}"
        docs.append(_tokens(text))

    N = len(docs)
    df: dict[str, int] = {}
    for d in docs:
        for t in set(d):
            df[t] = df.get(t, 0) + 1
    avgdl = sum(len(d) for d in docs) / max(1, N)
    k1, b = 1.5, 0.75

    def score(d: list[str]) -> float:
        if not d:
            return 0.0
        dl = len(d)
        tf: dict[str, int] = {}
        for t in d:
            tf[t] = tf.get(t, 0) + 1
        s = 0.0
        for q in q_tokens:
            if q not in tf:
                continue
            idf = math.log(1 + (N - df.get(q, 0) + 0.5) / (df.get(q, 0) + 0.5))
            f = tf[q]
            s += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
        return s

    scored = [(score(d), a) for d, a in zip(docs, pool, strict=True)]
    # Fallback: если BM25 не дал сигнала, используем простой keyword-overlap (jaccard)
    if all(s == 0.0 for s, _ in scored):
        q_kw = keyword_set(query)
        scored = [
            (len(q_kw & keyword_set(f"{a.get('claim','')} {a.get('title','')}")), a)
            for a in pool
        ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [a for s, a in scored[:k] if s > 0]


def format_atoms(atoms: list[dict]) -> str:
    """Компактный markdown для вставки в user-message — 1 строка на атом."""
    if not atoms:
        return ""
    lines = []
    for a in atoms:
        if a.get("kind") == "paper":
            lines.append(f"- [{a.get('id','?')}] {a.get('title','')[:80]} — {a.get('claim','')[:200]}")
        elif a.get("kind") == "repo":
            lines.append(
                f"- [repo: {a.get('id','?')} ★{a.get('stars',0)} {a.get('lang','')}] "
                f"— {a.get('claim','')[:200]}"
            )
        else:
            lines.append(f"- [{a.get('id','?')}] {a.get('claim','')[:200]}")
    return "\n".join(lines)
