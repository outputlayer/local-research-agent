"""File-based cross-session research memory: short typed notes."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .config import RESEARCH_MEMORY_DIR
from .utils import jaccard, keyword_set

_FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n?", re.DOTALL)
_SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9а-яА-Я._-]+")


@dataclass
class MemoryEntry:
    path: Path
    memory_type: str
    title: str
    description: str
    topic: str
    tags: list[str]
    created_at: str
    body: str


def ensure_memory_dir() -> Path:
    RESEARCH_MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    return RESEARCH_MEMORY_DIR


def _slugify(text: str, fallback: str = "memory") -> str:
    slug = _SAFE_NAME_RE.sub("-", (text or "").strip()).strip("-").lower()
    return slug[:80] or fallback


def _frontmatter(entry: MemoryEntry) -> str:
    lines = [
        "---",
        f"type: {entry.memory_type}",
        f"title: {entry.title}",
        f"description: {entry.description}",
        f"topic: {entry.topic}",
        f"tags: {', '.join(entry.tags)}",
        f"created_at: {entry.created_at}",
        "---",
        "",
    ]
    return "\n".join(lines)


def save_memory(
    *,
    memory_type: str,
    title: str,
    description: str,
    body: str,
    topic: str = "",
    tags: list[str] | None = None,
) -> Path:
    """Saves a memory entry to a separate markdown file."""
    ensure_memory_dir()
    now = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    entry = MemoryEntry(
        path=RESEARCH_MEMORY_DIR / f"{now}-{_slugify(title)}.md",
        memory_type=(memory_type or "note").strip().lower(),
        title=title.strip(),
        description=description.strip(),
        topic=topic.strip(),
        tags=[t.strip() for t in (tags or []) if t.strip()],
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        body=body.strip(),
    )
    entry.path.write_text(_frontmatter(entry) + entry.body + "\n", encoding="utf-8")
    return entry.path


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    raw = m.group(1)
    meta: dict[str, str] = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip()
    return meta, text[m.end():].strip()


def load_memories() -> list[MemoryEntry]:
    if not RESEARCH_MEMORY_DIR.exists():
        return []
    entries: list[MemoryEntry] = []
    for path in sorted(RESEARCH_MEMORY_DIR.glob("*.md")):
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            continue
        meta, body = _parse_frontmatter(raw)
        tags = [t.strip() for t in meta.get("tags", "").split(",") if t.strip()]
        entries.append(MemoryEntry(
            path=path,
            memory_type=meta.get("type", "note"),
            title=meta.get("title", path.stem),
            description=meta.get("description", ""),
            topic=meta.get("topic", ""),
            tags=tags,
            created_at=meta.get("created_at", ""),
            body=body,
        ))
    return entries


def select_relevant_memories(query: str, k: int = 3) -> list[MemoryEntry]:
    """Returns the top-k memory entries for a query; the selector is deliberately conservative."""
    q_kws = keyword_set(query)
    if not q_kws:
        return []
    scored: list[tuple[float, MemoryEntry]] = []
    for entry in load_memories():
        title_kws = keyword_set(entry.title)
        desc_kws = keyword_set(entry.description)
        topic_kws = keyword_set(entry.topic)
        tag_kws = {t.lower() for t in entry.tags if len(t) >= 4}
        body_preview = entry.body[:400]
        body_kws = keyword_set(body_preview)

        score = (
            3.0 * len(q_kws & title_kws)
            + 2.0 * len(q_kws & desc_kws)
            + 2.0 * len(q_kws & topic_kws)
            + 1.5 * len(q_kws & tag_kws)
            + 1.0 * len(q_kws & body_kws)
            + 5.0 * jaccard(q_kws, title_kws | desc_kws | topic_kws | tag_kws | body_kws)
        )
        if score <= 0:
            continue
        scored.append((score, entry))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [entry for _, entry in scored[:k]]


def format_memory_context(entries: list[MemoryEntry]) -> str:
    if not entries:
        return ""
    lines = []
    for entry in entries:
        lines.append(
            f"- [{entry.memory_type}] {entry.title} — {entry.description or entry.body[:160]}"
        )
    return "\n".join(lines)


def record_run_memory(
    *,
    query: str,
    stopped_reason: str | None,
    valid_ids: int,
    invalid_ids: list[str],
    suspicious_citations: list[str],
    lessons_tail: str = "",
) -> Path:
    """Short note about what helped or hindered the run."""
    quality = [
        f"valid_ids={valid_ids}",
        f"invalid_ids={len(invalid_ids)}",
        f"suspicious={len(suspicious_citations)}",
    ]
    if stopped_reason:
        quality.append(f"stop={stopped_reason}")
    body_lines = [
        f"Query: {query}",
        "",
        "Run quality:",
        "- " + "\n- ".join(quality),
    ]
    if lessons_tail.strip():
        body_lines.extend(["", "Latest lessons:", lessons_tail.strip()])
    return save_memory(
        memory_type="run-summary",
        title=f"Run summary: {query[:80]}",
        description=", ".join(quality),
        body="\n".join(body_lines),
        topic=query,
        tags=["research", "run", "quality"],
    )
