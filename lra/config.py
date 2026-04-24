"""Config and paths to artifacts."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
_CFG_FILE = ROOT / "chat_config.json"


@dataclass
class Settings:
    """Typed config. A misspelled key is an explicit error, not a silent KeyError."""
    model: str
    system_prompt: str = ""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_tokens: int = 4096
    repetition_penalty: float = 1.1
    max_history: int = 40
    cache_ttl_hours: int = 24
    log_level: str = "INFO"
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.model:
            raise ValueError("config: 'model' is required")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature out of [0,2]: {self.temperature}")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p out of (0,1]: {self.top_p}")
        if self.top_k < 0:
            raise ValueError(f"top_k < 0: {self.top_k}")
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens <= 0: {self.max_tokens}")

    @classmethod
    def load(cls, path: Path | None = None) -> Settings:
        p = path or _CFG_FILE
        raw: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls) if f.name != "extra"}
        extra = {k: v for k, v in raw.items() if k not in known}
        kwargs = {k: v for k, v in raw.items() if k in known}
        return cls(**kwargs, extra=extra)

    def __getitem__(self, key: str):
        """Backward compatibility: CFG[key] still works like a dict."""
        if hasattr(self, key):
            return getattr(self, key)
        return self.extra[key]

    def get(self, key: str, default=None):
        try:
            return self[key]
        except (AttributeError, KeyError):
            return default

    def __setitem__(self, key: str, value) -> None:
        """CFG[key] = value — for runtime flags (hitl, notes_strict, etc.).
        Known dataclass fields are updated directly; everything else goes into extra."""
        if hasattr(self, key) and key != "extra":
            setattr(self, key, value)
        else:
            self.extra[key] = value

    def pop(self, key: str, default=None):
        """CFG.pop(key) — remove a runtime flag from extra (for tests)."""
        return self.extra.pop(key, default)


CFG = Settings.load()

RESEARCH_DIR = ROOT / "research"
ARCHIVE_DIR = RESEARCH_DIR / "archive"
CACHE_DIR = ROOT / ".cache"
DRAFT_PATH = RESEARCH_DIR / "draft.md"
NOTES_PATH = RESEARCH_DIR / "notes.md"
PLAN_PATH = RESEARCH_DIR / "plan.md"
SYNTHESIS_PATH = RESEARCH_DIR / "synthesis.md"
RUN_LOG_PATH = RESEARCH_DIR / "run.log"
RESEARCH_MEMORY_DIR = RESEARCH_DIR / "memory"
# Reflexion memory is GLOBAL — lives across sessions, not wiped on a new query
LESSONS_PATH = RESEARCH_DIR / "lessons.md"
QUERYLOG_PATH = RESEARCH_DIR / "querylog.md"
# Log of rejected AppendNotes (domain gate) — evidence is not lost; we just do not
# push to KB. For debugging: which papers the explorer dragged in from an adjacent domain.
REJECTED_PATH = RESEARCH_DIR / "rejected.jsonl"

# ── Freshness knobs ────────────────────────────────────────────────────────
# Freshness threshold: GitHub — one year, arxiv — two. Old entries are not
# dropped outright (the fallback still returns the top with a "stale" note), but
# in the first pass we cut off anything older than the threshold.
GITHUB_RECENT_DAYS = 365
ARXIV_RECENT_DAYS = 730
# github_search with a long query (>5 significant words) almost always returns 0 —
# we reject on input and ask the model to shorten.
MAX_GITHUB_QUERY_WORDS = 5
