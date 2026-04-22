"""Конфиг и пути к артефактам."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
_CFG_FILE = ROOT / "chat_config.json"


@dataclass
class Settings:
    """Типизированный конфиг. Опечатка ключа = явная ошибка, не молчаливый KeyError."""
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
            raise ValueError("config: 'model' обязателен")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature вне [0,2]: {self.temperature}")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p вне (0,1]: {self.top_p}")
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
        """Обратная совместимость: CFG[key] продолжает работать как словарь."""
        if hasattr(self, key):
            return getattr(self, key)
        return self.extra[key]

    def get(self, key: str, default=None):
        try:
            return self[key]
        except (AttributeError, KeyError):
            return default

    def __setitem__(self, key: str, value) -> None:
        """CFG[key] = value — для runtime-флагов (hitl, notes_strict, etc.).
        Известные поля дата-класса обновляются напрямую, остальное — в extra."""
        if hasattr(self, key) and key != "extra":
            setattr(self, key, value)
        else:
            self.extra[key] = value

    def pop(self, key: str, default=None):
        """CFG.pop(key) — удалить runtime-флаг из extra (для тестов)."""
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
# Reflexion-память ГЛОБАЛЬНА — живёт между сессиями, не стирается при новом запросе
LESSONS_PATH = RESEARCH_DIR / "lessons.md"
QUERYLOG_PATH = RESEARCH_DIR / "querylog.md"
# Лог отклонённых AppendNotes (domain gate) — не теряем evidence, просто не
# льём в KB. Для отладки: какие papers explorer принёс из смежного домена.
REJECTED_PATH = RESEARCH_DIR / "rejected.jsonl"

# ── Freshness knobs ────────────────────────────────────────────────────────
# Порог актуальности: GitHub — год, arxiv — два. Старые записи не игнорируем
# намертво (fallback отдаёт топ с пометкой "устарело"), но в первом проходе
# отсекаем всё что свежее порога.
GITHUB_RECENT_DAYS = 365
ARXIV_RECENT_DAYS = 730
# github_search с длинным query (>5 значащих слов) почти всегда даёт 0 —
# реджектим на входе и просим модель сократить.
MAX_GITHUB_QUERY_WORDS = 5
