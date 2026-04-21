"""Метрики итераций: сколько времени, сколько id прибавилось, сходимость критика.

Экспортируются в research/metrics.json для внешней аналитики и integration-тестов.
"""
from __future__ import annotations
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from .config import RESEARCH_DIR


METRICS_PATH = RESEARCH_DIR / "metrics.json"


@dataclass
class IterationMetric:
    iteration: int
    focus: str
    prefetch_seconds: float = 0.0
    explorer_seconds: float = 0.0
    replanner_seconds: float = 0.0
    notes_grew_chars: int = 0
    new_arxiv_ids: int = 0
    hf_from_cache: bool = False
    gh_from_cache: bool = False
    focus_changed: bool = False


@dataclass
class CriticRound:
    round: int
    issues_found: int
    approved: bool
    converged_by_similarity: bool = False
    seconds: float = 0.0


@dataclass
class RunMetrics:
    query: str
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    iterations: list[IterationMetric] = field(default_factory=list)
    critic_rounds: list[CriticRound] = field(default_factory=list)
    synthesis_seconds: float = 0.0
    writer_seconds: float = 0.0
    valid_ids: int = 0
    invalid_ids: list[str] = field(default_factory=list)
    suspicious_citations: list[str] = field(default_factory=list)
    final_draft_chars: int = 0
    stopped_early_reason: Optional[str] = None

    @property
    def total_seconds(self) -> float:
        end = self.finished_at or time.time()
        return end - self.started_at

    def finish(self, path: Optional[Path] = None) -> Path:
        """Сохраняет метрики в JSON. Вызывается в конце research_loop."""
        self.finished_at = time.time()
        target = path or METRICS_PATH
        target.parent.mkdir(exist_ok=True)
        target.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return target


# ---- Парсинг критического ответа для точной конвергенции ----

_ISSUE_MARKERS = (
    "нет цитаты", "отсутствует в notes", "противоречит notes",
    "дублирование", "SMART_PLAGIARISM", "Novel Insights",
)
_ACTION_RE = re.compile(r"^\s*(?:[\-\*]|\d+[\.\)])\s+", re.MULTILINE)


def count_critic_issues(text: str) -> int:
    """Оценивает число конкретных правок в ответе критика.

    Стратегия:
    - если есть APPROVED — 0.
    - иначе считаем bullet/numbered-пункты; если их нет, считаем упоминания маркеров проблем.
    Капаем на 10, чтобы не реагировать на шум.
    """
    if not text:
        return 0
    if "APPROVED" in text.upper():
        return 0
    bullets = len(_ACTION_RE.findall(text))
    if bullets >= 1:
        return min(bullets, 10)
    marker_hits = sum(m.lower() in text.lower() for m in _ISSUE_MARKERS)
    return min(marker_hits, 10)
