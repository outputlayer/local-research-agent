"""Iteration metrics: timing, id growth, critic convergence.

Exported to research/metrics.json for external analytics and integration tests.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import RESEARCH_DIR
from .utils import extract_ids

METRICS_PATH = RESEARCH_DIR / "metrics.json"
_REPO_CITATION_RE = re.compile(r"\[repo:\s*([^\]\n]+?)\]")
_NUMERIC_RE = re.compile(r"\b\d+(?:\.\d+)?(?:x|%|k|m|b)?\b", re.IGNORECASE)
_SPECULATION_MARKERS = (
    "hypothetical",
    "insufficient evidence",
    "extrapolation",
    "next logical step",
    "speculative",
)


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
    finished_at: float | None = None
    iterations: list[IterationMetric] = field(default_factory=list)
    critic_rounds: list[CriticRound] = field(default_factory=list)
    synthesis_seconds: float = 0.0
    writer_seconds: float = 0.0
    valid_ids: int = 0
    invalid_ids: list[str] = field(default_factory=list)
    suspicious_citations: list[str] = field(default_factory=list)
    final_draft_chars: int = 0
    stopped_early_reason: str | None = None
    unique_cited_paper_ids: int = 0
    unique_cited_repos: int = 0
    source_diversity: int = 0
    numeric_evidence_count: int = 0
    speculation_markers_count: int = 0
    citation_coverage_ratio: float = 0.0

    @property
    def total_seconds(self) -> float:
        end = self.finished_at or time.time()
        return end - self.started_at

    def finish(self, path: Path | None = None) -> Path:
        """Saves metrics to JSON. Called at the end of research_loop."""
        self.finished_at = time.time()
        target = path or METRICS_PATH
        target.parent.mkdir(exist_ok=True)
        target.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return target


def summarize_evidence_quality(draft_text: str, notes_text: str, *, valid_ids: int = 0) -> dict[str, int | float]:
    """Counts simple quality metrics over draft+notes without external calls."""
    cited_ids = extract_ids(draft_text or "")
    cited_repos = {m.group(1).strip() for m in _REPO_CITATION_RE.finditer(draft_text or "")}

    notes_blocks = [block for block in re.split(r"\n\s*\n", notes_text or "") if block.strip()]
    numeric_evidence_count = 0
    if cited_ids:
        for block in notes_blocks:
            if any(f"[{pid}]" in block for pid in cited_ids):
                numeric_evidence_count += len(_NUMERIC_RE.findall(block))

    draft_lower = (draft_text or "").lower()
    speculation_markers_count = sum(draft_lower.count(marker) for marker in _SPECULATION_MARKERS)
    unique_cited_paper_ids = len(cited_ids)
    unique_cited_repos = len(cited_repos)
    source_diversity = unique_cited_paper_ids + unique_cited_repos
    citation_coverage_ratio = (
        valid_ids / unique_cited_paper_ids if unique_cited_paper_ids else 0.0
    )

    return {
        "unique_cited_paper_ids": unique_cited_paper_ids,
        "unique_cited_repos": unique_cited_repos,
        "source_diversity": source_diversity,
        "numeric_evidence_count": numeric_evidence_count,
        "speculation_markers_count": speculation_markers_count,
        "citation_coverage_ratio": round(citation_coverage_ratio, 4),
    }


# ---- Parsing the critic response for precise convergence ----

_ISSUE_MARKERS = (
    "no citation", "missing from notes", "contradicts notes",
    "duplication", "SMART_PLAGIARISM", "Novel Insights",
)
_ACTION_RE = re.compile(r"^\s*(?:[\-\*]|\d+[\.\)])\s+", re.MULTILINE)


def count_critic_issues(text: str) -> int:
    """Estimates the number of concrete edits in the critic response.

    Strategy:
    - if APPROVED is present — return 0.
    - otherwise count bullet/numbered items; if none, count occurrences of issue markers.
    Capped at 10 so we do not overreact to noise.
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
