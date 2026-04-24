"""Tests for iteration metrics and critic parser."""
import json

from lra.metrics import (
    CriticRound,
    IterationMetric,
    RunMetrics,
    count_critic_issues,
    summarize_evidence_quality,
)


def test_count_critic_issues_approved():
    assert count_critic_issues("APPROVED") == 0
    assert count_critic_issues("approved ") == 0
    assert count_critic_issues("") == 0


def test_count_critic_issues_bullets():
    text = """Here are the fixes:
- no citation in paragraph 1
- duplication of paragraph
- SMART_PLAGIARISM in section X
"""
    assert count_critic_issues(text) == 3


def test_count_critic_issues_numbered():
    text = """1. rephrase
2. add [arxiv-id]
3. remove duplicate"""
    assert count_critic_issues(text) == 3


def test_count_critic_issues_capped_at_10():
    text = "\n".join(f"- issue {i}" for i in range(20))
    assert count_critic_issues(text) == 10


def test_count_critic_issues_marker_fallback():
    """Without bullets, but with issue markers — count the markers."""
    text = "The text has duplication and is missing from notes an important fact."
    assert count_critic_issues(text) >= 1


def test_run_metrics_serialize(tmp_path):
    m = RunMetrics(query="test")
    m.iterations.append(IterationMetric(
        iteration=1, focus="foo",
        prefetch_seconds=0.5, explorer_seconds=2.0, replanner_seconds=1.0,
        notes_grew_chars=500, new_arxiv_ids=3,
        hf_from_cache=False, gh_from_cache=True, focus_changed=False,
    ))
    m.critic_rounds.append(CriticRound(round=1, issues_found=2, approved=False, seconds=1.5))
    m.stopped_early_reason = "PLAN_COMPLETE"
    path = m.finish(tmp_path / "metrics.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["query"] == "test"
    assert data["stopped_early_reason"] == "PLAN_COMPLETE"
    assert len(data["iterations"]) == 1
    assert data["iterations"][0]["new_arxiv_ids"] == 3
    assert data["iterations"][0]["gh_from_cache"] is True
    assert data["critic_rounds"][0]["issues_found"] == 2
    assert data["finished_at"] is not None


def test_run_metrics_total_seconds():
    m = RunMetrics(query="x")
    m.started_at = 100.0
    m.finished_at = 105.5
    assert m.total_seconds == 5.5


def test_summarize_evidence_quality_counts_sources_and_numbers():
    draft = (
        "# Report\n"
        "Result [2401.00001] uses metric 95%.\n"
        "Implementation [repo: foo/bar].\n"
        "This is a hypothetical reference architecture.\n"
    )
    notes = (
        "[2401.00001] dataset size 3B pulses and success 95%\n\n"
        "[2402.00002] unrelated note without citation in draft\n"
    )

    quality = summarize_evidence_quality(draft, notes, valid_ids=1)

    assert quality["unique_cited_paper_ids"] == 1
    assert quality["unique_cited_repos"] == 1
    assert quality["source_diversity"] == 2
    assert quality["numeric_evidence_count"] >= 2
    assert quality["speculation_markers_count"] >= 1
    assert quality["citation_coverage_ratio"] == 1.0
