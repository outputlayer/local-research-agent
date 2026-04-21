"""Тесты метрик итераций и парсера критики."""
import json

from lra.metrics import CriticRound, IterationMetric, RunMetrics, count_critic_issues


def test_count_critic_issues_approved():
    assert count_critic_issues("APPROVED") == 0
    assert count_critic_issues("approved ") == 0
    assert count_critic_issues("") == 0


def test_count_critic_issues_bullets():
    text = """Вот правки:
- нет цитаты в параграфе 1
- дублирование абзаца
- SMART_PLAGIARISM в секции X
"""
    assert count_critic_issues(text) == 3


def test_count_critic_issues_numbered():
    text = """1. переформулировать
2. добавить [arxiv-id]
3. удалить дубль"""
    assert count_critic_issues(text) == 3


def test_count_critic_issues_capped_at_10():
    text = "\n".join(f"- issue {i}" for i in range(20))
    assert count_critic_issues(text) == 10


def test_count_critic_issues_marker_fallback():
    """Без bullet'ов, но с маркерами проблем — считаем по маркерам."""
    text = "В тексте есть дублирование и отсутствует в notes важное."
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
