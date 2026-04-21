"""Тесты fuzzy-dedup для защиты от перефразировок — реальная проблема из
research/querylog.md от 2026-04-21 (WebWeaver/AgentCPM запросы повторялись
8+ раз с минимальными изменениями и проходили точный dedup)."""
from __future__ import annotations


def _setup(tmp_path, monkeypatch, queries: list[str]):
    from lra import config, memory
    qpath = tmp_path / "querylog.md"
    monkeypatch.setattr(config, "QUERYLOG_PATH", qpath)
    monkeypatch.setattr(memory, "QUERYLOG_PATH", qpath)
    content = "# Query log\n\n## Session test\n"
    for q in queries:
        content += f"- {q}\n"
    qpath.write_text(content, encoding="utf-8")
    return memory


def test_fuzzy_catches_word_order_permutation(tmp_path, monkeypatch):
    m = _setup(tmp_path, monkeypatch, [
        "gh-repos: WebWeaver agent planner writer python stars pushedAt",
    ])
    # Перестановка + синоним — реальный случай из querylog
    hit = m.is_similar_to_seen("gh-repos: WebWeaver agent python planner writer pushedAt stars")
    assert hit is not None, "fuzzy должен ловить перестановку слов"


def test_fuzzy_catches_added_noise_words(tmp_path, monkeypatch):
    m = _setup(tmp_path, monkeypatch, [
        "gh-repos: AgentCPM github repository python stars",
    ])
    # Добавлены "pushedAt" и "framework" — но основа та же
    hit = m.is_similar_to_seen("gh-repos: AgentCPM github repository python stars pushedAt framework")
    assert hit is not None


def test_fuzzy_allows_truly_different_topic(tmp_path, monkeypatch):
    m = _setup(tmp_path, monkeypatch, [
        "gh-repos: WebWeaver agent planner writer python",
    ])
    # Совсем другая тема
    hit = m.is_similar_to_seen("hf_papers: reinforcement learning from human feedback alignment")
    assert hit is None


def test_fuzzy_ignores_short_queries(tmp_path, monkeypatch):
    m = _setup(tmp_path, monkeypatch, [
        "gh-repos: quick",
    ])
    # Очень короткий запрос (< 3 keyword tokens) — не блокируем
    assert m.is_similar_to_seen("gh-repos: quick test x") is None


def test_fuzzy_no_querylog_returns_none(tmp_path, monkeypatch):
    from lra import config, memory
    monkeypatch.setattr(config, "QUERYLOG_PATH", tmp_path / "nope.md")
    monkeypatch.setattr(memory, "QUERYLOG_PATH", tmp_path / "nope.md")
    assert memory.is_similar_to_seen("anything with many keywords in it") is None


def test_fuzzy_identical_is_blocked(tmp_path, monkeypatch):
    m = _setup(tmp_path, monkeypatch, [
        "autonomous research agents architecture modern approach",
    ])
    hit = m.is_similar_to_seen("autonomous research agents architecture modern approach")
    assert hit == "autonomous research agents architecture modern approach"
