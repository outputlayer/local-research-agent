"""Автосейв результатов hf_papers/github_search в kb.jsonl — модель часто забывает
kb_add вручную, KB должен заполняться сам."""
from __future__ import annotations

import json
from unittest.mock import patch


def _patch(tmp_path, monkeypatch):
    from lra import cache, config, kb, memory, tools
    monkeypatch.setattr(config, "QUERYLOG_PATH", tmp_path / "querylog.md")
    monkeypatch.setattr(memory, "QUERYLOG_PATH", tmp_path / "querylog.md")
    monkeypatch.setattr(tools, "QUERYLOG_PATH", tmp_path / "querylog.md")
    monkeypatch.setattr(kb, "KB_PATH", tmp_path / "kb.jsonl")
    monkeypatch.setattr(kb, "RESEARCH_DIR", tmp_path)
    monkeypatch.setattr(config, "RESEARCH_DIR", tmp_path)
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / ".cache")
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / ".cache")
    # Gate в hf_papers читает PLAN_PATH → нужно изолировать от реального
    # research/plan.md (он может содержать EW-тему и рубить тестовые fake papers).
    monkeypatch.setattr(config, "PLAN_PATH", tmp_path / "plan.md")
    monkeypatch.setattr(tools, "PLAN_PATH", tmp_path / "plan.md")
    return kb


def test_hf_papers_autosaves_to_kb(tmp_path, monkeypatch):
    kb = _patch(tmp_path, monkeypatch)
    from lra import tools
    from lra.cli import CliResult

    fake_papers = [
        {"id": "2401.00001", "title": "Test A",
         "authors": [{"name": "Alice"}, {"name": "Bob"}],
         "published_at": "2024-01-01", "summary": "summary A"},
        {"id": "2401.00002", "title": "Test B",
         "authors": [{"name": "Carol"}],
         "published_at": "2024-01-02", "summary": "summary B"},
    ]
    with patch("lra.tools.cli_run.run",
               return_value=CliResult(json.dumps(fake_papers), "", 0)):
        out = tools.HfPapers().call({"query": "novel unique topic keywords alpha", "limit": 5})
    assert "auto-saved to kb: 2" in out
    atoms = kb.load()
    ids = {a["id"] for a in atoms}
    assert ids == {"2401.00001", "2401.00002"}
    a1 = next(a for a in atoms if a["id"] == "2401.00001")
    assert a1["kind"] == "paper"
    assert a1["title"] == "Test A"
    assert "Alice" in a1["authors"]
    assert a1["claim"].startswith("summary")


def test_github_search_autosaves_repos_above_threshold(tmp_path, monkeypatch):
    kb = _patch(tmp_path, monkeypatch)
    from lra import tools
    from lra.cli import CliResult

    fake_repos = [
        {"fullName": "popular/repo", "url": "https://github.com/popular/repo",
         "description": "useful framework", "stargazersCount": 500,
         "language": "Python", "pushedAt": "2024-06-01"},
        {"fullName": "tiny/fork", "url": "https://github.com/tiny/fork",
         "description": "", "stargazersCount": 3,
         "language": "Python", "pushedAt": "2020-01-01"},
    ]
    with patch("lra.tools.cli_run.run",
               return_value=CliResult(json.dumps(fake_repos), "", 0)):
        out = tools.GithubSearch().call(
            {"query": "totally distinct github topic beta", "type": "repos", "limit": 5})
    # Только репо с stars >= 10 должен попасть в kb
    assert "auto-saved to kb: 1" in out
    atoms = kb.load()
    assert len(atoms) == 1
    a = atoms[0]
    assert a["id"] == "popular/repo"
    assert a["kind"] == "repo"
    assert a["stars"] == 500
    assert a["lang"] == "Python"


def test_github_search_code_does_not_autosave(tmp_path, monkeypatch):
    kb = _patch(tmp_path, monkeypatch)
    from lra import tools
    from lra.cli import CliResult

    fake_code = [{"path": "src/foo.py",
                  "url": "https://github.com/a/b/blob/main/src/foo.py",
                  "repository": {"fullName": "a/b"}}]
    with patch("lra.tools.cli_run.run",
               return_value=CliResult(json.dumps(fake_code), "", 0)):
        out = tools.GithubSearch().call(
            {"query": "code-only different search gamma", "type": "code", "limit": 3})
    assert "авто-сохранено" not in out
    assert kb.load() == []
