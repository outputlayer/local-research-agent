"""Юниты для github_search tool — subprocess замокан, CLI не вызывается."""
import json
import types

import pytest


@pytest.fixture
def gh_tool(tmp_path, monkeypatch):
    """Перенаправляет querylog в tmp_path без reload (чтобы не переregistrировать tools)."""
    from lra import memory, tools as tools_mod
    ql = tmp_path / "querylog.md"
    monkeypatch.setattr(memory, "QUERYLOG_PATH", ql)
    monkeypatch.setattr(memory, "RESEARCH_DIR", tmp_path)
    monkeypatch.setattr(memory, "ARCHIVE_DIR", tmp_path / "archive")
    monkeypatch.setattr(tools_mod, "QUERYLOG_PATH", ql)
    return tools_mod.GithubSearch(), tools_mod, tmp_path


def _fake_run_ok(payload):
    """Возвращает mock subprocess.run, отдающий JSON payload с returncode=0."""
    def _run(cmd, **kw):
        return types.SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")
    return _run


def _fake_run_err(stderr, returncode=1):
    def _run(cmd, **kw):
        return types.SimpleNamespace(returncode=returncode, stdout="", stderr=stderr)
    return _run


def test_empty_query_rejected(gh_tool):
    tool, _, _ = gh_tool
    assert "обязателен" in tool.call({"query": ""})


def test_repos_happy_path(gh_tool, monkeypatch):
    tool, tools_mod, _ = gh_tool
    payload = [{
        "fullName": "hf/transformers",
        "url": "https://github.com/hf/transformers",
        "description": "State-of-the-art ML",
        "stargazersCount": 120000,
        "language": "Python",
        "pushedAt": "2026-04-20T00:00:00Z",
    }]
    monkeypatch.setattr(tools_mod.subprocess, "run", _fake_run_ok(payload))
    out = tool.call({"query": "transformers", "type": "repos", "limit": 1})
    assert "hf/transformers" in out
    assert "120000" in out
    assert "Python" in out
    assert "2026-04-20" in out


def test_code_search(gh_tool, monkeypatch):
    tool, tools_mod, _ = gh_tool
    payload = [{
        "path": "src/agent.py",
        "url": "https://github.com/x/y/blob/main/src/agent.py",
        "repository": {"fullName": "x/y"},
    }]
    monkeypatch.setattr(tools_mod.subprocess, "run", _fake_run_ok(payload))
    out = tool.call({"query": "def research_loop", "type": "code"})
    assert "x/y" in out
    assert "src/agent.py" in out


def test_invalid_type_falls_back_to_repos(gh_tool, monkeypatch):
    tool, tools_mod, _ = gh_tool
    captured = {}

    def _spy(cmd, **kw):
        captured["cmd"] = cmd
        return types.SimpleNamespace(returncode=0, stdout="[]", stderr="")

    monkeypatch.setattr(tools_mod.subprocess, "run", _spy)
    tool.call({"query": "x", "type": "garbage"})
    assert captured["cmd"][2] == "repos"  # gh search repos x


def test_no_results_message(gh_tool, monkeypatch):
    tool, tools_mod, _ = gh_tool
    monkeypatch.setattr(tools_mod.subprocess, "run", _fake_run_ok([]))
    out = tool.call({"query": "nonexistent-" + "x" * 20})
    assert "нет результатов" in out.lower()


def test_auth_error_suggests_login(gh_tool, monkeypatch):
    tool, tools_mod, _ = gh_tool
    monkeypatch.setattr(
        tools_mod.subprocess, "run",
        _fake_run_err("error: you must authenticate"),
    )
    out = tool.call({"query": "whatever"})
    assert "gh auth login" in out


def test_missing_cli_message(gh_tool, monkeypatch):
    tool, tools_mod, _ = gh_tool

    def _missing(cmd, **kw):
        raise FileNotFoundError()

    monkeypatch.setattr(tools_mod.subprocess, "run", _missing)
    out = tool.call({"query": "x"})
    assert "brew install gh" in out or "не найден" in out


def test_dedup_via_querylog(gh_tool, monkeypatch):
    tool, tools_mod, tmp = gh_tool
    monkeypatch.setattr(tools_mod.subprocess, "run", _fake_run_ok([]))
    # Первый вызов проходит
    tool.call({"query": "autonomous agents", "type": "repos"})
    # Второй — ожидаем отказ
    out2 = tool.call({"query": "autonomous agents", "type": "repos"})
    assert "ОТКАЗ" in out2
    # В querylog запись появилась с префиксом gh-
    assert "gh-repos" in (tmp / "querylog.md").read_text()
