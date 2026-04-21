"""Юниты для github_search tool — cli.run замокан, реальный CLI не вызывается."""
import json

import pytest

from lra.cli import CliResult


@pytest.fixture
def gh_tool(tmp_path, monkeypatch):
    from lra import memory, tools as tools_mod
    ql = tmp_path / "querylog.md"
    monkeypatch.setattr(memory, "QUERYLOG_PATH", ql)
    monkeypatch.setattr(memory, "RESEARCH_DIR", tmp_path)
    monkeypatch.setattr(memory, "ARCHIVE_DIR", tmp_path / "archive")
    monkeypatch.setattr(tools_mod, "QUERYLOG_PATH", ql)
    return tools_mod.GithubSearch(), tools_mod, tmp_path


def _ok(payload_json: str):
    return lambda cmd, **kw: CliResult(payload_json, "", 0)


def _err(stderr: str, rc: int = 1):
    return lambda cmd, **kw: CliResult("", stderr, rc)


def test_empty_query_rejected(gh_tool):
    tool, _, _ = gh_tool
    assert "обязателен" in tool.call({"query": ""})


def test_repos_happy_path(gh_tool, monkeypatch):
    tool, tools_mod, _ = gh_tool
    payload = json.dumps([{
        "fullName": "hf/transformers",
        "url": "https://github.com/hf/transformers",
        "description": "State-of-the-art ML",
        "stargazersCount": 120000,
        "language": "Python",
        "pushedAt": "2026-04-20T00:00:00Z",
    }])
    monkeypatch.setattr(tools_mod.cli_run, "run", _ok(payload))
    out = tool.call({"query": "transformers", "type": "repos", "limit": 1})
    assert "hf/transformers" in out
    assert "120000" in out
    assert "Python" in out
    assert "2026-04-20" in out


def test_code_search(gh_tool, monkeypatch):
    tool, tools_mod, _ = gh_tool
    payload = json.dumps([{
        "path": "src/agent.py",
        "url": "https://github.com/x/y/blob/main/src/agent.py",
        "repository": {"fullName": "x/y"},
    }])
    monkeypatch.setattr(tools_mod.cli_run, "run", _ok(payload))
    out = tool.call({"query": "def research_loop", "type": "code"})
    assert "x/y" in out
    assert "src/agent.py" in out


def test_invalid_type_falls_back_to_repos(gh_tool, monkeypatch):
    tool, tools_mod, _ = gh_tool
    captured = {}

    def _spy(cmd, **kw):
        captured["cmd"] = cmd
        return CliResult("[]", "", 0)

    monkeypatch.setattr(tools_mod.cli_run, "run", _spy)
    tool.call({"query": "x", "type": "garbage"})
    assert captured["cmd"][2] == "repos"


def test_no_results_message(gh_tool, monkeypatch):
    tool, tools_mod, _ = gh_tool
    monkeypatch.setattr(tools_mod.cli_run, "run", _ok("[]"))
    out = tool.call({"query": "nonexistent-" + "x" * 20})
    assert "нет результатов" in out.lower()


def test_auth_error_suggests_login(gh_tool, monkeypatch):
    tool, tools_mod, _ = gh_tool
    monkeypatch.setattr(tools_mod.cli_run, "run", _err("error: you must authenticate"))
    out = tool.call({"query": "whatever"})
    assert "gh auth login" in out


def test_missing_cli_message(gh_tool, monkeypatch):
    tool, tools_mod, _ = gh_tool
    monkeypatch.setattr(tools_mod.cli_run, "run", _err("команда не найдена: gh", rc=127))
    out = tool.call({"query": "x"})
    assert "brew install gh" in out or "не найден" in out


def test_dedup_via_querylog(gh_tool, monkeypatch):
    tool, tools_mod, tmp = gh_tool
    monkeypatch.setattr(tools_mod.cli_run, "run", _ok("[]"))
    tool.call({"query": "autonomous agents", "type": "repos"})
    out2 = tool.call({"query": "autonomous agents", "type": "repos"})
    assert "ОТКАЗ" in out2
    assert "gh-repos" in (tmp / "querylog.md").read_text(encoding="utf-8")
