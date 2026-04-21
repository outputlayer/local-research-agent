"""Юниты для github_search tool — cli.run замокан, реальный CLI не вызывается."""
import json

import pytest

from lra.cli import CliResult


@pytest.fixture
def gh_tool(tmp_path, monkeypatch):
    from lra import memory
    from lra import tools as tools_mod
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


def test_inline_stars_qualifier_is_extracted_to_flag(gh_tool, monkeypatch):
    """Модель часто лепит 'stars:>=10' в query — должно вырезаться и уходить в --stars."""
    tool, tools_mod, _ = gh_tool
    captured = {}

    def _spy(cmd, **kw):
        captured["cmd"] = cmd
        return CliResult("[]", "", 0)

    monkeypatch.setattr(tools_mod.cli_run, "run", _spy)
    tool.call({"query": "multi-agent orchestration stars:>=10", "type": "repos"})
    cmd = captured["cmd"]
    # qualifier удалён из позиционного аргумента query
    assert "stars:>=10" not in cmd[3]
    assert cmd[3] == "multi-agent orchestration"
    # и добавлен как флаг
    assert "--stars" in cmd
    assert ">=10" in cmd[cmd.index("--stars") + 1]


def test_inline_language_qualifier_is_extracted_to_flag(gh_tool, monkeypatch):
    tool, tools_mod, _ = gh_tool
    captured = {}

    def _spy(cmd, **kw):
        captured["cmd"] = cmd
        return CliResult("[]", "", 0)

    monkeypatch.setattr(tools_mod.cli_run, "run", _spy)
    tool.call({"query": "RAG pipeline language:python", "type": "repos"})
    cmd = captured["cmd"]
    assert "language:python" not in cmd[3]
    assert cmd[3] == "RAG pipeline"
    assert "--language" in cmd
    assert cmd[cmd.index("--language") + 1] == "python"


def test_explicit_min_stars_param_takes_precedence(gh_tool, monkeypatch):
    tool, tools_mod, _ = gh_tool
    captured = {}

    def _spy(cmd, **kw):
        captured["cmd"] = cmd
        return CliResult("[]", "", 0)

    monkeypatch.setattr(tools_mod.cli_run, "run", _spy)
    tool.call({"query": "x stars:>=5", "type": "repos", "min_stars": 100})
    cmd = captured["cmd"]
    # explicit param побеждает inline qualifier
    assert ">=100" in cmd[cmd.index("--stars") + 1]


def test_empty_query_after_stripping_qualifiers_returns_error(gh_tool, monkeypatch):
    tool, _, _ = gh_tool
    out = tool.call({"query": "stars:>=10 language:python", "type": "repos"})
    assert "пустой" in out.lower() or "ошибка" in out.lower()


def test_no_results_hint_for_long_query(gh_tool, monkeypatch):
    tool, tools_mod, _ = gh_tool
    monkeypatch.setattr(tools_mod.cli_run, "run", _ok("[]"))
    out = tool.call({
        "query": "LangGraph multi-agent orchestration state machine workflow",
        "type": "repos",
    })
    assert "нет результатов" in out.lower()
    assert "сократи" in out.lower()
