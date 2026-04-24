"""Unit tests for the github_search tool — cli.run is mocked; the real CLI is not called."""
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
    assert "is required" in tool.call({"query": ""})


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
    assert "no" in out.lower()


def test_auth_error_suggests_login(gh_tool, monkeypatch):
    tool, tools_mod, _ = gh_tool
    monkeypatch.setattr(tools_mod.cli_run, "run", _err("error: you must authenticate"))
    out = tool.call({"query": "whatever"})
    assert "gh auth login" in out


def test_missing_cli_message(gh_tool, monkeypatch):
    tool, tools_mod, _ = gh_tool
    monkeypatch.setattr(tools_mod.cli_run, "run", _err("command not found: gh", rc=127))
    out = tool.call({"query": "x"})
    assert "brew install gh" in out or "not found" in out


def test_dedup_via_querylog(gh_tool, monkeypatch):
    tool, tools_mod, tmp = gh_tool
    monkeypatch.setattr(tools_mod.cli_run, "run", _ok("[]"))
    tool.call({"query": "autonomous agents", "type": "repos"})
    out2 = tool.call({"query": "autonomous agents", "type": "repos"})
    assert "REJECTED" in out2
    assert "gh-repos" in (tmp / "querylog.md").read_text(encoding="utf-8")


def test_inline_stars_qualifier_is_extracted_to_flag(gh_tool, monkeypatch):
    """The model often puts 'stars:>=10' into query — it must be stripped and moved to --stars."""
    tool, tools_mod, _ = gh_tool
    captured = {}

    def _spy(cmd, **kw):
        captured["cmd"] = cmd
        return CliResult("[]", "", 0)

    monkeypatch.setattr(tools_mod.cli_run, "run", _spy)
    tool.call({"query": "multi-agent orchestration stars:>=10", "type": "repos"})
    cmd = captured["cmd"]
    # qualifier removed from the positional query arg
    assert "stars:>=10" not in cmd[3]
    assert cmd[3] == "multi-agent orchestration"
    # and added as a flag
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
    # explicit param beats inline qualifier
    assert ">=100" in cmd[cmd.index("--stars") + 1]


def test_empty_query_after_stripping_qualifiers_returns_error(gh_tool, monkeypatch):
    tool, _, _ = gh_tool
    out = tool.call({"query": "stars:>=10 language:python", "type": "repos"})
    assert "empty" in out.lower() or "error" in out.lower()


def test_long_query_rejected_before_network(gh_tool, monkeypatch):
    """Long (>MAX_GITHUB_QUERY_WORDS) queries are rejected before calling gh CLI."""
    tool, tools_mod, _ = gh_tool
    called = {"n": 0}

    def _counter(argv, timeout=None):
        called["n"] += 1
        return _ok("[]")(argv, timeout=timeout)

    monkeypatch.setattr(tools_mod.cli_run, "run", _counter)
    out = tool.call({
        "query": "LangGraph multi-agent orchestration state machine workflow",
        "type": "repos",
    })
    assert called["n"] == 0, "a long query must not invoke gh CLI"
    assert "reject" in out.lower()
    assert "too long" in out.lower()
    # Must include a concrete hint — a shortened variant
    assert "langgraph" in out.lower()


def test_short_query_zero_results_still_works(gh_tool, monkeypatch):
    """After rejecting long ones — a short query with zero results still gives a hint."""
    tool, tools_mod, _ = gh_tool
    monkeypatch.setattr(tools_mod.cli_run, "run", _ok("[]"))
    out = tool.call({"query": "langgraph orchestration", "type": "repos"})
    assert "no" in out.lower() or "reject" in out.lower()
