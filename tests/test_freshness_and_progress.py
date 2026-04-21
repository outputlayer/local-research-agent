"""Freshness-фильтры (github/hf_papers) + render_md с прогресс-строкой."""
import json
from datetime import UTC, datetime, timedelta, timezone

import pytest

from lra.cli import CliResult

# ── Fixtures ────────────────────────────────────────────────────────────────

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


@pytest.fixture
def hf_tool(tmp_path, monkeypatch):
    from lra import memory
    from lra import tools as tools_mod
    ql = tmp_path / "querylog.md"
    kb = tmp_path / "kb.jsonl"
    monkeypatch.setattr(memory, "QUERYLOG_PATH", ql)
    monkeypatch.setattr(memory, "RESEARCH_DIR", tmp_path)
    monkeypatch.setattr(memory, "ARCHIVE_DIR", tmp_path / "archive")
    monkeypatch.setattr(tools_mod, "QUERYLOG_PATH", ql)
    from lra import kb as kb_mod
    monkeypatch.setattr(kb_mod, "KB_PATH", kb)
    return tools_mod.HfPapers(), tools_mod, tmp_path


# ── github_search: --updated фильтр и fallback ──────────────────────────────

def test_github_uses_updated_flag_for_repos(gh_tool, monkeypatch):
    tool, tools_mod, _ = gh_tool
    all_argvs: list[list[str]] = []

    # Возвращаем непустой результат на первом же вызове, чтобы fallback не сработал
    def _capture(argv, timeout=None):
        all_argvs.append(list(argv))
        payload = json.dumps([{
            "fullName": "owner/repo", "url": "https://github.com/owner/repo",
            "description": "x", "stargazersCount": 20, "language": "Python",
            "pushedAt": "2025-01-01T00:00:00Z",
        }])
        return CliResult(payload, "", 0)

    monkeypatch.setattr(tools_mod.cli_run, "run", _capture)
    tool.call({"query": "rag pipelines", "type": "repos"})
    assert len(all_argvs) == 1
    argv = all_argvs[0]
    assert "--updated" in argv
    val = argv[argv.index("--updated") + 1]
    assert val.startswith(">=")
    cutoff_str = val.removeprefix(">=")
    cutoff = datetime.fromisoformat(cutoff_str).date()
    year_ago = (datetime.now(UTC) - timedelta(days=370)).date()
    assert cutoff >= year_ago


def test_github_falls_back_without_updated_if_first_empty(gh_tool, monkeypatch):
    """Если с --updated ноль результатов → второй запрос без фильтра + stale-note."""
    tool, tools_mod, _ = gh_tool
    calls = []

    fresh_result = json.dumps([])
    stale_result = json.dumps([{
        "fullName": "old/repo", "url": "https://github.com/old/repo",
        "description": "legacy", "stargazersCount": 50,
        "language": "Python", "pushedAt": "2022-05-01T00:00:00Z",
    }])

    def _run(argv, timeout=None):
        calls.append(list(argv))
        # первый — с --updated, пустой; второй — без, с результатом
        if "--updated" in argv:
            return CliResult(fresh_result, "", 0)
        return CliResult(stale_result, "", 0)

    monkeypatch.setattr(tools_mod.cli_run, "run", _run)
    out = tool.call({"query": "legacy stuff", "type": "repos"})
    assert len(calls) == 2
    assert "--updated" in calls[0]
    assert "--updated" not in calls[1]
    assert "old/repo" in out
    assert "fallback" in out.lower() or "старые" in out.lower() or "старые" in out.lower()


def test_github_code_search_no_freshness_filter(gh_tool, monkeypatch):
    """type=code не поддерживает --updated у gh, не должны его добавлять."""
    tool, tools_mod, _ = gh_tool
    captured = {}

    def _capture(argv, timeout=None):
        captured["argv"] = list(argv)
        return CliResult("[]", "", 0)

    monkeypatch.setattr(tools_mod.cli_run, "run", _capture)
    tool.call({"query": "lora adapter", "type": "code"})
    assert "--updated" not in captured["argv"]


# ── hf_papers: фильтр по published_at ───────────────────────────────────────

def test_hf_papers_filters_old(hf_tool, monkeypatch):
    tool, tools_mod, _ = hf_tool
    old = (datetime.now(UTC) - timedelta(days=1500)).date().isoformat()
    new = (datetime.now(UTC) - timedelta(days=90)).date().isoformat()
    payload = json.dumps([
        {"id": "2020.00001", "title": "Old paper", "summary": "stale",
         "authors": [{"name": "A"}], "published_at": f"{old}T00:00:00Z"},
        {"id": "2025.00001", "title": "Fresh paper", "summary": "hot",
         "authors": [{"name": "B"}], "published_at": f"{new}T00:00:00Z"},
    ])
    monkeypatch.setattr(tools_mod.cli_run, "run",
                        lambda argv, **kw: CliResult(payload, "", 0))
    out = tool.call({"query": "something", "limit": 5})
    assert "2025.00001" in out
    # Старая не должна проходить (>2 лет)
    assert "2020.00001" not in out


def test_hf_papers_fallback_when_all_old(hf_tool, monkeypatch):
    """Если все статьи старше порога — всё равно показываем (лучше что-то чем ничего) с пометкой."""
    tool, tools_mod, _ = hf_tool
    old = (datetime.now(UTC) - timedelta(days=1500)).date().isoformat()
    payload = json.dumps([
        {"id": "2020.00001", "title": "Ancient", "summary": "yep",
         "authors": [{"name": "A"}], "published_at": f"{old}T00:00:00Z"},
    ])
    monkeypatch.setattr(tools_mod.cli_run, "run",
                        lambda argv, **kw: CliResult(payload, "", 0))
    out = tool.call({"query": "niche", "limit": 5})
    assert "2020.00001" in out
    # должна быть явная пометка что fallback
    assert "старше" in out.lower() or "fallback" in out.lower()


# ── render_md: прогресс-строка ──────────────────────────────────────────────

def test_render_md_shows_progress_counts(tmp_path, monkeypatch):
    from lra import plan as plan_mod
    md = tmp_path / "plan.md"
    monkeypatch.setattr(plan_mod, "PLAN_JSON_PATH", tmp_path / "plan.json")
    monkeypatch.setattr(plan_mod, "PLAN_PATH", md)

    plan = plan_mod.reset("test topic")  # 5 open tasks by default
    # Закрываем две задачи, блокируем одну
    plan.close_task("T1", why="test")
    plan.close_task("T2", why="test")
    plan.block_task("T3", iter_=0, why="test")
    plan_mod.save(plan)

    text = md.read_text(encoding="utf-8")
    assert "Прогресс:" in text
    assert "2/5 done" in text or "done (40%)" in text
    assert "blocked=1" in text


def test_render_md_full_progress_on_all_done(tmp_path, monkeypatch):
    from lra import plan as plan_mod
    md = tmp_path / "plan.md"
    monkeypatch.setattr(plan_mod, "PLAN_JSON_PATH", tmp_path / "plan.json")
    monkeypatch.setattr(plan_mod, "PLAN_PATH", md)

    plan = plan_mod.reset("x")
    for t in list(plan.tasks):
        plan.close_task(t.id, why="done")
    plan_mod.save(plan)
    text = md.read_text(encoding="utf-8")
    assert "100%" in text
    assert "PLAN_COMPLETE" in text
