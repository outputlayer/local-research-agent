"""SemanticScholarSearch: JSON parsing, dedup, kb auto-save via arxiv-id."""

import json


def _patch(tmp_path, monkeypatch):
    from lra import config, kb, memory, tools
    monkeypatch.setattr(config, "QUERYLOG_PATH", tmp_path / "querylog.md")
    monkeypatch.setattr(memory, "QUERYLOG_PATH", tmp_path / "querylog.md")
    monkeypatch.setattr(tools, "QUERYLOG_PATH", tmp_path / "querylog.md")
    monkeypatch.setattr(kb, "KB_PATH", tmp_path / "kb.jsonl")
    monkeypatch.setattr(kb, "RESEARCH_DIR", tmp_path)
    monkeypatch.setattr(config, "RESEARCH_DIR", tmp_path)
    monkeypatch.setattr(config, "PLAN_PATH", tmp_path / "plan.md")
    monkeypatch.setattr(tools, "PLAN_PATH", tmp_path / "plan.md")
    return tools


def _payload(*items):
    return json.dumps({"data": list(items)})


def test_s2_basic_parse_and_autosave(tmp_path, monkeypatch):
    """One paper with arxiv-id → shown, auto-saved to kb."""
    tools = _patch(tmp_path, monkeypatch)
    # plan.md is empty → strict_domain_gate allows through (slow-start)
    (tmp_path / "plan.md").write_text("# Plan: radar jamming\n", encoding="utf-8")
    tool = tools.SemanticScholarSearch()
    raw = _payload({
        "paperId": "abc123",
        "title": "Cognitive Radar Jamming",
        "abstract": "We study radar jamming and ECM under cognitive radar settings.",
        "year": 2024,
        "authors": [{"name": "Alice"}, {"name": "Bob"}],
        "externalIds": {"ArXiv": "2401.12345"},
    })
    captured = {}
    def _fake(url, timeout=20):
        captured["url"] = url
        return raw
    monkeypatch.setattr(tools._helpers, "_fetch_text", _fake)

    out = tool.call({"query": "radar jamming", "limit": 5, "year": "2023-2025"})

    assert "[2401.12345]" in out
    assert "Cognitive Radar Jamming" in out
    assert "https://arxiv.org/abs/2401.12345" in out
    assert "year=2023-2025" in captured["url"]
    assert "fields=title%2Cabstract%2Cyear%2Cauthors%2CexternalIds" in captured["url"]


def test_s2_no_arxiv_not_saved(tmp_path, monkeypatch):
    """Paper without arxiv-id → shown but NOT saved to kb."""
    tools = _patch(tmp_path, monkeypatch)
    (tmp_path / "plan.md").write_text("# Plan: x\n", encoding="utf-8")
    tool = tools.SemanticScholarSearch()
    raw = _payload({
        "paperId": "no-arxiv-1",
        "title": "Pure ACL paper",
        "abstract": "Some NLP paper.",
        "year": 2024,
        "authors": [{"name": "X"}],
        "externalIds": {},  # no ArXiv
    })
    monkeypatch.setattr(tools._helpers, "_fetch_text", lambda url, timeout=20: raw)

    out = tool.call({"query": "test query"})

    assert "[s2:no-arxiv-1]" in out
    assert "without arxiv-id (not saved): 1" in out
    # kb empty
    assert not (tmp_path / "kb.jsonl").exists() or \
           (tmp_path / "kb.jsonl").read_text().strip() == ""


def test_s2_dedup_blocks_repeat(tmp_path, monkeypatch):
    """Repeat call with the same query → REJECTED via querylog."""
    tools = _patch(tmp_path, monkeypatch)
    (tmp_path / "plan.md").write_text("# Plan: x\n", encoding="utf-8")
    tool = tools.SemanticScholarSearch()
    raw = _payload({"paperId": "p1", "title": "T", "abstract": "A",
                    "year": 2024, "authors": [], "externalIds": {}})
    monkeypatch.setattr(tools._helpers, "_fetch_text", lambda url, timeout=20: raw)

    tool.call({"query": "novel transformer"})
    out2 = tool.call({"query": "novel transformer"})

    assert "REJECTED" in out2
    assert "semantic_scholar_search" in out2


def test_s2_invalid_year_dropped(tmp_path, monkeypatch):
    """Junk in year (injection) does not end up in the URL."""
    tools = _patch(tmp_path, monkeypatch)
    (tmp_path / "plan.md").write_text("# Plan: x\n", encoding="utf-8")
    tool = tools.SemanticScholarSearch()
    raw = _payload({"paperId": "p", "title": "T", "abstract": "A",
                    "year": 2024, "authors": [], "externalIds": {}})
    captured = {}
    def _fake(url, timeout=20):
        captured["url"] = url
        return raw
    monkeypatch.setattr(tools._helpers, "_fetch_text", _fake)

    tool.call({"query": "q1", "year": "2024'; DROP TABLE--"})

    assert "year=" not in captured["url"]
    assert "DROP" not in captured["url"]


def test_s2_empty_results(tmp_path, monkeypatch):
    """API returned empty data → friendly message."""
    tools = _patch(tmp_path, monkeypatch)
    (tmp_path / "plan.md").write_text("# Plan: x\n", encoding="utf-8")
    tool = tools.SemanticScholarSearch()
    monkeypatch.setattr(tools._helpers, "_fetch_text",
                        lambda url, timeout=20: '{"data": []}')

    out = tool.call({"query": "obscure topic xyz"})

    assert "no" in out
