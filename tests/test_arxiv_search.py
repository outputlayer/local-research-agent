"""Fallback search over the arXiv API: feed parsing, freshness, autosave into KB."""

from datetime import UTC, datetime, timedelta


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


def _feed(*entries: dict[str, str]) -> str:
    parts = [
        "<?xml version='1.0' encoding='UTF-8'?>",
        "<feed xmlns='http://www.w3.org/2005/Atom'>",
    ]
    for item in entries:
        authors = "".join(
            f"<author><name>{name}</name></author>" for name in item.get("authors", ["A"])
        )
        parts.append(
            "<entry>"
            f"<id>http://arxiv.org/abs/{item['id']}</id>"
            f"<title>{item['title']}</title>"
            f"<summary>{item['summary']}</summary>"
            f"<published>{item['published_at']}T00:00:00Z</published>"
            f"{authors}"
            "</entry>"
        )
    parts.append("</feed>")
    return "".join(parts)


def test_arxiv_search_filters_old_and_autosaves(tmp_path, monkeypatch):
    tools = _patch(tmp_path, monkeypatch)
    tool = tools.ArxivSearch()
    old = (datetime.now(UTC) - timedelta(days=1500)).date().isoformat()
    new = (datetime.now(UTC) - timedelta(days=60)).date().isoformat()
    xml = _feed(
        {"id": "2401.00001v2", "title": "Fresh Radar EW", "summary": "electronic warfare radar sensing",
         "published_at": new, "authors": ["Alice", "Bob"]},
        {"id": "2201.00001v1", "title": "Old Radar EW", "summary": "old radar result",
         "published_at": old, "authors": ["Carol"]},
    )
    monkeypatch.setattr(tools._helpers, "_fetch_text", lambda url, timeout=20: xml)

    out = tool.call({"query": "electronic warfare radar", "limit": 5})

    assert "2401.00001" in out
    assert "2201.00001" not in out
    assert "auto-saved to kb: 1" in out


def test_arxiv_search_fallback_when_all_old(tmp_path, monkeypatch):
    tools = _patch(tmp_path, monkeypatch)
    tool = tools.ArxivSearch()
    old = (datetime.now(UTC) - timedelta(days=1500)).date().isoformat()
    xml = _feed(
        {"id": "2201.00001v1", "title": "Legacy Paper", "summary": "legacy result",
         "published_at": old, "authors": ["Alice"]},
    )
    monkeypatch.setattr(tools._helpers, "_fetch_text", lambda url, timeout=20: xml)

    out = tool.call({"query": "legacy niche radar", "limit": 5})

    assert "2201.00001" in out
    assert "older" in out.lower() or "fallback" in out.lower() or "stale" in out.lower()


def test_arxiv_search_domain_gate_skips_offtopic_autosave(tmp_path, monkeypatch):
    tools = _patch(tmp_path, monkeypatch)
    tool = tools.ArxivSearch()
    (tmp_path / "plan.md").write_text(
        "# Plan: Modern approaches in electronic warfare and ELINT\n\n## [TODO]\n- [T1] radar jamming\n",
        encoding="utf-8",
    )
    fresh = (datetime.now(UTC) - timedelta(days=30)).date().isoformat()
    xml = _feed(
        {"id": "2501.00001v1", "title": "Audio Dialogue Support",
         "summary": "emotional support dialogue generation with empathy",
         "published_at": fresh, "authors": ["Alice"]},
    )
    monkeypatch.setattr(tools._helpers, "_fetch_text", lambda url, timeout=20: xml)

    out = tool.call({"query": "support dialogue", "limit": 5})

    assert "2501.00001" in out
    assert "filtered by domain gate: 1" in out


def test_arxiv_search_categories_in_query(tmp_path, monkeypatch):
    """categories=['eess.SP','cs.IT'] → search_query contains cat:eess.SP OR cat:cs.IT."""
    tools = _patch(tmp_path, monkeypatch)
    tool = tools.ArxivSearch()
    fresh = (datetime.now(UTC) - timedelta(days=30)).date().isoformat()
    xml = _feed(
        {"id": "2502.99999v1", "title": "Cat-filtered paper",
         "summary": "signal processing electronic warfare",
         "published_at": fresh, "authors": ["X"]},
    )
    captured = {}
    def _fake_fetch(url, timeout=20):
        captured["url"] = url
        return xml
    monkeypatch.setattr(tools._helpers, "_fetch_text", _fake_fetch)

    out = tool.call({"query": "jamming detection", "limit": 3,
                     "categories": ["eess.SP", "cs.IT"]})

    assert "2502.99999" in out
    url = captured["url"]
    # URL-encoded: spaces as +, AND/OR/cat: inside
    assert "cat%3Aeess.SP" in url
    assert "cat%3Acs.IT" in url
    assert "+OR+" in url
    assert "+AND+all%3Ajamming" in url


def test_arxiv_search_categories_sanitized(tmp_path, monkeypatch):
    """Invalid categories are dropped (protection against query-string injection)."""
    tools = _patch(tmp_path, monkeypatch)
    tool = tools.ArxivSearch()
    fresh = (datetime.now(UTC) - timedelta(days=30)).date().isoformat()
    xml = _feed(
        {"id": "2503.11111v1", "title": "P", "summary": "signal processing",
         "published_at": fresh, "authors": ["Y"]},
    )
    captured = {}
    def _fake(url, timeout=20):
        captured["url"] = url
        return xml
    monkeypatch.setattr(tools._helpers, "_fetch_text", _fake)

    # Mixed: one valid, two junk
    tool.call({"query": "test query A", "limit": 2,
               "categories": ["eess.SP", "drop table users", "x' OR '1"]})

    url = captured["url"]
    assert "cat%3Aeess.SP" in url
    # no junk leaked
    assert "drop" not in url.lower()
    assert "1%27" not in url
