"""Unit tests for _normalize_citations — fixes the writer's ugly citation format."""
from lra.pipeline import _normalize_citations


def test_linky_arxiv_to_flat():
    s = "see [`2510.15624`](arxiv-id) and [`2506.09440`](arxiv-id)"
    assert _normalize_citations(s) == "see [2510.15624] and [2506.09440]"


def test_linky_repo_to_flat_with_repo():
    s = "workspace from [`2510.15624`](repo: freephdlabor) describes a workflow"
    assert _normalize_citations(s) == "workspace from [2510.15624, repo: freephdlabor] describes a workflow"


def test_bare_backtick_id():
    s = "MoE from `2506.09440` is transferred to `2603.19029`"
    assert _normalize_citations(s) == "MoE from [2506.09440] is transferred to [2603.19029]"


def test_inner_backticks_stripped():
    s = "[`synthesis`] and [`REUSE`] mentioned"
    assert _normalize_citations(s) == "[synthesis] and [REUSE] mentioned"


def test_nested_brackets_flattened():
    s = "### MoE [[2506.09440], [2603.19029]] (arxiv-id)"
    assert _normalize_citations(s) == "### MoE [2506.09440, 2603.19029]"


def test_dangling_suffix_removed():
    s = "### Data Agents [2602.04261] (arxiv-id)"
    assert _normalize_citations(s) == "### Data Agents [2602.04261]"


def test_dangling_repo_suffix_removed():
    s = "### freephdlabor [2510.15624] (repo: freephdlabor)"
    assert _normalize_citations(s) == "### freephdlabor [2510.15624]"


def test_combined_pipeline():
    raw = (
        "## Approaches\n\n"
        "### Dynamic [`2510.15624`](repo: freephdlabor)\n"
        "- see `2506.09440` and [`synthesis`]\n"
        "### MoE [[`2506.09440`], [`2603.19029`]] (arxiv-id)\n"
    )
    out = _normalize_citations(raw)
    assert "`" not in out
    assert "](arxiv-id)" not in out
    assert "](repo:" not in out
    assert "[2510.15624, repo: freephdlabor]" in out
    assert "[2506.09440, 2603.19029]" in out
    assert "[synthesis]" in out


def test_plain_citation_unchanged():
    s = "Already correct format: [2509.17158] and [repo: microsoft/autogen]"
    assert _normalize_citations(s) == s
