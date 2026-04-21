"""Unit-тесты _normalize_citations — чинит уродливый формат цитат writer'а."""
from lra.pipeline import _normalize_citations


def test_linky_arxiv_to_flat():
    s = "см. [`2510.15624`](arxiv-id) и [`2506.09440`](arxiv-id)"
    assert _normalize_citations(s) == "см. [2510.15624] и [2506.09440]"


def test_linky_repo_to_flat_with_repo():
    s = "workspace из [`2510.15624`](repo: freephdlabor) описывает воркфлоу"
    assert _normalize_citations(s) == "workspace из [2510.15624, repo: freephdlabor] описывает воркфлоу"


def test_bare_backtick_id():
    s = "MoE из `2506.09440` переносится в `2603.19029`"
    assert _normalize_citations(s) == "MoE из [2506.09440] переносится в [2603.19029]"


def test_inner_backticks_stripped():
    s = "[`synthesis`] и [`REUSE`] упомянуты"
    assert _normalize_citations(s) == "[synthesis] и [REUSE] упомянуты"


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
        "## Подходы\n\n"
        "### Dynamic [`2510.15624`](repo: freephdlabor)\n"
        "- см. `2506.09440` и [`synthesis`]\n"
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
    s = "Это уже правильный формат: [2509.17158] и [repo: microsoft/autogen]"
    assert _normalize_citations(s) == s
