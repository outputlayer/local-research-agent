"""P7: canonicalize the ## Sources section from body citations."""
from lra.pipeline import _canonicalize_sources_section


def test_removes_invalid_ids_from_sources():
    """invalid_ids do not land in the canonical Sources list (body keeps them as-is)."""
    text = """# Report

Here is a body citation [2401.00001] and [2607.15491] invalid one.

## Sources
[2401.00001]
[2607.15491]
"""
    out, changed = _canonicalize_sources_section(text, ["2607.15491"])
    assert changed
    # Pull just the Sources section from out (everything after "## Sources")
    _, _, sources_part = out.partition("## Sources")
    assert "2401.00001" in sources_part
    # invalid dropped from the canonical section
    assert "2607.15491" not in sources_part


def test_syncs_with_body_adds_missing():
    """Writer forgot an id in Sources that is in the body — canonicalizer adds it."""
    text = """# R

Body mentions [2602.10434] and [2602.03856] but:

## Sources
[2602.03856]
"""
    out, changed = _canonicalize_sources_section(text, [])
    assert changed
    assert "2602.10434" in out
    assert "2602.03856" in out


def test_repos_from_body_included():
    """[repo: owner/name] in body → lands in the Repositories section."""
    text = """# R

See [repo: alan-turing/foo] for details.

## Sources
[2401.00001]
"""
    out, _ = _canonicalize_sources_section(text, [])
    assert "alan-turing/foo" in out
    assert "github.com/alan-turing/foo" in out
    assert "**Repositories:**" in out


def test_no_section_no_change():
    """No Sources section in draft → unchanged."""
    text = "# R\n\nJust text.\n"
    out, changed = _canonicalize_sources_section(text, [])
    assert not changed
    assert out == text


def test_dedup_within_body():
    """Same id, multiple mentions → one bullet."""
    text = """# R

[2401.00001] first. Also [2401.00001] second time and [2401.00001] third.

## Sources
old garbage
"""
    out, _ = _canonicalize_sources_section(text, [])
    assert out.count("2401.00001](https://arxiv.org/abs/2401.00001") == 1


def test_trailing_content_preserved():
    """Content AFTER the Sources section (if any) is not lost."""
    text = """# R

[2401.00001] fact.

## Sources
old stuff

## Appendix
extra data
"""
    out, _ = _canonicalize_sources_section(text, [])
    assert "## Appendix" in out
    assert "extra data" in out


def test_renders_canonical_urls():
    """Arxiv-id is rendered as a markdown link with the arxiv.org URL."""
    text = """# R

See [2401.00001].

## Sources
[2401.00001]
"""
    out, _ = _canonicalize_sources_section(text, [])
    assert "[2401.00001](https://arxiv.org/abs/2401.00001)" in out
