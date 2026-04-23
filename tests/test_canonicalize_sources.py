"""P7: канонизация секции ## Источники по body-цитатам."""
from lra.pipeline import _canonicalize_sources_section


def test_removes_invalid_ids_from_sources():
    """invalid_ids не попадают в канонический список Sources (в body остаются как есть)."""
    text = """# Report

Here is a body citation [2401.00001] and [2607.15491] invalid one.

## Источники
[2401.00001]
[2607.15491]
"""
    out, changed = _canonicalize_sources_section(text, ["2607.15491"])
    assert changed
    # Вытащим только секцию Sources из out (всё после "## Источники")
    _, _, sources_part = out.partition("## Источники")
    assert "2401.00001" in sources_part
    # invalid выкинут из канонической секции
    assert "2607.15491" not in sources_part


def test_syncs_with_body_adds_missing():
    """Writer забыл в Sources id который есть в body — канонизатор добавит."""
    text = """# R

Body mentions [2602.10434] and [2602.03856] but:

## Источники
[2602.03856]
"""
    out, changed = _canonicalize_sources_section(text, [])
    assert changed
    assert "2602.10434" in out
    assert "2602.03856" in out


def test_repos_from_body_included():
    """[repo: owner/name] в body → попадает в секцию Репозитории."""
    text = """# R

See [repo: alan-turing/foo] for details.

## Sources
[2401.00001]
"""
    out, _ = _canonicalize_sources_section(text, [])
    assert "alan-turing/foo" in out
    assert "github.com/alan-turing/foo" in out
    assert "**Репозитории:**" in out


def test_no_section_no_change():
    """Нет секции Sources в draft → без изменений."""
    text = "# R\n\nJust text.\n"
    out, changed = _canonicalize_sources_section(text, [])
    assert not changed
    assert out == text


def test_dedup_within_body():
    """Один id, несколько упоминаний → один bullet."""
    text = """# R

[2401.00001] first. Also [2401.00001] second time and [2401.00001] third.

## Источники
old garbage
"""
    out, _ = _canonicalize_sources_section(text, [])
    assert out.count("2401.00001](https://arxiv.org/abs/2401.00001") == 1


def test_trailing_content_preserved():
    """Контент ПОСЛЕ секции Sources (если вдруг есть) не теряется."""
    text = """# R

[2401.00001] fact.

## Источники
old stuff

## Приложение
extra data
"""
    out, _ = _canonicalize_sources_section(text, [])
    assert "## Приложение" in out
    assert "extra data" in out


def test_renders_canonical_urls():
    """Arxiv-id рендерится как markdown-link с arxiv.org URL."""
    text = """# R

See [2401.00001].

## Sources
[2401.00001]
"""
    out, _ = _canonicalize_sources_section(text, [])
    assert "[2401.00001](https://arxiv.org/abs/2401.00001)" in out
