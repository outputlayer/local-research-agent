"""read_notes_focused: anti-drift filter for notes.md blocks by jaccard with focus."""
import json

import pytest


@pytest.fixture
def _isolated_notes(tmp_path, monkeypatch):
    from lra import config, tools
    notes = tmp_path / "notes.md"
    monkeypatch.setattr(config, "NOTES_PATH", notes)
    monkeypatch.setattr(tools, "NOTES_PATH", notes)
    return notes


def _call(focus, max_chars=None, min_jaccard=None):
    from lra.tools import ReadNotesFocused
    args = {"focus": focus}
    if max_chars is not None:
        args["max_chars"] = max_chars
    if min_jaccard is not None:
        args["min_jaccard"] = min_jaccard
    return ReadNotesFocused().call(json.dumps(args))


def test_missing_notes(_isolated_notes):
    assert _call("anything") == "(no notes)"


def test_empty_focus(_isolated_notes):
    _isolated_notes.write_text("## section\n- [2401.00001] fact", encoding="utf-8")
    assert _call("   ") == "(focus is required)"


def test_filters_irrelevant_blocks(_isolated_notes):
    _isolated_notes.write_text(
        "- [2401.00001] semantic drift in agent workspace memory\n\n"
        "- [2402.00002] quantum cryptography protocols random factor\n\n"
        "- [2403.00003] context filtering via jaccard similarity workspace\n",
        encoding="utf-8",
    )
    out = _call("semantic drift workspace memory filter", min_jaccard=0.05)
    # relevant blocks about drift/workspace must pass; quantum — no
    assert "2401.00001" in out
    assert "2403.00003" in out
    assert "2402.00002" not in out
    assert "focus-filter" in out  # header presence


def test_respects_max_chars(_isolated_notes):
    big_block = "- [2401.00001] workspace drift " + ("filler " * 200)
    _isolated_notes.write_text(big_block + "\n\n" + big_block.replace("2401", "2402"), encoding="utf-8")
    out = _call("workspace drift filler", max_chars=500, min_jaccard=0.01)
    assert len(out) <= 500 + 200  # budget ~ max_chars (slack for header/score comments)


def test_no_matches_returns_empty_hint(_isolated_notes):
    _isolated_notes.write_text("- [2401.00001] quantum cryptography unrelated stuff", encoding="utf-8")
    out = _call("agents workflow orchestrator", min_jaccard=0.5)
    assert "0/1" in out or "0/" in out


def test_empty_focus_keywords_fallback(_isolated_notes):
    """If focus is present but has no keywords (short tokens) → returns tail."""
    _isolated_notes.write_text("- [2401.00001] fact about agents\n", encoding="utf-8")
    out = _call("a b c", max_chars=1000)  # all tokens <5 letters → keyword_set is empty
    assert "2401.00001" in out


def test_header_matches_actually_included_count(_isolated_notes):
    """Regression: header must reflect the blocks that actually fit, not the relevant ones,
    otherwise the LLM thinks it got N blocks while the budget only fit M<N."""
    blocks = [
        f"- [240{i}.0000{i}] semantic drift workspace agents filter block {i} "
        + ("wordy " * 30)
        for i in range(1, 6)
    ]
    _isolated_notes.write_text("\n\n".join(blocks), encoding="utf-8")
    out = _call("semantic drift workspace", max_chars=400, min_jaccard=0.01)
    included_count = out.count("score=")
    # header "N/M" — N must match what actually fit
    import re
    m = re.search(r"(\d+)/(\d+)", out)
    assert m is not None
    header_n = int(m.group(1))
    assert header_n == included_count, (
        f"header={header_n} but actually included={included_count}"
    )
    # And it must honestly indicate truncated
    if header_n < 5:
        assert "truncated" in out
