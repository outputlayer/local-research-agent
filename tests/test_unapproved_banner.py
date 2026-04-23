"""P9: unapproved banner в draft.md когда critic ни разу не approved."""
from lra import pipeline
from lra.pipeline import _UNAPPROVED_BANNER_MARK, _prepend_unapproved_banner


def _set_draft(tmp_path, monkeypatch, text):
    path = tmp_path / "draft.md"
    path.write_text(text, encoding="utf-8")
    monkeypatch.setattr(pipeline, "DRAFT_PATH", path)
    return path


def test_banner_prepended_when_unapproved(tmp_path, monkeypatch):
    path = _set_draft(tmp_path, monkeypatch, "# Report\n\nbody\n")
    _prepend_unapproved_banner(4, ["2607.15491"], ["2510.26941 (overlap=3)"])
    out = path.read_text(encoding="utf-8")
    assert out.startswith(_UNAPPROVED_BANNER_MARK)
    assert "НЕ approved" in out
    assert "4 раундов" in out
    assert "2607.15491" in out
    assert "2510.26941" in out
    # Original body preserved
    assert "# Report" in out
    assert "body" in out


def test_banner_idempotent(tmp_path, monkeypatch):
    path = _set_draft(tmp_path, monkeypatch, "# R\n\nbody\n")
    _prepend_unapproved_banner(2, [], [])
    size_after_first = path.stat().st_size
    _prepend_unapproved_banner(2, [], [])
    size_after_second = path.stat().st_size
    assert size_after_first == size_after_second


def test_banner_without_invalid_or_suspicious(tmp_path, monkeypatch):
    path = _set_draft(tmp_path, monkeypatch, "# R\n\nbody\n")
    _prepend_unapproved_banner(3, [], [])
    out = path.read_text(encoding="utf-8")
    assert "НЕ approved" in out
    assert "Галлюцинированные" not in out
    assert "Подозрительные" not in out


def test_no_draft_no_op(tmp_path, monkeypatch):
    """Нет draft.md → noop без exception."""
    monkeypatch.setattr(pipeline, "DRAFT_PATH", tmp_path / "nope.md")
    _prepend_unapproved_banner(2, [], [])  # не должно упасть
