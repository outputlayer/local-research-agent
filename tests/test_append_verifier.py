"""Pre-append verifier: блокирует запись в notes.md с неверифицированным arxiv-id."""
import pytest


@pytest.fixture
def _isolated_research(tmp_path, monkeypatch):
    from lra import config, kb, memory, tools
    monkeypatch.setattr(config, "RESEARCH_DIR", tmp_path)
    monkeypatch.setattr(config, "NOTES_PATH", tmp_path / "notes.md")
    monkeypatch.setattr(config, "PLAN_PATH", tmp_path / "plan.md")
    monkeypatch.setattr(config, "REJECTED_PATH", tmp_path / "rejected.jsonl")
    monkeypatch.setattr(kb, "KB_PATH", tmp_path / "kb.jsonl")
    monkeypatch.setattr(memory, "NOTES_PATH", tmp_path / "notes.md")
    monkeypatch.setattr(tools, "NOTES_PATH", tmp_path / "notes.md")
    monkeypatch.setattr(tools, "PLAN_PATH", tmp_path / "plan.md")
    monkeypatch.setattr(tools, "REJECTED_PATH", tmp_path / "rejected.jsonl")
    return tmp_path


def test_verify_ids_empty_kb(_isolated_research):
    from lra.tools import verify_ids_against_kb
    known, unknown = verify_ids_against_kb("см. [2401.00001] и [2402.12345]")
    assert known == set()
    assert unknown == {"2401.00001", "2402.12345"}


def test_verify_ids_partial(_isolated_research):
    from lra import kb
    from lra.tools import verify_ids_against_kb
    kb.add(kb.Atom(id="2401.00001", kind="paper", topic="t", claim="c"))
    known, unknown = verify_ids_against_kb("[2401.00001] ok, [2402.99999] hallucinated")
    assert known == {"2401.00001"}
    assert unknown == {"2402.99999"}


def test_verify_no_ids_in_content(_isolated_research):
    from lra.tools import verify_ids_against_kb
    known, unknown = verify_ids_against_kb("no ids here, just text")
    assert known == set() and unknown == set()


def test_append_notes_blocks_unknown_id(_isolated_research):
    from lra import tools
    result = tools.AppendNotes().call(
        {"content": "## hallucination\n[2401.99999] this id does not exist"}
    )
    assert "REJECTED" in result
    assert "2401.99999" in result
    # notes.md не должен быть создан
    assert not _isolated_research.joinpath("notes.md").exists()


def test_append_notes_allows_verified_id(_isolated_research):
    from lra import kb, tools
    kb.add(kb.Atom(id="2401.00001", kind="paper", topic="t", claim="c"))
    result = tools.AppendNotes().call({"content": "[2401.00001] valid fact"})
    assert "notes.md" in result and "REJECTED" not in result
    notes = _isolated_research.joinpath("notes.md").read_text(encoding="utf-8")
    assert "[2401.00001]" in notes


def test_append_notes_allows_repo_only(_isolated_research):
    """Записи с repo: но без arxiv-id не требуют верификации."""
    from lra import tools
    result = tools.AppendNotes().call(
        {"content": "[repo: foo/bar ★150 Python] — модульная архитектура"}
    )
    assert "REJECTED" not in result
    assert "notes.md" in result


def test_append_notes_mixed_blocks_if_any_unknown(_isolated_research):
    from lra import kb, tools
    kb.add(kb.Atom(id="2401.00001", kind="paper", topic="t", claim="c"))
    result = tools.AppendNotes().call(
        {"content": "[2401.00001] ok and [2401.99999] bad"}
    )
    assert "REJECTED" in result
    assert "2401.99999" in result


def test_append_notes_lenient_mode(_isolated_research, monkeypatch):
    """Если notes_strict=False — verifier отключён."""
    from lra import config, tools
    # Отключим strict через monkeypatch на CFG
    original_get = config.CFG.get
    monkeypatch.setattr(config.CFG, "get",
                        lambda k, d=None: False if k == "notes_strict" else original_get(k, d))
    result = tools.AppendNotes().call(
        {"content": "[2401.99999] would be blocked in strict mode"}
    )
    assert "REJECTED" not in result
