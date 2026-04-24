"""Tests for the human-in-the-loop (HITL) pause-point in _hitl_review.

Checks:
- HITL is not activated if CFG.hitl=False (default) — safe default for tests/resume
- HITL is not activated if stdin is not a TTY — even when CFG.hitl=True (does not break non-interactive)
- On approve (a/s/empty line) the writer is NOT invoked again
- On revise (r <comment>) the writer is invoked once with the comment
"""
import io

import pytest


@pytest.fixture
def patched_paths(tmp_path, monkeypatch):
    from lra import config, pipeline, tools
    paths = {
        "RESEARCH_DIR": tmp_path,
        "DRAFT_PATH": tmp_path / "draft.md",
        "NOTES_PATH": tmp_path / "notes.md",
    }
    for mod in (config, tools, pipeline):
        for name, path in paths.items():
            if hasattr(mod, name):
                monkeypatch.setattr(mod, name, path)
    # draft must exist for _hitl_review to pass the guard
    (tmp_path / "draft.md").write_text("# Report\n\ntest draft content\n", encoding="utf-8")
    return tmp_path


def _make_writer_stub(monkeypatch):
    """Patches _run_agent so it records into the calls list and does nothing."""
    from lra import pipeline
    calls: list[dict] = []

    def fake_run_agent(bot, messages, icon):
        calls.append({"bot": bot, "messages": list(messages), "icon": icon})
        return [{"role": "assistant", "content": "rewritten"}]

    monkeypatch.setattr(pipeline, "_run_agent", fake_run_agent)
    return calls


def test_hitl_disabled_by_default_no_prompt(patched_paths, monkeypatch):
    """CFG.hitl missing → _hitl_review must exit silently and NOT call input()."""
    from lra import pipeline
    from lra.config import CFG
    CFG.pop("hitl", None)  # default=False
    calls = _make_writer_stub(monkeypatch)

    def boom_input(*a, **kw):
        raise AssertionError("input() must not be called when CFG.hitl=False")
    monkeypatch.setattr("builtins.input", boom_input)

    pipeline._hitl_review("test", writer=None, writer_msgs=[], valid=2, invalid=[], suspicious=[])
    assert calls == []


def test_hitl_skipped_when_stdin_not_tty(patched_paths, monkeypatch):
    """Even with CFG.hitl=True, if stdin is not a TTY — exit silently."""
    from lra import pipeline
    from lra.config import CFG
    CFG["hitl"] = True
    calls = _make_writer_stub(monkeypatch)

    monkeypatch.setattr("sys.stdin", io.StringIO(""))  # isatty()=False

    def boom_input(*a, **kw):
        raise AssertionError("input() must not be called when stdin is not a TTY")
    monkeypatch.setattr("builtins.input", boom_input)

    try:
        pipeline._hitl_review("test", writer=None, writer_msgs=[], valid=2, invalid=[], suspicious=[])
    finally:
        CFG.pop("hitl", None)
    assert calls == []


def test_hitl_approve_no_rewrite(patched_paths, monkeypatch):
    """CFG.hitl=True + TTY + answer 'a' → writer is NOT invoked."""
    from lra import pipeline
    from lra.config import CFG
    CFG["hitl"] = True
    calls = _make_writer_stub(monkeypatch)

    class FakeTTY(io.StringIO):
        def isatty(self):
            return True
    monkeypatch.setattr("sys.stdin", FakeTTY(""))

    answers = iter(["a"])
    monkeypatch.setattr("builtins.input", lambda *a, **kw: next(answers))

    writer_msgs: list[dict] = []
    try:
        pipeline._hitl_review("test", writer=object(), writer_msgs=writer_msgs,
                              valid=2, invalid=[], suspicious=[])
    finally:
        CFG.pop("hitl", None)
    assert calls == []
    assert writer_msgs == []


def test_hitl_revise_triggers_one_writer_pass(patched_paths, monkeypatch):
    """Answer 'r' + comment → exactly one writer-pass with a HITL message."""
    from lra import pipeline
    from lra.config import CFG
    CFG["hitl"] = True
    calls = _make_writer_stub(monkeypatch)

    class FakeTTY(io.StringIO):
        def isatty(self):
            return True
    monkeypatch.setattr("sys.stdin", FakeTTY(""))

    # First input: "r" — choose revise. Second: the comment itself.
    answers = iter(["r", "add a benchmarks section"])
    monkeypatch.setattr("builtins.input", lambda *a, **kw: next(answers))

    writer_msgs: list[dict] = []
    try:
        pipeline._hitl_review("my topic", writer=object(), writer_msgs=writer_msgs,
                              valid=2, invalid=[], suspicious=[])
    finally:
        CFG.pop("hitl", None)
    assert len(calls) == 1, f"expected exactly one writer-pass, got {len(calls)}"
    # HITL message added to writer_msgs BEFORE the call
    assert any("HITL COMMENT" in m.get("content", "") and
               "benchmarks" in m.get("content", "") for m in writer_msgs)


def test_hitl_revise_empty_comment_approves(patched_paths, monkeypatch):
    """Answer 'r' + empty comment → writer is NOT invoked (accept as is)."""
    from lra import pipeline
    from lra.config import CFG
    CFG["hitl"] = True
    calls = _make_writer_stub(monkeypatch)

    class FakeTTY(io.StringIO):
        def isatty(self):
            return True
    monkeypatch.setattr("sys.stdin", FakeTTY(""))

    answers = iter(["r", ""])
    monkeypatch.setattr("builtins.input", lambda *a, **kw: next(answers))

    try:
        pipeline._hitl_review("t", writer=object(), writer_msgs=[],
                              valid=1, invalid=[], suspicious=[])
    finally:
        CFG.pop("hitl", None)
    assert calls == []


def test_hitl_inline_revise_comment(patched_paths, monkeypatch):
    """Format 'r <comment>' on one line — without the second input()."""
    from lra import pipeline
    from lra.config import CFG
    CFG["hitl"] = True
    calls = _make_writer_stub(monkeypatch)

    class FakeTTY(io.StringIO):
        def isatty(self):
            return True
    monkeypatch.setattr("sys.stdin", FakeTTY(""))

    answers = iter(["r remove the benchmarks section"])
    monkeypatch.setattr("builtins.input", lambda *a, **kw: next(answers))

    writer_msgs: list[dict] = []
    try:
        pipeline._hitl_review("t", writer=object(), writer_msgs=writer_msgs,
                              valid=1, invalid=[], suspicious=[])
    finally:
        CFG.pop("hitl", None)
    assert len(calls) == 1
    assert any("remove the benchmarks section" in m.get("content", "") for m in writer_msgs)


def test_hitl_keyboard_interrupt_safely_accepts(patched_paths, monkeypatch):
    """KeyboardInterrupt at the prompt → accept as is, writer is NOT invoked."""
    from lra import pipeline
    from lra.config import CFG
    CFG["hitl"] = True
    calls = _make_writer_stub(monkeypatch)

    class FakeTTY(io.StringIO):
        def isatty(self):
            return True
    monkeypatch.setattr("sys.stdin", FakeTTY(""))

    def ctrl_c(*a, **kw):
        raise KeyboardInterrupt
    monkeypatch.setattr("builtins.input", ctrl_c)

    try:
        pipeline._hitl_review("t", writer=object(), writer_msgs=[],
                              valid=1, invalid=[], suspicious=[])
    finally:
        CFG.pop("hitl", None)
    assert calls == []
