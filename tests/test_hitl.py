"""Тесты human-in-the-loop (HITL) pause-point в _hitl_review.

Проверяем:
- HITL не активируется если CFG.hitl=False (default) — safe default для тестов/resume
- HITL не активируется если stdin не TTY — даже при CFG.hitl=True (не ломает неинтерактив)
- При approve (a/s/пустая строка) writer НЕ вызывается повторно
- При revise (r <комментарий>) writer вызывается один раз с комментарием
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
    # draft должен существовать чтобы _hitl_review прошёл guard
    (tmp_path / "draft.md").write_text("# Отчёт\n\ntest draft content\n", encoding="utf-8")
    return tmp_path


def _make_writer_stub(monkeypatch):
    """Подменяет _run_agent так, чтобы писать в список calls и ничего не делать."""
    from lra import pipeline
    calls: list[dict] = []

    def fake_run_agent(bot, messages, icon):
        calls.append({"bot": bot, "messages": list(messages), "icon": icon})
        return [{"role": "assistant", "content": "rewritten"}]

    monkeypatch.setattr(pipeline, "_run_agent", fake_run_agent)
    return calls


def test_hitl_disabled_by_default_no_prompt(patched_paths, monkeypatch):
    """CFG.hitl отсутствует → _hitl_review должен тихо выйти, input() НЕ вызвать."""
    from lra import pipeline
    from lra.config import CFG
    CFG.pop("hitl", None)  # default=False
    calls = _make_writer_stub(monkeypatch)

    def boom_input(*a, **kw):
        raise AssertionError("input() не должен вызываться при CFG.hitl=False")
    monkeypatch.setattr("builtins.input", boom_input)

    pipeline._hitl_review("test", writer=None, writer_msgs=[], valid=2, invalid=[], suspicious=[])
    assert calls == []


def test_hitl_skipped_when_stdin_not_tty(patched_paths, monkeypatch):
    """Даже с CFG.hitl=True, если stdin не TTY — выходим тихо."""
    from lra import pipeline
    from lra.config import CFG
    CFG["hitl"] = True
    calls = _make_writer_stub(monkeypatch)

    monkeypatch.setattr("sys.stdin", io.StringIO(""))  # isatty()=False

    def boom_input(*a, **kw):
        raise AssertionError("input() не должен вызываться при stdin не TTY")
    monkeypatch.setattr("builtins.input", boom_input)

    try:
        pipeline._hitl_review("test", writer=None, writer_msgs=[], valid=2, invalid=[], suspicious=[])
    finally:
        CFG.pop("hitl", None)
    assert calls == []


def test_hitl_approve_no_rewrite(patched_paths, monkeypatch):
    """CFG.hitl=True + TTY + ответ 'a' → writer НЕ вызывается."""
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
    """Ответ 'r' + комментарий → ровно один writer-pass с HITL-сообщением."""
    from lra import pipeline
    from lra.config import CFG
    CFG["hitl"] = True
    calls = _make_writer_stub(monkeypatch)

    class FakeTTY(io.StringIO):
        def isatty(self):
            return True
    monkeypatch.setattr("sys.stdin", FakeTTY(""))

    # Первый input: "r" — выбор revise. Второй: сам комментарий.
    answers = iter(["r", "добавь секцию про benchmarks"])
    monkeypatch.setattr("builtins.input", lambda *a, **kw: next(answers))

    writer_msgs: list[dict] = []
    try:
        pipeline._hitl_review("моя тема", writer=object(), writer_msgs=writer_msgs,
                              valid=2, invalid=[], suspicious=[])
    finally:
        CFG.pop("hitl", None)
    assert len(calls) == 1, f"expected exactly one writer-pass, got {len(calls)}"
    # HITL message added to writer_msgs BEFORE the call
    assert any("HITL COMMENT" in m.get("content", "") and
               "benchmarks" in m.get("content", "") for m in writer_msgs)


def test_hitl_revise_empty_comment_approves(patched_paths, monkeypatch):
    """Ответ 'r' + пустой комментарий → writer НЕ вызывается (принимаем как есть)."""
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
    """Формат 'r <комментарий>' на одной строке — без второго input()."""
    from lra import pipeline
    from lra.config import CFG
    CFG["hitl"] = True
    calls = _make_writer_stub(monkeypatch)

    class FakeTTY(io.StringIO):
        def isatty(self):
            return True
    monkeypatch.setattr("sys.stdin", FakeTTY(""))

    answers = iter(["r убери секцию benchmarks"])
    monkeypatch.setattr("builtins.input", lambda *a, **kw: next(answers))

    writer_msgs: list[dict] = []
    try:
        pipeline._hitl_review("t", writer=object(), writer_msgs=writer_msgs,
                              valid=1, invalid=[], suspicious=[])
    finally:
        CFG.pop("hitl", None)
    assert len(calls) == 1
    assert any("убери секцию benchmarks" in m.get("content", "") for m in writer_msgs)


def test_hitl_keyboard_interrupt_safely_accepts(patched_paths, monkeypatch):
    """KeyboardInterrupt на prompt'е → принимаем как есть, writer НЕ вызывается."""
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
