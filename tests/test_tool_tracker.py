"""ToolCallTracker — детерминированная блокировка повторяющихся tool calls.

Мотивация: run.log 14:37-15:41 показал compact_notes в loop x16 с
прогрессивной деградацией JSON (escape накапливался). Ловим такие
циклы сразу на 3-м подряд идентичном вызове.
"""
from __future__ import annotations

import pytest

from lra.tool_tracker import ToolCallTracker, check_call, reset_tracker


def test_tracker_allows_first_three_different_calls():
    t = ToolCallTracker(max_repeats=3)
    assert t.check("hf_papers", {"query": "a"}) == (True, 1)
    assert t.check("hf_papers", {"query": "b"}) == (True, 1)
    assert t.check("hf_papers", {"query": "c"}) == (True, 1)


def test_tracker_allows_two_repeats_blocks_third():
    """max_repeats=3: 1-й ok, 2-й ok, 3-й подряд одинаковый → block."""
    t = ToolCallTracker(max_repeats=3)
    params = {"content": "same stuff"}
    allowed1, n1 = t.check("compact_notes", params)
    allowed2, n2 = t.check("compact_notes", params)
    allowed3, n3 = t.check("compact_notes", params)
    assert (allowed1, allowed2, allowed3) == (True, True, False)
    assert (n1, n2, n3) == (1, 2, 3)


def test_tracker_normalizes_string_and_dict_params():
    """Модель иногда даёт params как JSON-строку, иногда как dict — должны хешироваться одинаково."""
    t = ToolCallTracker(max_repeats=3)
    t.check("append_notes", {"content": "x"})
    t.check("append_notes", '{"content": "x"}')
    # 3-й подряд тот же (независимо от формата) — блок
    allowed, n = t.check("append_notes", {"content": "x"})
    assert not allowed
    assert n == 3


def test_tracker_json_key_order_invariant():
    t = ToolCallTracker(max_repeats=3)
    t.check("x", '{"a":1,"b":2}')
    t.check("x", '{"b":2,"a":1}')
    allowed, n = t.check("x", '{"a": 1, "b": 2}')
    assert not allowed and n == 3


def test_tracker_different_tools_dont_interfere():
    t = ToolCallTracker(max_repeats=3)
    for _ in range(3):
        assert t.check("hf_papers", {"q": "x"})[0] in (True, False)
    # compact_notes пока ни разу — первый вызов разрешён
    assert t.check("compact_notes", {"content": "y"}) == (True, 1)


def test_tracker_non_consecutive_repeat_not_blocked():
    """Если между повторами есть другой вызов — не блок."""
    t = ToolCallTracker(max_repeats=3)
    t.check("x", {"a": 1})
    t.check("y", {"b": 2})
    t.check("x", {"a": 1})
    t.check("y", {"b": 2})
    allowed, n = t.check("x", {"a": 1})
    assert allowed  # в окне ["x","y"]: 1 повтор x + текущий = 2 < 3 → ok
    assert n == 2


def test_tracker_reset():
    t = ToolCallTracker(max_repeats=3)
    t.check("x", {"a": 1})
    t.check("x", {"a": 1})
    t.reset()
    # после reset — всё сначала
    assert t.check("x", {"a": 1}) == (True, 1)


def test_tracker_handles_non_json_serializable():
    """str(params) fallback не должен падать."""
    t = ToolCallTracker(max_repeats=3)

    class Weird:
        def __repr__(self):
            return "weird"

    w = Weird()
    allowed, _ = t.check("x", w)
    assert allowed


def test_global_check_call_and_reset():
    reset_tracker()
    assert check_call("foo", {"a": 1}) == (True, 1)
    assert check_call("foo", {"a": 1}) == (True, 2)
    allowed, _ = check_call("foo", {"a": 1})
    assert not allowed
    reset_tracker()
    assert check_call("foo", {"a": 1}) == (True, 1)


# ── интеграция с обёрнутыми tool'ами ─────────────────────────────────────


def test_wrapped_tool_returns_loop_error_on_third_repeat(tmp_path, monkeypatch):
    """_wrap_with_logging должен вернуть строку ошибки вместо реального execute."""
    from lra import tool_tracker
    from lra.tools import AppendNotes

    tool_tracker.reset_tracker()
    tool = AppendNotes()
    params = '{"content": "[2501.00000] specific paper title for loop test"}'
    # первые 2 — идут в обычный execute (могут вернуть что-то вроде "OK" или error)
    r1 = tool.call(params)
    r2 = tool.call(params)
    r3 = tool.call(params)
    assert isinstance(r3, str)
    assert "loop detected" in r3.lower()
    # первые 2 не должны содержать loop-ошибку
    assert "loop detected" not in (r1 or "").lower()
    assert "loop detected" not in (r2 or "").lower()


@pytest.fixture(autouse=True)
def _reset_tracker_between_tests():
    reset_tracker()
    yield
    reset_tracker()
