"""ToolCallTracker — deterministic blocking of repeated tool calls.

Motivation: run.log 14:37-15:41 showed compact_notes in an x16 loop with
progressive JSON degradation (escapes accumulating). Catch such
loops on the 3rd identical call in a row.
"""
from __future__ import annotations

import pytest

from lra.tool_tracker import ToolCallTracker, check_call, reset_tracker, set_tool_budget


def test_tracker_allows_first_three_different_calls():
    t = ToolCallTracker(max_repeats=3)
    assert t.check("hf_papers", {"query": "a"}) == (True, 1)
    assert t.check("hf_papers", {"query": "b"}) == (True, 1)
    assert t.check("hf_papers", {"query": "c"}) == (True, 1)


def test_tracker_allows_two_repeats_blocks_third():
    """max_repeats=3: 1st ok, 2nd ok, 3rd identical in a row → block."""
    t = ToolCallTracker(max_repeats=3)
    params = {"content": "same stuff"}
    allowed1, n1 = t.check("compact_notes", params)
    allowed2, n2 = t.check("compact_notes", params)
    allowed3, n3 = t.check("compact_notes", params)
    assert (allowed1, allowed2, allowed3) == (True, True, False)
    assert (n1, n2, n3) == (1, 2, 3)


def test_tracker_normalizes_string_and_dict_params():
    """The model sometimes sends params as a JSON string, sometimes as a dict — both must hash equally."""
    t = ToolCallTracker(max_repeats=3)
    t.check("append_notes", {"content": "x"})
    t.check("append_notes", '{"content": "x"}')
    # 3rd identical in a row (regardless of format) — block
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
    # compact_notes never called — first call allowed
    assert t.check("compact_notes", {"content": "y"}) == (True, 1)


def test_tracker_non_consecutive_repeat_not_blocked():
    """If there is a different call between repeats — not a block."""
    t = ToolCallTracker(max_repeats=3)
    t.check("x", {"a": 1})
    t.check("y", {"b": 2})
    t.check("x", {"a": 1})
    t.check("y", {"b": 2})
    allowed, n = t.check("x", {"a": 1})
    assert allowed  # window ["x","y"]: 1 repeat of x + current = 2 < 3 → ok
    assert n == 2


def test_tracker_reset():
    t = ToolCallTracker(max_repeats=3)
    t.check("x", {"a": 1})
    t.check("x", {"a": 1})
    t.reset()
    # after reset — start over
    assert t.check("x", {"a": 1}) == (True, 1)


def test_tracker_handles_non_json_serializable():
    """str(params) fallback must not crash."""
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


def test_budget_blocks_after_n_total_calls():
    """set_budget(tool, n) must block call n+1 even with different params."""
    t = ToolCallTracker(max_repeats=10)  # large repeats — blocks only the budget
    t.set_budget("compact_notes", 3)
    assert t.check("compact_notes", {"content": "a"}) == (True, 1)
    assert t.check("compact_notes", {"content": "b"}) == (True, 1)
    assert t.check("compact_notes", {"content": "c"}) == (True, 1)
    # 4th call (n+1) must be blocked by the budget
    allowed, _ = t.check("compact_notes", {"content": "d"})
    assert not allowed


def test_budget_resets_with_reset():
    """After reset() the budget (totals) is cleared, _max_per_tool is kept."""
    t = ToolCallTracker(max_repeats=10)
    t.set_budget("compact_notes", 2)
    t.check("compact_notes", {"content": "x"})
    t.check("compact_notes", {"content": "y"})
    allowed_before, _ = t.check("compact_notes", {"content": "z"})
    assert not allowed_before  # budget exhausted
    t.reset()
    # after reset — the budget is available again (totals = 0)
    allowed_after, _ = t.check("compact_notes", {"content": "z"})
    assert allowed_after


def test_set_tool_budget_public_api():
    """set_tool_budget() public API works via the global tracker."""
    reset_tracker()
    set_tool_budget("compact_notes", 2)
    assert check_call("compact_notes", {"content": "a"})[0] is True
    assert check_call("compact_notes", {"content": "b"})[0] is True
    assert check_call("compact_notes", {"content": "c"})[0] is False  # > budget
    reset_tracker()  # cleanup


# ── integration with wrapped tools ─────────────────────────────────────


def test_wrapped_tool_returns_loop_error_on_third_repeat(tmp_path, monkeypatch):
    """_wrap_with_logging must return an error string instead of the real execute."""
    from lra import tool_tracker
    from lra.tools import AppendNotes

    tool_tracker.reset_tracker()
    tool = AppendNotes()
    params = '{"content": "[2501.00000] specific paper title for loop test"}'
    # the first 2 — go to the normal execute (may return something like "OK" or error)
    r1 = tool.call(params)
    r2 = tool.call(params)
    r3 = tool.call(params)
    assert isinstance(r3, str)
    assert "loop detected" in r3.lower()
    # the first 2 must not contain a loop error
    assert "loop detected" not in (r1 or "").lower()
    assert "loop detected" not in (r2 or "").lower()


@pytest.fixture(autouse=True)
def _reset_tracker_between_tests():
    reset_tracker()
    yield
    reset_tracker()
