"""Loop detection for tool calls.

Motivation (run.log 14:37-15:41): the 9B Qwen got stuck on `compact_notes`,
issuing 16 calls in a row with the same content. Each iteration the model
additionally serialized the params as a string, accumulating `\\\\\\\"` escapes —
progressive degradation with no chance to escape.

The detector deterministically blocks the N-th consecutive call to the same
tool with identical params (SHA1 hash of the normalized string). The block
itself is returned as a text error — the LLM sees it and must change strategy.
"""
from __future__ import annotations

import hashlib
import json
from collections import deque

# Threshold: if the last N calls are all identical → block. We take 3 —
# "first call ok, repeat ok (the model may have missed the result), third
# identical call in a row — an obvious loop".
_MAX_REPEATS = 3
_HISTORY_LEN = 32  # not strictly needed; scan only the last _MAX_REPEATS


class ToolCallTracker:
    """Tracks the most recent tool calls and blocks repeats.

    No thread-safety guarantee — qwen-agent invokes tools sequentially,
    the GIL is enough to protect the deque.
    """

    def __init__(self, max_repeats: int = _MAX_REPEATS) -> None:
        self.history: deque[str] = deque(maxlen=_HISTORY_LEN)
        self.max_repeats = max_repeats
        self._call_totals: dict[str, int] = {}
        self._max_per_tool: dict[str, int] = {}

    def set_budget(self, tool_name: str, max_calls: int) -> None:
        """Sets the maximum number of tool calls per session.

        Independent of the identical-params detector. Used for
        compact_notes — a progressive loop (different params each time due to
        accumulating escapes) is not caught by the identical-check.
        """
        self._max_per_tool[tool_name] = max_calls

    @staticmethod
    def _hash_params(params) -> str:
        """Normalizes params (dict/str/bytes) into a short stable hash.

        Key point: the model sometimes sends params as a JSON string,
        sometimes as an already parsed dict — both variants must hash equally.
        """
        if isinstance(params, str):
            # try to parse JSON so that {"a":1}/{"a": 1} produce the same hash
            s = params.strip()
            try:
                obj = json.loads(s)
                s = json.dumps(obj, sort_keys=True, ensure_ascii=False)
            except Exception:
                pass
        else:
            try:
                s = json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)
            except Exception:
                s = str(params)
        return hashlib.sha1(s.encode("utf-8", errors="replace")).hexdigest()[:12]

    def check(self, tool_name: str, params) -> tuple[bool, int]:
        """Registers a call, returns (allowed, repeat_count).

        repeat_count — how many times this same (tool+params) was seen in
        the last `max_repeats` slots INCLUDING the current call.
        allowed=False if repeat_count >= max_repeats OR the set_budget
        limit for this tool is exceeded.
        """
        key = f"{tool_name}:{self._hash_params(params)}"
        # look at the last (max_repeats - 1) entries — if they all match
        # the current key, then this call will be the max_repeats-th
        # identical call in a row → block.
        window = list(self.history)[-(self.max_repeats - 1):]
        repeat_count = 1 + sum(1 for k in window if k == key)
        # Per-tool budget check (total calls per session).
        total = self._call_totals.get(tool_name, 0) + 1
        budget = self._max_per_tool.get(tool_name)
        if budget is not None and total > budget:
            # Do not append to history (the call is blocked).
            return False, repeat_count
        allowed = repeat_count < self.max_repeats
        if allowed:
            self.history.append(key)
            self._call_totals[tool_name] = total
        return allowed, repeat_count

    def reset(self) -> None:
        self.history.clear()
        self._call_totals.clear()
        # _max_per_tool — do not reset: budgets are set once at startup


# Global singleton, reset between runs via reset_tracker().
# Tests can swap it via monkeypatch.setattr(tool_tracker, "_TRACKER", ...).
_TRACKER = ToolCallTracker()


def check_call(tool_name: str, params) -> tuple[bool, int]:
    """Public API — delegates to the global tracker."""
    return _TRACKER.check(tool_name, params)


def set_tool_budget(tool_name: str, max_calls: int) -> None:
    """Sets the maximum per-session call budget for tool_name."""
    _TRACKER.set_budget(tool_name, max_calls)


def reset_tracker() -> None:
    """Reset between runs (called in research_loop / resume_research)."""
    _TRACKER.reset()
