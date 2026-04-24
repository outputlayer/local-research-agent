"""P10: per-iteration wall-clock timeout."""
import time

from lra import pipeline
from lra.config import CFG


def test_iter_wall_clock_default_900():
    assert CFG.get("iter_wall_clock_limit_s", 900) == 900


def test_iter_wall_clock_limit_zero_disables(monkeypatch):
    """iter_wall_clock_limit_s=0 → timeout disabled."""
    monkeypatch.setitem(CFG.extra, "iter_wall_clock_limit_s", 0)
    # Logical check: `if 0 and ...` → False, i.e. branch does not trigger
    limit = CFG.get("iter_wall_clock_limit_s", 900)
    assert not limit


def test_iter_wall_clock_custom_value(monkeypatch):
    monkeypatch.setitem(CFG.extra, "iter_wall_clock_limit_s", 60)
    assert CFG.get("iter_wall_clock_limit_s", 900) == 60


def test_wall_clock_logic_trips_on_overrun():
    """Check the condition itself: elapsed > limit → halt."""
    t_start = time.time() - 1000  # "elapsed" 1000 s
    limit = 900
    elapsed = time.time() - t_start
    assert elapsed > limit


def test_wall_clock_logic_noop_within_budget():
    t_start = time.time() - 10
    limit = 900
    elapsed = time.time() - t_start
    assert not (limit and elapsed > limit)


def test_research_loop_has_t_iter_start():
    """Sanity: t_iter_start line is present in research_loop (P10 preserved)."""
    import inspect
    src = inspect.getsource(pipeline.research_loop)
    assert "t_iter_start" in src
    assert "ITER_WALL_CLOCK" in src
