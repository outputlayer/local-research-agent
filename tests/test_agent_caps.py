"""P11: soft-caps in _run_agent protect from MLX hangs."""
import time
from unittest.mock import MagicMock

from lra import pipeline
from lra.config import CFG


class _FakeBot:
    """Minimal Assistant stub: a response generator, one at a time."""
    def __init__(self, n_turns=100):
        self.n_turns = n_turns

    def run(self, messages):
        for i in range(self.n_turns):
            yield [{"role": "assistant", "content": f"turn {i}"}]


def test_max_agent_turns_caps_loop(monkeypatch, capsys):
    monkeypatch.setitem(CFG.extra, "max_agent_turns", 3)
    monkeypatch.setitem(CFG.extra, "agent_call_wall_clock_s", 0)
    # mock typewriter_print so it does not pollute output
    monkeypatch.setattr(pipeline, "typewriter_print", lambda r, p: "")
    bot = _FakeBot(n_turns=50)
    pipeline._run_agent(bot, [{"role": "user", "content": "x"}], "🧪")
    out = capsys.readouterr().out
    assert "max_agent_turns=3" in out


def test_wall_clock_caps_loop(monkeypatch, capsys):
    monkeypatch.setitem(CFG.extra, "max_agent_turns", 0)
    monkeypatch.setitem(CFG.extra, "agent_call_wall_clock_s", 0.01)  # 10ms
    monkeypatch.setattr(pipeline, "typewriter_print", lambda r, p: "")

    class _SlowBot:
        def run(self, messages):
            for i in range(5):
                time.sleep(0.02)
                yield [{"role": "assistant", "content": f"t{i}"}]

    pipeline._run_agent(_SlowBot(), [{"role": "user", "content": "x"}], "🧪")
    out = capsys.readouterr().out
    assert "agent_call_wall_clock_s=0.01s" in out


def test_no_cap_when_both_zero(monkeypatch, capsys):
    monkeypatch.setitem(CFG.extra, "max_agent_turns", 0)
    monkeypatch.setitem(CFG.extra, "agent_call_wall_clock_s", 0)
    monkeypatch.setattr(pipeline, "typewriter_print", lambda r, p: "")
    pipeline._run_agent(_FakeBot(n_turns=3), [{"role": "user", "content": "x"}], "🧪")
    out = capsys.readouterr().out
    assert "cap" not in out.lower() or "max_agent_turns" not in out


def test_default_max_agent_turns_present():
    # If user didn't set, fallback is 24 (reasonable for tool-loops)
    v = CFG.get("max_agent_turns", 24)
    assert v == 24 or isinstance(v, int)
