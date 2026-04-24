"""Tests for prefetch_iteration — parallel warmup of the disk cache."""
import time

from lra.cli import CliResult


def test_prefetch_runs_hf_and_gh_in_parallel(monkeypatch):
    """prefetch must call hf+gh in parallel: 2 sequential 0.1s requests ≈ 0.1s, not 0.2s."""
    from lra import pipeline

    def slow_hf_or_gh(cmd, **kw):
        time.sleep(0.1)
        return CliResult('[]', '', 0)

    monkeypatch.setattr(pipeline.cli_run, "run", slow_hf_or_gh)

    t0 = time.time()
    pf = pipeline.prefetch_iteration("transformers")
    elapsed = time.time() - t0

    assert pf["hf"] is True
    assert pf["gh"] is True
    # 2 parallel 0.1s must take ~0.1s + overhead, but substantially less than 0.2s
    assert elapsed < 0.18, f"expected parallel execution (<0.18s), got {elapsed:.3f}s"


def test_prefetch_reports_cache_hits(monkeypatch, tmp_path):
    """Repeat prefetch with the same FOCUS → both hf_cached+gh_cached=True."""
    from lra import cache, pipeline
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)

    call_count = {"n": 0}
    def fake_run(cmd, **kw):
        # Real cli.run with patched CACHE_DIR
        call_count["n"] += 1
        return CliResult('[]', '', 0)

    # Use the REAL cli.run but with mocked subprocess
    import subprocess
    class FakeProc:
        stdout = '[]'
        stderr = ''
        returncode = 0
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: FakeProc())

    pf1 = pipeline.prefetch_iteration("topic-xyz-abc")
    pf2 = pipeline.prefetch_iteration("topic-xyz-abc")

    assert pf1["hf_cached"] is False
    assert pf1["gh_cached"] is False
    assert pf2["hf_cached"] is True
    assert pf2["gh_cached"] is True


def test_prefetch_swallows_errors(monkeypatch):
    """If hf or gh fails — prefetch does not raise, just returns ok=False."""
    from lra import pipeline

    def failing(cmd, **kw):
        return CliResult('', 'command failed', 1)

    monkeypatch.setattr(pipeline.cli_run, "run", failing)
    pf = pipeline.prefetch_iteration("x")
    assert pf["hf"] is False
    assert pf["gh"] is False
    assert "elapsed" in pf
