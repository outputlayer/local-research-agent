"""Тесты для prefetch_iteration — параллельный прогрев disk-cache."""
import time

from lra.cli import CliResult


def test_prefetch_runs_hf_and_gh_in_parallel(monkeypatch):
    """prefetch должен вызвать hf+gh параллельно: 2 последовательных 0.1с запроса ≈ 0.1с, не 0.2с."""
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
    # 2 параллельных 0.1с должны занять ~0.1с + накладные, но существенно меньше 0.2с
    assert elapsed < 0.18, f"expected parallel execution (<0.18s), got {elapsed:.3f}s"


def test_prefetch_reports_cache_hits(monkeypatch, tmp_path):
    """Повторный prefetch с одним и тем же FOCUS → оба hf_cached+gh_cached=True."""
    from lra import cache, pipeline
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)

    call_count = {"n": 0}
    def fake_run(cmd, **kw):
        # Настоящий cli.run с патченным CACHE_DIR
        call_count["n"] += 1
        return CliResult('[]', '', 0)

    # Используем НАСТОЯЩИЙ cli.run, но с замоканным subprocess
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
    """Если hf или gh падает — prefetch не кидает, просто возвращает ok=False."""
    from lra import pipeline

    def failing(cmd, **kw):
        return CliResult('', 'command failed', 1)

    monkeypatch.setattr(pipeline.cli_run, "run", failing)
    pf = pipeline.prefetch_iteration("x")
    assert pf["hf"] is False
    assert pf["gh"] is False
    assert "elapsed" in pf
