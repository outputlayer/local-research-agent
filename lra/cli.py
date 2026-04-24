"""Wrapper around subprocess.run with a disk cache and uniform error handling."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass

from . import cache
from .logger import get_logger

log = get_logger("cli")


@dataclass
class CliResult:
    stdout: str
    stderr: str
    returncode: int
    from_cache: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def run(
    cmd: list[str],
    *,
    timeout: int = 30,
    use_cache: bool = True,
    cache_ttl_hours: float | None = None,
    input_str: str | None = None,
) -> CliResult:
    """Runs a CLI command with an optional disk cache.

    - FileNotFoundError → returncode=127, stderr=hint
    - TimeoutExpired   → returncode=124, stderr=hint
    Only successful results (returncode==0) are cached.
    """
    if use_cache and input_str is None:
        hit = cache.get(cmd, ttl_hours=cache_ttl_hours)
        if hit is not None:
            log.debug("cache hit: %s", " ".join(cmd[:3]))
            return CliResult(hit["stdout"], hit["stderr"], hit["returncode"], from_cache=True)

    log.debug("cli run: %s", " ".join(cmd[:4]))
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            input=input_str,
        )
    except FileNotFoundError:
        return CliResult("", f"command not found: {cmd[0]}", 127)
    except subprocess.TimeoutExpired:
        return CliResult("", f"timeout {timeout}s: {cmd[0]}", 124)

    result = CliResult(r.stdout, r.stderr, r.returncode)
    if use_cache and input_str is None and result.ok:
        cache.put(cmd, r.stdout, r.stderr, r.returncode)
    return result
