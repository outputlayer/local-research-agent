"""Unified logging setup. Writes to research/run.log + stderr (WARNING+)."""
from __future__ import annotations

import logging
from pathlib import Path

from .config import CFG, RUN_LOG_PATH

_CONFIGURED = False


def get_logger(name: str = "lra") -> logging.Logger:
    """Lazy setup: the first call configures the package root logger."""
    global _CONFIGURED
    logger = logging.getLogger(name if name.startswith("lra") else f"lra.{name}")
    if _CONFIGURED:
        return logger

    root = logging.getLogger("lra")
    root.setLevel(getattr(logging, CFG.log_level, logging.INFO))
    root.propagate = False

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                            datefmt="%H:%M:%S")

    # File handler — everything goes into run.log
    try:
        Path(RUN_LOG_PATH).parent.mkdir(exist_ok=True)
        fh = logging.FileHandler(RUN_LOG_PATH, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except Exception:
        pass

    # Stderr — only WARNING+, so it does not interfere with the "typewriter" pipeline output
    sh = logging.StreamHandler()
    sh.setLevel(logging.WARNING)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    # The noisy qwen-agent warning — 'Invalid json tool-calling arguments' —
    # spams when the model emits a duplicate key or adds a prefix around JSON. Our
    # `parse_args` is tolerant to all these cases (json5 + regex-fallback), so this
    # warning is useless to the user and pollutes pipeline output. We raise the
    # level of that specific logger to ERROR — real parse errors (the ones that
    # actually break a call) will still come through.
    # NOTE: qwen_agent uses the logger name 'qwen_agent_logger' (see qwen_agent/log.py);
    # the 'nous_fncall_prompt.py' filename in logs is %(filename)s in the formatter, not the logger name.
    logging.getLogger("qwen_agent_logger").setLevel(logging.ERROR)

    _CONFIGURED = True
    return logger
