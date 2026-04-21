"""Единая настройка логирования. Пишет в research/run.log + stderr (WARNING+)."""
from __future__ import annotations
import logging
from pathlib import Path

from .config import CFG, RUN_LOG_PATH


_CONFIGURED = False


def get_logger(name: str = "lra") -> logging.Logger:
    """Ленивая настройка: первый вызов конфигурирует root-logger пакета."""
    global _CONFIGURED
    logger = logging.getLogger(name if name.startswith("lra") else f"lra.{name}")
    if _CONFIGURED:
        return logger

    root = logging.getLogger("lra")
    root.setLevel(getattr(logging, CFG.log_level, logging.INFO))
    root.propagate = False

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                            datefmt="%H:%M:%S")

    # File handler — пишем всё в run.log
    try:
        Path(RUN_LOG_PATH).parent.mkdir(exist_ok=True)
        fh = logging.FileHandler(RUN_LOG_PATH, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except Exception:
        pass

    # Stderr — только WARNING+, чтобы не мешать "typewriter" выводу пайплайна
    sh = logging.StreamHandler()
    sh.setLevel(logging.WARNING)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    _CONFIGURED = True
    return logger
