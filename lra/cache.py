"""Простой TTL-кеш для CLI-вызовов (hf/gh). Ключ=hash(команды), значение=stdout+returncode."""
from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path
from typing import Optional

from .config import CACHE_DIR, CFG


def _ensure_cache_dir() -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR


def _key(cmd: list[str]) -> str:
    h = hashlib.sha256("|".join(cmd).encode("utf-8")).hexdigest()[:16]
    return h


def get(cmd: list[str], ttl_hours: Optional[float] = None) -> Optional[dict]:
    """Возвращает {stdout, stderr, returncode} или None если кеш-мисс / протух."""
    ttl = (ttl_hours if ttl_hours is not None else CFG.cache_ttl_hours) * 3600
    f = _ensure_cache_dir() / f"{_key(cmd)}.json"
    if not f.exists():
        return None
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return None
    if time.time() - data.get("ts", 0) > ttl:
        return None
    return data


def put(cmd: list[str], stdout: str, stderr: str, returncode: int) -> None:
    """Кладёт результат в кеш. Не кидает исключения — кеш best-effort."""
    try:
        f = _ensure_cache_dir() / f"{_key(cmd)}.json"
        f.write_text(
            json.dumps({
                "cmd": cmd,
                "stdout": stdout,
                "stderr": stderr,
                "returncode": returncode,
                "ts": time.time(),
            }, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass


def clear() -> int:
    """Стирает весь кеш. Возвращает число удалённых файлов."""
    if not CACHE_DIR.exists():
        return 0
    n = 0
    for f in CACHE_DIR.glob("*.json"):
        f.unlink(missing_ok=True)
        n += 1
    return n
