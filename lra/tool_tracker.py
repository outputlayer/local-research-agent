"""Loop detection для tool calls.

Мотивация (run.log 14:37-15:41): 9B Qwen застрял в петле на `compact_notes`,
делая 16 вызовов подряд с тем же контентом. Каждая итерация модель
дополнительно сериализовала params как строку, накапливая `\\\\\\\"` escape'ы —
прогрессивная деградация без шанса выйти.

Детектор детерминированно блокирует N-ный подряд вызов одного и того же
tool'а с одинаковыми params (хеш SHA1 от нормализованной строки). Сам
факт блокировки возвращается как текстовая ошибка — LLM видит её и должен
сменить стратегию.
"""
from __future__ import annotations

import hashlib
import json
from collections import deque

# Порог: если последние N вызовов все одинаковы → блок. Берём 3 — это
# "первый вызов ок, повтор ок (модель могла не увидеть результат), третий
# подряд идентичный — явный loop".
_MAX_REPEATS = 3
_HISTORY_LEN = 32  # больше не нужно, scan только последних _MAX_REPEATS


class ToolCallTracker:
    """Track последних tool calls и блокирует repeats.

    Потокобезопасности не гарантирует — qwen-agent вызывает tools
    последовательно, GIL-защиты достаточно для deque.
    """

    def __init__(self, max_repeats: int = _MAX_REPEATS) -> None:
        self.history: deque[str] = deque(maxlen=_HISTORY_LEN)
        self.max_repeats = max_repeats

    @staticmethod
    def _hash_params(params) -> str:
        """Нормализует params (dict/str/bytes) в короткий стабильный хеш.

        Ключ: model иногда даёт params как JSON-строку, иногда как уже
        распарсенный dict — оба варианта должны хешироваться одинаково.
        """
        if isinstance(params, str):
            # попробуем распарсить JSON, чтобы {"a":1}/{"a": 1} дали одинаковый хеш
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
        """Регистрирует вызов, возвращает (allowed, repeat_count).

        repeat_count — сколько раз этот же (tool+params) встретился в
        последних `max_repeats` слотах ВКЛЮЧАЯ текущий вызов.
        allowed=False если repeat_count >= max_repeats.
        """
        key = f"{tool_name}:{self._hash_params(params)}"
        # смотрим последние (max_repeats - 1) записей — если все они
        # совпадают с текущим ключом, то этот вызов будет max_repeats-м
        # подряд одинаковым → блок.
        window = list(self.history)[-(self.max_repeats - 1):]
        repeat_count = 1 + sum(1 for k in window if k == key)
        allowed = repeat_count < self.max_repeats
        if allowed:
            self.history.append(key)
        return allowed, repeat_count

    def reset(self) -> None:
        self.history.clear()


# Глобальный singleton, сбрасывается между прогонами через reset_tracker().
# Тесты могут заменить через monkeypatch.setattr(tool_tracker, "_TRACKER", ...).
_TRACKER = ToolCallTracker()


def check_call(tool_name: str, params) -> tuple[bool, int]:
    """Публичное API — делегирует в глобальный tracker."""
    return _TRACKER.check(tool_name, params)


def reset_tracker() -> None:
    """Сброс между прогонами (вызывается в research_loop / resume_research)."""
    _TRACKER.reset()
