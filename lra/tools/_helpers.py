"""Shared helpers для tools: verifier, domain gate, arxiv feed parsing, wrap-logger.

Вынесено из монолитного `lra/tools.py`, чтобы снизить размер основного модуля
и подготовить дальнейшее дробление на тематические подпакеты (search, notes,
draft, plan, kb и т.д.). Все @register_tool классы импортируют отсюда
верификаторы/gate и helpers для HTTP-фетчинга arXiv.
"""
from __future__ import annotations

import json
import re
from datetime import UTC
from urllib.request import urlopen
from xml.etree import ElementTree as ET

from qwen_agent.tools.base import BaseTool

from .. import config as _cfg
from .. import kb as kb_mod
from ..logger import get_logger
from ..memory import ensure_dir
from ..utils import (
    extract_ids,
    extract_topic_keywords_tiered,
    keyword_set,
)

log = get_logger("tools")

_ARXIV_FEED_NS = {"atom": "http://www.w3.org/2005/Atom"}


def verify_ids_against_kb(content: str) -> tuple[set[str], set[str]]:
    """Возвращает (known_ids, unknown_ids) — какие arxiv-id из content есть в kb.jsonl.

    Используется pre-append verifier'ом в AppendNotes: explorer не должен добавлять в notes
    факты с id, которых ни hf_papers, ни kb_search ни разу не возвращали в этой сессии.
    Это отсекает галлюцинации на этапе записи, а не на финальной валидации draft.md.
    """
    ids = extract_ids(content)
    if not ids:
        return set(), set()
    known_in_kb = {a.get("id", "") for a in kb_mod.load() if a.get("kind") == "paper"}
    return ids & known_in_kb, ids - known_in_kb


def domain_gate(content: str) -> tuple[bool, str, set[str], set[str]]:
    """Two-tier domain gate для AppendNotes и hf_papers kb auto-save.

    Правило: paper проходит ⇔ ≥2 совпадений с HEADER plan.md.
    HEADER = первая строка ('# Plan: ...') — это исходный topic пользователя
    и наиболее стабильный носитель core-терминов. Seeds из [Tn]-задач дрейфуют
    и содержат мусор ('support' в 'electronic support measures' даёт false
    positive на emotional-support-conversations paper) — используются только
    для диагностики причины отказа в rejected.jsonl, но не для прохода.

    Slow-start bypass: если в header <2 кандидатов — plan ещё generic, gate слеп.

    Returns (passed, reason, overlap_header, overlap_seeds). reason ∈
    reason ∈ {"no_plan", "slow_start", "no_core_hit", "weak_overlap", "passed"}.

    Адаптивный порог: при бедном header'е (≤ 4 core-kws) достаточно 1 overlap,
    иначе требуется ≥2. Это ловит кейсы узких тем типа
    `# Plan: electronic warfare (EW) and ELINT` (4 слова: electronic/warfare/
    elint/intelligence), где paper «cognitive radar jamming» даёт только 1 hit
    (`warfare` → 0, но `electronic` → 1) и иначе бы резался.
    """
    if not _cfg.PLAN_PATH.exists():
        return True, "no_plan", set(), set()
    header_kws, seed_kws = extract_topic_keywords_tiered(_cfg.PLAN_PATH.read_text(encoding="utf-8"))
    if len(header_kws) < 2:
        return True, "slow_start", header_kws, set()
    # Собираем материал из kb для id, упомянутых в content.
    ids_in_content = extract_ids(content)
    abstracts: list[str] = [content]
    if ids_in_content:
        kb_by_id = {a.get("id"): a for a in kb_mod.load() if a.get("kind") == "paper"}
        for aid in ids_in_content:
            atom = kb_by_id.get(aid)
            if atom:
                abstracts.append(f"{atom.get('title', '')} {atom.get('claim', '')}")
    content_kws = keyword_set(" ".join(abstracts))
    o_header = content_kws & header_kws
    o_seeds = content_kws & seed_kws
    if not o_header:
        return False, "no_core_hit", set(), o_seeds
    # Адаптивный порог: бедный header ⇒ 1 hit достаточно, богатый ⇒ ≥2.
    min_hits = 1 if len(header_kws) <= 4 else 2
    if len(o_header) < min_hits:
        return False, "weak_overlap", o_header, o_seeds
    return True, "passed", o_header, o_seeds


def gate_paper_for_kb(paper_id: str, title: str, abstract: str) -> tuple[bool, str, set[str], set[str]]:
    """Облегчённый gate для hf_papers/kb_add ДО записи в kb.jsonl.

    Иначе explorer наполняет KB off-topic paper'ами (ComVo с auto-save остаётся
    в KB даже если AppendNotes его режет, т.к. KB-запись идёт РАНЬШЕ в hf_papers).
    Работает на сыром abstract (kb ещё не знает этот id).

    Returns (passed, reason, o_header, header_kws) — o_header/header_kws нужны
    вызывающему для записи диагностики в rejected.jsonl (см. `_log_kb_rejected`).
    Адаптивный порог идентичен `domain_gate`: ≤4 core-kws → 1 hit, иначе ≥2.
    """
    if not _cfg.CFG.get("strict_domain_gate", True) or not _cfg.PLAN_PATH.exists():
        return True, "bypass", set(), set()
    header_kws, _ = extract_topic_keywords_tiered(_cfg.PLAN_PATH.read_text(encoding="utf-8"))
    if len(header_kws) < 2:
        return True, "slow_start", set(), header_kws
    kws = keyword_set(f"{title} {abstract}")
    o_h = kws & header_kws
    if not o_h:
        return False, "no_core_hit", o_h, header_kws
    min_hits = 1 if len(header_kws) <= 4 else 2
    if len(o_h) < min_hits:
        return False, "weak_overlap", o_h, header_kws
    return True, "passed", o_h, header_kws


def _log_kb_rejected(paper_id: str, title: str, reason: str,
                     o_header: set[str], header_kws: set[str],
                     source: str) -> None:
    """Логирует skip'нутый paper из `gate_paper_for_kb` в rejected.jsonl.

    Без этого skip уходит только в log.debug и причина не видна пользователю —
    см. session 2026-04-22: 17 поисков, 1 paper в KB, 0 reject entries от gate_paper_for_kb.
    """
    from datetime import datetime
    ensure_dir()
    entry = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "reason": f"kb_autosave:{reason}",
        "source": source,  # hf_papers / arxiv_search
        "paper_id": paper_id,
        "title": (title or "")[:200],
        "overlap_header": sorted(o_header),
        "header_keywords": sorted(header_kws)[:20],
    }
    with _cfg.REJECTED_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _log_rejected(content: str, ids: set[str], reason: str,
                  header_kws: set[str], seed_kws: set[str],
                  o_header: set[str], o_seeds: set[str]) -> None:
    """Пишет отклонённую заметку в research/rejected.jsonl для анализа."""
    from datetime import datetime
    ensure_dir()
    entry = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "reason": reason,
        "ids": sorted(ids),
        "overlap_header": sorted(o_header),
        "overlap_seeds": sorted(o_seeds),
        "header_keywords": sorted(header_kws)[:15],
        "seed_keywords_sample": sorted(seed_kws)[:15],
        "content_preview": content[:300],
    }
    with _cfg.REJECTED_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _fetch_text(url: str, timeout: int = 20) -> str:
    with urlopen(url, timeout=timeout) as resp:  # noqa: S310 - controlled arXiv endpoint
        return resp.read().decode("utf-8", errors="replace")


def _parse_arxiv_feed(xml_text: str) -> list[dict[str, str]]:
    root = ET.fromstring(xml_text)
    entries: list[dict[str, str]] = []
    for entry in root.findall("atom:entry", _ARXIV_FEED_NS):
        id_text = (entry.findtext("atom:id", default="", namespaces=_ARXIV_FEED_NS) or "").strip()
        m = re.search(r"(\d{4}\.\d{4,5})(?:v\d+)?", id_text)
        if not m:
            continue
        authors = [
            (node.findtext("atom:name", default="", namespaces=_ARXIV_FEED_NS) or "").strip()
            for node in entry.findall("atom:author", _ARXIV_FEED_NS)
        ]
        entries.append({
            "id": m.group(1),
            "title": " ".join((entry.findtext("atom:title", default="", namespaces=_ARXIV_FEED_NS) or "").split()),
            "summary": " ".join((entry.findtext("atom:summary", default="", namespaces=_ARXIV_FEED_NS) or "").split()),
            "published_at": (entry.findtext("atom:published", default="", namespaces=_ARXIV_FEED_NS) or "").strip(),
            "authors": ", ".join(a for a in authors[:4] if a),
        })
    return entries


def _wrap_with_logging(cls):
    """Оборачивает `cls.call`, чтобы каждый вызов попадал в лог единообразно.

    Идемпотентен (повторный вызов — no-op через `_tool_logged` флаг).
    Loop-detection через lra.tool_tracker — блокирует N-ный подряд идентичный вызов.
    """
    orig = cls.call
    if getattr(orig, "_tool_logged", False):
        return cls
    tool_name = getattr(cls, "name", cls.__name__)

    def call(self, params: str = "", **kwargs):  # type: ignore[override]
        try:
            preview_src = params if isinstance(params, str) else json.dumps(params, ensure_ascii=False)
        except Exception:
            preview_src = str(params)
        preview = (preview_src or "").replace("\n", " ")[:160]
        log.info("[TOOL_CALL] %s(%s)", tool_name, preview)
        # Loop detection: блокируем N-ный подряд идентичный вызов.
        # See lra/tool_tracker.py для мотивации (run.log compact_notes x16).
        try:
            from lra import tool_tracker
            allowed, n = tool_tracker.check_call(tool_name, params)
            if not allowed:
                log.warning("[TOOL_LOOP] %s заблокирован: %d-й идентичный вызов подряд", tool_name, n)
                return (
                    f"ошибка: loop detected — {tool_name} вызван {n} раз подряд "
                    f"с одинаковыми params. Смени стратегию: попробуй другой "
                    f"tool, измени params или перейди к следующему шагу плана."
                )
        except Exception as _loop_err:
            # tracker-баг не должен ронять tool execution
            log.debug("tool_tracker error: %s", _loop_err)
        try:
            return orig(self, params, **kwargs)
        except Exception as e:
            log.warning("[TOOL_ERR]  %s: %s", tool_name, e)
            raise

    call._tool_logged = True  # type: ignore[attr-defined]
    cls.call = call
    return cls


def _wrap_module_tools(module_globals: dict, module_name: str) -> None:
    """Оборачивает все BaseTool-подклассы модуля в wrap_with_logging.

    Вызывается в конце каждого sub-модуля tools/, чтобы только свои @register_tool
    классы получили логгер (не трогаем импортированные из чужих модулей).
    """
    for _obj in list(module_globals.values()):
        if isinstance(_obj, type) and issubclass(_obj, BaseTool) and _obj is not BaseTool:
            if _obj.__module__ == module_name:
                _wrap_with_logging(_obj)
