"""Оркестратор пайплайна: explorer ↔ replanner → synthesizer → writer ↔ critic → validator."""
from __future__ import annotations
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from qwen_agent.agents import Assistant
from qwen_agent.utils.output_beautify import typewriter_print

# Импорт tools обязателен для @register_tool — не удалять даже если не используется прямо
from . import cli as cli_run
from . import tools  # noqa: F401
from .config import (CFG, DRAFT_PATH, NOTES_PATH, PLAN_PATH, RESEARCH_DIR)
from .logger import get_logger
from .memory import reset_research
from .prompts import (COMPRESSOR_PROMPT, CRITIC_PROMPT, EXPLORER_PROMPT,
                      REPLANNER_PROMPT, SYNTHESIZER_PROMPT, WRITER_PROMPT)
from .utils import count_arxiv_ids, jaccard, keyword_set
from .validator import validate_draft_ids

log = get_logger("pipeline")


def prefetch_iteration(focus: str, limit: int = 5, hf_timeout: int = 30, gh_timeout: int = 20) -> dict:
    """Прогревает disk-cache hf_papers + github_search параллельно до запуска explorer-LLM.

    Когда explorer затем вызовет те же команды, они отработают из кеша (~0с вместо
    ~10-30с каждая). Тихо игнорирует ошибки CLI — это прогрев, не критичная операция.
    Возвращает {'hf': bool, 'gh': bool, 'elapsed': float} для наблюдаемости.
    """
    t0 = time.time()
    hf_cmd = ["hf", "papers", "search", focus, "--limit", str(limit * 2), "--format", "json"]
    gh_cmd = ["gh", "search", "repos", focus, "--limit", str(limit),
              "--json", "fullName,url,description,stargazersCount,language,pushedAt"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_hf = pool.submit(cli_run.run, hf_cmd, timeout=hf_timeout)
        fut_gh = pool.submit(cli_run.run, gh_cmd, timeout=gh_timeout)
        r_hf = fut_hf.result()
        r_gh = fut_gh.result()

    elapsed = time.time() - t0
    log.info("prefetch '%s': hf=%s gh=%s in %.1fs",
             focus[:40], "✓" if r_hf.ok else "✗", "✓" if r_gh.ok else "✗", elapsed)
    return {"hf": r_hf.ok, "gh": r_gh.ok, "elapsed": elapsed,
            "hf_cached": r_hf.from_cache, "gh_cached": r_gh.from_cache}


def build_bot(system_message: str, tool_names: list, max_tokens: Optional[int] = None) -> Assistant:
    llm_cfg = {
        "model": CFG["model"],
        "model_type": "mlx",
        "generate_cfg": {
            "temperature": CFG["temperature"],
            "top_p": CFG["top_p"],
            "top_k": CFG["top_k"],
            "max_tokens": max_tokens or CFG["max_tokens"],
            "fncall_prompt_type": "nous",
        },
    }
    return Assistant(llm=llm_cfg, system_message=system_message, function_list=tool_names)


def _run_agent(bot: Assistant, messages: list, icon: str) -> list:
    print(icon + " ", end="", flush=True)
    plain, resp = "", []
    for resp in bot.run(messages=messages):
        plain = typewriter_print(resp, plain)
    print()
    return resp


def _current_focus(query: str) -> str:
    if not PLAN_PATH.exists():
        return query
    for line in PLAN_PATH.read_text(encoding='utf-8').splitlines():
        if line.strip().startswith("[FOCUS]"):
            return line.strip().replace("[FOCUS]", "").strip(" -—:") or query
    return query


def research_loop(query: str, depth: int = 6, critic_rounds: int = 2):
    """Полный пайплайн: explorer/replanner (×depth) → compressor → synthesizer →
    writer/critic (×critic_rounds) → validator цитат."""
    reset_research(query)
    print(f"📁 Рабочая папка: {RESEARCH_DIR}\n")

    explorer = build_bot(EXPLORER_PROMPT,
                         ["hf_papers", "github_search",
                          "read_plan", "read_notes", "append_notes",
                          "read_lessons", "append_lessons", "read_querylog"],
                         max_tokens=3072)
    replanner = build_bot(REPLANNER_PROMPT,
                          ["read_notes", "read_plan", "write_plan"],
                          max_tokens=3072)

    print(f"🕳️  Фаза 1: кроличья нора (до {depth} итераций, адаптивный план)")
    low_gain_streak = 0
    for i in range(1, depth + 1):
        focus = _current_focus(query)
        print(f"\n── итерация {i}/{depth} ──  🎯 FOCUS: {focus[:80]}")
        # Прогреваем disk-cache параллельно — hf+gh одновременно, до первого LLM-вызова
        pf = prefetch_iteration(focus)
        cached_marks = []
        if pf["hf_cached"]: cached_marks.append("hf")
        if pf["gh_cached"]: cached_marks.append("gh")
        cache_note = f" (из кеша: {','.join(cached_marks)})" if cached_marks else ""
        print(f"   ⚡ prefetch hf+gh параллельно: {pf['elapsed']:.1f}с{cache_note}")
        ids_before = count_arxiv_ids(NOTES_PATH.read_text(encoding='utf-8') if NOTES_PATH.exists() else "")
        before = NOTES_PATH.stat().st_size if NOTES_PATH.exists() else 0
        msg = [{"role": "user",
                "content": f"Исходная тема: {query}\n"
                           f"Текущий [FOCUS] из plan.md: {focus}\n"
                           "Сделай одну итерацию по [FOCUS]. ОБЯЗАТЕЛЬНО append_notes и append_lessons."}]
        _run_agent(explorer, msg, "🔎")
        after = NOTES_PATH.stat().st_size if NOTES_PATH.exists() else 0
        grew = after - before
        ids_after = count_arxiv_ids(NOTES_PATH.read_text(encoding='utf-8') if NOTES_PATH.exists() else "")
        new_ids = ids_after - ids_before
        print(f"   📝 notes: +{grew} симв ({after} всего)  📊 новых arxiv-id: {len(new_ids)}")
        if grew < 100:
            print("   ⚠️  заметки не росли — retry со строгим требованием")
            retry = [{"role": "user", "content": (
                f"ПРОВАЛ: ты НЕ вызвал append_notes. Сделай hf_papers по '{focus}' (limit=5) "
                "и СРАЗУ после этого append_notes с минимум 3 фактами и [arxiv-id]. "
                "Затем append_lessons одной строкой.")}]
            _run_agent(explorer, retry, "🔁")
            ids_after = count_arxiv_ids(NOTES_PATH.read_text(encoding='utf-8') if NOTES_PATH.exists() else "")
            new_ids = ids_after - ids_before

        low_gain_streak = low_gain_streak + 1 if len(new_ids) < 2 else 0

        print("   🧭 replanner обновляет план...")
        plan_before = PLAN_PATH.read_text(encoding='utf-8') if PLAN_PATH.exists() else ""
        _run_agent(replanner,
                   [{"role": "user",
                     "content": f"Исходная тема: {query}. Обнови plan.md: Digest, Direction check, "
                                "новый [FOCUS], пересортируй [TODO]. Используй write_plan."}],
                   "🧭")
        plan_text = PLAN_PATH.read_text(encoding='utf-8') if PLAN_PATH.exists() else ""
        if plan_text == plan_before:
            print("   ⚠️  план не обновился — retry")
            _run_agent(replanner,
                       [{"role": "user", "content": (
                           "ПРОВАЛ: ты НЕ вызвал write_plan. Прочитай read_notes, затем ОБЯЗАТЕЛЬНО "
                           "вызови write_plan с новым Digest, новым [FOCUS] и обновлённым [TODO].")}],
                       "🔁")
            plan_text = PLAN_PATH.read_text(encoding='utf-8') if PLAN_PATH.exists() else ""
        new_focus = _current_focus(query)
        if new_focus != focus:
            print(f"   🔀 вектор скорректирован: {focus[:50]} → {new_focus[:50]}")
        if "PLAN_COMPLETE" in plan_text:
            print("✅ План исчерпан"); break
        if "[TODO]" not in plan_text and i > 1:
            print("✅ Нет больше [TODO]"); break
        if low_gain_streak >= 2 and i >= 3:
            print("✅ Ранний стоп: 2 итерации подряд < 2 новых id"); break

    # Фаза 1.5 — компрессия
    notes_size = NOTES_PATH.stat().st_size if NOTES_PATH.exists() else 0
    if notes_size > 8000:
        print(f"\n🗜️  Фаза 1.5: компрессор ({notes_size} симв → ~5000)")
        compressor = build_bot(COMPRESSOR_PROMPT, ["read_notes", "compact_notes"])
        _run_agent(compressor, [{"role": "user", "content": "Сократи заметки до ~5000 симв."}], "🗜️")

    # Фаза 2.0 — синтез
    print("\n💡 Фаза 2.0: синтезатор (мосты/противоречия/пробелы/экстраполяция/testable)")
    synthesizer = build_bot(SYNTHESIZER_PROMPT,
                            ["read_plan", "read_notes", "write_synthesis", "run_python"],
                            max_tokens=4096)
    _run_agent(synthesizer,
               [{"role": "user", "content": f"Произведи пять типов инсайтов по теме: {query}"}],
               "💡")

    # Фаза 2 — writer
    print("\n✍️  Фаза 2: writer — финальный черновик")
    writer = build_bot(WRITER_PROMPT,
                       ["read_plan", "read_notes", "read_synthesis", "read_draft",
                        "write_draft", "append_draft"],
                       max_tokens=6144)
    writer_msgs = [{"role": "user", "content": f"Собери финальный отчёт по теме: {query}"}]
    resp = _run_agent(writer, writer_msgs, "✍️ ")
    writer_msgs.extend(resp)

    # Фаза 3 — critic с конвергенцией
    print(f"\n🔍 Фаза 3: критик ({critic_rounds} раунд(ов))")
    critic = build_bot(CRITIC_PROMPT,
                       ["read_draft", "read_notes", "read_synthesis"],
                       max_tokens=2048)
    prev_critique = ""
    for i in range(1, critic_rounds + 1):
        print(f"\n── критика {i}/{critic_rounds} ──")
        c_resp = _run_agent(critic,
                            [{"role": "user", "content": f"Оцени черновик по теме: {query}"}],
                            "🔍")
        critique = " ".join(m.get("content", "") for m in c_resp if m.get("role") == "assistant").strip()
        if "APPROVED" in critique.upper() and len(critique) < 60:
            print("✅ критик одобрил"); break
        if prev_critique:
            sim = jaccard(keyword_set(prev_critique), keyword_set(critique))
            if sim > 0.70:
                print(f"✅ критик зациклился (сходство {sim:.0%}) — выходим"); break
        prev_critique = critique
        writer_msgs.append({"role": "user",
                            "content": f"Критика:\n{critique}\n\nПерепиши через write_draft."})
        resp = _run_agent(writer, writer_msgs, "✍️ ")
        writer_msgs.extend(resp)

    if DRAFT_PATH.exists():
        print("\n🔐 Фаза 4: валидация цитат (hf info + keyword overlap с notes)")
        valid, invalid, suspicious = validate_draft_ids()
        print(f"   ✓ валидных: {valid}")
        if invalid:
            print(f"   ✗ не найдены: {invalid}")
            print("   ⚠️  возможные галлюцинации — проверь вручную")
        if suspicious:
            print(f"   ⚠️  слабое совпадение цитаты с notes: {suspicious}")
            print("       (id существует, но текст вокруг него в draft'е не отражает факты из notes)")
        print(f"\n📄 Итог: {DRAFT_PATH}\n" + "─" * 60)
        print(DRAFT_PATH.read_text(encoding='utf-8'))
        print("─" * 60)
        print(f"Файлы: {NOTES_PATH.name}, {PLAN_PATH.name}, {DRAFT_PATH.name}")
    else:
        print("⚠️  writer не сохранил черновик")
