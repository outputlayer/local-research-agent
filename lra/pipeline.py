"""Оркестратор пайплайна: explorer ↔ replanner → synthesizer → writer ↔ critic → validator."""
from __future__ import annotations
from typing import Optional

from qwen_agent.agents import Assistant
from qwen_agent.utils.output_beautify import typewriter_print

# Импорт tools обязателен для @register_tool — не удалять даже если не используется прямо
from . import tools  # noqa: F401
from .config import (CFG, DRAFT_PATH, NOTES_PATH, PLAN_PATH, RESEARCH_DIR)
from .memory import reset_research
from .prompts import (COMPRESSOR_PROMPT, CRITIC_PROMPT, EXPLORER_PROMPT,
                      REPLANNER_PROMPT, SYNTHESIZER_PROMPT, WRITER_PROMPT)
from .utils import count_arxiv_ids, jaccard, keyword_set
from .validator import validate_draft_ids


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
    for line in PLAN_PATH.read_text().splitlines():
        if line.strip().startswith("[FOCUS]"):
            return line.strip().replace("[FOCUS]", "").strip(" -—:") or query
    return query


def research_loop(query: str, depth: int = 6, critic_rounds: int = 2):
    """Полный пайплайн: explorer/replanner (×depth) → compressor → synthesizer →
    writer/critic (×critic_rounds) → validator цитат."""
    reset_research(query)
    print(f"📁 Рабочая папка: {RESEARCH_DIR}\n")

    explorer = build_bot(EXPLORER_PROMPT,
                         ["hf_papers", "read_plan", "read_notes", "append_notes",
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
        ids_before = count_arxiv_ids(NOTES_PATH.read_text() if NOTES_PATH.exists() else "")
        before = NOTES_PATH.stat().st_size if NOTES_PATH.exists() else 0
        msg = [{"role": "user",
                "content": f"Исходная тема: {query}\n"
                           f"Текущий [FOCUS] из plan.md: {focus}\n"
                           "Сделай одну итерацию по [FOCUS]. ОБЯЗАТЕЛЬНО append_notes и append_lessons."}]
        _run_agent(explorer, msg, "🔎")
        after = NOTES_PATH.stat().st_size if NOTES_PATH.exists() else 0
        grew = after - before
        ids_after = count_arxiv_ids(NOTES_PATH.read_text() if NOTES_PATH.exists() else "")
        new_ids = ids_after - ids_before
        print(f"   📝 notes: +{grew} симв ({after} всего)  📊 новых arxiv-id: {len(new_ids)}")
        if grew < 100:
            print("   ⚠️  заметки не росли — retry со строгим требованием")
            retry = [{"role": "user", "content": (
                f"ПРОВАЛ: ты НЕ вызвал append_notes. Сделай hf_papers по '{focus}' (limit=5) "
                "и СРАЗУ после этого append_notes с минимум 3 фактами и [arxiv-id]. "
                "Затем append_lessons одной строкой.")}]
            _run_agent(explorer, retry, "🔁")
            ids_after = count_arxiv_ids(NOTES_PATH.read_text() if NOTES_PATH.exists() else "")
            new_ids = ids_after - ids_before

        low_gain_streak = low_gain_streak + 1 if len(new_ids) < 2 else 0

        print("   🧭 replanner обновляет план...")
        plan_before = PLAN_PATH.read_text() if PLAN_PATH.exists() else ""
        _run_agent(replanner,
                   [{"role": "user",
                     "content": f"Исходная тема: {query}. Обнови plan.md: Digest, Direction check, "
                                "новый [FOCUS], пересортируй [TODO]. Используй write_plan."}],
                   "🧭")
        plan_text = PLAN_PATH.read_text() if PLAN_PATH.exists() else ""
        if plan_text == plan_before:
            print("   ⚠️  план не обновился — retry")
            _run_agent(replanner,
                       [{"role": "user", "content": (
                           "ПРОВАЛ: ты НЕ вызвал write_plan. Прочитай read_notes, затем ОБЯЗАТЕЛЬНО "
                           "вызови write_plan с новым Digest, новым [FOCUS] и обновлённым [TODO].")}],
                       "🔁")
            plan_text = PLAN_PATH.read_text() if PLAN_PATH.exists() else ""
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
        print(DRAFT_PATH.read_text())
        print("─" * 60)
        print(f"Файлы: {NOTES_PATH.name}, {PLAN_PATH.name}, {DRAFT_PATH.name}")
    else:
        print("⚠️  writer не сохранил черновик")
