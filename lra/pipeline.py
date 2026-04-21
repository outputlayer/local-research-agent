"""Оркестратор пайплайна: explorer ↔ replanner → synthesizer → writer ↔ critic → validator."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from qwen_agent.agents import Assistant
from qwen_agent.utils.output_beautify import typewriter_print

# Импорт tools обязателен для @register_tool — не удалять даже если не используется прямо
from . import cli as cli_run
from . import kb as kb_mod
from . import plan as plan_mod
from . import tools  # noqa: F401
from .config import CFG, DRAFT_PATH, NOTES_PATH, PLAN_PATH, RESEARCH_DIR
from .logger import get_logger
from .memory import reset_research
from .metrics import CriticRound, IterationMetric, RunMetrics, count_critic_issues
from .prompts import (
    COMPRESSOR_PROMPT,
    CRITIC_PROMPT,
    EXPLORER_PROMPT,
    REPLANNER_PROMPT,
    SYNTHESIZER_PROMPT,
    WRITER_PROMPT,
)
from .utils import extract_ids, jaccard, keyword_set
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


def build_bot(system_message: str, tool_names: list, max_tokens: int | None = None) -> Assistant:
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
    """Фокус берётся из plan.json (источник истины). Если его нет (legacy или
    plan.json повреждён), fallback — parsing plan.md.
    """
    plan = plan_mod.load()
    if plan:
        focus = plan.focus_task()
        if focus:
            return focus.title
        return query
    if not PLAN_PATH.exists():
        return query
    for line in PLAN_PATH.read_text(encoding='utf-8').splitlines():
        if line.strip().startswith("[FOCUS]"):
            return line.strip().replace("[FOCUS]", "").strip(" -—:") or query
    return query


def _rotate_focus_fallback(query: str) -> bool:
    """Программная ротация фокуса через plan.json. Закрывает текущий focus как dropped
    (если он ещё открыт) и ставит фокус на первую open-задачу. Возвращает True если удалось.
    """
    plan = plan_mod.load()
    if not plan:
        return False
    if plan.current_focus_id:
        # помечаем как заблокированный — replanner провалился на нём
        t = plan.get(plan.current_focus_id)
        if t and t.status == "in_progress":
            plan.block_task(t.id, why="replanner провалился дважды, ротируем")
        plan.current_focus_id = None
    next_t = next((t for t in plan.tasks if t.status == "open"), None)
    if not next_t:
        plan_mod.save(plan)
        return False
    plan.set_focus(next_t.id, why="fallback ротация после провала replanner'а")
    plan_mod.save(plan)
    return True


def research_loop(query: str, depth: int = 6, critic_rounds: int = 2):
    """Полный пайплайн: explorer/replanner (×depth) → compressor → synthesizer →
    writer/critic (×critic_rounds) → validator цитат."""
    reset_research(query)
    metrics = RunMetrics(query=query)
    print(f"📁 Рабочая папка: {RESEARCH_DIR}\n")

    explorer = build_bot(EXPLORER_PROMPT,
                         ["hf_papers", "github_search",
                          "read_plan", "read_notes", "append_notes",
                          "read_lessons", "append_lessons", "read_querylog",
                          "kb_add", "kb_search",
                          "plan_add_task", "plan_close_task", "plan_split_task"],
                         max_tokens=3072)
    replanner = build_bot(REPLANNER_PROMPT,
                          ["read_notes", "read_plan", "write_plan"],
                          max_tokens=3072)

    print(f"🕳️  Фаза 1: кроличья нора (до {depth} итераций, адаптивный план)")
    low_gain_streak = 0
    empty_iter_streak = 0  # подряд итераций, где explorer не вырастил notes (grew<100 после retry)
    stuck_focus_streak = 0  # подряд итераций с тем же FOCUS
    prev_focus = None
    for i in range(1, depth + 1):
        focus = _current_focus(query)
        print(f"\n── итерация {i}/{depth} ──  🎯 FOCUS: {focus[:80]}")
        # Прогреваем disk-cache параллельно — hf+gh одновременно, до первого LLM-вызова
        pf = prefetch_iteration(focus)
        cached_marks = []
        if pf["hf_cached"]:
            cached_marks.append("hf")
        if pf["gh_cached"]:
            cached_marks.append("gh")
        cache_note = f" (из кеша: {','.join(cached_marks)})" if cached_marks else ""
        print(f"   ⚡ prefetch hf+gh параллельно: {pf['elapsed']:.1f}с{cache_note}")
        ids_before = extract_ids(NOTES_PATH.read_text(encoding='utf-8') if NOTES_PATH.exists() else "")
        before = NOTES_PATH.stat().st_size if NOTES_PATH.exists() else 0
        # Инжектим в сообщение explorer'а top-3 атома из KB, релевантных FOCUS'у —
        # это даёт постоянный контекст поверх read_notes (который режется на 20k симв).
        kb_context = kb_mod.format_atoms(kb_mod.search(focus, k=3))
        kb_block = f"\n\nУже известно по схожим темам (KB top-3):\n{kb_context}\n" if kb_context else ""
        t_exp = time.time()
        msg = [{"role": "user",
                "content": f"Исходная тема: {query}\n"
                           f"Текущий [FOCUS] из plan.md: {focus}{kb_block}\n"
                           "Сделай одну итерацию по [FOCUS]. ОБЯЗАТЕЛЬНО append_notes и append_lessons. "
                           "Для КАЖДОЙ новой статьи/репозитория вызови kb_add — это нужно для поиска "
                           "по накопленному знанию на будущих итерациях."}]
        _run_agent(explorer, msg, "🔎")
        explorer_seconds = time.time() - t_exp
        after = NOTES_PATH.stat().st_size if NOTES_PATH.exists() else 0
        grew = after - before
        ids_after = extract_ids(NOTES_PATH.read_text(encoding='utf-8') if NOTES_PATH.exists() else "")
        new_ids = ids_after - ids_before
        print(f"   📝 notes: +{grew} симв ({after} всего)  📊 новых arxiv-id: {len(new_ids)}")
        if grew < 100:
            print("   ⚠️  заметки не росли — retry со строгим требованием")
            retry = [{"role": "user", "content": (
                f"ПРОВАЛ: ты НЕ вызвал append_notes. Сделай hf_papers по '{focus}' (limit=5) "
                "и СРАЗУ после этого append_notes с минимум 3 фактами и [arxiv-id]. "
                "Затем append_lessons одной строкой.")}]
            _run_agent(explorer, retry, "🔁")
            after = NOTES_PATH.stat().st_size if NOTES_PATH.exists() else 0
            grew = after - before
            ids_after = extract_ids(NOTES_PATH.read_text(encoding='utf-8') if NOTES_PATH.exists() else "")
            new_ids = ids_after - ids_before

        low_gain_streak = low_gain_streak + 1 if len(new_ids) < 2 else 0

        print("   🧭 replanner обновляет план...")
        plan_before = PLAN_PATH.read_text(encoding='utf-8') if PLAN_PATH.exists() else ""
        t_rep = time.time()
        _run_agent(replanner,
                   [{"role": "user",
                     "content": f"Исходная тема: {query}. Обнови plan.md: Digest, Direction check, "
                                "новый [FOCUS], пересортируй [TODO]. Используй write_plan."}],
                   "🧭")
        replanner_seconds = time.time() - t_rep
        plan_text = PLAN_PATH.read_text(encoding='utf-8') if PLAN_PATH.exists() else ""
        if plan_text == plan_before:
            print("   ⚠️  план не обновился — retry")
            _run_agent(replanner,
                       [{"role": "user", "content": (
                           "ПРОВАЛ: ты НЕ вызвал write_plan. Прочитай read_notes, затем ОБЯЗАТЕЛЬНО "
                           "вызови write_plan с новым Digest, новым [FOCUS] и обновлённым [TODO].")}],
                       "🔁")
            plan_text = PLAN_PATH.read_text(encoding='utf-8') if PLAN_PATH.exists() else ""
            if plan_text == plan_before:
                # Replanner провалился оба раза — ротируем FOCUS программно, чтобы не зацикливаться.
                if _rotate_focus_fallback(query):
                    print("   🛟 fallback: FOCUS ротирован программно из [TODO]")
                    plan_text = PLAN_PATH.read_text(encoding='utf-8')
        # Если replanner выставил новый [FOCUS] через write_plan (legacy-путь),
        # синхронизируем plan.json — иначе источник истины разъедется с plan.md.
        _plan = plan_mod.load()
        if _plan and plan_text:
            if plan_mod.sync_focus_from_md(_plan, plan_text, iter_=i):
                plan_mod.save(_plan)
        new_focus = _current_focus(query)
        focus_changed = new_focus != focus
        if focus_changed:
            print(f"   🔀 вектор скорректирован: {focus[:50]} → {new_focus[:50]}")

        # Трекинг застревания: тот же FOCUS подряд + пустой explorer подряд
        if prev_focus is not None and new_focus == prev_focus:
            stuck_focus_streak += 1
        else:
            stuck_focus_streak = 0
        prev_focus = new_focus
        if grew < 100:
            empty_iter_streak += 1
        else:
            empty_iter_streak = 0

        # ── Детерминированный guard: инкрементирует attempts, блокирует исчерпанные
        # задачи, авто-ротирует фокус, сигналит HALT если все задачи закрыты/заблокированы.
        _plan = plan_mod.load()
        if _plan:
            rep = plan_mod.guard(
                _plan, iter_=i,
                notes_grew=grew, new_ids=len(new_ids),
                focus_unchanged_streak=stuck_focus_streak,
                empty_iter_streak=empty_iter_streak,
            )
            if rep.blocked_ids or rep.rotated_focus or rep.warnings:
                print(f"   🛡️  guard: {rep.summary()}")
            if rep.halt:
                print(f"✅ Ранний стоп: guard сообщил {rep.halt_reason}")
                metrics.stopped_early_reason = f"GUARD_{rep.halt_reason}"
                metrics.iterations.append(IterationMetric(
                    iteration=i, focus=focus[:200],
                    prefetch_seconds=pf["elapsed"],
                    explorer_seconds=explorer_seconds,
                    replanner_seconds=replanner_seconds,
                    notes_grew_chars=grew,
                    new_arxiv_ids=len(new_ids),
                    hf_from_cache=pf.get("hf_cached", False),
                    gh_from_cache=pf.get("gh_cached", False),
                    focus_changed=focus_changed,
                ))
                break

        metrics.iterations.append(IterationMetric(
            iteration=i, focus=focus[:200],
            prefetch_seconds=pf["elapsed"],
            explorer_seconds=explorer_seconds,
            replanner_seconds=replanner_seconds,
            notes_grew_chars=grew,
            new_arxiv_ids=len(new_ids),
            hf_from_cache=pf.get("hf_cached", False),
            gh_from_cache=pf.get("gh_cached", False),
            focus_changed=focus_changed,
        ))

        if "PLAN_COMPLETE" in plan_text:
            print("✅ План исчерпан")
            metrics.stopped_early_reason = "PLAN_COMPLETE"
            break
        if "[TODO]" not in plan_text and i > 1:
            print("✅ Нет больше [TODO]")
            metrics.stopped_early_reason = "NO_TODO"
            break
        if empty_iter_streak >= 2:
            print("✅ Ранний стоп: 2 итерации подряд explorer не вырастил notes (агент застрял)")
            metrics.stopped_early_reason = "EMPTY_ITERATIONS"
            break
        if stuck_focus_streak >= 2 and i >= 3:
            print(f"✅ Ранний стоп: FOCUS не менялся 3 итерации подряд ({prev_focus[:50]})")
            metrics.stopped_early_reason = "FOCUS_STUCK"
            break
        if low_gain_streak >= 2 and i >= 3:
            print("✅ Ранний стоп: 2 итерации подряд < 2 новых id")
            metrics.stopped_early_reason = "LOW_GAIN"
            break

    # Фаза 1.5 — компрессия
    notes_size = NOTES_PATH.stat().st_size if NOTES_PATH.exists() else 0
    if notes_size > 8000:
        print(f"\n🗜️  Фаза 1.5: компрессор ({notes_size} симв → ~5000)")
        compressor = build_bot(COMPRESSOR_PROMPT, ["read_notes", "compact_notes"])
        _run_agent(compressor, [{"role": "user", "content": "Сократи заметки до ~5000 симв."}], "🗜️")

    # Фаза 2.0 — синтез
    print("\n💡 Фаза 2.0: синтезатор (мосты/противоречия/пробелы/экстраполяция/testable)")
    synthesizer = build_bot(SYNTHESIZER_PROMPT,
                            ["read_plan", "read_notes", "write_synthesis", "run_python",
                             "kb_search"],
                            max_tokens=4096)
    t_syn = time.time()
    _run_agent(synthesizer,
               [{"role": "user", "content": f"Произведи пять типов инсайтов по теме: {query}"}],
               "💡")
    metrics.synthesis_seconds = time.time() - t_syn

    # Фаза 2 — writer
    print("\n✍️  Фаза 2: writer — финальный черновик")
    writer = build_bot(WRITER_PROMPT,
                       ["read_plan", "read_notes", "read_synthesis", "read_draft",
                        "write_draft", "append_draft"],
                       max_tokens=6144)
    # Инжект KB — передаём writer'у свежий снимок атомов (репо + top papers) отдельным
    # блоком, чтобы секции "## Репозитории" и "## Источники" формировались из KB, а не
    # из уходящего в компрессию notes.md.
    kb_all = kb_mod.load()
    repos = sorted([a for a in kb_all if a.get("kind") == "repo"],
                   key=lambda a: a.get("stars", 0), reverse=True)[:8]
    papers = kb_mod.search(query, k=12, atoms=kb_all) or \
        [a for a in kb_all if a.get("kind") == "paper"][:12]
    kb_blocks: list[str] = []
    if repos:
        kb_blocks.append("Репозитории (для секции '## Репозитории'):")
        for r in repos:
            kb_blocks.append(
                f"- [repo: {r.get('id','?')} ★{r.get('stars',0)} {r.get('lang','')}] "
                f"{r.get('url','')} — {r.get('claim','')[:180]}")
    else:
        kb_blocks.append("Репозитории: в KB нет репо с ★≥10 (используй точную строку-заглушку из промпта).")
    if papers:
        kb_blocks.append("\nPapers (для таблиц '## Подходы' / '## Бенчмарки'):")
        for p in papers:
            kb_blocks.append(f"- [{p.get('id','?')}] {p.get('title','')[:90]} — {p.get('claim','')[:180]}")
    kb_context = "\n".join(kb_blocks)
    writer_msgs = [{"role": "user",
                    "content": (f"Собери финальный отчёт по теме: {query}\n\n"
                                f"KB context (authoritative список источников — "
                                f"используй id и repo ИМЕННО отсюда):\n{kb_context}")}]
    t_wr = time.time()
    resp = _run_agent(writer, writer_msgs, "✍️ ")
    metrics.writer_seconds = time.time() - t_wr
    writer_msgs.extend(resp)

    # Sanity-check: writer должен был вызвать write_draft. Если файл не создан или <200 симв —
    # retry со строгим сообщением. Без этого критик будет зря крутиться на пустоте.
    def _draft_too_small() -> bool:
        return not DRAFT_PATH.exists() or DRAFT_PATH.stat().st_size < 200

    if _draft_too_small():
        print("   ⚠️  writer не сохранил draft (или <200 симв) — retry со строгим сообщением")
        writer_msgs.append({"role": "user", "content": (
            "ПРОВАЛ: ты НЕ вызвал write_draft (либо получился слишком короткий черновик). "
            "СЕЙЧАС же вызови write_draft с заголовком и '## Краткий ответ' (3-6 булетов "
            "с [id] из KB context выше), затем append_draft для остальных секций по порядку: "
            "'## Подходы', '## Бенчмарки и метрики', '## Реализации', '## Ключевые инсайты' "
            "(6 тегов из synthesis.md), '## Открытые вопросы', '## Источники'. "
            "НЕ отвечай текстом — используй инструменты.")})
        resp = _run_agent(writer, writer_msgs, "🔁")
        writer_msgs.extend(resp)

    # Фаза 3 — critic с конвергенцией
    print(f"\n🔍 Фаза 3: критик ({critic_rounds} раунд(ов))")
    critic = build_bot(CRITIC_PROMPT,
                       ["read_draft", "read_notes", "read_synthesis"],
                       max_tokens=2048)
    prev_critique = ""
    for i in range(1, critic_rounds + 1):
        print(f"\n── критика {i}/{critic_rounds} ──")
        # Если draft всё ещё пустой — критик бесполезен, но дадим ему шанс вскрыть это
        # через тулы. Явно инструктируем использовать read_draft/read_notes/read_synthesis
        # (наблюдали кейс когда критик забывал про тулы и просил текст в чат).
        t_cr = time.time()
        c_resp = _run_agent(critic,
                            [{"role": "user", "content": (
                                f"Оцени черновик по теме: {query}\n\n"
                                "ОБЯЗАТЕЛЬНО сначала вызови read_draft, потом read_notes и "
                                "read_synthesis — у тебя ЕСТЬ эти инструменты. Не проси "
                                "прислать текст, не отвечай 'нет доступа к файлам'. "
                                "После чтения — либо список до 5 правок по формату из system-prompt, "
                                "либо ровно слово APPROVED на отдельной строке.")}],
                            "🔍")
        c_seconds = time.time() - t_cr
        critique = " ".join(m.get("content", "") for m in c_resp if m.get("role") == "assistant").strip()
        issues = count_critic_issues(critique)
        # APPROVED валидно ТОЛЬКО если:
        #   - draft.md существует и непустой (иначе критик галлюцинирует)
        #   - критик вернул короткий ответ с APPROVED как финальное слово (не в середине объяснения)
        #   - 0 структурных правок
        draft_ok = DRAFT_PATH.exists() and DRAFT_PATH.stat().st_size >= 200
        # "APPROVED" должно быть отдельным словом в последних 50 символах ответа
        # (чтобы не ловить упоминание в середине текста вроде «ответь APPROVED если...»)
        tail = critique[-80:].upper().strip()
        approved_keyword = tail.endswith("APPROVED") or tail == "APPROVED"
        approved = draft_ok and issues == 0 and approved_keyword
        converged = False
        print(f"   📋 правок у критика: {issues}  (draft_ok={draft_ok})")
        if approved:
            metrics.critic_rounds.append(CriticRound(round=i, issues_found=0, approved=True, seconds=c_seconds))
            print("✅ критик одобрил")
            break
        if prev_critique:
            sim = jaccard(keyword_set(prev_critique), keyword_set(critique))
            if sim > 0.70:
                converged = True
                metrics.critic_rounds.append(CriticRound(
                    round=i, issues_found=issues, approved=False,
                    converged_by_similarity=True, seconds=c_seconds))
                print(f"✅ критик зациклился (сходство {sim:.0%}) — выходим")
                break
        metrics.critic_rounds.append(CriticRound(
            round=i, issues_found=issues, approved=False,
            converged_by_similarity=converged, seconds=c_seconds))
        prev_critique = critique
        writer_msgs.append({"role": "user",
                            "content": f"Критика:\n{critique}\n\nПерепиши через write_draft."})
        resp = _run_agent(writer, writer_msgs, "✍️ ")
        writer_msgs.extend(resp)

    if DRAFT_PATH.exists():
        print("\n🔐 Фаза 4: валидация цитат (hf info + keyword overlap с notes)")
        valid, invalid, suspicious = validate_draft_ids()
        metrics.valid_ids = valid
        metrics.invalid_ids = list(invalid)
        metrics.suspicious_citations = list(suspicious)
        print(f"   ✓ валидных: {valid}")
        if invalid:
            print(f"   ✗ не найдены: {invalid}")
            print("   ⚠️  возможные галлюцинации — проверь вручную")
        if suspicious:
            print(f"   ⚠️  слабое совпадение цитаты с notes: {suspicious}")
            print("       (id существует, но текст вокруг него в draft'е не отражает факты из notes)")
        metrics.final_draft_chars = DRAFT_PATH.stat().st_size
        print(f"\n📄 Итог: {DRAFT_PATH}\n" + "─" * 60)
        print(DRAFT_PATH.read_text(encoding='utf-8'))
        print("─" * 60)
        print(f"Файлы: {NOTES_PATH.name}, {PLAN_PATH.name}, {DRAFT_PATH.name}")
    else:
        print("⚠️  writer не сохранил черновик")

    # Всегда сохраняем метрики — даже если draft не создался
    metrics_path = metrics.finish()
    print(f"📊 Метрики прогона: {metrics_path.relative_to(RESEARCH_DIR.parent)} "
          f"({metrics.total_seconds:.1f}с, {len(metrics.iterations)} итераций)")
