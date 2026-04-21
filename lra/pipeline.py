"""Оркестратор пайплайна: explorer ↔ replanner → synthesizer → writer ↔ critic → validator."""
from __future__ import annotations

import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from qwen_agent.agents import Assistant
from qwen_agent.utils.output_beautify import typewriter_print

# Импорт tools обязателен для @register_tool — не удалять даже если не используется прямо
# Импорт llm — для регистрации @register_llm("mlx") (иначе resume_research при запуске
# без предварительной загрузки модели через agent.py упадёт на 'Please set model_type').
from . import cli as cli_run
from . import kb as kb_mod
from . import llm as _llm_register  # noqa: F401
from . import plan as plan_mod
from . import tools  # noqa: F401
from .config import CFG, DRAFT_PATH, NOTES_PATH, PLAN_PATH, RESEARCH_DIR, SYNTHESIS_PATH
from .logger import get_logger
from .memory import reset_research
from .metrics import CriticRound, IterationMetric, RunMetrics, count_critic_issues
from .prompts import (
    COMPRESSOR_PROMPT,
    CRITIC_PROMPT,
    EXPLORER_PROMPT,
    FACT_CRITIC_PROMPT,
    INITIAL_PLANNER_PROMPT,
    REPLANNER_PROMPT,
    STRUCTURE_CRITIC_PROMPT,
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


def _bootstrap_initial_plan(query: str) -> bool:
    """Один LLM-вызов BOOTSTRAP-планера → переписываем plan.json специфичными задачами.

    Тихо degrade'ится в статический план (уже записанный reset_research) при:
    - ошибке LLM / таймауте,
    - невалидном JSON,
    - недостаточном количестве задач (<3).

    Возвращает True если bootstrap успешно применён, False если оставлен статический план.
    """
    try:
        planner = build_bot(INITIAL_PLANNER_PROMPT, [], max_tokens=1024)
        msgs = [{"role": "user", "content": f"Тема: {query}"}]
        resp = _run_agent(planner, msgs, "🧭")
        raw = ""
        if resp and isinstance(resp, list):
            for m in reversed(resp):
                c = m.get("content", "") if isinstance(m, dict) else ""
                if c:
                    raw = c
                    break
        parsed = plan_mod.parse_bootstrap_json(raw)
        if not parsed:
            log.info("bootstrap planner: невалидный JSON, остаёмся на статическом плане")
            return False
        topic_type, seeds = parsed
        plan = plan_mod.bootstrap_from_seeds(query, seeds, topic_type=topic_type)
        if plan is None:
            log.info("bootstrap planner: seeds не прошли валидацию (n=%d)", len(seeds))
            return False
        print(f"   🧭 bootstrap plan: topic_type={topic_type}, задач={len(seeds)}")
        return True
    except Exception as exc:
        log.warning("bootstrap planner упал (%s), остаёмся на статическом плане", exc)
        return False


def research_loop(query: str, depth: int = 6, critic_rounds: int = 2):
    """Полный пайплайн: explorer/replanner (×depth) → compressor → synthesizer →
    writer/critic (×critic_rounds) → validator цитат."""
    reset_research(query)
    metrics = RunMetrics(query=query)
    print(f"📁 Рабочая папка: {RESEARCH_DIR}\n")

    # Bootstrap-планер (опционально): один LLM-вызов до фазы 1 генерирует 4-6
    # тематически-специфичных seed-задач и классифицирует тему (engineering / theoretical
    # / mixed). При ошибке/невалидном JSON тихо остаёмся на статическом плане,
    # который уже создал reset_research → plan_mod.reset().
    if CFG.get("dynamic_initial_plan", True):
        _bootstrap_initial_plan(query)

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
            # Конкретный план действий вместо абстрактного «повтори». Различаем
            # три причины пустой итерации: (1) дубликат запросов → kb_search,
            # (2) длинный github-query был заreject'ен → сократить,
            # (3) задача исчерпана → plan_close_task + следующий TODO.
            retry = [{"role": "user", "content": (
                f"ПРОВАЛ: итерация пустая (grew={grew}, new_ids=0). Выполни ТОЧНО этот порядок:\n"
                f"1) kb_search '{focus[:60]}' (k=5) — проверь что уже в базе.\n"
                f"2) hf_papers с ПЕРЕФОРМУЛИРОВАННЫМ запросом по '{focus[:60]}' "
                "(смени термины, год, автора — НЕ ту же фразу что в querylog); limit=5.\n"
                "3) Если и это вернуло дубликат — СРАЗУ append_notes (2-3 факта из kb_search "
                "результатов) и append_lessons строкой «[iter] исчерпано: <focus>; "
                "следующий шаг: plan_close_task». Не зацикливайся на github_search.\n"
                "4) plan_close_task(id=<focus id>, evidence='...', why='исчерпано').")}]
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
                           "ПРОВАЛ: ты НЕ вызвал write_plan. Сейчас: (1) read_notes — "
                           "посмотри что добавилось за последнюю итерацию, (2) write_plan "
                           "ОБЯЗАТЕЛЬНО: обнови Digest (3-5 новых пунктов с [arxiv-id]), "
                           "выставь новый [FOCUS] (берём из [TODO] подтему с МЕНЬШИМ "
                           "количеством evidence), обнови [TODO]/[DONE]. "
                           "Формат tool_call arguments — строгий JSON без markdown.")}],
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
                            ["read_plan", "read_notes", "write_synthesis", "kb_search"],
                            max_tokens=4096)
    t_syn = time.time()
    _run_agent(synthesizer,
               [{"role": "user", "content": f"Произведи пять типов инсайтов по теме: {query}"}],
               "💡")
    metrics.synthesis_seconds = time.time() - t_syn

    _finalize_draft(query, metrics, critic_rounds)


def _build_kb_context(query: str) -> str:
    """Собирает authoritative-блок из kb.jsonl для writer'а и fallback'а."""
    kb_all = kb_mod.load()
    repos = sorted([a for a in kb_all if a.get("kind") == "repo"],
                   key=lambda a: a.get("stars", 0), reverse=True)[:8]
    papers = kb_mod.search(query, k=12, atoms=kb_all) or \
        [a for a in kb_all if a.get("kind") == "paper"][:12]
    blocks: list[str] = []
    if repos:
        blocks.append("Репозитории (для секции '## Реализации'):")
        for r in repos:
            blocks.append(
                f"- [repo: {r.get('id','?')} ★{r.get('stars',0)} {r.get('lang','')}] "
                f"{r.get('url','')} — {r.get('claim','')[:180]}")
    else:
        blocks.append("Репозитории: в KB нет репо с ★≥10 (используй точную строку-заглушку из промпта).")
    if papers:
        blocks.append("\nPapers (для секций '## Подходы' / '## Бенчмарки и метрики'):")
        for p in papers:
            blocks.append(f"- [{p.get('id','?')}] {p.get('title','')[:90]} — {p.get('claim','')[:180]}")
    return "\n".join(blocks)


def _fallback_draft_from_kb(query: str) -> None:
    """Программный fallback если writer не смог 2 раза подряд: собираем минимальный
    draft.md прямо из kb.jsonl + synthesis.md. Пользователь получит хоть что-то
    вместо пустоты. Все утверждения — прямые выдержки из claim'ов KB, так что
    validator пропустит [id] без проблем.
    """
    kb_all = kb_mod.load()
    papers = [a for a in kb_all if a.get("kind") == "paper"][:15]
    repos = sorted([a for a in kb_all if a.get("kind") == "repo"],
                   key=lambda a: a.get("stars", 0), reverse=True)[:5]
    synth = SYNTHESIS_PATH.read_text(encoding="utf-8") if SYNTHESIS_PATH.exists() else ""

    lines = [f"# {query}", "",
             "> Fallback-черновик: собран программно из kb.jsonl и synthesis.md "
             "(writer не вызвал write_draft после retry).", ""]
    lines.append("## Краткий ответ")
    for p in papers[:6]:
        claim_short = (p.get('claim', '') or '').replace('\n', ' ')[:180]
        lines.append(f"- [{p.get('id','?')}] {p.get('title','')[:70]}: {claim_short}")
    lines.append("")
    lines.append("## Подходы")
    for p in papers[:8]:
        lines.append(f"\n### {p.get('title','?')[:80]} [{p.get('id','?')}]")
        claim = (p.get('claim', '') or '').strip()
        lines.append(claim[:500] if claim else "_(claim отсутствует в kb)_")
    lines.append("")
    lines.append("## Реализации")
    if repos:
        for i, r in enumerate(repos, 1):
            lines.append(f"{i}. [{r.get('id','?')} ★{r.get('stars',0)}] ({r.get('lang','')}) — "
                         f"{(r.get('claim','') or '')[:150]}")
    else:
        lines.append("Публичных реализаций с ★≥10 в собранной выборке не обнаружено.")
    lines.append("")
    lines.append("## Ключевые инсайты")
    lines.append(synth or "_(synthesis.md отсутствует)_")
    lines.append("")
    lines.append("## Источники")
    lines.append("### Papers")
    for i, p in enumerate(papers, 1):
        lines.append(f"{i}. [{p.get('id','?')}] {p.get('title','')[:100]}")
    if repos:
        lines.append("\n### Repositories")
        for i, r in enumerate(repos, 1):
            lines.append(f"{i}. {r.get('id','?')} ({r.get('url','')})")
    DRAFT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"   🛟 fallback-draft собран программно: {DRAFT_PATH} "
          f"({DRAFT_PATH.stat().st_size} симв)")


_CITATION_LINKY_RE = re.compile(r"\[`?([^`\]\s]+?)`?\]\((arxiv-id|repo:\s*[^)]+)\)")
_CITATION_BT_RE = re.compile(r"`(\d{4}\.\d{4,6}(?:v\d+)?)`")
_CITATION_INNER_BT_RE = re.compile(r"\[`([^`\]]+)`\]")
_CITATION_NESTED_RE = re.compile(r"\[((?:\[[^\[\]]+\]\s*,\s*)+\[[^\[\]]+\])\]")
_CITATION_DANGLING_RE = re.compile(r"(\][^\n]{0,40}?)\s*\((arxiv-id|repo:\s*[^)]+)\)")


def _normalize_citations(text: str) -> str:
    r"""Чинит уродливый стиль `[`id`](arxiv-id)` / `[`id`](repo: X)` → плоский `[id]` / `[id, repo: X]`.

    Нормализует распространённые формы, которые модель любит генерить:
      1. ``[`2510.15624`](arxiv-id)`` → ``[2510.15624]``
      2. ``[`2510.15624`](repo: freephdlabor)`` → ``[2510.15624, repo: freephdlabor]``
      3. голый ```2506.09440``` → ``[2506.09440]``
      4. ``[`synthesis`]`` → ``[synthesis]``
      5. ``[[x], [y]]`` → ``[x, y]`` (склейка после шага 1)
      6. висячий ``(arxiv-id)`` / ``(repo: X)`` после ``[id]`` — удаляется
    """

    def _linky(m: re.Match) -> str:
        ident, target = m.group(1).strip(), m.group(2).strip()
        if target.lower().startswith("repo:"):
            repo = target.split(":", 1)[1].strip()
            return f"[{ident}, repo: {repo}]"
        return f"[{ident}]"

    text = _CITATION_LINKY_RE.sub(_linky, text)
    text = _CITATION_INNER_BT_RE.sub(lambda m: f"[{m.group(1)}]", text)
    text = _CITATION_BT_RE.sub(lambda m: f"[{m.group(1)}]", text)

    # Склейка вложенных [[a], [b]] → [a, b]
    def _flatten(m: re.Match) -> str:
        inner = m.group(1)
        parts = re.findall(r"\[([^\[\]]+)\]", inner)
        return "[" + ", ".join(p.strip() for p in parts) + "]"

    text = _CITATION_NESTED_RE.sub(_flatten, text)
    # Висячий (arxiv-id)/(repo: X) сразу после ] — убираем
    text = _CITATION_DANGLING_RE.sub(r"\1", text)
    return text


def _normalize_draft_file() -> bool:
    """Применяет _normalize_citations к draft.md inplace. Возвращает True если были правки."""
    if not DRAFT_PATH.exists():
        return False
    original = DRAFT_PATH.read_text(encoding="utf-8")
    fixed = _normalize_citations(original)
    if fixed != original:
        DRAFT_PATH.write_text(fixed, encoding="utf-8")
        return True
    return False


def _hitl_review(query: str, writer: Assistant, writer_msgs: list,
                 valid: int, invalid: list, suspicious: list) -> None:
    """Human-in-the-loop пауза после validator'а. Печатает превью + метрики и
    просит пользователя либо утвердить черновик, либо дать комментарий на один
    дополнительный проход writer'а, либо пропустить.

    Вызывается только если CFG.hitl=True И stdin — TTY (иначе в тестах/resume
    сломает неинтерактивный запуск).
    """
    if not CFG.get("hitl", False) or not sys.stdin.isatty() or not DRAFT_PATH.exists():
        return
    print("\n" + "═" * 60)
    print("🧑 HITL pause-point — отчёт готов, но можно внести правки")
    print("═" * 60)
    draft_text = DRAFT_PATH.read_text(encoding="utf-8")
    preview = draft_text if len(draft_text) < 2000 else draft_text[:1000] + "\n...\n" + draft_text[-800:]
    print(preview)
    print("─" * 60)
    print(f"Метрики: valid={valid}, invalid={len(invalid)}, suspicious={len(suspicious)}, "
          f"chars={len(draft_text)}")
    print("Команды: [a] approve как есть  |  [r] revise (попросить writer'а переписать)  "
          "|  [s] skip (то же что approve)")
    try:
        choice = input("Выбор [a/r/s]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("   ⚠️  HITL прерван — принимаем как есть")
        return
    if choice.startswith("r"):
        comment = choice[1:].strip(" :") if len(choice) > 1 else ""
        if not comment:
            try:
                comment = input("Что переписать? ").strip()
            except (EOFError, KeyboardInterrupt):
                print("   ⚠️  комментарий не введён — принимаем как есть")
                return
        if not comment:
            print("   ⚠️  пустой комментарий — принимаем как есть")
            return
        print(f"   🔁 HITL revise: «{comment}» — запускаем один дополнительный writer-pass")
        writer_msgs.append({"role": "user",
                            "content": (f"HITL-КОММЕНТАРИЙ от человека по теме «{query}»:\n{comment}\n\n"
                                        "Перепиши черновик через write_draft + append_draft, "
                                        "учитывая комментарий. Секции и формат сохраняй.")})
        writer_msgs.extend(_run_agent(writer, writer_msgs, "✍️ "))
        _normalize_draft_file()
        print("   ✓ writer переписал draft по HITL-комментарию")
    else:
        print("   ✓ HITL approved — финализируем как есть")


def _run_critic_round(critic: Assistant, critic_name: str, query: str,
                      user_hint: str, i: int, total: int,
                      prev_critique: str, metrics: RunMetrics) -> tuple[str, bool, bool]:
    """Один раунд критики. Возвращает (critique_text, approved, converged).

    Выделено, чтобы не дублировать логику между legacy-single-critic и specialized-critics.
    """
    print(f"\n── {critic_name} {i}/{total} ──")
    t_cr = time.time()
    c_resp = _run_agent(critic,
                        [{"role": "user", "content": (
                            f"Оцени черновик по теме: {query}\n\n{user_hint}\n\n"
                            "После чтения — либо список до 5 правок по формату из system-prompt, "
                            "либо ровно слово APPROVED на отдельной строке.")}],
                        "🔍")
    c_seconds = time.time() - t_cr
    critique = " ".join(m.get("content", "") for m in c_resp if m.get("role") == "assistant").strip()
    issues = count_critic_issues(critique)
    draft_ok = DRAFT_PATH.exists() and DRAFT_PATH.stat().st_size >= 200
    tail = critique[-80:].upper().strip()
    approved_keyword = tail.endswith("APPROVED") or tail == "APPROVED"
    approved = draft_ok and issues == 0 and approved_keyword
    converged = False
    if prev_critique and not approved:
        sim = jaccard(keyword_set(prev_critique), keyword_set(critique))
        if sim > 0.70:
            converged = True
            print(f"   🔁 критик зациклился (сходство {sim:.0%})")
    print(f"   📋 правок: {issues}  (draft_ok={draft_ok}, approved={approved})")
    metrics.critic_rounds.append(CriticRound(
        round=len(metrics.critic_rounds) + 1, issues_found=(0 if approved else issues),
        approved=approved, converged_by_similarity=converged, seconds=c_seconds,
    ))
    return critique, approved, converged


def _run_legacy_critic(query: str, writer: Assistant, writer_msgs: list,
                       metrics: RunMetrics, critic_rounds: int) -> None:
    """Один combined critic (CRITIC_PROMPT) с циклом rewrite."""
    print(f"\n🔍 Фаза 3: критик ({critic_rounds} раунд(ов))")
    critic = build_bot(CRITIC_PROMPT,
                       ["read_draft", "read_notes", "read_synthesis"],
                       max_tokens=2048)
    hint = ("ОБЯЗАТЕЛЬНО сначала вызови read_draft, потом read_notes и read_synthesis — "
            "у тебя ЕСТЬ эти инструменты. Не проси прислать текст.")
    prev_critique = ""
    for i in range(1, critic_rounds + 1):
        critique, approved, converged = _run_critic_round(
            critic, "критика", query, hint, i, critic_rounds, prev_critique, metrics)
        if approved or converged:
            break
        prev_critique = critique
        writer_msgs.append({"role": "user",
                            "content": f"Критика:\n{critique}\n\nПерепиши через write_draft."})
        writer_msgs.extend(_run_agent(writer, writer_msgs, "✍️ "))
        _normalize_draft_file()


def _run_specialized_critics(query: str, writer: Assistant, writer_msgs: list,
                             metrics: RunMetrics, critic_rounds: int) -> None:
    """Fact-critic → writer rewrite цикл → structure-critic → writer rewrite цикл.

    Это отражает insight из [2506.18096]: разделение verifier на фактологический и
    структурный даёт более точные правки, чем один 'универсальный' critic.
    """
    print(f"\n🔍 Фаза 3: specialized critics (fact → structure, до {critic_rounds} раунд(ов) каждый)")
    fact_critic = build_bot(FACT_CRITIC_PROMPT,
                            ["read_draft", "read_notes", "read_synthesis"],
                            max_tokens=2048)
    struct_critic = build_bot(STRUCTURE_CRITIC_PROMPT,
                              ["read_draft"],
                              max_tokens=1536)

    # Sub-phase A — fact-critic
    fact_hint = ("ОБЯЗАТЕЛЬНО сначала вызови read_draft, потом read_notes и read_synthesis. "
                 "Проверяй только фактологию: есть ли [id], упоминается ли id в notes, "
                 "согласован ли факт рядом с id с notes. Структуру НЕ трогай.")
    prev = ""
    for i in range(1, critic_rounds + 1):
        critique, approved, converged = _run_critic_round(
            fact_critic, "fact-critic", query, fact_hint, i, critic_rounds, prev, metrics)
        if approved or converged:
            break
        prev = critique
        writer_msgs.append({"role": "user",
                            "content": (f"FACT-КРИТИКА (только фактология):\n{critique}\n\n"
                                        "Перепиши через write_draft + append_draft.")})
        writer_msgs.extend(_run_agent(writer, writer_msgs, "✍️ "))
        _normalize_draft_file()

    # Sub-phase B — structure-critic
    struct_hint = ("ОБЯЗАТЕЛЬНО вызови read_draft. Проверяй ТОЛЬКО структуру и формат "
                   "(7 секций, 6 тегов в '## Ключевые инсайты', плоские цитаты без backticks). "
                   "Фактологию НЕ трогай.")
    prev = ""
    for i in range(1, critic_rounds + 1):
        critique, approved, converged = _run_critic_round(
            struct_critic, "structure-critic", query, struct_hint, i, critic_rounds, prev, metrics)
        if approved or converged:
            break
        prev = critique
        writer_msgs.append({"role": "user",
                            "content": (f"STRUCTURE-КРИТИКА (только формат и структура):\n{critique}\n\n"
                                        "Перепиши через write_draft + append_draft.")})
        writer_msgs.extend(_run_agent(writer, writer_msgs, "✍️ "))
        _normalize_draft_file()


def _finalize_draft(query: str, metrics: RunMetrics, critic_rounds: int) -> None:
    """Фаза 2 (writer) + 3 (critic) + 4 (validator). Выделена для переиспользования
    в resume_research: позволяет дописать отчёт если phase 1 уже прошла но draft пустой.
    """
    # Фаза 2 — writer
    print("\n✍️  Фаза 2: writer — финальный черновик")
    writer = build_bot(WRITER_PROMPT,
                       ["read_plan", "read_notes", "read_synthesis", "read_draft",
                        "write_draft", "append_draft"],
                       max_tokens=6144)
    kb_context = _build_kb_context(query)
    writer_msgs = [{"role": "user",
                    "content": (f"Собери финальный отчёт по теме: {query}\n\n"
                                f"KB context (authoritative список источников — "
                                f"используй id и repo ИМЕННО отсюда):\n{kb_context}")}]
    t_wr = time.time()
    resp = _run_agent(writer, writer_msgs, "✍️ ")
    metrics.writer_seconds = time.time() - t_wr
    writer_msgs.extend(resp)

    # Sanity-check: writer должен был вызвать write_draft. Если файл не создан или <200 симв —
    # retry со строгим сообщением. Если и после retry пусто — fallback программно.
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

    if _draft_too_small():
        print("   ❌ writer провалился и после retry — fallback на программный сбор из KB")
        _fallback_draft_from_kb(query)

    # Нормализация цитат: `[\`id\`](arxiv-id)` / `[\`id\`](repo: X)` → `[id]` / `[id, repo: X]`
    if _normalize_draft_file():
        print("   🧹 нормализованы цитаты в draft.md")

    # Фаза 3 — критика. Если CFG.specialized_critics=True (default) — два
    # специализированных критика подряд (fact → structure), каждый со своим фокусом.
    # Legacy-режим: один combined CRITIC_PROMPT.
    if CFG.get("specialized_critics", True) and critic_rounds >= 1:
        _run_specialized_critics(query, writer, writer_msgs, metrics, critic_rounds)
    else:
        _run_legacy_critic(query, writer, writer_msgs, metrics, critic_rounds)

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
        # HITL pause-point: даём пользователю шанс ткнуть «перепиши про X» до финализации.
        _hitl_review(query, writer, writer_msgs, valid, invalid, suspicious)
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


def resume_research(query: str | None = None, critic_rounds: int = 2):
    """Продолжает прерванный прогон: пропускает explorer/replanner, использует
    существующие notes.md/plan.md/kb.jsonl/synthesis.md и идёт сразу к writer/critic.

    Если synthesis.md отсутствует — запускает только фазу синтеза, затем writer.
    Query берётся из plan.md если не задан явно.
    """
    # Восстанавливаем query из plan.md если не передан
    if query is None and PLAN_PATH.exists():
        for ln in PLAN_PATH.read_text(encoding="utf-8").splitlines():
            if ln.startswith("# Plan:"):
                query = ln.replace("# Plan:", "").strip()
                break
    if not query:
        raise RuntimeError("не могу восстановить query: передай явно или убедись что plan.md существует")

    print(f"🔄 RESUME: продолжаем прогон по теме: {query}")
    print(f"📁 Рабочая папка: {RESEARCH_DIR}")
    have = {
        "notes.md": NOTES_PATH.exists() and NOTES_PATH.stat().st_size > 100,
        "plan.md": PLAN_PATH.exists(),
        "kb.jsonl": kb_mod.KB_PATH.exists(),
        "synthesis.md": SYNTHESIS_PATH.exists() and SYNTHESIS_PATH.stat().st_size > 100,
    }
    for k, v in have.items():
        print(f"   {'✓' if v else '✗'} {k}")
    if not have["notes.md"]:
        raise RuntimeError("notes.md пуст или отсутствует — нечего продолжать, запусти новый research")

    metrics = RunMetrics(query=query)

    if not have["synthesis.md"]:
        print("\n💡 synthesis.md отсутствует — запускаем фазу синтеза")
        synthesizer = build_bot(SYNTHESIZER_PROMPT,
                                ["read_plan", "read_notes", "write_synthesis", "kb_search"],
                                max_tokens=4096)
        t_syn = time.time()
        _run_agent(synthesizer,
                   [{"role": "user", "content": f"Произведи пять типов инсайтов по теме: {query}"}],
                   "💡")
        metrics.synthesis_seconds = time.time() - t_syn

    _finalize_draft(query, metrics, critic_rounds)
