"""Pipeline orchestrator: explorer ↔ replanner → synthesizer → writer ↔ critic → validator."""
from __future__ import annotations

import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from qwen_agent.agents import Assistant
from qwen_agent.utils.output_beautify import typewriter_print

# Importing tools is required for @register_tool side effects — do not remove
# even if the name is unused here. Importing llm registers @register_llm("mlx")
# (otherwise resume_research launched without agent.py bootstrapping fails with
# "Please set model_type").
from . import cli as cli_run
from . import kb as kb_mod
from . import llm as _llm_register  # noqa: F401
from . import plan as plan_mod
from . import research_memory as research_memory_mod
from . import tools  # noqa: F401
from .config import (
    CFG,
    DRAFT_PATH,
    LESSONS_PATH,  # noqa: F401  # re-exported for tests (pipeline.LESSONS_PATH)
    NOTES_PATH,
    PLAN_PATH,
    REJECTED_PATH,  # noqa: F401  # re-exported for tests (pipeline.REJECTED_PATH)
    RESEARCH_DIR,
    SYNTHESIS_PATH,
)
from .logger import get_logger
from .metrics import (
    CriticRound,
    IterationMetric,
    RunMetrics,
    count_critic_issues,
    summarize_evidence_quality,
)
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
    """Warms disk-cache of hf_papers + github_search in parallel before the explorer LLM runs.

    When the explorer later issues the same commands they hit the cache (~0s
    vs ~10-30s each). CLI errors are silently ignored — this is a warmup,
    not a critical op. Returns {'hf': bool, 'gh': bool, 'elapsed': float}
    for observability.
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
    """Runs a qwen-agent tool-loop and streams progress.

    P11: a soft cap on the number of LLM turns per call protects against
    MLX pathological loops where the explorer can spin for 20+ minutes
    inside a single `_run_agent` invocation. When the cap is hit we just
    stop consuming the generator and return the latest response we have.
    """
    print(icon + " ", end="", flush=True)
    plain, resp = "", []
    max_turns = CFG.get("max_agent_turns", 24)
    turns = 0
    start = time.time()
    soft_wall = CFG.get("agent_call_wall_clock_s", 420)  # 7 min per _run_agent
    for resp in bot.run(messages=messages):
        plain = typewriter_print(resp, plain)
        turns += 1
        if max_turns and turns >= max_turns:
            print(f"\n   ⏱️  P11: cap max_agent_turns={max_turns} reached — stopping tool loop")
            break
        if soft_wall and (time.time() - start) > soft_wall:
            print(f"\n   ⏱️  P11: cap agent_call_wall_clock_s={soft_wall}s reached — stopping tool loop")
            break
    print()
    return resp


def _current_focus(query: str) -> str:
    """Focus is taken from plan.json (source of truth). If missing (legacy or
    corrupted plan.json), fallback to parsing plan.md.
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
    """Programmatic focus rotation via plan.json. Closes the current focus as
    blocked (if still open) and sets focus to the first open task. Returns
    True if rotation succeeded.
    """
    plan = plan_mod.load()
    if not plan:
        return False
    if plan.current_focus_id:
        # mark as blocked — the replanner failed on it
        t = plan.get(plan.current_focus_id)
        if t and t.status == "in_progress":
            plan.block_task(t.id, why="replanner failed twice, rotating")
        plan.current_focus_id = None
    next_t = next((t for t in plan.tasks if t.status == "open"), None)
    if not next_t:
        plan_mod.save(plan)
        return False
    plan.set_focus(next_t.id, why="fallback rotation after replanner failure")
    plan_mod.save(plan)
    return True


def _bootstrap_initial_plan(query: str) -> bool:
    """One LLM call to BOOTSTRAP-planner → rewrites plan.json with specific tasks.

    Recovery strategy on errors (in order):
    1. Full INITIAL_PLANNER_PROMPT (1 try).
    2. Simplified retry-prompt: JSON only, with a list of terms and titles (1 try).
    3. Static-vocabulary fallback: keyword_set(query) → the domain gate runs
       on specific words derived from the topic itself instead of relaxing
       to header-only.

    Returns True if bootstrap was applied (including the static-vocab fallback),
    False when only a fully static plan without vocabulary remains.
    """
    # ── Attempt 1: full prompt ───────────────────────────────────
    parsed = _try_bootstrap_call(query, INITIAL_PLANNER_PROMPT, label="bootstrap")
    if parsed:
        return _apply_bootstrap_parsed(query, parsed)

    # ── Attempt 2: simplified retry ──────────────────────────────
    retry_prompt = (
        "Return STRICTLY one JSON object (no markdown fence, no explanation) of the form:\n"
        '{"topic_type": "engineering|theoretical|mixed",\n'
        ' "core_vocabulary": ["<8-12 narrow domain terms>"],\n'
        ' "tasks": [{"title": "<40-90 chars>", "why": "<1 line>"}, ...]}\n'
        "At least 4 tasks in tasks. No text outside the JSON."
    )
    parsed = _try_bootstrap_call(query, retry_prompt, label="bootstrap-retry")
    if parsed:
        return _apply_bootstrap_parsed(query, parsed)

    # ── Fallback 3: static vocabulary derived from query ─────────
    from .utils import derive_static_vocabulary
    static_vocab = derive_static_vocabulary(query)
    if static_vocab:
        # Reset already created the static plan. Just write core_vocabulary.
        plan = plan_mod.load()
        if plan is not None:
            plan.core_vocabulary = static_vocab
            plan_mod.save(plan)
            log.warning("bootstrap planner failed twice; static-vocab fallback applied "
                        "(%d terms from query): %s", len(static_vocab), ", ".join(static_vocab))
            print(f"   🛟 bootstrap fallback: static vocabulary from query "
                  f"({len(static_vocab)} terms) — gate works but without LLM refinement")
            return True
    log.warning("bootstrap planner: all strategies failed, plan without core_vocabulary "
                "(fail-closed gate will block all appends until CFG['allow_no_vocab']=True)")
    print("   ⚠️  bootstrap plan: static plan without vocabulary — the agent cannot write "
          "to notes until you set CFG['allow_no_vocab']=True")
    return False


def _try_bootstrap_call(query: str, prompt: str, *, label: str) -> tuple | None:
    """One LLM call with the given prompt. Returns parse_bootstrap_json output or None."""
    try:
        planner = build_bot(prompt, [], max_tokens=1024)
        msgs = [{"role": "user", "content": f"Topic: {query}"}]
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
            log.warning("%s: invalid JSON. Raw reply (first 300 chars): %s",
                        label, (raw or "(empty)")[:300])
        return parsed
    except Exception as exc:
        log.warning("%s failed (%s)", label, exc)
        return None


def _apply_bootstrap_parsed(query: str, parsed: tuple) -> bool:
    """Applies the parsed bootstrap result to plan.json. True on success."""
    topic_type, seeds, core_vocab = parsed
    plan = plan_mod.bootstrap_from_seeds(query, seeds, topic_type=topic_type,
                                         core_vocabulary=core_vocab)
    if plan is None:
        log.warning("bootstrap planner: seeds did not pass validation (n=%d)", len(seeds))
        return False
    vocab_n = len(plan.core_vocabulary)
    if vocab_n == 0:
        # LLM returned valid JSON but empty vocabulary — try static fallback.
        from .utils import derive_static_vocabulary
        static_vocab = derive_static_vocabulary(query)
        if static_vocab:
            plan.core_vocabulary = static_vocab
            plan_mod.save(plan)
            vocab_n = len(static_vocab)
            log.warning("bootstrap planner: LLM vocab empty, static-vocab from query applied "
                        "(%d terms)", vocab_n)
    print(f"   🧭 bootstrap plan: topic_type={topic_type}, tasks={len(seeds)}, "
          f"core_vocabulary={vocab_n}")
    return True


def research_loop(query: str, depth: int = 6, critic_rounds: int = 2):
    """Full pipeline: explorer/replanner (×depth) → compressor → synthesizer →
    writer/critic (×critic_rounds) → citation validator."""
    from lra import tool_tracker
    tool_tracker.reset_tracker()
    tool_tracker.set_tool_budget("compact_notes", 4)  # prevents compact_notes loop (see tool_tracker.py)
    metrics = RunMetrics(query=query)
    print(f"📁 Working directory: {RESEARCH_DIR}\n")

    # Bootstrap planner (optional): one LLM call before phase 1 generates 4-6
    # topic-specific seed tasks and classifies the topic (engineering /
    # theoretical / mixed). On any error/invalid JSON we silently fall back to
    # the static plan that reset_research → plan_mod.reset() has already
    # produced.
    if CFG.get("dynamic_initial_plan", True):
        _bootstrap_initial_plan(query)

    explorer = build_bot(EXPLORER_PROMPT,
                         ["hf_papers", "arxiv_search", "semantic_scholar_search", "github_search",
                          "read_plan", "read_notes", "read_notes_focused", "append_notes",
                          "read_lessons", "append_lessons", "read_querylog",
                          "kb_add", "kb_search",
                          "plan_add_task", "plan_close_task", "plan_split_task"],
                         max_tokens=3072)
    replanner = build_bot(REPLANNER_PROMPT,
                          ["read_notes", "read_plan", "write_plan"],
                          max_tokens=3072)

    print(f"🕳️  Phase 1: rabbit hole (up to {depth} iterations, adaptive plan)")
    low_gain_streak = 0
    empty_iter_streak = 0  # consecutive iterations where explorer did not grow notes (grew<100 after retry)
    stuck_focus_streak = 0  # consecutive iterations with the same FOCUS
    prev_focus = None
    for i in range(1, depth + 1):
        focus = _current_focus(query)
        print(f"\n── iteration {i}/{depth} ──  🎯 FOCUS: {focus[:80]}")
        t_iter_start = time.time()
        # Warm disk cache in parallel — hf+gh concurrently, before the first LLM call.
        pf = prefetch_iteration(focus)
        cached_marks = []
        if pf["hf_cached"]:
            cached_marks.append("hf")
        if pf["gh_cached"]:
            cached_marks.append("gh")
        cache_note = f" (from cache: {','.join(cached_marks)})" if cached_marks else ""
        print(f"   ⚡ prefetch hf+gh in parallel: {pf['elapsed']:.1f}s{cache_note}")
        ids_before = extract_ids(NOTES_PATH.read_text(encoding='utf-8') if NOTES_PATH.exists() else "")
        before = NOTES_PATH.stat().st_size if NOTES_PATH.exists() else 0
        # Inject top-3 atoms from KB relevant to FOCUS into the explorer's message —
        # this gives persistent context on top of read_notes (which is capped at 20k chars).
        kb_context = kb_mod.format_atoms(kb_mod.search(focus, k=3))
        kb_block = f"\n\nAlready known on similar topics (KB top-3):\n{kb_context}\n" if kb_context else ""
        memory_context = _build_memory_context(query, focus)
        memory_block = (
            f"\n\nRelevant cross-session memory:\n{memory_context}\n"
            if memory_context else ""
        )
        status_block = "\n\n" + _build_status_context(query, focus)
        t_exp = time.time()
        msg = [{"role": "user",
                "content": f"Original topic: {query}\n"
                           f"Current [FOCUS] from plan.md: {focus}{kb_block}{memory_block}{status_block}\n"
                           "Do one iteration on [FOCUS]. MANDATORY append_notes and append_lessons. "
                           "For EVERY new paper/repository call kb_add — needed for search "
                           "over accumulated knowledge in future iterations."}]
        _run_agent(explorer, msg, "🔎")
        explorer_seconds = time.time() - t_exp
        after = NOTES_PATH.stat().st_size if NOTES_PATH.exists() else 0
        grew = after - before
        ids_after = extract_ids(NOTES_PATH.read_text(encoding='utf-8') if NOTES_PATH.exists() else "")
        new_ids = ids_after - ids_before
        print(f"   📝 notes: +{grew} chars ({after} total)  📊 new arxiv-ids: {len(new_ids)}")
        if grew < 100:
            print("   ⚠️  notes did not grow — retry with a strict instruction")
            # Concrete plan of action instead of an abstract 'try again'. Three causes
            # of an empty iteration: (1) duplicate queries → kb_search, (2) long
            # github query rejected → shorten, (3) task exhausted → plan_close_task
            # + next TODO.
            retry = [{"role": "user", "content": (
                f"FAILURE: empty iteration (grew={grew}, new_ids=0). Execute EXACTLY this order:\n"
                f"1) kb_search '{focus[:60]}' (k=5) — check what is already in the base.\n"
                f"2) hf_papers with a REPHRASED query on '{focus[:60]}' "
                "(change terms, year, author — NOT the same phrase as in querylog); limit=5.\n"
                "3) If that also returned a duplicate — IMMEDIATELY append_notes (2-3 facts "
                "from kb_search results) and append_lessons with '[iter] exhausted: <focus>; "
                "next step: plan_close_task'. Do not loop on github_search.\n"
                "4) plan_close_task(id=<focus id>, evidence='...', why='exhausted').")}]
            _run_agent(explorer, retry, "🔁")
            after = NOTES_PATH.stat().st_size if NOTES_PATH.exists() else 0
            grew = after - before
            ids_after = extract_ids(NOTES_PATH.read_text(encoding='utf-8') if NOTES_PATH.exists() else "")
            new_ids = ids_after - ids_before

        low_gain_streak = low_gain_streak + 1 if len(new_ids) < 2 else 0

        print("   🧭 replanner updates the plan...")
        plan_before = PLAN_PATH.read_text(encoding='utf-8') if PLAN_PATH.exists() else ""
        t_rep = time.time()
        _run_agent(replanner,
                   [{"role": "user",
                     "content": f"Original topic: {query}. Update plan.md: Digest, Direction check, "
                                "new [FOCUS], resort [TODO]. Use write_plan."}],
                   "🧭")
        replanner_seconds = time.time() - t_rep
        plan_text = PLAN_PATH.read_text(encoding='utf-8') if PLAN_PATH.exists() else ""
        if plan_text == plan_before:
            print("   ⚠️  plan did not update — retry")
            _run_agent(replanner,
                       [{"role": "user", "content": (
                           "FAILURE: you did NOT call write_plan. Now: (1) read_notes — "
                           "look at what was added in the last iteration, (2) write_plan "
                           "MANDATORY: update Digest (3-5 new bullets with [arxiv-id]), "
                           "set a new [FOCUS] (take from [TODO] the sub-topic with the LEAST "
                           "evidence), update [TODO]/[DONE]. "
                           "tool_call arguments — strict JSON without markdown.")}],
                       "🔁")
            plan_text = PLAN_PATH.read_text(encoding='utf-8') if PLAN_PATH.exists() else ""
            if plan_text == plan_before:
                # Replanner failed twice in a row — rotate FOCUS programmatically so we don't loop.
                if _rotate_focus_fallback(query):
                    print("   🛟 fallback: FOCUS rotated programmatically from [TODO]")
                    plan_text = PLAN_PATH.read_text(encoding='utf-8')
        # If the replanner set a new [FOCUS] via write_plan (legacy path),
        # sync plan.json — otherwise the source of truth diverges from plan.md.
        _plan = plan_mod.load()
        if _plan and plan_text:
            if plan_mod.sync_focus_from_md(_plan, plan_text, iter_=i):
                plan_mod.save(_plan)
        new_focus = _current_focus(query)
        focus_changed = new_focus != focus
        if focus_changed:
            print(f"   🔀 vector corrected: {focus[:50]} → {new_focus[:50]}")

        # Stuck tracking: same FOCUS in a row + empty explorer in a row
        if prev_focus is not None and new_focus == prev_focus:
            stuck_focus_streak += 1
        else:
            stuck_focus_streak = 0
        prev_focus = new_focus
        if grew < 100:
            empty_iter_streak += 1
        else:
            empty_iter_streak = 0

        # ── Deterministic guard: bumps attempts, blocks exhausted tasks, auto-
        # rotates focus, signals HALT when all tasks are closed/blocked.
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
                print(f"✅ Early stop: guard reported {rep.halt_reason}")
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
            print("✅ Plan exhausted")
            metrics.stopped_early_reason = "PLAN_COMPLETE"
            break
        if "[TODO]" not in plan_text and i > 1:
            print("✅ No more [TODO]")
            metrics.stopped_early_reason = "NO_TODO"
            break
        # P10: per-iteration wall-clock timeout. A previous run had an iteration
        # of 9.6 hours (MLX anomaly); such an overrun must not drag N more
        # iterations with it. We do not interrupt MLX mid-call but after it
        # returns we look at the total iteration time and halt if over the limit.
        iter_wall_limit = CFG.get("iter_wall_clock_limit_s", 900)
        iter_elapsed = time.time() - t_iter_start
        if iter_wall_limit and iter_elapsed > iter_wall_limit:
            print(f"✅ Early stop: iter {i} took {iter_elapsed:.0f}s > "
                  f"limit {iter_wall_limit}s (ITER_WALL_CLOCK)")
            metrics.stopped_early_reason = "ITER_WALL_CLOCK"
            break
        if empty_iter_streak >= 2:
            print("✅ Early stop: 2 consecutive iterations with no note growth (agent stuck)")
            metrics.stopped_early_reason = "EMPTY_ITERATIONS"
            break
        if stuck_focus_streak >= 2 and i >= 3:
            print(f"✅ Early stop: FOCUS unchanged for 3 iterations ({prev_focus[:50]})")
            metrics.stopped_early_reason = "FOCUS_STUCK"
            break
        if low_gain_streak >= 2 and i >= 3:
            print("✅ Early stop: 2 consecutive iterations with < 2 new ids")
            metrics.stopped_early_reason = "LOW_GAIN"
            break

    # Phase 1.5 — compression
    notes_size = NOTES_PATH.stat().st_size if NOTES_PATH.exists() else 0
    if notes_size > 8000:
        print(f"\n🗜️  Phase 1.5: compressor ({notes_size} chars → ~5000)")
        compressor = build_bot(COMPRESSOR_PROMPT, ["read_notes", "compact_notes"])
        _run_agent(compressor, [{"role": "user", "content": "Compress notes to ~5000 chars."}], "🗜️")

    # Phase 2.0 — synthesis
    print("\n💡 Phase 2.0: synthesizer (bridges / contradictions / gaps / extrapolation / testable)")
    synthesizer = build_bot(SYNTHESIZER_PROMPT,
                            ["read_plan", "read_notes", "read_notes_focused", "write_synthesis", "kb_search"],
                            max_tokens=4096)
    synth_memory = _build_memory_context(query, "synthesis insights")
    synth_memory_block = (
        f"\n\nRelevant cross-session memory:\n{synth_memory}"
        if synth_memory else ""
    )
    synth_status_block = "\n\n" + _build_status_context(query)
    t_syn = time.time()
    _run_agent(synthesizer,
               [{"role": "user",
                 "content": f"Produce the six insight blocks on the topic: {query}"
                            f"{synth_memory_block}{synth_status_block}"}],
               "💡")
    metrics.synthesis_seconds = time.time() - t_syn

    _finalize_draft(query, metrics, critic_rounds)


# Re-export from context_builders. Historically these helpers lived in pipeline
# and their fully-qualified name `lra.pipeline._build_*` is used by tests
# (`pipeline._build_status_context`) and external consumers (agent.py). When
# we moved them into a dedicated module we kept the old names as re-exports —
# signatures and behavior are identical.
from .context_builders import (  # noqa: E402
    _build_kb_context,
    _build_memory_context,
    _build_status_context,
    _fallback_draft_from_kb,
    _latest_lessons_tail,
)

__all_context_builders__ = (
    "_build_kb_context",
    "_build_memory_context",
    "_build_status_context",
    "_fallback_draft_from_kb",
    "_latest_lessons_tail",
)

_CITATION_LINKY_RE = re.compile(r"\[`?([^`\]\s]+?)`?\]\((arxiv-id|repo:\s*[^)]+)\)")
_CITATION_BT_RE = re.compile(r"`(\d{4}\.\d{4,6}(?:v\d+)?)`")
_CITATION_INNER_BT_RE = re.compile(r"\[`([^`\]]+)`\]")
_CITATION_NESTED_RE = re.compile(r"\[((?:\[[^\[\]]+\]\s*,\s*)+\[[^\[\]]+\])\]")
_CITATION_DANGLING_RE = re.compile(r"(\][^\n]{0,40}?)\s*\((arxiv-id|repo:\s*[^)]+)\)")


def _normalize_citations(text: str) -> str:
    r"""Fixes the ugly `[`id`](arxiv-id)` / `[`id`](repo: X)` style → flat `[id]` / `[id, repo: X]`.

    Normalizes common forms the model likes to emit:
      1. ``[`2510.15624`](arxiv-id)`` → ``[2510.15624]``
      2. ``[`2510.15624`](repo: freephdlabor)`` → ``[2510.15624, repo: freephdlabor]``
      3. bare ```2506.09440``` → ``[2506.09440]``
      4. ``[`synthesis`]`` → ``[synthesis]``
      5. ``[[x], [y]]`` → ``[x, y]`` (after step 1)
      6. dangling ``(arxiv-id)`` / ``(repo: X)`` after ``[id]`` — removed
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

    # Flatten nested [[a], [b]] → [a, b]
    def _flatten(m: re.Match) -> str:
        inner = m.group(1)
        parts = re.findall(r"\[([^\[\]]+)\]", inner)
        return "[" + ", ".join(p.strip() for p in parts) + "]"

    text = _CITATION_NESTED_RE.sub(_flatten, text)
    # Dangling (arxiv-id)/(repo: X) right after ] — strip.
    text = _CITATION_DANGLING_RE.sub(r"\1", text)
    return text


def _normalize_draft_file() -> bool:
    """Applies _normalize_citations to draft.md in place. Returns True if anything changed."""
    if not DRAFT_PATH.exists():
        return False
    original = DRAFT_PATH.read_text(encoding="utf-8")
    fixed = _normalize_citations(original)
    if fixed != original:
        DRAFT_PATH.write_text(fixed, encoding="utf-8")
        return True
    return False


# ── P7: canonicalize the "## Sources" section against in-body citations ──
_SOURCES_HEADING_RE = re.compile(
    r"(^|\n)##+\s*(Sources|References|Источники)\s*\n.*?(?=\n##\s|\Z)",
    re.DOTALL | re.IGNORECASE,
)
_BODY_ARXIV_RE = re.compile(r"\[(\d{4}\.\d{4,6})(?:v\d+)?\]")
_BODY_REPO_RE = re.compile(r"\[repo:\s*([^\]]+)\]|(?:^|\s)([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+)(?=[\s.,\]])")


def _canonicalize_sources_section(text: str, invalid_ids: list[str]) -> tuple[str, bool]:
    """Replaces the content of ## Sources with a canonical list of body citations.

    1. Collect all [arxiv-id] from the body (the whole draft before the Sources section).
    2. Drop invalid_ids (flagged by the validator as non-existent).
    3. Collect [repo: owner/name] from the body.
    4. Regenerate the Sources section in the canonical format.

    Protects from body↔sources drift: the writer tends not to update this
    section when rewriting the body. Also scrubs 2607.15491-style
    hallucinations out of the final.

    Returns (new_text, changed).
    """
    m = _SOURCES_HEADING_RE.search(text)
    if not m:
        return text, False
    sources_start = m.start(0)
    body = text[:sources_start]
    # arxiv-ids from the body
    body_arxiv = []
    seen_arx: set[str] = set()
    invalid_set = {i.strip() for i in (invalid_ids or [])}
    for aid in _BODY_ARXIV_RE.findall(body):
        if aid in invalid_set:
            continue
        if aid in seen_arx:
            continue
        seen_arx.add(aid)
        body_arxiv.append(aid)
    # repo-ids: only the [repo: X/Y] form (bare slashes in the text are too noisy for the draft)
    body_repos: list[str] = []
    seen_repo: set[str] = set()
    for repo_bracket, _ in _BODY_REPO_RE.findall(body):
        r = (repo_bracket or "").strip()
        if not r:
            continue
        if r in seen_repo:
            continue
        seen_repo.add(r)
        body_repos.append(r)
    # Render
    lines: list[str] = ["", "## Sources", ""]
    if body_arxiv:
        for aid in body_arxiv:
            lines.append(f"- [{aid}](https://arxiv.org/abs/{aid})")
    else:
        lines.append("_no arxiv citations in the report_")
    if body_repos:
        lines.append("")
        lines.append("**Repositories:**")
        for r in body_repos:
            lines.append(f"- [{r}](https://github.com/{r})")
    lines.append("")
    new_section = "\n".join(lines)
    # Preserve the tail after the section, if any.
    tail = text[m.end(0):]
    new_text = body.rstrip() + "\n" + new_section + (tail if tail else "")
    return new_text, new_text != text


def _canonicalize_sources_file(invalid_ids: list[str]) -> bool:
    """Applies _canonicalize_sources_section to draft.md in place."""
    if not DRAFT_PATH.exists():
        return False
    original = DRAFT_PATH.read_text(encoding="utf-8")
    fixed, changed = _canonicalize_sources_section(original, invalid_ids)
    if changed:
        DRAFT_PATH.write_text(fixed, encoding="utf-8")
    return changed


_UNAPPROVED_BANNER_MARK = "<!-- lra-unapproved-banner -->"


def _prepend_unapproved_banner(n_rounds: int, invalid: list[str], suspicious: list[str]) -> None:
    """Prepends a warning banner to draft.md when no critic round approved it.

    Idempotent: if the banner marker (HTML comment) is already present, we do not
    duplicate it.
    """
    if not DRAFT_PATH.exists():
        return
    original = DRAFT_PATH.read_text(encoding="utf-8")
    if _UNAPPROVED_BANNER_MARK in original:
        return
    parts = [
        f"{_UNAPPROVED_BANNER_MARK}",
        "> ⚠️ **Draft NOT approved by critic**",
        f"> {n_rounds} critic rounds completed, none ended with APPROVED.",
    ]
    if invalid:
        parts.append(f"> Hallucinated ids (removed from Sources): {', '.join(invalid)}.")
    if suspicious:
        parts.append(f"> Suspicious citations (weak overlap with notes): {', '.join(suspicious)}.")
    parts.extend([
        "> Verify facts manually before use.",
        "",
        "",
    ])
    banner = "\n".join(parts)
    DRAFT_PATH.write_text(banner + original, encoding="utf-8")


def _hitl_review(query: str, writer: Assistant, writer_msgs: list,
                 valid: int, invalid: list, suspicious: list) -> None:
    """Human-in-the-loop pause after the validator. Prints a preview + metrics
    and asks the user either to approve the draft, give a comment for one
    extra writer pass, or skip.

    Invoked only if CFG.hitl=True AND stdin is a TTY (otherwise tests / resume
    would break non-interactive runs).
    """
    if not CFG.get("hitl", False) or not sys.stdin.isatty() or not DRAFT_PATH.exists():
        return
    print("\n" + "═" * 60)
    print("🧑 HITL pause-point — report ready, fixes still possible")
    print("═" * 60)
    draft_text = DRAFT_PATH.read_text(encoding="utf-8")
    preview = draft_text if len(draft_text) < 2000 else draft_text[:1000] + "\n...\n" + draft_text[-800:]
    print(preview)
    print("─" * 60)
    print(f"Metrics: valid={valid}, invalid={len(invalid)}, suspicious={len(suspicious)}, "
          f"chars={len(draft_text)}")
    print("Commands: [a] approve as-is  |  [r] revise (ask the writer to rewrite)  "
          "|  [s] skip (same as approve)")
    try:
        choice = input("Choice [a/r/s]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("   ⚠️  HITL interrupted — accepting as-is")
        return
    if choice.startswith("r"):
        comment = choice[1:].strip(" :") if len(choice) > 1 else ""
        if not comment:
            try:
                comment = input("What to rewrite? ").strip()
            except (EOFError, KeyboardInterrupt):
                print("   ⚠️  no comment entered — accepting as-is")
                return
        if not comment:
            print("   ⚠️  empty comment — accepting as-is")
            return
        print(f"   🔁 HITL revise: '{comment}' — running one extra writer pass")
        writer_msgs.append({"role": "user",
                            "content": (f"HITL COMMENT from a human on topic '{query}':\n{comment}\n\n"
                                        "Rewrite the draft via write_draft + append_draft taking "
                                        "this comment into account. Keep sections and format.")})
        writer_msgs.extend(_run_agent(writer, writer_msgs, "✍️ "))
        _normalize_draft_file()
        print("   ✓ writer rewrote the draft per HITL comment")
    else:
        print("   ✓ HITL approved — finalizing as-is")


def _run_critic_round(critic: Assistant, critic_name: str, query: str,
                      user_hint: str, i: int, total: int,
                      prev_critique: str, metrics: RunMetrics) -> tuple[str, bool, bool]:
    """One critic round. Returns (critique_text, approved, converged).

    Extracted to avoid duplicating the logic between legacy-single-critic and
    specialized-critics modes.
    """
    print(f"\n── {critic_name} {i}/{total} ──")
    t_cr = time.time()
    c_resp = _run_agent(critic,
                        [{"role": "user", "content": (
                            f"Review the draft on topic: {query}\n\n{user_hint}\n\n"
                            "After reading — either a list of up to 5 fixes in the format from "
                            "the system prompt, or exactly the word APPROVED on its own line.")}],
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
            print(f"   🔁 critic looping (similarity {sim:.0%})")
    print(f"   📋 fixes: {issues}  (draft_ok={draft_ok}, approved={approved})")
    metrics.critic_rounds.append(CriticRound(
        round=len(metrics.critic_rounds) + 1, issues_found=(0 if approved else issues),
        approved=approved, converged_by_similarity=converged, seconds=c_seconds,
    ))
    return critique, approved, converged


def _run_legacy_critic(query: str, writer: Assistant, writer_msgs: list,
                       metrics: RunMetrics, critic_rounds: int) -> None:
    """One combined critic (CRITIC_PROMPT) with rewrite loop."""
    print(f"\n🔍 Phase 3: critic ({critic_rounds} round(s))")
    critic = build_bot(CRITIC_PROMPT,
                       ["read_draft", "read_notes", "read_synthesis"],
                       max_tokens=2048)
    hint = ("MANDATORY: first call read_draft, then read_notes and read_synthesis — "
            "you HAVE these tools. Do not ask for text to be sent.")
    prev_critique = ""
    for i in range(1, critic_rounds + 1):
        critique, approved, converged = _run_critic_round(
            critic, "critic", query, hint, i, critic_rounds, prev_critique, metrics)
        if approved or converged:
            break
        prev_critique = critique
        writer_msgs.append({"role": "user",
                            "content": f"Critique:\n{critique}\n\nRewrite via write_draft."})
        writer_msgs.extend(_run_agent(writer, writer_msgs, "✍️ "))
        _normalize_draft_file()


def _run_specialized_critics(query: str, writer: Assistant, writer_msgs: list,
                             metrics: RunMetrics, critic_rounds: int) -> None:
    """Fact-critic → writer rewrite loop → structure-critic → writer rewrite loop.

    Reflects the insight from [2506.18096]: splitting the verifier into a
    factual and a structural critic yields more precise edits than one
    'universal' critic.
    """
    print(f"\n🔍 Phase 3: specialized critics (fact → structure, up to {critic_rounds} round(s) each)")
    fact_critic = build_bot(FACT_CRITIC_PROMPT,
                            ["read_draft", "read_notes", "read_synthesis"],
                            max_tokens=2048)
    struct_critic = build_bot(STRUCTURE_CRITIC_PROMPT,
                              ["read_draft"],
                              max_tokens=1536)

    # Sub-phase A — fact-critic
    fact_hint = ("MANDATORY: first call read_draft, then read_notes and read_synthesis. "
                 "Check factuality only: is [id] present, is the id mentioned in notes, "
                 "is the fact next to the id consistent with notes. Do NOT touch structure.")
    prev = ""
    for i in range(1, critic_rounds + 1):
        critique, approved, converged = _run_critic_round(
            fact_critic, "fact-critic", query, fact_hint, i, critic_rounds, prev, metrics)
        if approved or converged:
            break
        prev = critique
        writer_msgs.append({"role": "user",
                            "content": (f"FACT CRITIQUE (factuality only):\n{critique}\n\n"
                                        "Rewrite via write_draft + append_draft.")})
        writer_msgs.extend(_run_agent(writer, writer_msgs, "✍️ "))
        _normalize_draft_file()

    # Sub-phase B — structure-critic
    struct_hint = ("MANDATORY: call read_draft. Check ONLY structure and format "
                   "(7 sections, 6 tags in '## Key Insights', flat citations without backticks). "
                   "Do NOT touch factuality.")
    prev = ""
    for i in range(1, critic_rounds + 1):
        critique, approved, converged = _run_critic_round(
            struct_critic, "structure-critic", query, struct_hint, i, critic_rounds, prev, metrics)
        if approved or converged:
            break
        prev = critique
        writer_msgs.append({"role": "user",
                            "content": (f"STRUCTURE CRITIQUE (format and structure only):\n{critique}\n\n"
                                        "Rewrite via write_draft + append_draft.")})
        writer_msgs.extend(_run_agent(writer, writer_msgs, "✍️ "))
        _normalize_draft_file()


def _finalize_draft(query: str, metrics: RunMetrics, critic_rounds: int) -> None:
    """Phase 2 (writer) + 3 (critic) + 4 (validator). Split out for reuse in
    resume_research: lets us finish the report when phase 1 already ran but
    the draft is empty.
    """
    # Phase 2 — writer
    print("\n✍️  Phase 2: writer — final draft")
    writer = build_bot(WRITER_PROMPT,
                       ["read_plan", "read_notes", "read_notes_focused", "read_synthesis", "read_draft",
                        "write_draft", "append_draft"],
                       max_tokens=6144)
    kb_context = _build_kb_context(query)
    memory_context = _build_memory_context(query, "final report")
    memory_block = (
        f"\n\nRelevant cross-session memory (use only as auxiliary context, "
        f"not as a replacement for notes/KB):\n{memory_context}"
        if memory_context else ""
    )
    status_block = "\n\n" + _build_status_context(query)
    writer_msgs = [{"role": "user",
                    "content": (f"Assemble the final report on topic: {query}\n\n"
                                f"KB context (authoritative list of sources — "
                                f"use ids and repos ONLY from here):\n{kb_context}"
                                f"{memory_block}{status_block}")}]
    t_wr = time.time()
    resp = _run_agent(writer, writer_msgs, "✍️ ")
    metrics.writer_seconds = time.time() - t_wr
    writer_msgs.extend(resp)

    # Sanity check: writer was supposed to call write_draft. If the file is
    # missing or <200 chars, retry with a strict message. If still empty
    # after retry — fall back programmatically.
    def _draft_too_small() -> bool:
        return not DRAFT_PATH.exists() or DRAFT_PATH.stat().st_size < 200

    if _draft_too_small():
        print("   ⚠️  writer did not save a draft (or <200 chars) — retry with a strict message")
        writer_msgs.append({"role": "user", "content": (
            "FAILURE: you did NOT call write_draft (or the draft is too short). "
            "RIGHT NOW call write_draft with the title and '## TL;DR' (3-6 bullets "
            "with [id] from the KB context above), then append_draft for the remaining "
            "sections in order: '## Approaches', '## Benchmarks and Metrics', "
            "'## Implementations', '## Key Insights' (6 tags from synthesis.md), "
            "'## Open Questions', '## Sources'. Do NOT reply with text — use the tools.")})
        resp = _run_agent(writer, writer_msgs, "🔁")
        writer_msgs.extend(resp)

    if _draft_too_small():
        print("   ❌ writer failed even after retry — falling back to programmatic draft from KB")
        _fallback_draft_from_kb(query)

    # Citation normalization: `[`id`](arxiv-id)` / `[`id`](repo: X)` → `[id]` / `[id, repo: X]`
    if _normalize_draft_file():
        print("   🧹 citations normalized in draft.md")

    # Phase 3 — critique. If CFG.specialized_critics=True (default) — two
    # specialized critics in sequence (fact → structure), each with its own
    # focus. Legacy mode: one combined CRITIC_PROMPT.
    if CFG.get("specialized_critics", True) and critic_rounds >= 1:
        _run_specialized_critics(query, writer, writer_msgs, metrics, critic_rounds)
    else:
        _run_legacy_critic(query, writer, writer_msgs, metrics, critic_rounds)

    if DRAFT_PATH.exists():
        print("\n🔐 Phase 4: citation validation (hf info + keyword overlap with notes)")
        valid, invalid, suspicious = validate_draft_ids()
        metrics.valid_ids = valid
        metrics.invalid_ids = list(invalid)
        metrics.suspicious_citations = list(suspicious)
        print(f"   ✓ valid: {valid}")
        if invalid:
            print(f"   ✗ not found: {invalid}")
            print("   ⚠️  possible hallucinations — review manually")
        if suspicious:
            print(f"   ⚠️  weak citation-to-notes overlap: {suspicious}")
            print("       (id exists but the text around it in the draft does not reflect facts from notes)")
        # P7: canonicalize ## Sources against body citations, minus invalid_ids.
        # Writers often do not update this section and it keeps hallucinated ids.
        if _canonicalize_sources_file(list(invalid)):
            removed = ", ".join(invalid) if invalid else "0"
            print(f"   🧾 ## Sources canonicalized (removed invalid: {removed})")
        # P9: if NO critic round returned approved=True the draft was
        # finalized by force (not enough rounds or critic could not confirm).
        # Prepend an explicit warning banner to the draft and mark metrics
        # as an early stop by quality.
        approved_any = any(getattr(cr, "approved", False) for cr in metrics.critic_rounds)
        if metrics.critic_rounds and not approved_any:
            n = len(metrics.critic_rounds)
            _prepend_unapproved_banner(n, invalid, suspicious)
            if not metrics.stopped_early_reason:
                metrics.stopped_early_reason = f"CRITIC_UNAPPROVED_AFTER_{n}_ROUNDS"
            print(f"   ⚠️  P9: draft NOT approved after {n} critic rounds — "
                  f"warning banner prepended, stopped_early_reason={metrics.stopped_early_reason}")
        # HITL pause-point: let the user say 'rewrite X' before finalizing.
        _hitl_review(query, writer, writer_msgs, valid, invalid, suspicious)
        metrics.final_draft_chars = DRAFT_PATH.stat().st_size
        quality = summarize_evidence_quality(
            DRAFT_PATH.read_text(encoding="utf-8"),
            NOTES_PATH.read_text(encoding="utf-8") if NOTES_PATH.exists() else "",
            valid_ids=valid,
        )
        for key, value in quality.items():
            setattr(metrics, key, value)
        print(f"\n📄 Result: {DRAFT_PATH}\n" + "─" * 60)
        print(DRAFT_PATH.read_text(encoding='utf-8'))
        print("─" * 60)
        print(f"Files: {NOTES_PATH.name}, {PLAN_PATH.name}, {DRAFT_PATH.name}")
    else:
        print("⚠️  writer did not save a draft")

    try:
        research_memory_mod.record_run_memory(
            query=query,
            stopped_reason=metrics.stopped_early_reason,
            valid_ids=metrics.valid_ids,
            invalid_ids=metrics.invalid_ids,
            suspicious_citations=metrics.suspicious_citations,
            lessons_tail=_latest_lessons_tail(),
        )
    except Exception as exc:
        log.warning("failed to record run-summary memory (%s)", exc)

    # Always save metrics — even if the draft was not created.
    metrics_path = metrics.finish()
    print(f"📊 Run metrics: {metrics_path.relative_to(RESEARCH_DIR.parent)} "
          f"({metrics.total_seconds:.1f}s, {len(metrics.iterations)} iterations)")


def resume_research(query: str | None = None, critic_rounds: int = 2):
    """Continues an interrupted run: skips explorer/replanner, uses existing
    notes.md/plan.md/kb.jsonl/synthesis.md and goes straight to writer/critic.

    If synthesis.md is missing, it runs only the synthesis phase, then writer.
    Query is restored from plan.md when not given explicitly.
    """
    # Restore query from plan.md if not passed.
    if query is None and PLAN_PATH.exists():
        for ln in PLAN_PATH.read_text(encoding="utf-8").splitlines():
            if ln.startswith("# Plan:"):
                query = ln.replace("# Plan:", "").strip()
                break
    if not query:
        raise RuntimeError("cannot restore query: pass it explicitly or ensure plan.md exists")

    from lra import tool_tracker
    tool_tracker.reset_tracker()
    tool_tracker.set_tool_budget("compact_notes", 4)  # prevents compact_notes loop
    print(f"🔄 RESUME: continuing run on topic: {query}")
    print(f"📁 Working directory: {RESEARCH_DIR}")
    have = {
        "notes.md": NOTES_PATH.exists() and NOTES_PATH.stat().st_size > 100,
        "plan.md": PLAN_PATH.exists(),
        "kb.jsonl": kb_mod.KB_PATH.exists(),
        "synthesis.md": SYNTHESIS_PATH.exists() and SYNTHESIS_PATH.stat().st_size > 100,
    }
    for k, v in have.items():
        print(f"   {'✓' if v else '✗'} {k}")
    if not have["notes.md"]:
        raise RuntimeError("notes.md is empty or missing — nothing to continue, start a new research")

    metrics = RunMetrics(query=query)

    if not have["synthesis.md"]:
        print("\n💡 synthesis.md missing — running synthesis phase")
        synthesizer = build_bot(SYNTHESIZER_PROMPT,
                                ["read_plan", "read_notes", "write_synthesis", "kb_search"],
                                max_tokens=4096)
        t_syn = time.time()
        _run_agent(synthesizer,
                   [{"role": "user", "content": f"Produce the six insight blocks on the topic: {query}"}],
                   "💡")
        metrics.synthesis_seconds = time.time() - t_syn

    _finalize_draft(query, metrics, critic_rounds)
