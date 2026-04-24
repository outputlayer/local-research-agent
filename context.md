# context.md

Code map for AI sessions. Read AFTER `AGENTS.md` (invariants) and BEFORE doing work.
All paths are workspace-relative.

## Package `lra/` — 15 modules

| File | LOC | What it does | Key exports |
|---|---|---|---|
| `lra/config.py` | ~90 | `Settings` dataclass from `chat_config.json` + runtime `CFG['flag']` | `CFG`, `DRAFT_PATH`, `NOTES_PATH`, `PLAN_PATH`, `SYNTHESIS_PATH`, `RESEARCH_DIR` |
| `lra/llm.py` | ~80 | `@register_llm("mlx")` — wrapper over mlx_lm with a global weights cache | `get_mlx(model_name)` |
| `lra/logger.py` | ~30 | structured logger → `research/run.log` | `get_logger(name)` |
| `lra/cache.py` | ~60 | disk cache for CLI output (hf/gh), TTL in hours | `get(cmd)`, `put(cmd, stdout, stderr, rc)` |
| `lra/cli.py` | ~70 | `subprocess.run` wrapper with cache and unified error handling | `run(cmd, timeout, use_cache)` → `CliResult` |
| `lra/memory.py` | 110 | file-backed memory: `reset_research()`, `seen_queries()`, `log_query()`, `is_similar_to_seen()` | same functions |
| `lra/kb.py` | 151 | knowledge base: `Atom(id, kind, topic, claim, url, stars, lang)` → jsonl | `add(atom)`, `load()`, `search(query)` |
| `lra/plan.py` | 487 | structured plan.json: task tree, `[FOCUS]`/`[TODO]`/`[DONE]`, markdown rendering | `Plan`, `Task`, `render_md()`, `sync_focus_from_md()` |
| `lra/metrics.py` | ~100 | `RunMetrics`, `IterationMetric`, `CriticRound` → metrics.json | `count_critic_issues(text)` |
| `lra/prompts.py` | 228 | 6 role prompts: EXPLORER, REPLANNER, SYNTHESIZER, WRITER, CRITIC, FACT_CRITIC, STRUCTURE_CRITIC, COMPRESSOR | all `*_PROMPT` constants |
| `lra/utils.py` | ~80 | `extract_ids(text)`, `normalize_query(q)`, `parse_args(params)`, `keyword_set(text)`, `jaccard(a,b)` | same |
| `lra/validator.py` | ~100 | `validate_draft_ids()` — checks arxiv-ids via `hf papers info` + keyword overlap draft↔notes | `validate_draft_ids()` → `(valid, invalid, suspicious)` |
| `lra/tools.py` | 689 | **22 @register_tool classes** — see table below | all `*Tool` classes, `verify_ids_against_kb(content)` |
| `lra/pipeline.py` | 736 | orchestrator. Phase 1 (explorer↔replanner) → compressor? → Phase 2 (synthesizer) → Phase 3 (writer ↔ critics) → Phase 4 (validator) → HITL? | `research_loop(query, depth, critic_rounds)`, `resume_research()` |
| `lra/__init__.py` | ~10 | `__all__` re-exports | — |

## `lra/tools.py` — 22 @register_tool classes

Groups (in file order):

**Search:** `HfPapers` (hf papers search), `GithubSearch` (gh search repos/code) — both with dedup via `querylog`, autosave into `kb.jsonl`.

**Notes:** `ReadNotes`, `ReadNotesFocused` (anti-drift: jaccard filter of blocks by focus query), `AppendNotes` (with pre-append verifier when `notes_strict=True`), `CompactNotes`.

**Draft:** `ReadDraft`, `WriteDraft`, `AppendDraft`.

**Plan:** `ReadPlan`, `WritePlan`, `PlanAddTask`, `PlanCloseTask`, `PlanSplitTask`.

**Synthesis:** `ReadSynthesis`, `WriteSynthesis`.

**KB:** `KbAdd`, `KbSearch`.

**Memory:** `ReadLessons`, `AppendLessons`, `ReadQueryLog`.

## `lra/pipeline.py` — orchestration functions

| Function | Lines | What it does |
|---|---|---|
| `research_loop(query, depth=6, critic_rounds=2)` | 129–323 | main entry point; phase 1 + call to `_finalize_draft` |
| `resume_research(query=None, critic_rounds=2)` | ~end | skips phase 1, uses existing notes/kb |
| `_run_iteration(...)` | — | one explorer → replanner cycle (phase 1 body) |
| `_finalize_draft(query, metrics, critic_rounds)` | ~570 | phase 2 (writer) + phase 3 (critics) + phase 4 (validator) + HITL |
| `_run_critic_round(critic, name, ...)` | 452–486 | one round of any critic (fact/structure/combined) |
| `_run_legacy_critic(...)` | 488–508 | `CRITIC_PROMPT` loop (when `specialized_critics=False`) |
| `_run_specialized_critics(...)` | 510–568 | fact-critic → writer rewrite → structure-critic → writer rewrite |
| `_hitl_review(query, writer, writer_msgs, valid, invalid, suspicious)` | ~452+ | HITL pause after validator (guarded by `CFG.hitl` + `sys.stdin.isatty()`) |
| `_fallback_draft_from_kb(query)` | 349–396 | if writer fails write_draft twice — assemble draft programmatically from kb |
| `_normalize_draft_file()` | ~430 | post-process citations in draft.md: ``[`id`](arxiv-id)`` → `[id]` |
| `_build_kb_context(query)` | — | assembles the authoritative list of sources for the writer from kb.jsonl |

## Data flow

```
User: "grounding LLMs"
  │
  ▼
research_loop("grounding LLMs", depth=6, critic_rounds=2)
  │
  ├── Phase 1 × depth iterations:
  │     EXPLORER → hf_papers + github_search → AppendNotes → kb.add
  │     REPLANNER → WritePlan (new [FOCUS] + [TODO])
  │     info-gain early stop: <2 new ids → halt
  │
  ├── compressor if notes.md > 8000 chars
  │
  ├── Phase 2: SYNTHESIZER → WriteSynthesis (6 tags)
  │
  ├── Phase 3: WRITER → WriteDraft/AppendDraft
  │     (if draft <200 chars → retry → fallback _fallback_draft_from_kb)
  │
  ├── Phase 3b: if CFG.specialized_critics=True:
  │     FACT_CRITIC → writer rewrite → STRUCTURE_CRITIC → writer rewrite
  │   else (legacy):
  │     CRITIC → writer rewrite (×critic_rounds)
  │
  ├── Phase 4: validate_draft_ids() — hf papers info for each arxiv-id
  │     + keyword overlap draft↔notes → (valid, invalid, suspicious)
  │
  ├── HITL if CFG.hitl=True AND sys.stdin.isatty():
  │     preview + [a/r/s]; revise → one writer pass with comment
  │
  └── metrics.finish() → research/metrics.json
```

## Artifact files (`research/`)

| Path | Writer | Reader | Global? |
|---|---|---|---|
| `research/draft.md` | writer | writer, critic, validator | no (wiped by `/clean`) |
| `research/notes.md` | explorer (AppendNotes), compressor | explorer, synthesizer, critic | no |
| `research/plan.md` | replanner, plan.py render | explorer, replanner | no |
| `research/plan.json` | plan.py (structured) | plan.py | no |
| `research/synthesis.md` | synthesizer | writer, critic | no |
| `research/kb.jsonl` | explorer (kb.add), github_search autosave | kb-context builder, pre-append verifier | no |
| `research/metrics.json` | pipeline `metrics.finish()` | — | no |
| `research/run.log` | logger | — | no |
| `research/lessons.md` | explorer (AppendLessons) | explorer | **yes** (NOT wiped by `/clean`) |
| `research/querylog.md` | tools (log_query) | dedup checks | **yes** (NOT wiped by `/clean`) |
| `research/archive/<ts>_<slug>/` | end-of-run snapshot | — | — |

## Tests (257, all in `tests/`)

| File | Tests | Coverage |
|---|---|---|
| `tests/test_utils.py` | ~15 | `extract_ids`, `normalize_query`, `keyword_set`, `jaccard` |
| `tests/test_memory.py` | ~10 | `reset_research`, querylog, dedup |
| `tests/test_validator.py` | ~8 | `validate_draft_ids` with mocked `hf papers info` |
| `tests/test_kb.py` | ~10 | `Atom`, `add`, `load`, `search` |
| `tests/test_kb_autosave.py` | ~5 | GithubSearch → kb autosave at ≥10★ |
| `tests/test_plan.py` | ~20 | add/close/split/drop_task, focus rotation |
| `tests/test_plan_tools.py` | ~10 | `PlanAddTask`/`PlanCloseTask`/`PlanSplitTask` tools |
| `tests/test_github_search.py` | ~15 | `GithubSearch.call` — qualifier parse, dedup, fallback hints |
| `tests/test_pipeline_integration.py` | ~5 | e2e with mocked LLM — all phases |
| `tests/test_append_verifier.py` | 8 | `notes_strict` — blocks unknown arxiv-id |
| `tests/test_hitl.py` | 7 | HITL: disabled default, non-TTY, approve, revise, Ctrl+C |
| `tests/test_notes_focused.py` | 6 | `read_notes_focused` — jaccard filter of notes.md blocks |
| `tests/test_metrics.py` | ~5 | `RunMetrics`, `count_critic_issues` |
| `tests/test_prompts.py` | ~3 | smoke prompts |
| the rest | ~12 | smoke / compressor / logger |

## Complexity hotspots (top-5 by lizard)

| Function | CCN | NLOC | Note |
|---|---|---|---|
| `research_loop` @ `lra/pipeline.py:129` | 42 | 171 | Linear phase composition — OK |
| `_fallback_draft_from_kb` @ `lra/pipeline.py:349` | 17 | 42 | Fallback when writer fails twice |
| `render_md` @ `lra/plan.py:265` | 22 | 50 | plan.json → markdown |
| `sync_focus_from_md` @ `lra/plan.py:439` | 17 | 38 | markdown-plan → structured Plan |
| `main` @ `agent.py:52` | 20 | 72 | REPL with 7 slash commands |

## External dependencies (pinned)

| Package | Version | Why |
|---|---|---|
| `mlx-lm` | 0.31.2 | MLX backend for Qwen3.5 on Apple Silicon |
| `qwen-agent` | 0.0.34 | `@register_tool`, `@register_llm`, `Assistant` orchestrator |
| `json5` | 0.14.0 | Tolerant parser for tool params from the LLM |
| `pytest` | 9.0.3 | — |

External CLIs (system): `hf` (HuggingFace), `gh` (GitHub). Checked in `agent.py::_check_clis`.
