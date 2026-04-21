# context.md

Карта кода для AI-сессий. Читать ПОСЛЕ `AGENTS.md` (инварианты) и ПЕРЕД работой.
Все пути — workspace-relative.

## Пакет `lra/` — 15 модулей

| Файл | LOC | Что делает | Ключевые экспорты |
|---|---|---|---|
| `lra/config.py` | ~90 | `Settings` dataclass из `chat_config.json` + runtime `CFG['flag']` | `CFG`, `DRAFT_PATH`, `NOTES_PATH`, `PLAN_PATH`, `SYNTHESIS_PATH`, `RESEARCH_DIR` |
| `lra/llm.py` | ~80 | `@register_llm("mlx")` — обёртка над mlx_lm с глобальным кэшем весов | `get_mlx(model_name)` |
| `lra/logger.py` | ~30 | structured logger → `research/run.log` | `get_logger(name)` |
| `lra/cache.py` | ~60 | disk-cache для CLI-вывода (hf/gh), TTL в часах | `get(cmd)`, `put(cmd, stdout, stderr, rc)` |
| `lra/cli.py` | ~70 | обёртка `subprocess.run` с кешем + единой обработкой ошибок | `run(cmd, timeout, use_cache)` → `CliResult` |
| `lra/memory.py` | 110 | файловая память: `reset_research()`, `seen_queries()`, `log_query()`, `is_similar_to_seen()` | те же функции |
| `lra/kb.py` | 151 | knowledge-base: `Atom(id, kind, topic, claim, url, stars, lang)` → jsonl | `add(atom)`, `load()`, `search(query)` |
| `lra/plan.py` | 487 | структурированный plan.json: дерево задач, `[FOCUS]`/`[TODO]`/`[DONE]`, markdown-рендер | `Plan`, `Task`, `render_md()`, `sync_focus_from_md()` |
| `lra/metrics.py` | ~100 | `RunMetrics`, `IterationMetric`, `CriticRound` → metrics.json | `count_critic_issues(text)` |
| `lra/prompts.py` | 228 | 6 role-prompts: EXPLORER, REPLANNER, SYNTHESIZER, WRITER, CRITIC, FACT_CRITIC, STRUCTURE_CRITIC, COMPRESSOR | все `*_PROMPT` константы |
| `lra/utils.py` | ~80 | `extract_ids(text)`, `normalize_query(q)`, `parse_args(params)`, `keyword_set(text)`, `jaccard(a,b)` | те же |
| `lra/validator.py` | ~100 | `validate_draft_ids()` — проверяет arxiv-id через `hf papers info` + keyword overlap draft↔notes | `validate_draft_ids()` → `(valid, invalid, suspicious)` |
| `lra/tools.py` | 689 | **22 @register_tool классов** — см. таблицу ниже | все `*Tool` классы, `verify_ids_against_kb(content)` |
| `lra/pipeline.py` | 736 | оркестратор. Phase 1 (explorer↔replanner) → compressor? → Phase 2 (synthesizer) → Phase 3 (writer ↔ critics) → Phase 4 (validator) → HITL? | `research_loop(query, depth, critic_rounds)`, `resume_research()` |
| `lra/__init__.py` | ~10 | `__all__` re-exports | — |

## `lra/tools.py` — 22 @register_tool классов

Группы (порядок как в файле):

**Поиск:** `HfPapers` (hf papers search), `GithubSearch` (gh search repos/code) — оба с dedup через `querylog`, autosave в `kb.jsonl`.

**Sandbox:** `RunPython` — `subprocess` с RLIMIT_CPU=5s, AS=512MB, socket monkey-patch, write только в `/tmp`. **Best-effort** (см. docstring).

**Notes:** `ReadNotes`, `AppendNotes` (с pre-append verifier если `notes_strict=True`), `CompactNotes`.

**Draft:** `ReadDraft`, `WriteDraft`, `AppendDraft`.

**Plan:** `ReadPlan`, `WritePlan`, `PlanAddTask`, `PlanCloseTask`, `PlanSplitTask`.

**Synthesis:** `ReadSynthesis`, `WriteSynthesis`.

**KB:** `KbAdd`, `KbSearch`.

**Memory:** `ReadLessons`, `AppendLessons`, `ReadQueryLog`.

## `lra/pipeline.py` — функции оркестрации

| Функция | Строки | Что делает |
|---|---|---|
| `research_loop(query, depth=6, critic_rounds=2)` | 129–323 | главная точка входа; phase 1 + вызов `_finalize_draft` |
| `resume_research(query=None, critic_rounds=2)` | ~в конце | пропускает phase 1, использует существующие notes/kb |
| `_run_iteration(...)` | — | один цикл explorer → replanner (phase 1 body) |
| `_finalize_draft(query, metrics, critic_rounds)` | ~570 | phase 2 (writer) + phase 3 (critics) + phase 4 (validator) + HITL |
| `_run_critic_round(critic, name, ...)` | 452–486 | один раунд любой критики (fact/structure/combined) |
| `_run_legacy_critic(...)` | 488–508 | цикл `CRITIC_PROMPT` (если `specialized_critics=False`) |
| `_run_specialized_critics(...)` | 510–568 | fact-critic → writer rewrite → structure-critic → writer rewrite |
| `_hitl_review(query, writer, writer_msgs, valid, invalid, suspicious)` | ~452+ | HITL пауза после валидатора (guarded `CFG.hitl` + `sys.stdin.isatty()`) |
| `_fallback_draft_from_kb(query)` | 349–396 | если writer дважды провалил write_draft — собираем draft программно из kb |
| `_normalize_draft_file()` | ~430 | пост-процесс цитат в draft.md: ``[`id`](arxiv-id)`` → `[id]` |
| `_build_kb_context(query)` | — | собирает authoritative список источников для writer'а из kb.jsonl |

## Data flow

```
User: "grounding LLMs"
  │
  ▼
research_loop("grounding LLMs", depth=6, critic_rounds=2)
  │
  ├── Phase 1 × depth итераций:
  │     EXPLORER → hf_papers + github_search → AppendNotes → kb.add
  │     REPLANNER → WritePlan (новый [FOCUS] + [TODO])
  │     info-gain early stop: <2 новых id → прерываем
  │
  ├── compressor если notes.md > 8000 симв
  │
  ├── Phase 2: SYNTHESIZER → WriteSynthesis (6 тегов)
  │
  ├── Phase 3: WRITER → WriteDraft/AppendDraft
  │     (если draft <200 симв → retry → fallback _fallback_draft_from_kb)
  │
  ├── Phase 3b: если CFG.specialized_critics=True:
  │     FACT_CRITIC → writer rewrite → STRUCTURE_CRITIC → writer rewrite
  │   иначе (legacy):
  │     CRITIC → writer rewrite (×critic_rounds)
  │
  ├── Phase 4: validate_draft_ids() — hf papers info на каждый arxiv-id
  │     + keyword overlap draft↔notes → (valid, invalid, suspicious)
  │
  ├── HITL если CFG.hitl=True И sys.stdin.isatty():
  │     превью + [a/r/s]; revise → один writer-pass с комментом
  │
  └── metrics.finish() → research/metrics.json
```

## Файлы артефактов (`research/`)

| Путь | Пишет | Читает | Глобальная? |
|---|---|---|---|
| `research/draft.md` | writer | writer, critic, validator | нет (стирается `/clean`) |
| `research/notes.md` | explorer (AppendNotes), compressor | explorer, synthesizer, critic | нет |
| `research/plan.md` | replanner, plan.py-render | explorer, replanner | нет |
| `research/plan.json` | plan.py (structured) | plan.py | нет |
| `research/synthesis.md` | synthesizer | writer, critic | нет |
| `research/kb.jsonl` | explorer (kb.add), github_search autosave | kb-context builder, pre-append verifier | нет |
| `research/metrics.json` | pipeline `metrics.finish()` | — | нет |
| `research/run.log` | logger | — | нет |
| `research/lessons.md` | explorer (AppendLessons) | explorer | **да** (НЕ `/clean`) |
| `research/querylog.md` | tools (log_query) | dedup проверки | **да** (НЕ `/clean`) |
| `research/archive/<ts>_<slug>/` | end-of-run snapshot | — | — |

## Тесты (133, все в `tests/`)

| Файл | Тестов | Что покрывает |
|---|---|---|
| `tests/test_utils.py` | ~15 | `extract_ids`, `normalize_query`, `keyword_set`, `jaccard` |
| `tests/test_memory.py` | ~10 | `reset_research`, querylog, dedup |
| `tests/test_validator.py` | ~8 | `validate_draft_ids` с моками `hf papers info` |
| `tests/test_kb.py` | ~10 | `Atom`, `add`, `load`, `search` |
| `tests/test_kb_autosave.py` | ~5 | GithubSearch → kb autosave при ≥10★ |
| `tests/test_plan.py` | ~20 | add/close/split/drop_task, focus rotation |
| `tests/test_plan_tools.py` | ~10 | `PlanAddTask`/`PlanCloseTask`/`PlanSplitTask` tools |
| `tests/test_github_search.py` | ~15 | `GithubSearch.call` — qualifier parse, dedup, fallback-hints |
| `tests/test_pipeline_integration.py` | ~5 | e2e с замоканным LLM — все фазы |
| `tests/test_append_verifier.py` | 8 | `notes_strict` — блокирует unknown arxiv-id |
| `tests/test_hitl.py` | 7 | HITL: disabled-default, non-TTY, approve, revise, Ctrl+C |
| `tests/test_metrics.py` | ~5 | `RunMetrics`, `count_critic_issues` |
| `tests/test_prompts.py` | ~3 | smoke prompts |
| остальное | ~12 | smoke / compressor / logger |

## Complexity hotspots (top-5 по lizard)

| Функция | CCN | NLOC | Примечание |
|---|---|---|---|
| `research_loop` @ `lra/pipeline.py:129` | 42 | 171 | Линейная композиция фаз — OK |
| `_fallback_draft_from_kb` @ `lra/pipeline.py:349` | 17 | 42 | Fallback when writer fails twice |
| `render_md` @ `lra/plan.py:265` | 22 | 50 | plan.json → markdown |
| `sync_focus_from_md` @ `lra/plan.py:439` | 17 | 38 | markdown-plan → structured Plan |
| `main` @ `agent.py:52` | 20 | 72 | REPL с 7 slash-командами |

## Внешние зависимости (pinned)

| Пакет | Версия | Зачем |
|---|---|---|
| `mlx-lm` | 0.31.2 | MLX-backend для Qwen3.5 на Apple Silicon |
| `qwen-agent` | 0.0.34 | `@register_tool`, `@register_llm`, `Assistant` оркестратор |
| `json5` | 0.14.0 | Толерантный парсер для tool-params от LLM |
| `pytest` | 9.0.3 | — |

Внешние CLI (системные): `hf` (HuggingFace), `gh` (GitHub). Проверка в `agent.py::_check_clis`.
