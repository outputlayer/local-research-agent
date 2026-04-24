# AGENTS.md

Contract for AI agents (GitHub Copilot / Claude Code / any qwen-agent host) that
will work in this repo. Read **at the start of every session**. Complements `context.md`
(code structure) and `README.md` (user interface).

## What is this project (one sentence)

A local research agent on MLX + Qwen-Agent: one command → multi-stage pipeline
(explorer → replanner → synthesizer → writer ↔ critic → validator) → report
`research/draft.md` with verified arxiv citations.

## Invariants — NEVER violate

| # | Prohibition | Reason |
|---|---|---|
| 1 | Do not delete `research/lessons.md` and `research/querylog.md` on `/clean` | Reflexion memory is cross-session — only `/forget` clears it |
| 2 | Do not write to `AppendNotes` an arxiv-id that is NOT in `kb.jsonl` (pre-append verifier)| Otherwise the explorer hallucinates references |
| 3 | Do not add `shell=True` to `subprocess.run` | All external CLIs (`hf`, `gh`) are invoked via argv-list in `lra/cli.py` |
| 4 | Do not change the text of `CRITIC_PROMPT` / `FACT_CRITIC_PROMPT` / `STRUCTURE_CRITIC_PROMPT` without approval | Quality thresholds are calibrated to the current versions |
| 5 | Do not commit files in `.skills/`, `.cache/`, `research/archive/` | All in `.gitignore`; third-party skills and caches |
| 6 | Do not call `print(...)` in library code `lra/*.py` (except explicit pipeline phases) | Logging goes through `lra/logger.py` so it does not interfere with pytest capture |

## Invariants — ALWAYS do

| # | Requirement | Verification command |
|---|---|---|
| A | Before commit: `ruff check lra tests agent.py` → `All checks passed!` | `ruff check lra tests agent.py` |
| B | Before commit: `pytest -q` → `257 passed` (or higher) | `source .venv/bin/activate && python -m pytest -q` |
| C | New @register_tool → add a test in `tests/` | `ls tests/test_*.py` |
| D | Behavior-neutral refactor → 4-check via the refactor-verify SKILL | see `.skills/vibesubin/plugins/vibesubin/skills/refactor-verify/SKILL.md` |
| E | New CFG flag → add it in `lra/config.py` and document it below in this file | `grep CFG.get lra/*.py` |

## Trade-offs — already decided, do not reopen

- **`tools.py` = single 689 LOC file**: a split into a `tools/` package is deferred — 14 monkeypatches in tests are attached to `lra.tools.*` attributes, ROI is low.
- **`research_loop` = 171 NLOC, CCN=42**: already split into `_run_iteration` / `_finalize_draft` / `_hitl_review`. Further splitting is linear phase composition and would increase code jumps.
- **Pinned deps via `==` exact, no lockfile**: simple CLI without auto-updates. A lockfile is over-engineering for 4 dependencies.
- **`CRITIC_PROMPT` kept alongside `FACT/STRUCTURE_CRITIC_PROMPT`**: legacy mode `specialized_critics=False`, needed for resume compatibility.

## How to work with this code

### Running tests
```bash
source .venv/bin/activate
python -m pytest -q
```
All 257 tests — no MLX, ~1 second.

### Running the agent
```bash
source .venv/bin/activate
python agent.py          # interactive REPL
# or: python -c "from lra.pipeline import research_loop; research_loop('topic', depth=6, critic_rounds=2)"
```

### Resume after a crash
```bash
python -c "from lra.pipeline import resume_research; resume_research()"
```
Skips the explorer, continues from synthesizer → writer → critic → validator, using
existing `notes.md`/`kb.jsonl`.

### Runtime flags (in `chat_config.json` under the `extra` key OR `CFG['flag'] = True` in the REPL)

| Flag | Default | What it does |
|---|---|---|
| `notes_strict` | `True` | Pre-append verifier blocks AppendNotes on unknown arxiv-ids |
| `strict_domain_gate` | `True` | Two-tier gate in AppendNotes and hf_papers kb auto-save: requires ≥2 overlap with HEADER from plan.md (the first line `# Plan:`). Seeds from [Tn] tasks drift and are used only for diagnosing the reason in `rejected.jsonl` (`reason`: `no_core_hit` / `weak_overlap`). Slow-start: if the header has <2 specific kws, the gate lets it through. |
| `specialized_critics` | `True` | Fact-critic + structure-critic instead of combined `CRITIC_PROMPT` |
| `dynamic_initial_plan` | `True` | LLM-bootstrap of topic-aware seeds before Phase 1; falls back to a static 5-seed plan on any parse/LLM error |
| `hitl` | `False` | Human-in-the-loop pause after the validator (REPL: `/hitl on`) |
| `iter_wall_clock_limit_s` | `900` | P10: if an iteration lasted > N seconds (prefetch+explorer+replanner), halt with `stopped_early_reason=ITER_WALL_CLOCK`. Set `0` to disable. |
| `max_agent_turns` | `24` | P11: max tool-loop turns in a single `_run_agent`. Protection against MLX pathological ReACT loops. `0` disables. |
| `agent_call_wall_clock_s` | `420` | P11: max wall-clock seconds for a single `_run_agent` call. `0` disables. |

## On any change — run

```bash
source .venv/bin/activate && ruff check lra tests agent.py && python -m pytest -q
```

If either step is red — DO NOT commit. Fix first.

## Vibesubin skills (local, in `.skills/vibesubin/plugins/vibesubin/skills/`)

For complex tasks read SKILL.md of the RELEVANT skill BEFORE starting work:

| Trigger | Skill |
|---|---|
| refactor / rename / split / safe delete | `refactor-verify` — 4-check verification is mandatory |
| "check security / secrets / vulnerabilities" | `audit-security` |
| "find dead code / what to delete?" | `fight-repo-rot` |
| write README / commit / PR / AGENTS.md | `write-for-ai` |

See `/memories/repo/vibesubin-skills.md` (Copilot memory) for the full table.

## Pointers

- `context.md` — package map: what lives where, key classes and functions
- `README.md` — user interface: install, commands, features
- `.skills/vibesubin/` — local dev skills (in `.gitignore`, do not commit)
- `research/` — run artifacts (draft.md, notes.md, plan.md, synthesis.md, kb.jsonl, metrics.json)
- `research/archive/` — snapshots of finished runs (timestamped)
