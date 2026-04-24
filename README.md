# local-research-agent

A local, self-contained research agent for Apple Silicon: **MLX + Qwen-Agent + Hugging Face Papers / arXiv / Semantic Scholar / GitHub**. No cloud LLM, no API keys required beyond public data sources. One command turns a free-form topic into a cited, critic-reviewed Markdown report.

## What it does

Given a topic, the agent runs a multi-stage pipeline and writes `research/draft.md`:

```
bootstrap planner        produce a topic-specific plan.md (vocab + 5 initial tasks)
        ↓
explorer ↔ replanner     ×depth iterations, adaptive plan
        ↓
compressor               if notes.md > 8 KB
        ↓
synthesizer              bridges / contradictions / gaps / extrapolations / testable
        ↓
writer ↔ critic          ×rounds, with convergence and approval tracking
        ↓
validator                arXiv-id existence + keyword overlap between draft and notes
        ↓
canonicalizer            normalize ## Sources against body citations, drop invalid ids
```

Artifacts land in `research/` and a timestamped copy is archived under `research/archive/<slug>/`.

## Key features

- **Modular package** — `lra/` is split into `config`, `llm`, `tools`, `pipeline`, `plan`, `validator`, `memory`, `research_memory`, `metrics`, `kb`, `cache`. `agent.py` is a thin REPL.
- **Single MLX weight load** — all six agent roles share one copy of the model via a global cache.
- **Cross-session Reflexion memory** — `lessons.md`, `querylog.md`, and per-run summaries in `research/memory/` persist across runs and are only cleared by `/forget`.
- **Knowledge base** — append-only `kb.jsonl` atoms (papers, repos) with dedup on `(kind, id)` and title-collision detection (`kb_collisions.jsonl`).
- **Query enforcement** — `hf_papers`, `arxiv_search`, `github_search`, and `semantic_scholar_search` reject exact duplicate queries, forcing the model to rephrase.
- **Adaptive replanner** — rewrites `plan.md` each iteration with `[FOCUS] / Digest / [IN_PROGRESS] / [TODO] / [DONE]`.
- **Specialized critics** — separate fact-critic and structure-critic, or a combined legacy critic, with Jaccard-convergence and an explicit `APPROVED` token.
- **Citation validator** — every `[arxiv-id]` in the draft must resolve and its surrounding text must overlap with notes.
- **Source canonicalization** — `## Sources` is rebuilt from the body so the writer cannot silently drift.
- **Unapproved-draft banner** — if no critic round ever emits `APPROVED`, a warning block is prepended to `draft.md` and `stopped_early_reason=CRITIC_UNAPPROVED_AFTER_N_ROUNDS` is written to `metrics.json`.
- **Wall-clock safety nets** — per-iteration cap (`iter_wall_clock_limit_s`, default 900 s) and per-agent-call caps (`max_agent_turns=24`, `agent_call_wall_clock_s=420`) protect against MLX tool-loop stalls.

## Requirements

- Apple Silicon (M1 or newer)
- Python ≥ 3.12 (3.14 tested)
- [`hf` CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) on `PATH`
- Optional: GitHub Personal Access Token in `GH_TOKEN` for `github_search`

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python agent.py
```

On first run the default model (`mlx-community/Qwen3.5-9B-MLX-4bit`, ~5 GB) is downloaded into the standard Hugging Face cache. The repository itself stores nothing large.

## REPL commands

```
<any topic>     run research_loop on the topic
/clean          wipe the working research/ folder (lessons/querylog preserved)
/forget         full memory reset, including lessons and querylog
/hitl on|off    toggle human-in-the-loop review after the validator
/resume         resume a crashed run from the existing notes.md/kb.jsonl
/exit           quit
```

Defaults: `depth=6`, `critic_rounds=2`. Override in `main()` or call `research_loop()` directly from Python.

## Non-interactive usage

```python
from lra.pipeline import research_loop, resume_research

research_loop("adversarial attacks on LLM agents", depth=6, critic_rounds=2)
# after a crash:
resume_research()
```

## Runtime flags

Set in `chat_config.json` under `"extra"` or at runtime via `CFG["flag"] = value`.

| Flag | Default | Purpose |
|------|---------|---------|
| `notes_strict` | `true` | Block `append_notes` on unknown arxiv-ids |
| `strict_domain_gate` | `true` | Require ≥ 2 overlap with `plan.md` header vocabulary |
| `specialized_critics` | `true` | Fact-critic + structure-critic instead of combined critic |
| `dynamic_initial_plan` | `true` | LLM-bootstrapped topic-specific plan (fallback to static 5-seed) |
| `hitl` | `false` | Pause for human review after the validator |
| `iter_wall_clock_limit_s` | `900` | Halt if one iteration exceeds this budget (`0` disables) |
| `max_agent_turns` | `24` | Max tool-loop turns per `_run_agent` call (`0` disables) |
| `agent_call_wall_clock_s` | `420` | Max wall-clock seconds per `_run_agent` call (`0` disables) |

## Project layout

```
agent.py                       thin CLI / REPL
chat_config.json               model, temperature, max_tokens, runtime flags
requirements.txt               pinned dependencies
lra/
  config.py                    typed Settings + artifact paths
  llm.py                       MLX backend + weight cache
  tools.py                     @register_tool functions (hf_papers, arxiv, github, kb, plan...)
  prompts.py                   prompts for explorer / replanner / synthesizer / writer / critic / validator
  plan.py                      plan.json source of truth
  pipeline.py                  research_loop, build_bot, phases
  validator.py                 arxiv-id + overlap validation
  kb.py                        append-only knowledge base + collision detection
  memory.py                    short-term run memory
  research_memory.py           cross-session Reflexion memory
  metrics.py                   RunMetrics, IterationMetric, CriticRound
  cache.py, logger.py, tool_tracker.py, utils.py
tests/                         257+ tests, run without MLX, ~1 s
research/                      working directory (gitignored)
  draft.md, notes.md, plan.md, plan.json, synthesis.md,
  kb.jsonl, kb_collisions.jsonl, lessons.md, querylog.md,
  rejected.jsonl, metrics.json
  archive/                     timestamped snapshots of completed runs
```

## Development

```bash
source .venv/bin/activate
ruff check lra tests agent.py
python -m pytest -q
```

A commit should pass both. See [AGENTS.md](AGENTS.md) for the full contract used by AI assistants working on this repo.

## Status

Actively maintained, single-developer project. Tests: **257 passing** at `3f40073`.

## License

MIT. See [LICENSE](LICENSE).
