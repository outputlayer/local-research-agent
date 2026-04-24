# ROADMAP — progression 7.5 → 8.5

Three commits in a row, each must end with `ruff check ... && pytest -q → 211+ passed`.
Every stage is a behavior-neutral refactor (refactor-verify skill).

---

## Stage 1: extract context builders into `lra/context_builders.py` ✅ DONE

**Goal:** trim ~150 LOC off `pipeline.py` (currently 922 LOC), increase cohesion.

**What to move** (pure functions over file artifacts):
- `_build_kb_context(query)` — authoritative list of sources from `kb.jsonl` for the writer
- `_build_memory_context(query, phase_hint)` — wrapper over `research_memory`
- `_build_status_context(query, focus="")` — compact research status
- `_latest_lessons_tail(max_lines, max_chars)` — tail of `lessons.md`
- `_fallback_draft_from_kb(query)` — if it stays pure, otherwise leave it

**Compatibility:**
- Keep `from .context_builders import _build_status_context, ...` in `lra/pipeline.py`
  (re-export) so that tests and `agent.py` do not break.
- Monkeypatches `monkeypatch.setattr(pipeline, "REJECTED_PATH", ...)` work through
  the `REJECTED_PATH = REJECTED_PATH` re-export in pipeline — verify.

**Verify (refactor-verify 4-check):**
1. `ruff check lra tests agent.py` → `All checks passed!`
2. `pytest -q` → 211 passed with no changes
3. `git diff --stat` — only pipeline.py (deletions) and new context_builders.py
4. No behavior changes: signatures, return values, side effects are identical.

**Commit:** `refactor(pipeline): extract context builders to lra/context_builders.py`

---

## Stage 2: split `lra/tools.py` (1070 LOC) → `lra/tools/` package ✅ DONE

**Goal:** close the debt from `AGENTS.md` (trade-off #1: "tools.py split deferred due to
14 monkeypatches"). Solvable through a re-export shim.

**Layout:**
```
lra/tools/
  __init__.py          # re-export ALL classes + verify_ids_against_kb
  search.py            # HfPapers, ArxivSearch, GithubSearch
  notes.py             # ReadNotes, ReadNotesFocused, AppendNotes, CompactNotes
  draft.py             # ReadDraft, WriteDraft, AppendDraft
  plan.py              # ReadPlan, WritePlan, PlanAddTask, PlanCloseTask, PlanSplitTask
  synthesis.py         # ReadSynthesis, WriteSynthesis
  kb_tools.py          # KbAdd, KbSearch (named _tools to not collide with lra.kb)
  memory_tools.py      # ReadLessons, AppendLessons, ReadQueryLog
  _verifier.py         # verify_ids_against_kb + pre-append helpers
```

**Compatibility (critical):**
- `lra.tools.HfPapers`, `lra.tools.verify_ids_against_kb` — all 14 monkeypatches keep
  working via re-export in `__init__.py`.
- The `@register_tool` decorator references `lra.tools` — make sure importing submodules
  at `import lra.tools` registers all tools (done in `__init__.py`).

**Verify:**
1. `ruff check` → All checks passed
2. `pytest -q` → 211 passed
3. `grep -r "lra.tools\." tests/` — all legacy paths still work
4. `python -c "from lra.tools import HfPapers, AppendNotes, verify_ids_against_kb"` — OK

**Commit:** `refactor(tools): split monolithic tools.py into lra/tools/ package`

---

## Stage 3: GitHub Actions CI ✅ DONE

**Goal:** enforce "ruff + pytest before commit" discipline for every future PR.

**File:** `.github/workflows/ci.yml`

```yaml
name: CI
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'  # 3.14 wheels are not all built yet
          cache: pip
      - name: Install deps (no MLX — it's macOS+apple silicon only)
        run: |
          pip install -r requirements.txt || true
          pip install ruff pytest qwen-agent
      - name: Ruff
        run: ruff check lra tests agent.py
      - name: Pytest
        run: python -m pytest -q
```

**Important:** MLX is a macOS-only dependency. Either exclude `lra/llm.py` from the
import test, or use `pip install -r requirements.txt || true` and rely on the fact
that all tests use `@pytest.fixture` stubs.
**Verify:** all 211 tests import `lra.*` without real MLX (they already do).

**Verify:**
1. The file is valid YAML — check via `python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"`.
2. Commit + push → watch for the green badge.

**Commit:** `ci: add GitHub Actions workflow (ruff + pytest on push/PR)`

---

## Execution order

1. Stage 1 → commit → Stage 2 → commit → Stage 3 → commit → done.
2. Between stages always `ruff check && pytest -q` — no commit lands red.
3. Do not push to origin without an explicit user command.
