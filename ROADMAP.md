# ROADMAP — движение 7.5 → 8.5

Три коммита подряд, каждый должен заканчиваться `ruff check ... && pytest -q → 211+ passed`.
Каждый этап — поведенчески-нейтральный рефактор (refactor-verify skill).

---

## Этап 1: вынести билдеры контекста в `lra/context_builders.py` ✅ DONE

**Цель:** снять ~150 LOC с `pipeline.py` (сейчас 922 LOC), повысить cohesion.

**Что переносим** (pure-функции файловых артефактов):
- `_build_kb_context(query)` — authoritative список источников из `kb.jsonl` для writer'а
- `_build_memory_context(query, phase_hint)` — wrapper над `research_memory`
- `_build_status_context(query, focus="")` — compact research status
- `_latest_lessons_tail(max_lines, max_chars)` — tail `lessons.md`
- `_fallback_draft_from_kb(query)` — если остаётся pure, иначе оставить

**Совместимость:**
- В `lra/pipeline.py` оставить `from .context_builders import _build_status_context, ...`
  (re-export) чтобы тесты и `agent.py` не сломались.
- Монкипатчи `monkeypatch.setattr(pipeline, "REJECTED_PATH", ...)` работают через
  re-export `REJECTED_PATH = REJECTED_PATH` в pipeline — проверить.

**Verify (refactor-verify 4-check):**
1. `ruff check lra tests agent.py` → `All checks passed!`
2. `pytest -q` → 211 passed без изменений
3. `git diff --stat` — только pipeline.py (вычитания) и новый context_builders.py
4. Нет изменений поведения: сигнатуры, возвращаемые значения, side-effects идентичны.

**Коммит:** `refactor(pipeline): extract context builders to lra/context_builders.py`

---

## Этап 2: split `lra/tools.py` (1070 LOC) → `lra/tools/` package ✅ DONE

**Цель:** закрыть debt из `AGENTS.md` (trade-off #1: "tools.py split отложен из-за
14 monkeypatch"). Задача решаема через re-export shim.

**Структура:**
```
lra/tools/
  __init__.py          # re-export ВСЕХ классов + verify_ids_against_kb
  search.py            # HfPapers, ArxivSearch, GithubSearch
  notes.py             # ReadNotes, ReadNotesFocused, AppendNotes, CompactNotes
  draft.py             # ReadDraft, WriteDraft, AppendDraft
  plan.py              # ReadPlan, WritePlan, PlanAddTask, PlanCloseTask, PlanSplitTask
  synthesis.py         # ReadSynthesis, WriteSynthesis
  kb_tools.py          # KbAdd, KbSearch (имя _tools чтобы не коллидировать с lra.kb)
  memory_tools.py      # ReadLessons, AppendLessons, ReadQueryLog
  _verifier.py         # verify_ids_against_kb + pre-append helpers
```

**Совместимость (критично):**
- `lra.tools.HfPapers`, `lra.tools.verify_ids_against_kb` — все 14 монкипатчей продолжают
  работать через re-export в `__init__.py`.
- `@register_tool` декоратор ссылается на `lra.tools` — нужно проверить, что импорт
  submodules при `import lra.tools` регистрирует все tools (делается в `__init__.py`).

**Verify:**
1. `ruff check` → All checks passed
2. `pytest -q` → 211 passed
3. `grep -r "lra.tools\." tests/` — все старые пути работают
4. `python -c "from lra.tools import HfPapers, AppendNotes, verify_ids_against_kb"` — OK

**Коммит:** `refactor(tools): split monolithic tools.py into lra/tools/ package`

---

## Этап 3: GitHub Actions CI ✅ DONE

**Цель:** принудить дисциплину "ruff + pytest перед коммитом" для всех future PR.

**Файл:** `.github/workflows/ci.yml`

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
          python-version: '3.12'  # 3.14 пока не все wheels собраны
          cache: pip
      - name: Install deps (без MLX — он только macOS+apple silicon)
        run: |
          pip install -r requirements.txt || true
          pip install ruff pytest qwen-agent
      - name: Ruff
        run: ruff check lra tests agent.py
      - name: Pytest
        run: python -m pytest -q
```

**Важно:** MLX — macOS-only dependency. Нужно либо исключить `lra/llm.py` из
теста импорта, либо использовать `pip install -r requirements.txt || true`
и полагаться на то, что все тесты пишут `@pytest.fixture`-заглушки.
**Проверить:** все 211 тестов импортируют `lra.*` без реального MLX (они уже так работают).

**Verify:**
1. Файл валидный YAML — можно проверить через `python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"`.
2. Коммит + push → смотрим зелёный бейдж.

**Коммит:** `ci: add GitHub Actions workflow (ruff + pytest on push/PR)`

---

## Execution order

1. Этап 1 → commit → Этап 2 → commit → Этап 3 → commit → done.
2. Между этапами обязательно `ruff check && pytest -q` — ни один коммит не уходит красным.
3. Не пушить в origin без явной команды пользователя.
