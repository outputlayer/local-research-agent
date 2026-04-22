# AGENTS.md

Контракт для AI-агентов (GitHub Copilot / Claude Code / любой qwen-agent host), которые
будут работать в этом репо. Читать **в начале каждой сессии**. Дополняет `context.md`
(структура кода) и `README.md` (пользовательский интерфейс).

## Что это за проект (одна фраза)

Локальный research-агент на MLX + Qwen-Agent: одна команда → многостадийный pipeline
(explorer → replanner → synthesizer → writer ↔ critic → validator) → отчёт
`research/draft.md` с верифицированными arxiv-цитатами.

## Инварианты — НИКОГДА не нарушай

| # | Запрет | Причина |
|---|---|---|
| 1 | Не удалять `research/lessons.md` и `research/querylog.md` при `/clean` | Reflexion-память кросс-сессионная — её стирает только `/forget` |
| 2 | Не писать в `AppendNotes` arxiv-id, которого НЕТ в `kb.jsonl` (pre-append verifier)| Explorer иначе галлюцинирует ссылки |
| 3 | Не добавлять `shell=True` в `subprocess.run` | Все внешние CLI (`hf`, `gh`) запускаются через argv-list в `lra/cli.py` |
| 4 | Не менять `CRITIC_PROMPT` / `FACT_CRITIC_PROMPT` / `STRUCTURE_CRITIC_PROMPT` текст без одобрения | Пороги качества откалиброваны на текущие версии |
| 5 | Не коммитить файлы в `.skills/`, `.cache/`, `research/archive/` | Всё в `.gitignore`; сторонние скиллы и кэши |
| 6 | Не вызывать `print(...)` в библиотечном коде `lra/*.py` (кроме явных фаз пайплайна) | Логирование через `lra/logger.py`, чтобы не мешало pytest capture |

## Инварианты — ВСЕГДА делай

| # | Требование | Команда верификации |
|---|---|---|
| A | Перед коммитом: `ruff check lra tests agent.py` → `All checks passed!` | `ruff check lra tests agent.py` |
| B | Перед коммитом: `pytest -q` → `162 passed` (или выше) | `source .venv/bin/activate && python -m pytest -q` |
| C | Новый @register_tool → добавить тест в `tests/` | `ls tests/test_*.py` |
| D | Рефактор поведенчески-нейтральный → 4-check по refactor-verify SKILL | см. `.skills/vibesubin/plugins/vibesubin/skills/refactor-verify/SKILL.md` |
| E | Новый CFG-флаг → добавить в `lra/config.py` и документировать в этом файле ниже | `grep CFG.get lra/*.py` |

## Trade-offs — уже решено, не переоткрывать

- **`tools.py` = один файл 689 LOC**: split в `tools/` package отложен — 14 monkeypatch в тестах прицеплены к `lra.tools.*` атрибутам, ROI низкий.
- **`research_loop` = 171 NLOC, CCN=42**: уже разбит на `_run_iteration` / `_finalize_draft` / `_hitl_review`. Дальнейший split — линейная композиция фаз, увеличит прыжки по коду.
- **Pinned deps через `==` exact, без lockfile**: простой CLI без автообновлений. Lockfile = over-engineering для 4 зависимостей.
- **CRITIC_PROMPT сохранён рядом с FACT/STRUCTURE_CRITIC_PROMPT**: legacy-режим `specialized_critics=False`, нужен для совместимости с resume.

## Как работать с этим кодом

### Запуск тестов
```bash
source .venv/bin/activate
python -m pytest -q
```
Все 162 теста — без MLX, ~1 секунда.

### Запуск агента
```bash
source .venv/bin/activate
python agent.py          # интерактивный REPL
# или: python -c "from lra.pipeline import research_loop; research_loop('тема', depth=6, critic_rounds=2)"
```

### Резюм после краша
```bash
python -c "from lra.pipeline import resume_research; resume_research()"
```
Пропустит explorer, продолжит с synthesizer → writer → critic → validator, используя
существующие `notes.md`/`kb.jsonl`.

### Runtime-флаги (в `chat_config.json` под ключом `extra` ИЛИ `CFG['flag'] = True` в REPL)

| Флаг | Default | Что делает |
|---|---|---|
| `notes_strict` | `True` | Pre-append verifier блокирует AppendNotes на неизвестных arxiv-id |
| `strict_domain_gate` | `True` | AppendNotes блокирует paper из смежного домена (< 2 overlap с topic keywords из plan.md). Slow-start: если plan содержит < 3 специфичных kws, gate пропускает. Отклонённые → `research/rejected.jsonl` |
| `specialized_critics` | `True` | Fact-critic + structure-critic вместо combined `CRITIC_PROMPT` |
| `dynamic_initial_plan` | `True` | LLM-bootstrap topic-aware seeds перед Phase 1; fallback на статический 5-seed plan при любой ошибке парсинга/LLM |
| `hitl` | `False` | Human-in-the-loop пауза после validator'а (REPL: `/hitl on`) |

## При любом изменении — прогон

```bash
source .venv/bin/activate && ruff check lra tests agent.py && python -m pytest -q
```

Если один из шагов красный — НЕ коммить. Сначала починить.

## Vibesubin skills (локальные, в `.skills/vibesubin/plugins/vibesubin/skills/`)

Для сложных задач читать SKILL.md СООТВЕТСТВУЮЩЕГО скилла ДО работы:

| Триггер | Скилл |
|---|---|
| refactor / rename / split / safe delete | `refactor-verify` — обязательна 4-check верификация |
| «проверь безопасность / секреты / уязвимости» | `audit-security` |
| «найди мёртвый код / что удалить?» | `fight-repo-rot` |
| написать README / commit / PR / AGENTS.md | `write-for-ai` |

См. `/memories/repo/vibesubin-skills.md` (Copilot memory) для полной таблицы.

## Pointers

- `context.md` — карта пакета, что где лежит, ключевые классы и функции
- `README.md` — пользовательский интерфейс: установка, команды, фичи
- `.skills/vibesubin/` — локальные dev-скиллы (в `.gitignore`, не коммитить)
- `research/` — артефакты прогонов (draft.md, notes.md, plan.md, synthesis.md, kb.jsonl, metrics.json)
- `research/archive/` — снапшоты завершённых прогонов (timestamped)
