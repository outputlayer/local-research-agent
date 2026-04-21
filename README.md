# local-research-agent

Локальный научный research-агент на Apple Silicon: **MLX + Qwen3.5-9B-4bit + Qwen-Agent**. Без облаков, без API-ключей, единственный внешний источник — Hugging Face Papers через `hf` CLI.

## Что делает

Один режим — кроличья нора по научной теме. За один запуск проходит 7 фаз:

```
explorer ↔ replanner  (×depth, адаптивный план)
          ↓
     compressor  (если notes > 8000 симв)
          ↓
     synthesizer  (5 типов инсайтов: мосты / противоречия / пробелы / экстраполяции / testable)
          ↓
     writer ↔ critic  (×rounds, с проверкой конвергенции)
          ↓
     validator  (arXiv-id через `hf papers info` + keyword overlap draft↔notes)
```

На выходе: `research/draft.md` с Novel Insights и нумерованными источниками, а в `research/archive/<timestamp>_<slug>/` — снапшот всех артефактов прогона.

## Ключевые фичи

- **Модульная архитектура**: пакет `lra/` — config, utils, memory, llm, tools, prompts, validator, pipeline. CLI `agent.py` тонкий
- **Покрытие тестами**: 35 pytest-тестов (utils / memory / validator / smoke), запускаются без MLX за ~1с
- **Одна загрузка модели** через глобальный кэш `_MLX_CACHE` — 6 ролей/агентов делят веса
- **Кросс-сессионная Reflexion-память**: `lessons.md` и `querylog.md` не стираются между запусками
- **Querylog enforcement**: `hf_papers` возвращает отказ на точный дубль запроса → модель вынуждена переформулировать
- **Info-gain early stop**: если 2 итерации подряд дают <2 новых arXiv-id, ранний выход
- **Адаптивный replanner**: после каждой итерации explorer'а переписывает `plan.md` со структурой `[FOCUS] / Digest / Direction check / [TODO] / [DONE]`
- **Critic convergence**: цикл writer↔critic выходит если жаккард-сходство соседних критик >70%
- **Валидатор цитат**: проверяет существование arXiv-id и семантическую связь текста вокруг id в draft'е с фактами в notes
- **Sandbox для `run_python`**: подпроцесс без сети, RLIMIT_CPU=5s, AS=512MB, запись только в `/tmp/`

## Tools (15)

`hf_papers`, `run_python`, `read/write/append_draft`, `read/append_notes`, `compact_notes`, `read/write_plan`, `read/write_synthesis`, `read/append_lessons`, `read_querylog`.

## Установка

Требуется Apple Silicon, Python ≥3.10, [`hf` CLI](https://huggingface.co/docs/huggingface_hub/guides/cli):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# модель подтянется при первом запуске (~5 GB)
python agent.py
```

Первый запуск скачает `mlx-community/Qwen3.5-9B-MLX-4bit` (в HF-кэш, сам репо ничего не хранит).

## Команды

```
<тема>     — запустить research_loop по теме
/clean     — очистить рабочую папку (lessons/querylog глобальные, НЕ трогаются)
/forget    — полный сброс памяти (включая глобальные lessons/querylog)
/exit      — выход
```

По умолчанию: `depth=6` итераций, `critic_rounds=2`. Меняется в `main()`.

## Структура

```
agent.py              — тонкий CLI (входная точка)
chat_config.json      — модель, температура, max_tokens, max_history
requirements.txt      — mlx-lm, qwen-agent, json5, pytest
lra/                  — пакет с логикой
  config.py           — конфиг + пути артефактов
  utils.py            — чистые функции (parse_args, keyword_set, jaccard, count_arxiv_ids)
  memory.py           — Reflexion-память (seen_queries, archive, reset)
  llm.py              — MLX-бэкенд для qwen-agent + кэш весов
  tools.py            — 15 tools (@register_tool)
  prompts.py          — промпты 6 ролей
  validator.py        — проверка arXiv-id и keyword-overlap
  pipeline.py         — research_loop, build_bot, фазы
tests/
  test_utils.py       — 14 тестов
  test_memory.py      — 8 тестов (через tmp_path)
  test_validator.py   — 5 тестов (с инжекцией текста)
  test_smoke.py       — 5 smoke-тестов (не грузит MLX)
research/             — рабочая папка (gitignored)
  draft.md            — финальный отчёт (stdout тоже)
  notes.md            — накопленные факты с [arxiv-id]
  plan.md             — живой план с [FOCUS]/[TODO]/[DONE]
  synthesis.md        — пять тегов инсайтов
  lessons.md          ★ Reflexion-память между сессиями
  querylog.md         ★ уже выполненные hf_papers запросы
  archive/            — снапшоты прошлых прогонов
```

★ — глобальные, переживают `/clean`.

## Тесты

```bash
pytest tests/ -v   # 35 тестов, ~1 секунда, без MLX
```

## Что НЕ включено

Намеренно отброшено как overkill для 9B модели:

- Tree-of-Thoughts (ветвление × 3-5 → быстрые галлюцинации)
- Multi-agent negotiation (у нас и так 6 специализированных ролей)
- RLHF / continual learning (нет датасета оценок)
- HTTP-сервер для MLX (своя обёртка `MlxLLM(BaseFnCallModel)` — без лишнего hop'а)

## Лицензия

MIT.
