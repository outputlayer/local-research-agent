#!/usr/bin/env python3
"""
Локальный научный research-агент на MLX + Qwen-Agent.

Единственный режим — `/research <тема>`: кроличья нора по hf_papers →
синтез новых инсайтов → черновик → критик → валидация arXiv-id.

Запуск:
    python agent.py
"""

from __future__ import annotations
import json, re, subprocess, sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import json5


def _parse_args(params) -> dict:
    """Толерантный парсер tool-аргументов: чинит литеральные переносы строк и
    гарантирует dict на выходе (LLM иногда присылает просто строку)."""
    if isinstance(params, dict):
        return params
    if not isinstance(params, str):
        return {"content": str(params)}

    def _wrap(obj):
        if isinstance(obj, dict):
            return obj
        # LLM прислал просто строку/число/список вместо объекта — оборачиваем в content
        return {"content": obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)}

    try:
        return _wrap(json5.loads(params))
    except Exception:
        pass
    # Экранируем переводы строк/табы внутри строковых значений и пробуем снова
    try:
        fixed = re.sub(
            r'"((?:[^"\\]|\\.)*)"',
            lambda m: '"' + m.group(1).replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t') + '"',
            params,
            flags=re.DOTALL,
        )
        return _wrap(json5.loads(fixed))
    except Exception:
        pass
    # Последняя попытка: достать поле по regex (non-greedy, до следующей ключа или конца объекта)
    for key in ("content", "code", "query", "url"):
        m = re.search(rf'"{key}"\s*:\s*"(.*?)"\s*(?:,\s*"|\}})', params, re.DOTALL)
        if m:
            return {key: m.group(1).replace('\\n', '\n').replace('\\"', '"')}
    # Совсем ничего — отдаём как content чтобы инструменты не падали
    return {"content": params}
from qwen_agent.agents import Assistant
from qwen_agent.llm.base import register_llm
from qwen_agent.llm.function_calling import BaseFnCallModel
from qwen_agent.llm.schema import ASSISTANT, Message
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.utils.output_beautify import typewriter_print

CFG = json.loads((Path(__file__).parent / "chat_config.json").read_text())

# Глобальный кеш весов — чтобы два Assistant (main + research) не грузили модель дважды
_MLX_CACHE: Dict[str, tuple] = {}


def _get_mlx(model_name: str):
    if model_name not in _MLX_CACHE:
        from mlx_lm import load
        _MLX_CACHE[model_name] = load(model_name)
    return _MLX_CACHE[model_name]


# ─── Кастомный LLM поверх mlx_lm (без HTTP-сервера) ────────────────

@register_llm("mlx")
class MlxLLM(BaseFnCallModel):
    """Локальный MLX-бэкенд для Qwen-Agent."""

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        self.mlx_model, self.tokenizer = _get_mlx(cfg["model"])

    def _build_prompt(self, messages: List[Message]) -> str:
        plain = [m.model_dump() if hasattr(m, "model_dump") else dict(m) for m in messages]
        # Qwen3.5 thinking отключаем: tool-calls идут через nous-шаблон Qwen-Agent
        return self.tokenizer.apply_chat_template(
            plain, add_generation_prompt=True, tokenize=False,
            enable_thinking=False,
        )

    def _mlx_generate(self, prompt: str, cfg: dict):
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler, make_logits_processors
        sampler = make_sampler(
            temp=cfg.get("temperature", CFG["temperature"]),
            top_p=cfg.get("top_p", CFG["top_p"]),
            top_k=cfg.get("top_k", CFG["top_k"]),
        )
        proc = make_logits_processors(repetition_penalty=CFG["repetition_penalty"])
        return stream_generate(
            self.mlx_model, self.tokenizer, prompt=prompt,
            max_tokens=cfg.get("max_tokens", CFG["max_tokens"]),
            sampler=sampler, logits_processors=proc,
        )

    def _chat_stream(self, messages, delta_stream, generate_cfg) -> Iterator[List[Message]]:
        prompt = self._build_prompt(messages)
        acc = ""
        for resp in self._mlx_generate(prompt, generate_cfg):
            if delta_stream:
                yield [Message(ASSISTANT, resp.text)]
            else:
                acc += resp.text
                yield [Message(ASSISTANT, acc)]

    def _chat_no_stream(self, messages, generate_cfg) -> List[Message]:
        prompt = self._build_prompt(messages)
        text = "".join(r.text for r in self._mlx_generate(prompt, generate_cfg))
        return [Message(ASSISTANT, text)]


# ─── Tools ─────────────────────────────────────────────────────────

_SANDBOX_PRELUDE = r"""
import resource, sys, builtins
# CPU 5s, adress space 512MB, no fork, no new files >10MB
resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
try: resource.setrlimit(resource.RLIMIT_AS, (512*1024*1024, 512*1024*1024))
except Exception: pass
try: resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
except Exception: pass
# блок сети через monkey-patch socket
import socket
def _blocked(*a, **k): raise OSError('сеть запрещена в песочнице')
socket.socket = _blocked  # type: ignore
socket.create_connection = _blocked  # type: ignore
# снять доступ к open для записи вне /tmp
_orig_open = builtins.open
def _safe_open(f, mode='r', *a, **k):
    if any(m in mode for m in ('w','a','x','+')) and not str(f).startswith('/tmp/'):
        raise PermissionError('запись разрешена только в /tmp/')
    return _orig_open(f, mode, *a, **k)
builtins.open = _safe_open
CODE = sys.stdin.read()
exec(compile(CODE, '<tool>', 'exec'), {'__name__':'__main__'})
"""


@register_tool("hf_papers")
class HfPapers(BaseTool):
    description = ("Поиск научных статей на Hugging Face Papers через локальный `hf` CLI. "
                   "Возвращает id (arxiv), заголовок, авторов, дату и abstract.")
    parameters = [
        {"name": "query", "type": "string", "description": "Запрос", "required": True},
        {"name": "limit", "type": "integer", "description": "Сколько результатов (1-10)", "required": False},
    ]

    def call(self, params: str, **kwargs) -> str:
        p = _parse_args(params)
        query = p["query"]
        limit = max(1, min(int(p.get("limit", 5)), 10))
        # Enforcement: точный дубликат запроса = отказ (экономим токены и заставляем переформулировать)
        if _normalize_query(query) in _seen_queries():
            return (f"ОТКАЗ: запрос '{query}' уже выполнялся в этой сессии. "
                    "Переформулируй (другие ключевые слова, автор, год, техника) или читай read_notes.")
        _log_query(query)
        try:
            r = subprocess.run(
                ["hf", "papers", "search", query, "--limit", str(limit * 2), "--format", "json"],
                capture_output=True, text=True, timeout=30,
            )
        except FileNotFoundError:
            return "ошибка: `hf` CLI не найден в PATH"
        except subprocess.TimeoutExpired:
            return "таймаут поиска"
        if r.returncode != 0:
            return f"ошибка: {r.stderr.strip()[:500]}"
        try:
            data = json.loads(r.stdout)
        except Exception:
            return f"не удалось распарсить JSON: {r.stdout[:300]}"
        if not data:
            return f"нет результатов: {query}"
        # сортируем от свежих к старым и режем до limit
        data.sort(key=lambda x: x.get("published_at", ""), reverse=True)
        data = data[:limit]
        lines = []
        for paper in data:
            authors = ", ".join(a["name"] for a in paper.get("authors", [])[:4])
            if len(paper.get("authors", [])) > 4:
                authors += " и др."
            title = " ".join(paper.get("title", "").split())
            summary = " ".join(paper.get("summary", "").split())[:800]
            date = paper.get("published_at", "")[:10]
            pid = paper.get("id", "")
            lines.append(f"[{pid}] {title}\n  {authors} · {date}\n  https://hf.co/papers/{pid}\n  {summary}")
        return "\n\n".join(lines)


@register_tool("run_python")
class RunPython(BaseTool):
    description = ("Выполняет Python-код в изолированной подпроцессной песочнице "
                   "(без сети, 5с CPU, 512MB RAM). Возвращает stdout+stderr.")
    parameters = [{"name": "code", "type": "string", "description": "Код", "required": True}]

    def call(self, params: str, **kwargs) -> str:
        code = _parse_args(params)["code"]
        try:
            r = subprocess.run(
                [sys.executable, "-I", "-c", _SANDBOX_PRELUDE],
                input=code, capture_output=True, text=True, timeout=10,
            )
        except subprocess.TimeoutExpired:
            return "таймаут (>10с)"
        out = (r.stdout + (r.stderr and f"\n[stderr]\n{r.stderr}" or "")).strip()
        return (out or "(вывода нет)")[:5000]


# ─── Research: explorer (rabbit-hole) → writer → critic ───────────────

RESEARCH_DIR = Path(__file__).parent / "research"
ARCHIVE_DIR = RESEARCH_DIR / "archive"
DRAFT_PATH = RESEARCH_DIR / "draft.md"
NOTES_PATH = RESEARCH_DIR / "notes.md"
PLAN_PATH = RESEARCH_DIR / "plan.md"
SYNTHESIS_PATH = RESEARCH_DIR / "synthesis.md"
# Reflexion-память ГЛОБАЛЬНА — живёт между сессиями, не стирается при новом запросе
LESSONS_PATH = RESEARCH_DIR / "lessons.md"
QUERYLOG_PATH = RESEARCH_DIR / "querylog.md"


def _ensure_dir():
    RESEARCH_DIR.mkdir(exist_ok=True)
    ARCHIVE_DIR.mkdir(exist_ok=True)


def _normalize_query(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


def _seen_queries() -> set[str]:
    if not QUERYLOG_PATH.exists():
        return set()
    return {_normalize_query(ln.lstrip("- ").strip())
            for ln in QUERYLOG_PATH.read_text().splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")}


def _log_query(query: str):
    """Регистрируем выполненные hf_papers запросы (Reflexion episodic memory)."""
    _ensure_dir()
    with QUERYLOG_PATH.open("a") as f:
        f.write(f"- {query}\n")


@register_tool("compact_notes")
class CompactNotes(BaseTool):
    description = "Перезаписывает research/notes.md сжатым содержанием."
    parameters = [{"name": "content", "type": "string", "description": "Сжатый markdown", "required": True}]

    def call(self, params: str, **kwargs) -> str:
        _ensure_dir()
        content = _parse_args(params)["content"]
        old = NOTES_PATH.stat().st_size if NOTES_PATH.exists() else 0
        NOTES_PATH.write_text(content)
        return f"notes.md: {old} → {len(content)} симв (сжато в {max(1, old // max(1, len(content)))}x)"


@register_tool("write_draft")
class WriteDraft(BaseTool):
    description = "Перезаписывает черновик research/draft.md с нуля. Используй для старта или полного переписывания."
    parameters = [{"name": "content", "type": "string", "description": "Markdown", "required": True}]

    def call(self, params: str, **kwargs) -> str:
        _ensure_dir()
        content = _parse_args(params)["content"]
        DRAFT_PATH.write_text(content)
        return f"draft.md сохранён ({len(content)} симв)"


@register_tool("append_draft")
class AppendDraft(BaseTool):
    description = ("Добавляет секцию в конец research/draft.md. "
                   "Предпочтительно write_draft+много append_draft, чем один огромный write_draft.")
    parameters = [{"name": "content", "type": "string", "description": "Markdown-секция", "required": True}]

    def call(self, params: str, **kwargs) -> str:
        _ensure_dir()
        content = _parse_args(params)["content"]
        with DRAFT_PATH.open("a") as f:
            f.write("\n\n" + content.strip() + "\n")
        return f"draft.md +{len(content)} симв (всего {DRAFT_PATH.stat().st_size})"


@register_tool("read_draft")
class ReadDraft(BaseTool):
    description = "Читает research/draft.md."
    parameters = []

    def call(self, params: str, **kwargs) -> str:
        return DRAFT_PATH.read_text() if DRAFT_PATH.exists() else "(пусто)"


@register_tool("append_notes")
class AppendNotes(BaseTool):
    description = ("Добавляет новый фрагмент знаний в research/notes.md "
                   "(структурированная запись: подтема, arXiv id, ключевые факты).")
    parameters = [{"name": "content", "type": "string", "description": "Markdown-фрагмент", "required": True}]

    def call(self, params: str, **kwargs) -> str:
        _ensure_dir()
        content = _parse_args(params)["content"]
        with NOTES_PATH.open("a") as f:
            f.write("\n\n" + content.strip() + "\n")
        return f"notes.md +{len(content)} симв (всего {NOTES_PATH.stat().st_size} симв)"


@register_tool("read_notes")
class ReadNotes(BaseTool):
    description = "Читает накопленные заметки research/notes.md."
    parameters = []

    def call(self, params: str, **kwargs) -> str:
        if not NOTES_PATH.exists():
            return "(заметок нет)"
        text = NOTES_PATH.read_text()
        # Qwen3 native 32K контекст — можем позволить себе большой хвост
        return text[-20000:] if len(text) > 20000 else text


@register_tool("write_plan")
class WritePlan(BaseTool):
    description = ("Перезаписывает план ресёрча в research/plan.md. "
                   "Формат: список подтем с маркерами [TODO]/[DONE]. "
                   "Добавляй новые подтемы, выявленные из заметок.")
    parameters = [{"name": "content", "type": "string", "description": "Markdown план", "required": True}]

    def call(self, params: str, **kwargs) -> str:
        _ensure_dir()
        content = _parse_args(params)["content"]
        PLAN_PATH.write_text(content)
        todos = content.count("[TODO]")
        done = content.count("[DONE]")
        return f"plan.md: {todos} TODO, {done} DONE"


@register_tool("read_plan")
class ReadPlan(BaseTool):
    description = "Читает текущий план research/plan.md."
    parameters = []

    def call(self, params: str, **kwargs) -> str:
        return PLAN_PATH.read_text() if PLAN_PATH.exists() else "(плана нет)"


@register_tool("write_synthesis")
class WriteSynthesis(BaseTool):
    description = ("Записывает research/synthesis.md — НОВЫЕ мысли, которых нет в исходных абстрактах: "
                   "мосты между paper'ами, противоречия, пробелы, экстраполяции, проверяемые гипотезы.")
    parameters = [{"name": "content", "type": "string", "description": "Markdown", "required": True}]

    def call(self, params: str, **kwargs) -> str:
        _ensure_dir()
        content = _parse_args(params)["content"]
        SYNTHESIS_PATH.write_text(content)
        tags = sum(content.count(t) for t in ("[BRIDGE]", "[CONTRADICTION]", "[GAP]", "[EXTRAPOLATION]", "[TESTABLE]"))
        return f"synthesis.md ({len(content)} симв, {tags} инсайтов)"


@register_tool("read_synthesis")
class ReadSynthesis(BaseTool):
    description = "Читает research/synthesis.md — новые инсайты после анализа notes."
    parameters = []

    def call(self, params: str, **kwargs) -> str:
        return SYNTHESIS_PATH.read_text() if SYNTHESIS_PATH.exists() else "(синтеза нет)"


@register_tool("append_lessons")
class AppendLessons(BaseTool):
    description = ("Записывает урок итерации в research/lessons.md (Reflexion-память). "
                   "Формат: что сработало, что нет, чего избегать в следующих запросах.")
    parameters = [{"name": "content", "type": "string", "description": "Markdown", "required": True}]

    def call(self, params: str, **kwargs) -> str:
        _ensure_dir()
        content = _parse_args(params)["content"]
        with LESSONS_PATH.open("a") as f:
            f.write("\n" + content.strip() + "\n")
        return f"lessons.md +{len(content)} симв"


@register_tool("read_lessons")
class ReadLessons(BaseTool):
    description = "Читает lessons.md — уроки прошлых итераций, что не повторять."
    parameters = []

    def call(self, params: str, **kwargs) -> str:
        return LESSONS_PATH.read_text() if LESSONS_PATH.exists() else "(уроков ещё нет)"


@register_tool("read_querylog")
class ReadQueryLog(BaseTool):
    description = "Читает список уже выполненных hf_papers запросов — НЕ повторяй их."
    parameters = []

    def call(self, params: str, **kwargs) -> str:
        if not QUERYLOG_PATH.exists():
            return "(запросов ещё не было)"
        text = QUERYLOG_PATH.read_text()
        # оставляем последние 30 уникальных
        lines = []
        seen = set()
        for ln in reversed(text.splitlines()):
            if ln.strip() and ln not in seen:
                seen.add(ln)
                lines.append(ln)
            if len(lines) >= 30:
                break
        return "\n".join(reversed(lines))


EXPLORER_PROMPT = (
    "Ты — научный исследователь, идущий в глубину («кроличья нора»). Источник только hf_papers.\n"
    "ПРАВИЛА:\n"
    "• Сначала read_plan — [FOCUS] задаёт цель. read_notes — видишь собранные id. "
    "read_querylog — видишь какие запросы УЖЕ делал (НЕ повторяй их дословно, переформулируй). "
    "read_lessons — уроки прошлых итераций, чего избегать.\n"
    "• Вызови hf_papers по [FOCUS] (limit=5). Запрос должен отличаться от всех в querylog.\n"
    "• Из абстрактов извлеки 2-4 факта, отметь новые термины/авторов.\n"
    "• MULTI-HOP: встретил важного автора — второй hf_papers по его имени (limit=3).\n"
    "• ОБЯЗАТЕЛЬНО append_notes минимум ОДИН раз. Формат: подтема, [arxiv-id] → факт, термины, авторы, даты.\n"
    "• В конце ОБЯЗАТЕЛЬНО append_lessons: одна строка вида «[iter N] сработало: X; НЕ сработало: Y; "
    "следующий шаг: Z». Это твоя память для будущих итераций.\n"
    "• НЕ трогай plan.md — его перепишет replanner.\n"
    "• Ответь: 'iteration done'."
)

REPLANNER_PROMPT = (
    "Ты — РЕПЛАНЕР научного исследования. Твоя роль — корректировать направление после каждой итерации.\n"
    "Прочитай read_notes (всё что собрано) и read_plan (текущий план).\n"
    "Затем перепиши plan.md через write_plan СТРОГО в следующем формате:\n\n"
    "```\n"
    "# Plan: <тема>\n\n"
    "[FOCUS] <одна конкретная цель для СЛЕДУЮЩЕЙ итерации explorer'а, 1 строка>\n\n"
    "## Digest (что мы уже знаем, макс 800 символов)\n"
    "- <ключевой факт 1 с [arxiv-id]>\n"
    "- <ключевой факт 2 с [arxiv-id]>\n"
    "- <...4-8 пунктов>\n\n"
    "## Direction check\n"
    "<2-3 предложения: соответствует ли текущий вектор исходной теме? Если находки уводят в "
    "более перспективное русло — явно скажи «корректирую вектор: X → Y» и обоснуй.>\n\n"
    "## [TODO]\n"
    "- <подтема 1, приоритет высший>\n"
    "- <подтема 2>\n"
    "- ... (3-6 штук, свежие ветки ПЕРВЫМИ)\n\n"
    "## [DONE]\n"
    "- <пройденные подтемы>\n"
    "```\n\n"
    "ПРАВИЛА ВЫБОРА [FOCUS]:\n"
    "• Бери из [TODO] ту подтему, где у тебя МЕНЬШЕ всего данных в notes — максимизируй прирост знаний.\n"
    "• Если в notes появился неожиданный термин/автор, которого нет в плане — заведи для него новый [TODO] "
    "и поставь [FOCUS] на него (pivot).\n"
    "• Если все ключевые ветки покрыты (3+ id в каждой) — напиши в [FOCUS]: `PLAN_COMPLETE` и очисти [TODO].\n"
    "• ЗАПРЕЩЕНО просто копировать предыдущий план — каждый вызов должен внести изменение.\n"
    "Ответь одной строкой: 'replan done'."
)


SYNTHESIZER_PROMPT = (
    "Ты — СИНТЕЗАТОР-изобретатель. Твоя задача: произвести НОВОЕ знание, которого НЕТ в исходных абстрактах.\n"
    "Прочитай read_notes и read_plan. Затем вызови write_synthesis с markdown, содержащим РОВНО ПЯТЬ блоков:\n\n"
    "## [BRIDGE]\n"
    "Возьми два paper'а из РАЗНЫХ подтем ([id_A] и [id_B]) и предложи применение техники из A к задаче из B. "
    "Формула: «Метод X из [id_A] может решить проблему Y из [id_B], потому что ...».\n\n"
    "## [CONTRADICTION]\n"
    "Найди два paper'а, чьи выводы конфликтуют (даже косвенно). Сформулируй противоречие одним предложением "
    "и предложи экспериментальное разрешение: «Чтобы выбрать между [id_A] и [id_B], нужно измерить ...».\n\n"
    "## [GAP]\n"
    "Назови ОДНУ тему/аспект/вопрос, который ни один paper в notes НЕ покрывает, но который логически следует "
    "из собранного. Обоснуй почему это пробел.\n\n"
    "## [EXTRAPOLATION]\n"
    "По датам публикаций и направлению изменений сформулируй предсказание: «Следующим логическим шагом после "
    "[id_последний] будет ... потому что тренд с [id_ранний] → [id_последний] показывает ...».\n\n"
    "## [TESTABLE]\n"
    "Сформулируй ОДНУ численно проверяемую гипотезу. Затем ВЫЗОВИ run_python с кодом, который её проверяет "
    "(например, статистика по датам из notes, или простая численная симуляция). Вставь результат в блок.\n\n"
    "ЖЕЛЕЗНЫЕ ПРАВИЛА:\n"
    "• Каждое утверждение должно цитировать минимум один [arxiv-id] ИЗ notes.md.\n"
    "• Запрещено пересказывать абстракт — только КОМБИНАЦИИ, ПРОТИВОРЕЧИЯ, ПРЕДСКАЗАНИЯ.\n"
    "• Если материала мало для какого-то блока — честно напиши «insufficient evidence», не выдумывай."
)

WRITER_PROMPT = (
    "Ты — научный writer. Прочитай read_plan, read_notes и read_synthesis.\n"
    "ВАЖНО: пиши черновик ИНКРЕМЕНТАЛЬНО:\n"
    "  1) write_draft — с заголовком и Введением (короткое).\n"
    "  2) append_draft — секция 'Тезисы' с 6-10 пунктами [arxiv-id] (одним вызовом).\n"
    "  3) append_draft — 'Взаимосвязи и противоречия'.\n"
    "  4) append_draft — '## Novel Insights' — ОБЯЗАТЕЛЬНАЯ секция, перенеси пять блоков из synthesis.md "
    "с сохранёнными тегами [BRIDGE]/[CONTRADICTION]/[GAP]/[EXTRAPOLATION]/[TESTABLE]. Это то, что отличает "
    "отчёт от пересказа абстрактов — не удаляй и не смягчай.\n"
    "  5) append_draft — 'Открытые вопросы'.\n"
    "  6) append_draft — 'Источники' (нумерованный список).\n"
    "Каждый вызов — не более ~1500 символов в content, иначе JSON обрежется.\n"
    "Используй ТОЛЬКО id из notes.md. При критике — write_draft заново + append_draft."
)

CRITIC_PROMPT = (
    "Ты — строгий научный рецензент. Прочитай read_draft, read_notes и read_synthesis.\n"
    "Для КАЖДОЙ правки укажи точно:\n"
    "  • ЦИТАТУ из черновика (фрагмент строки в кавычках);\n"
    "  • ПРОБЛЕМУ, одну из:\n"
    "      — нет цитаты [arxiv-id];\n"
    "      — id отсутствует в notes;\n"
    "      — утверждение противоречит notes;\n"
    "      — дублирование;\n"
    "      — SMART_PLAGIARISM: утверждение — пересказ одного абстракта без нового угла "
    "(нет моста к другому paper'у, нет противоречия, нет экстраполяции);\n"
    "      — секция Novel Insights отсутствует или потеряла теги [BRIDGE]/[CONTRADICTION]/[GAP]/[EXTRAPOLATION]/[TESTABLE].\n"
    "  • КОНКРЕТНОЕ действие (удалить / переформулировать с [<id>] / добавить мост между [<id>] и [<id>]).\n"
    "Общие реплики вроде 'стоит добавить деталей' ЗАПРЕЩЕНЫ. Максимум 5 правок.\n"
    "Если черновик полностью соответствует notes И содержит все пять тегов Novel Insights — ответь ровно 'APPROVED'."
)

COMPRESSOR_PROMPT = (
    "Ты — компрессор заметок. Прочитай read_notes. Сократи содержание сохранив:\n"
    "• все [arxiv-id] с одной строкой-фактом на каждый;\n"
    "• секции по подтемам;\n"
    "• новые термины.\n"
    "Вызови инструмент compact_notes с итоговым сжатым markdown. Цель: уместиться в ~3000 символов."
)


def _validate_draft_ids() -> tuple[int, list[str], list[str]]:
    """Проверяет arXiv-id в черновике и качество их цитирования.

    Возвращает: (валидных, несуществующих, подозрительных_цитат).
    Подозрительная цитата: id из notes, но контекст вокруг него в draft'е не пересекается
    с фактами из notes для этого id (≥3 общих ключевых слов).
    """
    if not DRAFT_PATH.exists():
        return 0, [], []
    draft_text = DRAFT_PATH.read_text()
    ids = set(re.findall(r"\b(\d{4}\.\d{4,5})\b", draft_text))
    if not ids:
        return 0, [], []
    notes_text = NOTES_PATH.read_text() if NOTES_PATH.exists() else ""
    notes_ids = set(re.findall(r"\b(\d{4}\.\d{4,5})\b", notes_text))
    invalid, suspicious = [], []
    valid = 0

    def _keywords(s: str) -> set[str]:
        return {w.lower() for w in re.findall(r"[A-Za-zА-Яа-я][A-Za-zА-Яа-я\-]{4,}", s)}

    for pid in ids:
        if pid not in notes_ids:
            # id не в notes — сверяем через hf CLI
            try:
                r = subprocess.run(["hf", "papers", "info", pid],
                                   capture_output=True, text=True, timeout=15)
                if r.returncode != 0 or "not found" in (r.stdout + r.stderr).lower():
                    invalid.append(pid); continue
            except Exception:
                invalid.append(pid); continue
            valid += 1
            continue
        # id есть в notes — проверяем семантическое совпадение цитаты
        # Берём контекст в draft: 120 символов вокруг каждого вхождения id
        draft_ctx = " ".join(re.findall(rf".{{0,120}}{re.escape(pid)}.{{0,120}}", draft_text, re.DOTALL))
        # Контекст в notes: строки содержащие pid + соседние
        notes_lines = notes_text.splitlines()
        notes_ctx_parts = []
        for i, ln in enumerate(notes_lines):
            if pid in ln:
                notes_ctx_parts.append(" ".join(notes_lines[max(0, i-1):i+3]))
        notes_ctx = " ".join(notes_ctx_parts)
        draft_kw = _keywords(draft_ctx) - {"paper", "paperов", "статья", "работа", "авторы", "model", "method"}
        notes_kw = _keywords(notes_ctx)
        overlap = draft_kw & notes_kw
        if len(overlap) < 3:
            suspicious.append(f"{pid} (overlap={len(overlap)})")
        valid += 1
    return valid, invalid, suspicious


def _run_agent(bot: Assistant, messages: list, icon: str) -> list:
    print(icon + " ", end="", flush=True)
    plain, resp = "", []
    for resp in bot.run(messages=messages):
        plain = typewriter_print(resp, plain)
    print()
    return resp


def _archive_previous(query_hint: str = ""):
    """Сохраняет предыдущий draft+notes+plan+synthesis в archive/<timestamp>_<slug>/."""
    if not DRAFT_PATH.exists() and not NOTES_PATH.exists():
        return None
    from datetime import datetime
    slug = re.sub(r"[^a-zA-Z0-9а-яА-Я]+", "-", query_hint)[:40].strip("-") or "run"
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    dest = ARCHIVE_DIR / f"{stamp}_{slug}"
    dest.mkdir(parents=True, exist_ok=True)
    for p in (DRAFT_PATH, NOTES_PATH, PLAN_PATH, SYNTHESIS_PATH):
        if p.exists():
            (dest / p.name).write_text(p.read_text())
    return dest


def _reset_research(query: str):
    """Готовит рабочую папку к новому запуску.
    draft/notes/plan/synthesis — архивируются и очищаются.
    lessons/querylog — СОХРАНЯЮТСЯ (кросс-сессионная Reflexion-память).
    """
    _ensure_dir()
    archived = _archive_previous(query)
    if archived:
        print(f"📦 Прошлый прогон сохранён: {archived.relative_to(RESEARCH_DIR.parent)}")
    for p in (DRAFT_PATH, NOTES_PATH, PLAN_PATH, SYNTHESIS_PATH):
        p.unlink(missing_ok=True)
    NOTES_PATH.write_text(f"# Notes: {query}\n")
    # lessons и querylog инициализируем только если их нет совсем
    if not LESSONS_PATH.exists():
        LESSONS_PATH.write_text("# Lessons (global, across sessions)\n")
    if not QUERYLOG_PATH.exists():
        QUERYLOG_PATH.write_text("# Query log (global, across sessions)\n")
    # Маркер новой сессии в обоих файлах
    from datetime import datetime
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    with LESSONS_PATH.open("a") as f:
        f.write(f"\n## Session {stamp}: {query}\n")
    with QUERYLOG_PATH.open("a") as f:
        f.write(f"\n## Session {stamp}: {query}\n")
    # Начальный план с FOCUS = исходная тема. Replanner потом перепишет структуру.
    PLAN_PATH.write_text(
        f"# Plan: {query}\n\n"
        f"[FOCUS] {query} — обзор архитектур и ключевых работ\n\n"
        "## Digest\n(пока пусто — первая итерация)\n\n"
        "## Direction check\n(будет обновлено replanner'ом)\n\n"
        "## [TODO]\n"
        f"- {query}: обзорные статьи\n"
        f"- {query}: ключевые методы\n"
        f"- {query}: ограничения и открытые вопросы\n\n"
        "## [DONE]\n"
    )


def _count_arxiv_ids(text: str) -> set[str]:
    return set(re.findall(r"\b(\d{4}\.\d{4,5})\b", text))


def research_loop(query: str, depth: int = 6, critic_rounds: int = 2):
    """
    Фаза 1 (explorer + replanner): до `depth` итераций кроличьей норы с адаптивным планом.
      Каждая итерация: explorer читает plan.md (FOCUS+Digest), идёт в [FOCUS],
      добавляет в notes.md. Затем replanner переписывает plan.md: обновляет Digest,
      корректирует вектор, выбирает новый [FOCUS].
    Фаза 1.5 (compressor): сжимает notes если > 8000 симв.
    Фаза 2.0 (synthesizer): мосты, противоречия, пробелы, экстраполяции, testable.
    Фаза 2 (writer): финальный черновик из notes+synthesis.
    Фаза 3 (critic): критика + правки в цикле.
    Фаза 4 (validator): проверка arXiv-id через hf papers info.
    """
    _reset_research(query)
    print(f"📁 Рабочая папка: {RESEARCH_DIR}\n")

    # Explorer — собирает факты + Reflexion-память (lessons + querylog)
    explorer = _build_bot(EXPLORER_PROMPT,
                          ["hf_papers", "read_plan", "read_notes", "append_notes",
                           "read_lessons", "append_lessons", "read_querylog"],
                          max_tokens=3072)
    # Replanner — единственный, кто пишет plan.md
    replanner = _build_bot(REPLANNER_PROMPT,
                           ["read_notes", "read_plan", "write_plan"],
                           max_tokens=3072)

    def _current_focus() -> str:
        if not PLAN_PATH.exists():
            return query
        for line in PLAN_PATH.read_text().splitlines():
            if line.strip().startswith("[FOCUS]"):
                return line.strip().replace("[FOCUS]", "").strip(" -—:") or query
        return query

    print(f"🕳️  Фаза 1: кроличья нора (до {depth} итераций, адаптивный план)")
    low_gain_streak = 0
    for i in range(1, depth + 1):
        focus = _current_focus()
        print(f"\n── итерация {i}/{depth} ──  🎯 FOCUS: {focus[:80]}")
        ids_before = _count_arxiv_ids(NOTES_PATH.read_text() if NOTES_PATH.exists() else "")
        before = NOTES_PATH.stat().st_size if NOTES_PATH.exists() else 0
        msg = [{"role": "user",
                "content": f"Исходная тема: {query}\n"
                           f"Текущий [FOCUS] из plan.md: {focus}\n"
                           "Сделай одну итерацию по [FOCUS]. ОБЯЗАТЕЛЬНО append_notes и append_lessons."}]
        _run_agent(explorer, msg, "🔎")
        after = NOTES_PATH.stat().st_size if NOTES_PATH.exists() else 0
        grew = after - before
        ids_after = _count_arxiv_ids(NOTES_PATH.read_text() if NOTES_PATH.exists() else "")
        new_ids = ids_after - ids_before
        print(f"   📝 notes: +{grew} симв ({after} всего)  📊 новых arxiv-id: {len(new_ids)}")
        if grew < 100:
            print("   ⚠️  заметки не росли — retry со строгим требованием")
            retry = [{"role": "user", "content": (
                f"ПРОВАЛ: ты НЕ вызвал append_notes. Сделай hf_papers по '{focus}' (limit=5) "
                "и СРАЗУ после этого append_notes с минимум 3 фактами и [arxiv-id]. "
                "Затем append_lessons одной строкой.")}]
            _run_agent(explorer, retry, "🔁")
            ids_after = _count_arxiv_ids(NOTES_PATH.read_text() if NOTES_PATH.exists() else "")
            new_ids = ids_after - ids_before

        # Info-gain tracking: если 2 итерации подряд дают < 2 новых id — ранний стоп
        if len(new_ids) < 2:
            low_gain_streak += 1
        else:
            low_gain_streak = 0

        # Replanner: корректирует план на основе новых заметок
        print(f"   🧭 replanner обновляет план...")
        plan_before = PLAN_PATH.read_text() if PLAN_PATH.exists() else ""
        _run_agent(replanner,
                   [{"role": "user",
                     "content": f"Исходная тема: {query}. Обнови plan.md: Digest, Direction check, "
                                "новый [FOCUS], пересортируй [TODO]. Используй write_plan."}],
                   "🧭")
        plan_text = PLAN_PATH.read_text() if PLAN_PATH.exists() else ""
        if plan_text == plan_before:
            print("   ⚠️  план не обновился — retry")
            _run_agent(replanner,
                       [{"role": "user", "content": (
                           f"ПРОВАЛ: ты НЕ вызвал write_plan. Текущий план в plan.md устарел. "
                           f"Прочитай read_notes, затем ОБЯЗАТЕЛЬНО вызови write_plan с новым Digest, "
                           f"новым [FOCUS] и обновлённым [TODO].")}],
                       "🔁")
            plan_text = PLAN_PATH.read_text() if PLAN_PATH.exists() else ""
        new_focus = _current_focus()
        if new_focus != focus:
            print(f"   🔀 вектор скорректирован: {focus[:50]} → {new_focus[:50]}")
        if "PLAN_COMPLETE" in plan_text:
            print("✅ План исчерпан"); break
        if "[TODO]" not in plan_text and i > 1:
            print("✅ Нет больше [TODO]"); break
        if low_gain_streak >= 2 and i >= 3:
            print(f"✅ Ранний стоп: 2 итерации подряд < 2 новых id (информационная отдача исчерпана)"); break

    # Фаза 1.5 — компрессия заметок для снижения контекста writer'а
    notes_size = NOTES_PATH.stat().st_size if NOTES_PATH.exists() else 0
    # локальная модель = бесплатно → сжимаем реже, только при реальном переполнении
    if notes_size > 8000:
        print(f"\n🗜️  Фаза 1.5: компрессор ({notes_size} симв → ~5000)")
        compressor = _build_bot(COMPRESSOR_PROMPT, ["read_notes", "compact_notes"])
        _run_agent(compressor, [{"role": "user", "content": "Сократи заметки до ~5000 симв."}], "🗜️")

    # Фаза 2.0 — синтезатор: новые инсайты поверх собранных заметок
    print(f"\n💡 Фаза 2.0: синтезатор (мосты/противоречия/пробелы/экстраполяция/testable)")
    synthesizer = _build_bot(SYNTHESIZER_PROMPT,
                             ["read_plan", "read_notes", "write_synthesis", "run_python"],
                             max_tokens=4096)
    _run_agent(synthesizer,
               [{"role": "user", "content": f"Произведи пять типов инсайтов по теме: {query}"}],
               "💡")

    # Фаза 2
    print(f"\n✍️  Фаза 2: writer — финальный черновик")
    writer = _build_bot(WRITER_PROMPT,
                        ["read_plan", "read_notes", "read_synthesis", "read_draft",
                         "write_draft", "append_draft"],
                        max_tokens=6144)
    writer_msgs = [{"role": "user", "content": f"Собери финальный отчёт по теме: {query}"}]
    resp = _run_agent(writer, writer_msgs, "✍️ ")
    writer_msgs.extend(resp)

    # Фаза 3 — критик с проверкой конвергенции (если две критики подряд почти одинаковы — выходим)
    print(f"\n🔍 Фаза 3: критик ({critic_rounds} раунд(ов))")
    critic = _build_bot(CRITIC_PROMPT,
                        ["read_draft", "read_notes", "read_synthesis"],
                        max_tokens=2048)
    prev_critique = ""

    def _crit_keywords(s: str) -> set[str]:
        return {w.lower() for w in re.findall(r"[\w\-]{5,}", s)}

    for i in range(1, critic_rounds + 1):
        print(f"\n── критика {i}/{critic_rounds} ──")
        c_resp = _run_agent(critic, [{"role": "user", "content": f"Оцени черновик по теме: {query}"}], "🔍")
        critique = " ".join(m.get("content", "") for m in c_resp if m.get("role") == "assistant").strip()
        if "APPROVED" in critique.upper() and len(critique) < 60:
            print("✅ критик одобрил"); break
        # Конвергенция: если новая критика пересекается с предыдущей на >70% ключевых слов — зациклились
        if prev_critique:
            a, b = _crit_keywords(prev_critique), _crit_keywords(critique)
            if a and b:
                sim = len(a & b) / max(1, len(a | b))
                if sim > 0.70:
                    print(f"✅ критик зациклился (сходство {sim:.0%}) — выходим"); break
        prev_critique = critique
        writer_msgs.append({"role": "user",
                            "content": f"Критика:\n{critique}\n\nПерепиши через write_draft."})
        resp = _run_agent(writer, writer_msgs, "✍️ ")
        writer_msgs.extend(resp)

    if DRAFT_PATH.exists():
        # Фаза 4 — валидация arXiv-id + семантическая сверка цитат с notes
        print(f"\n🔐 Фаза 4: валидация цитат (hf info + keyword overlap с notes)")
        valid, invalid, suspicious = _validate_draft_ids()
        print(f"   ✓ валидных: {valid}")
        if invalid:
            print(f"   ✗ не найдены: {invalid}")
            print("   ⚠️  возможные галлюцинации — проверь вручную")
        if suspicious:
            print(f"   ⚠️  слабое совпадение цитаты с notes: {suspicious}")
            print("       (id существует, но текст вокруг него в draft'е не отражает факты из notes)")
        print(f"\n📄 Итог: {DRAFT_PATH}\n" + "─" * 60)
        print(DRAFT_PATH.read_text())
        print("─" * 60)
        print(f"Файлы: {NOTES_PATH.name}, {PLAN_PATH.name}, {DRAFT_PATH.name}")
    else:
        print("⚠️  writer не сохранил черновик")


# ─── Запуск ────────────────────────────────────────────────────────


def _build_bot(system_message: str, tools: list, max_tokens: Optional[int] = None) -> Assistant:
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
    return Assistant(llm=llm_cfg, system_message=system_message, function_list=tools)


def main():
    print(f"⏳ Загружаю {CFG['model']} ...")
    # Прогреваем модель (первый _build_bot заполнит _MLX_CACHE)
    _get_mlx(CFG["model"])
    print("✅ Готово. Команды:")
    print("   <тема>              — запустить ресёрч (alias: /research <тема>)")
    print("   /clean              — очистить рабочую папку (lessons/querylog остаются)")
    print("   /forget             — стереть ВСЁ, включая глобальную Reflexion-память")
    print("   /exit               — выход\n")

    while True:
        try:
            q = input("🔬 ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋"); return
        if not q:
            continue
        if q == "/exit":
            return
        if q in ("/clean", "/clean-research"):
            # стираем только текущую рабочую папку
            for p in (DRAFT_PATH, NOTES_PATH, PLAN_PATH, SYNTHESIS_PATH):
                p.unlink(missing_ok=True)
            print("🗑️  research/ очищена (lessons/querylog сохранены — глобальная память)\n"); continue
        if q == "/forget":
            # полный сброс, включая Reflexion-память
            for p in (DRAFT_PATH, NOTES_PATH, PLAN_PATH, SYNTHESIS_PATH, LESSONS_PATH, QUERYLOG_PATH):
                p.unlink(missing_ok=True)
            print("🧠  Всё стёрто, включая глобальные lessons/querylog\n"); continue

        if q.startswith("/research"):
            q = q[len("/research"):].strip()
        if not q:
            print("⚠️  укажи тему\n"); continue
        research_loop(q, depth=6, critic_rounds=2)
        print()


if __name__ == "__main__":
    main()
