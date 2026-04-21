"""Все tools для qwen-agent."""
from __future__ import annotations
import json
import subprocess
import sys

from qwen_agent.tools.base import BaseTool, register_tool

from .config import (DRAFT_PATH, LESSONS_PATH, NOTES_PATH, PLAN_PATH,
                     QUERYLOG_PATH, SYNTHESIS_PATH)
from .memory import ensure_dir, log_query, seen_queries
from .utils import normalize_query, parse_args

_SANDBOX_PRELUDE = r"""
import resource, sys, builtins
# CPU 5s, address space 512MB, no fork
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
# запись только в /tmp
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
        p = parse_args(params)
        query = p["query"]
        limit = max(1, min(int(p.get("limit", 5)), 10))
        if normalize_query(query) in seen_queries():
            return (f"ОТКАЗ: запрос '{query}' уже выполнялся в этой сессии. "
                    "Переформулируй (другие ключевые слова, автор, год, техника) или читай read_notes.")
        log_query(query)
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
        code = parse_args(params)["code"]
        try:
            r = subprocess.run(
                [sys.executable, "-I", "-c", _SANDBOX_PRELUDE],
                input=code, capture_output=True, text=True, timeout=10,
            )
        except subprocess.TimeoutExpired:
            return "таймаут (>10с)"
        out = (r.stdout + (r.stderr and f"\n[stderr]\n{r.stderr}" or "")).strip()
        return (out or "(вывода нет)")[:5000]


@register_tool("compact_notes")
class CompactNotes(BaseTool):
    description = "Перезаписывает research/notes.md сжатым содержанием."
    parameters = [{"name": "content", "type": "string", "description": "Сжатый markdown", "required": True}]

    def call(self, params: str, **kwargs) -> str:
        ensure_dir()
        content = parse_args(params)["content"]
        old = NOTES_PATH.stat().st_size if NOTES_PATH.exists() else 0
        NOTES_PATH.write_text(content)
        return f"notes.md: {old} → {len(content)} симв (сжато в {max(1, old // max(1, len(content)))}x)"


@register_tool("write_draft")
class WriteDraft(BaseTool):
    description = "Перезаписывает черновик research/draft.md с нуля."
    parameters = [{"name": "content", "type": "string", "description": "Markdown", "required": True}]

    def call(self, params: str, **kwargs) -> str:
        ensure_dir()
        content = parse_args(params)["content"]
        DRAFT_PATH.write_text(content)
        return f"draft.md сохранён ({len(content)} симв)"


@register_tool("append_draft")
class AppendDraft(BaseTool):
    description = ("Добавляет секцию в конец research/draft.md. "
                   "Предпочтительно write_draft+много append_draft, чем один огромный write_draft.")
    parameters = [{"name": "content", "type": "string", "description": "Markdown-секция", "required": True}]

    def call(self, params: str, **kwargs) -> str:
        ensure_dir()
        content = parse_args(params)["content"]
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
                   "(подтема, arXiv id, ключевые факты).")
    parameters = [{"name": "content", "type": "string", "description": "Markdown-фрагмент", "required": True}]

    def call(self, params: str, **kwargs) -> str:
        ensure_dir()
        content = parse_args(params)["content"]
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
        return text[-20000:] if len(text) > 20000 else text


@register_tool("write_plan")
class WritePlan(BaseTool):
    description = "Перезаписывает план ресёрча в research/plan.md."
    parameters = [{"name": "content", "type": "string", "description": "Markdown план", "required": True}]

    def call(self, params: str, **kwargs) -> str:
        ensure_dir()
        content = parse_args(params)["content"]
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
    description = ("Записывает research/synthesis.md — НОВЫЕ мысли: "
                   "мосты, противоречия, пробелы, экстраполяции, testable гипотезы.")
    parameters = [{"name": "content", "type": "string", "description": "Markdown", "required": True}]

    def call(self, params: str, **kwargs) -> str:
        ensure_dir()
        content = parse_args(params)["content"]
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
                   "Формат: что сработало, что нет, чего избегать.")
    parameters = [{"name": "content", "type": "string", "description": "Markdown", "required": True}]

    def call(self, params: str, **kwargs) -> str:
        ensure_dir()
        content = parse_args(params)["content"]
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
        lines, seen = [], set()
        for ln in reversed(text.splitlines()):
            if ln.strip() and ln not in seen:
                seen.add(ln)
                lines.append(ln)
            if len(lines) >= 30:
                break
        return "\n".join(reversed(lines))
