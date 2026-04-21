"""Структурированный план ресёрча: `plan.json` — source of truth, `plan.md` — рендер для людей/LLM.

Модель: дерево `Task` с статусами, журнал `Revision` (аудит изменений плана),
детерминированный `guard()` без LLM-вызовов для обнаружения тупиков/лупов/оверэкспансии.

Интеграция:
- `reset_plan(query)` инициализирует начальный план (3 open задачи)
- `load()/save()` для чтения/записи plan.json; `render_md()` автоматически обновляет plan.md
- `guard()` вызывается после каждой итерации pipeline; возвращает рекомендации
- Инструменты `plan_add_task`/`plan_close_task`/`plan_split_task` (в `tools.py`) — единственный
  легитимный способ модели менять структуру плана.

Ограничения (hard caps):
- `MAX_OPEN_TASKS` — защита от оверэкспансии (модель не может бесконечно плодить подзадачи)
- `MAX_ATTEMPTS_PER_TASK` — после N неуспешных заходов задача авто-блокируется
- `MAX_REVISIONS` — защита от постоянной переписи плана
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from .config import PLAN_PATH, RESEARCH_DIR
from .logger import get_logger

log = get_logger("plan")

PLAN_JSON_PATH = RESEARCH_DIR / "plan.json"

# Hard caps. Подобраны эмпирически: держим план компактным, чтобы explorer не тонул.
MAX_OPEN_TASKS = 8
MAX_ATTEMPTS_PER_TASK = 3
MAX_REVISIONS = 20
# Если задача in_progress >= N итераций и evidence_refs пуст → инкрементим attempts.
EVIDENCE_STARVATION_ITERS = 2


# ── Модель ────────────────────────────────────────────────────────────────
@dataclass
class Task:
    """Единица работы в плане. Дерево через `parent` (2 уровня достаточно на практике)."""
    id: str
    title: str
    status: str = "open"  # open | in_progress | done | blocked | dropped
    parent: str | None = None
    origin: str = "initial"  # initial | emerged | split_from_X | corrective | goal_redefine
    attempts: int = 0
    evidence_refs: list[str] = field(default_factory=list)  # ["kb:id", "notes:arxiv-id"]
    created_iter: int = 0
    closed_iter: int | None = None
    last_active_iter: int = 0  # когда последний раз эта задача была in_progress
    note: str = ""


@dataclass
class Revision:
    """Запись в аудит-логе: ПОЧЕМУ план изменился. Не для LLM — для человека и анализа."""
    iter: int
    action: str  # add | close | split | drop | block | unblock | redefine_goal | rotate_focus | attempt
    target: str | list[str] | None = None
    why: str = ""
    ts: str = ""

    def __post_init__(self):
        if not self.ts:
            self.ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class Plan:
    """Корневой объект. Сохраняется целиком в plan.json каждый раз."""
    root_goal: str
    current_focus_id: str | None = None
    tasks: list[Task] = field(default_factory=list)
    revisions: list[Revision] = field(default_factory=list)
    version: int = 1

    # ── Навигация ─────────────────────────────────────────────────────────
    def get(self, task_id: str) -> Task | None:
        return next((t for t in self.tasks if t.id == task_id), None)

    def open_tasks(self) -> list[Task]:
        return [t for t in self.tasks if t.status in ("open", "in_progress")]

    def children(self, parent_id: str) -> list[Task]:
        return [t for t in self.tasks if t.parent == parent_id]

    def focus_task(self) -> Task | None:
        if self.current_focus_id:
            return self.get(self.current_focus_id)
        return None

    def focus_title(self) -> str:
        t = self.focus_task()
        return t.title if t else self.root_goal

    def _next_id(self, prefix: str = "T") -> str:
        """Генерирует следующий id. Использует числовой суффикс с учётом родителя для подзадач."""
        existing = {t.id for t in self.tasks}
        n = 1
        while f"{prefix}{n}" in existing:
            n += 1
        return f"{prefix}{n}"

    # ── Мутации (каждая → revision) ──────────────────────────────────────
    def add_task(self, title: str, *, parent: str | None = None,
                 origin: str = "emerged", iter_: int = 0,
                 note: str = "", why: str = "") -> Task:
        """Добавляет новую задачу. Уважает MAX_OPEN_TASKS."""
        if len(self.open_tasks()) >= MAX_OPEN_TASKS:
            raise ValueError(
                f"MAX_OPEN_TASKS={MAX_OPEN_TASKS} достигнут — сначала закрой/дропни "
                f"существующие ({len(self.open_tasks())} open)")
        if parent and not self.get(parent):
            raise ValueError(f"parent '{parent}' не найден")
        if parent:
            tid = self._child_id(parent)
        else:
            tid = self._next_id("T")
        task = Task(id=tid, title=title.strip(), parent=parent,
                    origin=origin, created_iter=iter_, note=note)
        self.tasks.append(task)
        self._revise(iter_, "add", tid, why or f"добавлена задача: {title[:80]}")
        return task

    def _child_id(self, parent: str) -> str:
        """Для T2 генерит T2.1, T2.2 и т.д."""
        siblings = [t.id for t in self.tasks if t.parent == parent]
        # уже есть T2.1, T2.2 — ищем T2.N
        n = 1
        while f"{parent}.{n}" in siblings:
            n += 1
        return f"{parent}.{n}"

    def close_task(self, task_id: str, *, iter_: int = 0,
                   evidence: list[str] | None = None, why: str = "") -> Task:
        t = self.get(task_id)
        if not t:
            raise ValueError(f"задача '{task_id}' не найдена")
        t.status = "done"
        t.closed_iter = iter_
        if evidence:
            t.evidence_refs.extend(evidence)
        self._revise(iter_, "close", task_id, why or "закрыта")
        # если это был focus — обнуляем
        if self.current_focus_id == task_id:
            self.current_focus_id = None
        return t

    def split_task(self, task_id: str, subtitles: list[str], *,
                   iter_: int = 0, why: str = "") -> list[Task]:
        parent = self.get(task_id)
        if not parent:
            raise ValueError(f"задача '{task_id}' не найдена")
        if not subtitles:
            raise ValueError("subtitles пуст")
        # превращаем parent в контейнер: open → dropped родительской «плоской» задачи
        # чтобы не путалась с подзадачами; но сохраняем её title как заголовок ветки
        parent.status = "dropped"
        parent.note = (parent.note + " | split-container").strip(" |")
        children: list[Task] = []
        for st in subtitles:
            # обходим MAX_OPEN_TASKS check для split (это не экспансия, а декомпозиция)
            tid = self._child_id(task_id)
            child = Task(id=tid, title=st.strip(), parent=task_id,
                         origin=f"split_from_{task_id}", created_iter=iter_)
            self.tasks.append(child)
            children.append(child)
        self._revise(iter_, "split", [c.id for c in children],
                     why or f"декомпозиция {task_id} на {len(children)} подзадач")
        return children

    def drop_task(self, task_id: str, *, iter_: int = 0, why: str = "") -> Task:
        t = self.get(task_id)
        if not t:
            raise ValueError(f"задача '{task_id}' не найдена")
        t.status = "dropped"
        t.closed_iter = iter_
        self._revise(iter_, "drop", task_id, why or "дропнута")
        if self.current_focus_id == task_id:
            self.current_focus_id = None
        return t

    def block_task(self, task_id: str, *, iter_: int = 0, why: str = "") -> Task:
        t = self.get(task_id)
        if not t:
            raise ValueError(f"задача '{task_id}' не найдена")
        t.status = "blocked"
        self._revise(iter_, "block", task_id, why or "заблокирована")
        return t

    def set_focus(self, task_id: str | None, *, iter_: int = 0, why: str = "") -> None:
        if task_id is not None:
            t = self.get(task_id)
            if not t:
                raise ValueError(f"задача '{task_id}' не найдена")
            if t.status in ("done", "dropped", "blocked"):
                raise ValueError(f"нельзя ставить фокус на {t.status} задачу")
            t.status = "in_progress"
            t.last_active_iter = iter_
        self.current_focus_id = task_id
        self._revise(iter_, "rotate_focus", task_id, why or "новый фокус")

    def increment_attempts(self, task_id: str, *, iter_: int = 0, why: str = "") -> int:
        t = self.get(task_id)
        if not t:
            return 0
        t.attempts += 1
        self._revise(iter_, "attempt", task_id,
                     why or f"попытка {t.attempts}/{MAX_ATTEMPTS_PER_TASK}")
        return t.attempts

    def link_evidence(self, task_id: str, refs: list[str]) -> None:
        t = self.get(task_id)
        if not t:
            return
        for r in refs:
            if r and r not in t.evidence_refs:
                t.evidence_refs.append(r)

    def _revise(self, iter_: int, action: str, target, why: str) -> None:
        self.revisions.append(Revision(iter=iter_, action=action, target=target, why=why))
        # trim чтобы не распух
        if len(self.revisions) > MAX_REVISIONS * 5:
            self.revisions = self.revisions[-MAX_REVISIONS * 3:]


# ── Persistence ───────────────────────────────────────────────────────────
def load(path: Path | None = None) -> Plan | None:
    p = path or PLAN_JSON_PATH
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        tasks = [Task(**t) for t in raw.get("tasks", [])]
        revisions = [Revision(**r) for r in raw.get("revisions", [])]
        return Plan(
            root_goal=raw["root_goal"],
            current_focus_id=raw.get("current_focus_id"),
            tasks=tasks, revisions=revisions,
            version=raw.get("version", 1),
        )
    except Exception as e:
        log.warning("plan.json повреждён (%s) — игнорируем и пересоздадим", e)
        return None


def save(plan: Plan, path: Path | None = None) -> None:
    p = path or PLAN_JSON_PATH
    p.parent.mkdir(exist_ok=True, parents=True)
    payload = {
        "root_goal": plan.root_goal,
        "current_focus_id": plan.current_focus_id,
        "version": plan.version,
        "tasks": [asdict(t) for t in plan.tasks],
        "revisions": [asdict(r) for r in plan.revisions[-MAX_REVISIONS:]],
    }
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    # рендерим markdown параллельно — ТОТ ЖЕ источник данных
    render_md(plan)


def render_md(plan: Plan, path: Path | None = None) -> None:
    """Рендерит `plan.md` из plan.json. Формат совместим с текущими read_plan / _current_focus
    (есть строка `[FOCUS] <title>`, секции `## [TODO]`, `## [DONE]`).
    """
    p = path or PLAN_PATH
    focus_t = plan.focus_task()
    focus_line = f"[FOCUS] {focus_t.title}" if focus_t else f"[FOCUS] {plan.root_goal}"

    todo = [t for t in plan.tasks if t.status == "open"]
    in_prog = [t for t in plan.tasks if t.status == "in_progress"]
    done = [t for t in plan.tasks if t.status == "done"]
    blocked = [t for t in plan.tasks if t.status == "blocked"]
    total = len(plan.tasks)
    pct = int(100 * len(done) / total) if total else 0

    lines = [
        f"# Plan: {plan.root_goal}",
        "",
        focus_line,
        "",
        f"**Прогресс: {len(done)}/{total} done ({pct}%)** · "
        f"open={len(todo)} · in_progress={len(in_prog)} · blocked={len(blocked)} · "
        f"ревизий={len(plan.revisions)}",
        "",
    ]

    # Digest — из последних revisions
    last_rev = plan.revisions[-5:]
    if last_rev:
        lines.append("## Digest (последние изменения плана)")
        for r in last_rev:
            lines.append(f"- iter {r.iter}: {r.action} {r.target} — {r.why}")
        lines.append("")

    if in_prog:
        lines.append("## [IN_PROGRESS]")
        for t in in_prog:
            lines.append(f"- [{t.id}] {t.title}  _(attempts={t.attempts}, evidence={len(t.evidence_refs)})_")
        lines.append("")

    lines.append("## [TODO]")
    if todo:
        for t in todo:
            pfx = f"[{t.id}] "
            tree = "  " if t.parent else ""
            lines.append(f"- {tree}{pfx}{t.title}")
    else:
        lines.append("(пусто — план исчерпан)")
        lines.append("")
        lines.append("PLAN_COMPLETE")
    lines.append("")

    if done:
        lines.append("## [DONE]")
        for t in done:
            evidence = f"  _(evidence={len(t.evidence_refs)})_" if t.evidence_refs else ""
            lines.append(f"- [{t.id}] {t.title}{evidence}")
        lines.append("")

    if blocked:
        lines.append("## [BLOCKED]")
        for t in blocked:
            lines.append(f"- [{t.id}] {t.title}  _(attempts={t.attempts})_")
        lines.append("")

    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── Инициализация ─────────────────────────────────────────────────────────
def reset(query: str, path: Path | None = None) -> Plan:
    """Пересоздаёт plan.json под новый запрос. Пять стартовых open-задач + focus на первую.

    Ветки подобраны так, чтобы покрывать и науку (papers/методы), и инженерию (репо/
    бенчмарки) — это заставляет explorer делать github_search и собирать конкретные
    метрики, а не только пересказывать абстракты.
    """
    plan = Plan(root_goal=query)
    t1 = plan.add_task(f"{query}: обзорные статьи и ключевые архитектуры",
                       origin="initial", iter_=0, why="seed: наука/методы")
    plan.add_task(f"{query}: реализации и open-source репозитории (★≥10)",
                  origin="initial", iter_=0, why="seed: инженерия/reuse")
    plan.add_task(f"{query}: бенчмарки и численные метрики (SR, accuracy, E2E)",
                  origin="initial", iter_=0, why="seed: evaluation")
    plan.add_task(f"{query}: ограничения и failure modes",
                  origin="initial", iter_=0, why="seed: critical view")
    plan.add_task(f"{query}: открытые вопросы и направления",
                  origin="initial", iter_=0, why="seed: gaps")
    plan.set_focus(t1.id, iter_=0, why="начальный фокус")
    save(plan, path=path)
    return plan


def bootstrap_from_seeds(query: str, seeds: list[dict], topic_type: str = "mixed",
                         path: Path | None = None) -> Plan | None:
    """Инициализирует plan.json из LLM-сгенерированных seed-задач.

    `seeds` — список dict c ключами `title` и `why` (опц). Должно быть 3-6 валидных
    элементов, иначе возвращаем None (caller упадёт на статический reset()).
    `topic_type` пишется в root_goal для трассируемости.

    Поведенческая гарантия: если функция возвращает Plan, файл plan.json уже записан
    и имеет ≥1 open-задачу + установленный focus. Если возвращает None — ни один файл
    не тронут (caller должен вызвать reset()).
    """
    # Валидация входа — строгая, поскольку это граница "LLM → наш код"
    if not isinstance(seeds, list) or not (3 <= len(seeds) <= 8):
        return None
    cleaned: list[tuple[str, str]] = []
    for s in seeds:
        if not isinstance(s, dict):
            continue
        title = str(s.get("title", "")).strip()
        why = str(s.get("why", "")).strip() or "bootstrap"
        # Минимальная длина заголовка — чтобы отсечь «test», «a», etc.
        if 10 <= len(title) <= 200:
            cleaned.append((title, why))
    if len(cleaned) < 3:
        return None

    tt = topic_type if topic_type in ("engineering", "theoretical", "mixed") else "mixed"
    plan = Plan(root_goal=f"[{tt}] {query}")
    first = None
    for title, why in cleaned:
        t = plan.add_task(title, origin="bootstrap", iter_=0, why=why)
        if first is None:
            first = t
    plan.set_focus(first.id, iter_=0, why=f"bootstrap focus (topic_type={tt})")
    save(plan, path=path)
    return plan


def parse_bootstrap_json(text: str) -> tuple[str, list[dict]] | None:
    """Парсит вывод INITIAL_PLANNER_PROMPT. Толерантен к ```json-ограде и префиксам.

    Возвращает (topic_type, seeds) или None если JSON невалиден/пуст.
    """
    if not text:
        return None
    # Убираем распространённые обёртки: ```json ... ``` или ``` ... ```
    stripped = text.strip()
    if stripped.startswith("```"):
        # режем первую и последнюю fenced-строку
        lines = stripped.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    # Находим первую `{` и последнюю `}` — даже если LLM наговорил вокруг
    lb = stripped.find("{")
    rb = stripped.rfind("}")
    if lb < 0 or rb <= lb:
        return None
    candidate = stripped[lb:rb + 1]
    try:
        import json5
        data = json5.loads(candidate)
    except Exception:
        try:
            data = json.loads(candidate)
        except Exception:
            return None
    if not isinstance(data, dict):
        return None
    tt = str(data.get("topic_type", "mixed")).strip().lower()
    tasks = data.get("tasks", [])
    if not isinstance(tasks, list):
        return None
    return tt, tasks



# ── Guard ─────────────────────────────────────────────────────────────────
@dataclass
class GuardReport:
    """Рекомендации детерминированного watchdog'а. LLM не вовлечён."""
    iter: int
    halt: bool = False
    halt_reason: str = ""
    blocked_ids: list[str] = field(default_factory=list)
    auto_dropped_ids: list[str] = field(default_factory=list)
    rotated_focus: bool = False
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = []
        if self.halt:
            parts.append(f"HALT={self.halt_reason}")
        if self.blocked_ids:
            parts.append(f"blocked={self.blocked_ids}")
        if self.auto_dropped_ids:
            parts.append(f"dropped={self.auto_dropped_ids}")
        if self.rotated_focus:
            parts.append("rotated_focus")
        if self.warnings:
            parts.append("warn=" + "; ".join(self.warnings))
        return " | ".join(parts) if parts else "ok"


def guard(plan: Plan, iter_: int, *,
          notes_grew: int = 0, new_ids: int = 0,
          focus_unchanged_streak: int = 0,
          empty_iter_streak: int = 0) -> GuardReport:
    """Детерминированный надзиратель. Работает ПОСЛЕ итерации.

    Сигналы от pipeline:
    - notes_grew: сколько символов добавил explorer за итерацию
    - new_ids: сколько новых arxiv-id обнаружено
    - focus_unchanged_streak: сколько итераций подряд тот же focus
    - empty_iter_streak: сколько итераций подряд notes не росли

    Действия (в порядке):
    1. Текущему focus'у инкрементим attempts если итерация оказалась «пустой»
    2. Если attempts у focus достигли предела → блокируем, снимаем фокус
    3. Если focus_unchanged_streak >= 3 И у задачи есть attempts → блокируем
    4. Если все open/in_progress задачи blocked/dropped → halt
    """
    report = GuardReport(iter=iter_)
    focus = plan.focus_task()

    # 1. Учёт пустой итерации
    if focus and notes_grew < 100 and new_ids == 0:
        attempts = plan.increment_attempts(focus.id, iter_=iter_,
                                           why="итерация без прироста notes/ids")
        report.warnings.append(f"{focus.id}: attempts={attempts}")
        if attempts >= MAX_ATTEMPTS_PER_TASK:
            plan.block_task(focus.id, iter_=iter_,
                            why=f"исчерпаны попытки ({MAX_ATTEMPTS_PER_TASK})")
            report.blocked_ids.append(focus.id)
            plan.current_focus_id = None

    # 2. Стагнация фокуса — даже если росло чуть-чуть, но ID не появились
    if focus and focus_unchanged_streak >= 3 and empty_iter_streak >= 2:
        if focus.status != "blocked":
            plan.block_task(focus.id, iter_=iter_,
                            why=f"фокус не меняется {focus_unchanged_streak} итераций")
            report.blocked_ids.append(focus.id)
            plan.current_focus_id = None

    # 3. Авто-ротация: если focus пуст но open ещё есть
    if plan.current_focus_id is None and plan.open_tasks():
        # берём первую open (не in_progress)
        next_t = next((t for t in plan.tasks if t.status == "open"), None)
        if next_t:
            plan.set_focus(next_t.id, iter_=iter_,
                           why="авто-ротация после блокировки/закрытия предыдущего")
            report.rotated_focus = True

    # 4. Halt-условия
    if not plan.open_tasks():
        report.halt = True
        report.halt_reason = "ALL_DONE_OR_BLOCKED"

    save(plan)
    return report


# ── Sync с plan.md (обратная совместимость с write_plan / _rotate_focus_fallback) ──
def sync_focus_from_md(plan: Plan, md_text: str, *, iter_: int = 0) -> bool:
    """Если write_plan изменил plan.md (legacy-путь replanner'а), пытаемся согласовать
    фокус с plan.json. Best-effort: ищем строку `[FOCUS] <text>` и если текст НЕ совпадает
    с текущим focus_task().title, создаём новую `corrective` задачу и ставим её в фокус.

    Возвращает True если применили корректировку.
    """
    focus_line = ""
    for ln in md_text.splitlines():
        s = ln.strip()
        if s.startswith("[FOCUS]"):
            focus_line = s.replace("[FOCUS]", "").strip(" -—:")
            break
    if not focus_line:
        return False
    # Sentinel: replanner объявил план исчерпанным — закрываем все open/in_progress задачи
    # чтобы render_md показал PLAN_COMPLETE и guard сигналил halt.
    if focus_line.strip().upper() == "PLAN_COMPLETE":
        changed = False
        for t in plan.tasks:
            if t.status in ("open", "in_progress"):
                plan.close_task(t.id, why="replanner: PLAN_COMPLETE")
                changed = True
        return changed
    current_title = plan.focus_title().strip()
    if focus_line == current_title:
        return False
    # Ищем существующую task с таким же title — если есть, ставим на неё фокус
    match = next((t for t in plan.tasks
                  if t.title == focus_line and t.status not in ("done", "dropped", "blocked")), None)
    if match:
        plan.set_focus(match.id, iter_=iter_, why="sync с plan.md (write_plan)")
        return True
    # Иначе — создаём corrective задачу (обходим MAX_OPEN_TASKS мягко: если лимит, всё равно добавляем)
    try:
        new_t = plan.add_task(focus_line, origin="corrective", iter_=iter_,
                              why="replanner выставил новый FOCUS через write_plan")
    except ValueError:
        # лимит — дропаем самую старую open задачу без evidence
        stale = next((t for t in plan.tasks
                      if t.status == "open" and not t.evidence_refs), None)
        if stale:
            plan.drop_task(stale.id, iter_=iter_, why="вытеснена corrective задачей (лимит open)")
            new_t = plan.add_task(focus_line, origin="corrective", iter_=iter_,
                                  why="replanner выставил новый FOCUS (после дропа stale)")
        else:
            return False
    plan.set_focus(new_t.id, iter_=iter_, why="новый corrective фокус")
    return True
