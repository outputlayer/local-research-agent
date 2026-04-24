"""Structured research plan: `plan.json` is the source of truth, `plan.md` is the render for humans/LLMs.

Model: a `Task` tree with statuses, a `Revision` journal (audit of plan changes),
and a deterministic `guard()` (no LLM calls) for detecting deadlocks/loops/over-expansion.

Integration:
- `reset_plan(query)` initializes the starting plan (3 open tasks)
- `load()/save()` for reading/writing plan.json; `render_md()` updates plan.md automatically
- `guard()` is invoked after each pipeline iteration and returns recommendations
- Tools `plan_add_task`/`plan_close_task`/`plan_split_task` (in `tools.py`) are the only
  legitimate way for the model to change the plan structure.

Hard caps:
- `MAX_OPEN_TASKS` — protects against over-expansion (the model cannot spawn sub-tasks forever)
- `MAX_ATTEMPTS_PER_TASK` — after N failed attempts the task is auto-blocked
- `MAX_REVISIONS` — protects against constant plan rewriting
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

# Hard caps. Picked empirically: keep the plan compact so the explorer does not drown.
MAX_OPEN_TASKS = 8
MAX_ATTEMPTS_PER_TASK = 3
MAX_REVISIONS = 20
# If a task has been in_progress for >= N iterations with empty evidence_refs, increment attempts.
EVIDENCE_STARVATION_ITERS = 2


# ── Model ────────────────────────────────────────────────────
@dataclass
class Task:
    """A unit of work in the plan. Tree via `parent` (2 levels are enough in practice)."""
    id: str
    title: str
    status: str = "open"  # open | in_progress | done | blocked | dropped
    parent: str | None = None
    origin: str = "initial"  # initial | emerged | split_from_X | corrective | goal_redefine
    attempts: int = 0
    evidence_refs: list[str] = field(default_factory=list)  # ["kb:id", "notes:arxiv-id"]
    created_iter: int = 0
    closed_iter: int | None = None
    last_active_iter: int = 0  # the last iteration when this task was in_progress
    note: str = ""


@dataclass
class Revision:
    """Audit-log entry: WHY the plan changed. Not for the LLM — for humans and analysis."""
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
    """Root object. Saved entirely to plan.json every time."""
    root_goal: str
    current_focus_id: str | None = None
    tasks: list[Task] = field(default_factory=list)
    revisions: list[Revision] = field(default_factory=list)
    version: int = 1
    # Total number of revisions over the plan's lifetime (not trimmed at save).
    # Unlike len(revisions), which is capped by MAX_REVISIONS inside save(),
    # this counter grows monotonically and reflects the real "age" of the plan in plan.md.
    total_revisions: int = 0
    # Core vocabulary: 8-15 specific domain terms from LLM bootstrap. Used
    # in two places: (a) domain_gate extends HEADER keywords with this list —
    # treats gate blindness when the header is too generic; (b) the explorer may
    # use it as a seed for concrete arxiv queries instead of a generic query.
    core_vocabulary: list[str] = field(default_factory=list)

    # ── Navigation ──────────────────────────────────────────────────────
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
        """Generates the next id. Uses a numeric suffix with parent awareness for sub-tasks."""
        existing = {t.id for t in self.tasks}
        n = 1
        while f"{prefix}{n}" in existing:
            n += 1
        return f"{prefix}{n}"

    # ── Mutations (each → revision) ──────────────────────────────────────
    def add_task(self, title: str, *, parent: str | None = None,
                 origin: str = "emerged", iter_: int = 0,
                 note: str = "", why: str = "") -> Task:
        """Adds a new task. Respects MAX_OPEN_TASKS."""
        if len(self.open_tasks()) >= MAX_OPEN_TASKS:
            raise ValueError(
                f"MAX_OPEN_TASKS={MAX_OPEN_TASKS} reached — first close/drop "
                f"existing ({len(self.open_tasks())} open)")
        if parent and not self.get(parent):
            raise ValueError(f"parent '{parent}' not found")
        if parent:
            tid = self._child_id(parent)
        else:
            tid = self._next_id("T")
        task = Task(id=tid, title=title.strip(), parent=parent,
                    origin=origin, created_iter=iter_, note=note)
        self.tasks.append(task)
        self._revise(iter_, "add", tid, why or f"added task: {title[:80]}")
        return task

    def _child_id(self, parent: str) -> str:
        """For T2 generates T2.1, T2.2, etc."""
        siblings = [t.id for t in self.tasks if t.parent == parent]
        # T2.1/T2.2 may exist — look for T2.N
        n = 1
        while f"{parent}.{n}" in siblings:
            n += 1
        return f"{parent}.{n}"

    def close_task(self, task_id: str, *, iter_: int = 0,
                   evidence: list[str] | None = None, why: str = "") -> Task:
        t = self.get(task_id)
        if not t:
            raise ValueError(f"task '{task_id}' not found")
        t.status = "done"
        t.closed_iter = iter_
        if evidence:
            t.evidence_refs.extend(evidence)
        self._revise(iter_, "close", task_id, why or "closed")
        # if this was the focus — reset it
        if self.current_focus_id == task_id:
            self.current_focus_id = None
        return t

    def split_task(self, task_id: str, subtitles: list[str], *,
                   iter_: int = 0, why: str = "") -> list[Task]:
        parent = self.get(task_id)
        if not parent:
            raise ValueError(f"task '{task_id}' not found")
        if not subtitles:
            raise ValueError("subtitles is empty")
        # turn parent into a container: open → dropped of the parent "flat" task
        # so it does not get confused with sub-tasks; we keep its title as the branch header
        parent.status = "dropped"
        parent.note = (parent.note + " | split-container").strip(" |")
        children: list[Task] = []
        for st in subtitles:
            # bypass MAX_OPEN_TASKS check for split (this is decomposition, not expansion)
            tid = self._child_id(task_id)
            child = Task(id=tid, title=st.strip(), parent=task_id,
                         origin=f"split_from_{task_id}", created_iter=iter_)
            self.tasks.append(child)
            children.append(child)
        self._revise(iter_, "split", [c.id for c in children],
                     why or f"decomposition of {task_id} into {len(children)} sub-tasks")
        return children

    def drop_task(self, task_id: str, *, iter_: int = 0, why: str = "") -> Task:
        t = self.get(task_id)
        if not t:
            raise ValueError(f"task '{task_id}' not found")
        t.status = "dropped"
        t.closed_iter = iter_
        self._revise(iter_, "drop", task_id, why or "dropped")
        if self.current_focus_id == task_id:
            self.current_focus_id = None
        return t

    def block_task(self, task_id: str, *, iter_: int = 0, why: str = "") -> Task:
        t = self.get(task_id)
        if not t:
            raise ValueError(f"task '{task_id}' not found")
        t.status = "blocked"
        self._revise(iter_, "block", task_id, why or "blocked")
        return t

    def set_focus(self, task_id: str | None, *, iter_: int = 0, why: str = "") -> None:
        if task_id is not None:
            t = self.get(task_id)
            if not t:
                raise ValueError(f"task '{task_id}' not found")
            if t.status in ("done", "dropped", "blocked"):
                raise ValueError(f"cannot focus a {t.status} task")
            t.status = "in_progress"
            t.last_active_iter = iter_
        self.current_focus_id = task_id
        self._revise(iter_, "rotate_focus", task_id, why or "new focus")

    def increment_attempts(self, task_id: str, *, iter_: int = 0, why: str = "") -> int:
        t = self.get(task_id)
        if not t:
            return 0
        t.attempts += 1
        self._revise(iter_, "attempt", task_id,
                     why or f"attempt {t.attempts}/{MAX_ATTEMPTS_PER_TASK}")
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
        self.total_revisions += 1
        # trim so it does not balloon
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
        stored_revs = raw.get("total_revisions", len(revisions))
        return Plan(
            root_goal=raw["root_goal"],
            current_focus_id=raw.get("current_focus_id"),
            tasks=tasks, revisions=revisions,
            version=raw.get("version", 1),
            core_vocabulary=raw.get("core_vocabulary", []),
            total_revisions=stored_revs,
        )
    except Exception as e:
        log.warning("plan.json is corrupted (%s) — ignoring and recreating", e)
        return None


def save(plan: Plan, path: Path | None = None) -> None:
    p = path or PLAN_JSON_PATH
    p.parent.mkdir(exist_ok=True, parents=True)
    payload = {
        "root_goal": plan.root_goal,
        "current_focus_id": plan.current_focus_id,
        "version": plan.version,
        "total_revisions": plan.total_revisions,
        "core_vocabulary": plan.core_vocabulary,
        "tasks": [asdict(t) for t in plan.tasks],
        "revisions": [asdict(r) for r in plan.revisions[-MAX_REVISIONS:]],
    }
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    # render markdown in parallel — SAME source of data
    render_md(plan)


def render_md(plan: Plan, path: Path | None = None) -> None:
    """Renders `plan.md` from plan.json. Format is compatible with current read_plan /
    _current_focus (has a `[FOCUS] <title>` line, `## [TODO]`, `## [DONE]` sections).
    """
    p = path or PLAN_PATH
    focus_t = plan.focus_task()
    focus_line = f"[FOCUS] {focus_t.title}" if focus_t else f"[FOCUS] {plan.root_goal}"

    todo = [t for t in plan.tasks if t.status == "open"]
    in_prog = [t for t in plan.tasks if t.status == "in_progress"]
    done = [t for t in plan.tasks if t.status == "done"]
    blocked = [t for t in plan.tasks if t.status == "blocked"]
    # Exclude dropped from the denominator — split containers and
    # displaced tasks must not make the visible progress smaller than real.
    total = len([t for t in plan.tasks if t.status != "dropped"])
    pct = int(100 * len(done) / total) if total else 0

    lines = [
        f"# Plan: {plan.root_goal}",
        "",
    ]
    # Core vocabulary (if set) — right after the title so domain_gate
    # can extend HEADER-keywords with specific domain terms from the LLM.
    if plan.core_vocabulary:
        vocab_line = "**Core vocabulary:** " + ", ".join(plan.core_vocabulary)
        lines.extend([vocab_line, ""])
    lines.extend([
        focus_line,
        "",
        f"**Progress: {len(done)}/{total} done ({pct}%)** · "
        f"open={len(todo)} · in_progress={len(in_prog)} · blocked={len(blocked)} · "
        f"revisions={plan.total_revisions}",
        "",
    ])

    # Digest — from the latest revisions
    last_rev = plan.revisions[-5:]
    if last_rev:
        lines.append("## Digest (latest plan changes)")
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
        lines.append("(empty — plan exhausted)")
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


# ── Initialization ────────────────────────────────────────────────────────
def reset(query: str, path: Path | None = None) -> Plan:
    """Recreates plan.json for a new query. Five starting open tasks + focus on the first.

    Branches are chosen to cover both science (papers/methods) and engineering (repos/
    benchmarks) — this forces the explorer to do github_search and gather concrete
    metrics rather than only retelling abstracts.
    """
    plan = Plan(root_goal=query)
    t1 = plan.add_task(f"{query}: survey papers and key architectures",
                       origin="initial", iter_=0, why="seed: science/methods")
    plan.add_task(f"{query}: implementations and open-source repositories (★≥10)",
                  origin="initial", iter_=0, why="seed: engineering/reuse")
    plan.add_task(f"{query}: benchmarks and numeric metrics (SR, accuracy, E2E)",
                  origin="initial", iter_=0, why="seed: evaluation")
    plan.add_task(f"{query}: limitations and failure modes",
                  origin="initial", iter_=0, why="seed: critical view")
    plan.add_task(f"{query}: open questions and directions",
                  origin="initial", iter_=0, why="seed: gaps")
    plan.set_focus(t1.id, iter_=0, why="initial focus")
    save(plan, path=path)
    return plan


def bootstrap_from_seeds(query: str, seeds: list[dict], topic_type: str = "mixed",
                         core_vocabulary: list[str] | None = None,
                         path: Path | None = None) -> Plan | None:
    """Initializes plan.json from LLM-generated seed tasks.

    `seeds` — a list of dicts with keys `title` and `why` (optional). Must be 3-6 valid
    items; otherwise returns None (the caller falls back to static reset()).
    `topic_type` is stored in root_goal for traceability.
    `core_vocabulary` — 8-15 LLM-provided domain terms that reinforce domain_gate
    when the header is generic (careful filtering: len≥4, not in STOPWORDS).

    Behavioral guarantee: if the function returns a Plan, plan.json is already written
    with ≥1 open task and an established focus. If it returns None, no file
    is touched (the caller must invoke reset()).
    """
    # Strict input validation because this is the "LLM → our code" boundary
    if not isinstance(seeds, list) or not (3 <= len(seeds) <= 8):
        return None
    cleaned: list[tuple[str, str]] = []
    for s in seeds:
        if not isinstance(s, dict):
            continue
        title = str(s.get("title", "")).strip()
        why = str(s.get("why", "")).strip() or "bootstrap"
        # Minimum title length — filters out "test", "a", etc.
        if 10 <= len(title) <= 200:
            cleaned.append((title, why))
    if len(cleaned) < 3:
        return None

    # core_vocabulary sanitization ("LLM → our code" boundary): strings of 3-40 chars
    # (3 to catch acronyms: SGD, RAG, ECM, ESM), dedup
    # (case-insensitive), up to 15 entries. Empty list is fine (fallback to header).
    cv_clean: list[str] = []
    seen_lower: set[str] = set()
    for term in core_vocabulary or []:
        if not isinstance(term, str):
            continue
        t = term.strip()
        if not (3 <= len(t) <= 40):
            continue
        lo = t.lower()
        if lo in seen_lower:
            continue
        seen_lower.add(lo)
        cv_clean.append(t)
        if len(cv_clean) >= 15:
            break

    tt = topic_type if topic_type in ("engineering", "theoretical", "mixed") else "mixed"
    plan = Plan(root_goal=f"[{tt}] {query}", core_vocabulary=cv_clean)
    first = None
    for title, why in cleaned:
        t = plan.add_task(title, origin="bootstrap", iter_=0, why=why)
        if first is None:
            first = t
    plan.set_focus(first.id, iter_=0, why=f"bootstrap focus (topic_type={tt})")
    save(plan, path=path)
    return plan


def parse_bootstrap_json(text: str) -> tuple[str, list[dict], list[str]] | None:
    """Parses INITIAL_PLANNER_PROMPT output. Tolerant of ```json fences and prefixes.

    Returns (topic_type, seeds, core_vocabulary) or None if JSON is invalid/empty.
    core_vocabulary is optional (may be empty for backward compatibility).
    """
    if not text:
        return None
    # Strip common wrappers: ```json ... ``` or ``` ... ```
    stripped = text.strip()
    if stripped.startswith("```"):
        # drop first and last fenced lines
        lines = stripped.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    # Find the first `{` and last `}` — even if the LLM narrates around them
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
    raw_vocab = data.get("core_vocabulary", [])
    vocab: list[str] = (
        [v for v in raw_vocab if isinstance(v, str)]
        if isinstance(raw_vocab, list) else []
    )
    return tt, tasks, vocab



# ── Guard ─────────────────────────────────────────────────────────────────
@dataclass
class GuardReport:
    """Recommendations of the deterministic watchdog. No LLM is involved."""
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
    """Deterministic watchdog. Runs AFTER an iteration.

    Signals from the pipeline:
    - notes_grew: how many characters the explorer appended this iteration
    - new_ids: how many new arxiv-ids were found
    - focus_unchanged_streak: how many consecutive iterations kept the same focus
    - empty_iter_streak: how many consecutive iterations the notes did not grow

    Actions (in order):
    1. Increment attempts on the current focus if the iteration was "empty"
    2. If focus attempts reach the cap → block it and clear focus
    3. If focus_unchanged_streak >= 3 AND the task has attempts → block it
    4. If all open/in_progress tasks are blocked/dropped → halt
    """
    report = GuardReport(iter=iter_)
    focus = plan.focus_task()

    # 1. Account for an empty iteration
    if focus and notes_grew < 100 and new_ids == 0:
        attempts = plan.increment_attempts(focus.id, iter_=iter_,
                                           why="iteration without notes/ids growth")
        report.warnings.append(f"{focus.id}: attempts={attempts}")
        if attempts >= MAX_ATTEMPTS_PER_TASK:
            plan.block_task(focus.id, iter_=iter_,
                            why=f"attempts exhausted ({MAX_ATTEMPTS_PER_TASK})")
            report.blocked_ids.append(focus.id)
            plan.current_focus_id = None

    # 2. Focus stagnation — even if there was a tiny growth, no new IDs appeared
    if focus and focus_unchanged_streak >= 3 and empty_iter_streak >= 2:
        if focus.status != "blocked":
            plan.block_task(focus.id, iter_=iter_,
                            why=f"focus has not changed for {focus_unchanged_streak} iterations")
            report.blocked_ids.append(focus.id)
            plan.current_focus_id = None

    # 3. Auto-rotation: if focus is empty but open tasks remain
    if plan.current_focus_id is None and plan.open_tasks():
        # pick the first open (non in_progress) task
        next_t = next((t for t in plan.tasks if t.status == "open"), None)
        if next_t:
            plan.set_focus(next_t.id, iter_=iter_,
                           why="auto-rotation after blocking/closing the previous focus")
            report.rotated_focus = True

    # 4. Halt conditions
    if not plan.open_tasks():
        report.halt = True
        report.halt_reason = "ALL_DONE_OR_BLOCKED"
    elif empty_iter_streak >= 4:
        # Global halt: ≥4 iterations in a row with no growth in notes/ids,
        # regardless of focus. Protection from the case where auto-rotation
        # dutifully rotates focus but none of them yields any results.
        report.halt = True
        report.halt_reason = f"GLOBAL_EMPTY_STREAK({empty_iter_streak})"

    save(plan)
    return report


# ── Sync with plan.md (backward compatibility with write_plan / _rotate_focus_fallback) ──
def sync_focus_from_md(plan: Plan, md_text: str, *, iter_: int = 0) -> bool:
    """If write_plan changed plan.md (the replanner legacy path), try to reconcile
    focus with plan.json. Best-effort: look for `[FOCUS] <text>` and if the text does NOT match
    the current focus_task().title, create a new `corrective` task and give it focus.

    Returns True if a correction was applied.
    """
    focus_line = ""
    for ln in md_text.splitlines():
        s = ln.strip()
        if s.startswith("[FOCUS]"):
            focus_line = s.replace("[FOCUS]", "").strip(" -—:")
            break
    if not focus_line:
        return False
    # Sentinel: replanner declared the plan exhausted — close all open/in_progress tasks
    # so render_md shows PLAN_COMPLETE and guard signals halt.
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
    # Look for an existing task with the same title — if found, focus on it
    match = next((t for t in plan.tasks
                  if t.title == focus_line and t.status not in ("done", "dropped", "blocked")), None)
    if match:
        plan.set_focus(match.id, iter_=iter_, why="sync with plan.md (write_plan)")
        return True
    # Otherwise — create a corrective task (bypass MAX_OPEN_TASKS softly: add anyway at the limit)
    try:
        new_t = plan.add_task(focus_line, origin="corrective", iter_=iter_,
                              why="replanner set a new FOCUS via write_plan")
    except ValueError:
        # limit reached — drop the oldest open task without evidence
        stale = next((t for t in plan.tasks
                      if t.status == "open" and not t.evidence_refs), None)
        if stale:
            plan.drop_task(stale.id, iter_=iter_, why="displaced by a corrective task (open limit)")
            new_t = plan.add_task(focus_line, origin="corrective", iter_=iter_,
                                  why="replanner set a new FOCUS (after dropping stale)")
        else:
            return False
    plan.set_focus(new_t.id, iter_=iter_, why="new corrective focus")
    return True
