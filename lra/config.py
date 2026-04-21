"""Конфиг и пути к артефактам."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CFG = json.loads((ROOT / "chat_config.json").read_text())

RESEARCH_DIR = ROOT / "research"
ARCHIVE_DIR = RESEARCH_DIR / "archive"
DRAFT_PATH = RESEARCH_DIR / "draft.md"
NOTES_PATH = RESEARCH_DIR / "notes.md"
PLAN_PATH = RESEARCH_DIR / "plan.md"
SYNTHESIS_PATH = RESEARCH_DIR / "synthesis.md"
# Reflexion-память ГЛОБАЛЬНА — живёт между сессиями, не стирается при новом запросе
LESSONS_PATH = RESEARCH_DIR / "lessons.md"
QUERYLOG_PATH = RESEARCH_DIR / "querylog.md"
