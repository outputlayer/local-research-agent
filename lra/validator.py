"""Validation of arXiv-ids in the draft: existence + semantic link with notes."""
from __future__ import annotations

import re
import subprocess

from .config import DRAFT_PATH, NOTES_PATH
from .utils import STOPWORDS, extract_ids, keyword_set


def validate_draft_ids(
    draft_text: str | None = None,
    notes_text: str | None = None,
    run_hf_info: bool = True,
) -> tuple[int, list[str], list[str]]:
    """Checks arXiv-ids in the draft and the quality of their citations.

    Parameters (optional, for testing):
      draft_text — content of draft.md (None → read from disk)
      notes_text — content of notes.md (None → read from disk)
      run_hf_info — whether to call `hf papers info` for external ids (False for unit tests)

    Returns: (valid, missing, suspicious_citations).
    A suspicious citation: an id present in notes, but its surrounding context in
    the draft does not overlap with facts from notes for that id (≥3 shared keywords).
    """
    if draft_text is None:
        if not DRAFT_PATH.exists():
            return 0, [], []
        draft_text = DRAFT_PATH.read_text(encoding='utf-8')
    if notes_text is None:
        notes_text = NOTES_PATH.read_text(encoding='utf-8') if NOTES_PATH.exists() else ""

    ids = extract_ids(draft_text)
    if not ids:
        return 0, [], []
    notes_ids = extract_ids(notes_text)
    invalid, suspicious = [], []
    valid = 0

    for pid in ids:
        if pid not in notes_ids:
            if not run_hf_info:
                invalid.append(pid)
                continue
            try:
                r = subprocess.run(
                    ["hf", "papers", "info", pid],
                    capture_output=True, text=True, timeout=15,
                )
                if r.returncode != 0 or "not found" in (r.stdout + r.stderr).lower():
                    invalid.append(pid)
                    continue
            except Exception:
                invalid.append(pid)
                continue
            valid += 1
            continue
        draft_ctx = " ".join(
            re.findall(rf".{{0,120}}{re.escape(pid)}.{{0,120}}", draft_text, re.DOTALL)
        )
        notes_lines = notes_text.splitlines()
        notes_ctx_parts = []
        for i, ln in enumerate(notes_lines):
            if pid in ln:
                notes_ctx_parts.append(" ".join(notes_lines[max(0, i - 1):i + 3]))
        notes_ctx = " ".join(notes_ctx_parts)
        draft_kw = keyword_set(draft_ctx) - STOPWORDS
        notes_kw = keyword_set(notes_ctx)
        overlap = draft_kw & notes_kw
        # Threshold: used to be 3 — too lenient. Technical jargon
        # (complex-valued/adversarial/FMCW) easily yields 3 overlaps even for a
        # fully fabricated claim. 5 — empirically cuts off
        # citation-laundering without false positives on correct paraphrases.
        if len(overlap) < 5:
            suspicious.append(f"{pid} (overlap={len(overlap)})")
        valid += 1
    return valid, invalid, suspicious
