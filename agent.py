#!/usr/bin/env python3
"""Thin CLI. All logic lives in the `lra/` package."""
from __future__ import annotations

import shutil
import subprocess

from lra.config import (
    ARCHIVE_DIR,
    CACHE_DIR,
    CFG,
    DRAFT_PATH,
    LESSONS_PATH,
    NOTES_PATH,
    PLAN_PATH,
    QUERYLOG_PATH,
    RESEARCH_DIR,
    RUN_LOG_PATH,
    SYNTHESIS_PATH,
)
from lra.kb import KB_PATH
from lra.llm import get_mlx
from lra.pipeline import _build_status_context, research_loop, resume_research
from lra.plan import PLAN_JSON_PATH


def _check_clis():
    """Checks presence and auth of external CLIs. Requires nothing — only prints status."""
    if shutil.which("hf"):
        print("   ✓ hf CLI found")
    else:
        print("   ⚠️  hf CLI not found (pip install huggingface_hub[cli]) — hf_papers will not work")
    if not shutil.which("gh"):
        print("   ⚠️  gh CLI not found (brew install gh) — github_search will not work")
        return
    try:
        r = subprocess.run(
            ["gh", "api", "user", "--jq", ".login"],
            capture_output=True, text=True, timeout=5,
        )
        login = r.stdout.strip()
        if r.returncode == 0 and login:
            print(f"   ✓ gh CLI: authenticated as {login}")
        else:
            print("   ⚠️  gh CLI found but NOT authenticated. Run: gh auth login")
    except Exception as e:
        print(f"   ⚠️  gh auth check failed: {e}")


def main():
    print(f"⏳ Loading {CFG['model']} ...")
    get_mlx(CFG["model"])
    print("🔧 External CLI check:")
    _check_clis()
    print("✅ Ready. Commands:")
    print("   <topic>             — start research (alias: /research <topic>)")
    print("   /resume             — finish report from existing notes/synthesis/kb (skip explorer)")
    print("   /status [topic]     — show current research status (plan, rejected evidence)")
    print("   /hitl on|off        — toggle human-in-the-loop pause after validator")
    print("   /clean              — clean research dir (lessons/querylog kept)")
    print("   /forget             — wipe + global Reflexion memory (archive and cache kept)")
    print("   /reset              — FULL wipe: research/*, archive/*, .cache/*, run.log")
    print("   /exit               — quit\n")

    while True:
        try:
            q = input("🔬 ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋")
            return
        if not q:
            continue
        if q == "/exit":
            return
        if q == "/resume":
            resume_research()
            print()
            continue
        if q.startswith("/status"):
            topic = q[len("/status"):].strip() or "(current research)"
            print(_build_status_context(topic))
            print()
            continue
        if q.startswith("/hitl"):
            arg = q[len("/hitl"):].strip().lower()
            if arg in ("on", "true", "1", "yes"):
                CFG["hitl"] = True
                print("🧑 HITL enabled: pipeline will pause after the validator asking for edits\n")
            elif arg in ("off", "false", "0", "no", ""):
                CFG["hitl"] = False
                print("🤖 HITL disabled (automatic mode)\n")
            else:
                print(f"⚠️  unknown argument: {arg!r}. Use: /hitl on | /hitl off\n")
            continue
        if q in ("/clean", "/clean-research"):
            for p in (DRAFT_PATH, NOTES_PATH, PLAN_PATH, PLAN_JSON_PATH, SYNTHESIS_PATH, KB_PATH):
                p.unlink(missing_ok=True)
            print("🗑️  research/ cleaned (lessons/querylog/archive kept)\n")
            continue
        if q == "/forget":
            for p in (DRAFT_PATH, NOTES_PATH, PLAN_PATH, PLAN_JSON_PATH, SYNTHESIS_PATH, KB_PATH,
                      LESSONS_PATH, QUERYLOG_PATH):
                p.unlink(missing_ok=True)
            print("🧠  Wiped + global lessons/querylog (archive and cache kept)\n")
            continue
        if q == "/reset":
            confirm = input("⚠️  /reset will delete research/* + archive/* + .cache/*. "
                            "Type 'yes' to confirm: ").strip().lower()
            if confirm != "yes":
                print("❌ cancelled\n")
                continue
            for p in (DRAFT_PATH, NOTES_PATH, PLAN_PATH, PLAN_JSON_PATH, SYNTHESIS_PATH, KB_PATH,
                      LESSONS_PATH, QUERYLOG_PATH, RUN_LOG_PATH):
                p.unlink(missing_ok=True)
            for d in (ARCHIVE_DIR, CACHE_DIR):
                if d.exists():
                    shutil.rmtree(d, ignore_errors=True)
            RESEARCH_DIR.mkdir(exist_ok=True)
            ARCHIVE_DIR.mkdir(exist_ok=True)
            print("💥  Full reset: research/*, archive/*, .cache/*, run.log wiped\n")
            continue

        if q.startswith("/research"):
            q = q[len("/research"):].strip()
        if not q:
            print("⚠️  please specify a topic\n")
            continue
        research_loop(q, depth=6, critic_rounds=2)
        print()


if __name__ == "__main__":
    main()
