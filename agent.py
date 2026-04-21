#!/usr/bin/env python3
"""Тонкий CLI. Логика — в пакете `lra/`."""
from __future__ import annotations

import shutil
import subprocess

from lra.config import (CFG, DRAFT_PATH, LESSONS_PATH, NOTES_PATH, PLAN_PATH,
                        QUERYLOG_PATH, SYNTHESIS_PATH)
from lra.llm import get_mlx
from lra.pipeline import research_loop


def _check_clis():
    """Проверяет наличие и авторизацию внешних CLI. Ничего не требует — только печатает статус."""
    # hf
    if shutil.which("hf"):
        print("   ✓ hf CLI найден")
    else:
        print("   ⚠️  hf CLI не найден (pip install huggingface_hub[cli]) — hf_papers не будет работать")
    # gh
    if not shutil.which("gh"):
        print("   ⚠️  gh CLI не найден (brew install gh) — github_search не будет работать")
        return
    try:
        r = subprocess.run(["gh", "auth", "status"], capture_output=True, text=True, timeout=5)
        out = (r.stdout + r.stderr).strip()
        if r.returncode == 0 and "Logged in" in out:
            # Достаём имя аккаунта
            acc = ""
            for ln in out.splitlines():
                if "account" in ln.lower():
                    acc = ln.strip(); break
            print(f"   ✓ gh CLI: {acc or 'авторизован'}")
        else:
            print("   ⚠️  gh CLI найден, но НЕ авторизован. Выполни: gh auth login")
    except Exception as e:
        print(f"   ⚠️  gh auth check failed: {e}")


def main():
    print(f"⏳ Загружаю {CFG['model']} ...")
    get_mlx(CFG["model"])
    print("🔧 Проверка внешних CLI:")
    _check_clis()
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
            for p in (DRAFT_PATH, NOTES_PATH, PLAN_PATH, SYNTHESIS_PATH):
                p.unlink(missing_ok=True)
            print("🗑️  research/ очищена (lessons/querylog сохранены — глобальная память)\n")
            continue
        if q == "/forget":
            for p in (DRAFT_PATH, NOTES_PATH, PLAN_PATH, SYNTHESIS_PATH,
                      LESSONS_PATH, QUERYLOG_PATH):
                p.unlink(missing_ok=True)
            print("🧠  Всё стёрто, включая глобальные lessons/querylog\n")
            continue

        if q.startswith("/research"):
            q = q[len("/research"):].strip()
        if not q:
            print("⚠️  укажи тему\n"); continue
        research_loop(q, depth=6, critic_rounds=2)
        print()


if __name__ == "__main__":
    main()
