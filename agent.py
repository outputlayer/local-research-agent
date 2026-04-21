#!/usr/bin/env python3
"""Тонкий CLI. Логика — в пакете `lra/`."""
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
from lra.pipeline import research_loop
from lra.plan import PLAN_JSON_PATH


def _check_clis():
    """Проверяет наличие и авторизацию внешних CLI. Ничего не требует — только печатает статус."""
    # hf
    if shutil.which("hf"):
        print("   ✓ hf CLI найден")
    else:
        print("   ⚠️  hf CLI не найден (pip install huggingface_hub[cli]) — hf_papers не будет работать")
    # gh — используем `gh api user --jq .login` (не зависит от локали)
    if not shutil.which("gh"):
        print("   ⚠️  gh CLI не найден (brew install gh) — github_search не будет работать")
        return
    try:
        r = subprocess.run(
            ["gh", "api", "user", "--jq", ".login"],
            capture_output=True, text=True, timeout=5,
        )
        login = r.stdout.strip()
        if r.returncode == 0 and login:
            print(f"   ✓ gh CLI: авторизован как {login}")
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
    print("   /forget             — стереть + глобальную Reflexion-память (archive и cache сохраняются)")
    print("   /reset              — ПОЛНАЯ очистка: research/*, archive/*, .cache/*, run.log")
    print("   /exit               — выход\n")

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
        if q in ("/clean", "/clean-research"):
            for p in (DRAFT_PATH, NOTES_PATH, PLAN_PATH, PLAN_JSON_PATH, SYNTHESIS_PATH, KB_PATH):
                p.unlink(missing_ok=True)
            print("🗑️  research/ очищена (lessons/querylog/archive сохранены)\n")
            continue
        if q == "/forget":
            for p in (DRAFT_PATH, NOTES_PATH, PLAN_PATH, PLAN_JSON_PATH, SYNTHESIS_PATH, KB_PATH,
                      LESSONS_PATH, QUERYLOG_PATH):
                p.unlink(missing_ok=True)
            print("🧠  Стёрто + глобальные lessons/querylog (archive и cache остаются)\n")
            continue
        if q == "/reset":
            # Полная очистка: всё в research/ (кроме самой папки), весь кеш, архив.
            # Используем shutil.rmtree только для поддиректорий — саму research/ сохраняем.
            confirm = input("⚠️  /reset удалит research/* + archive/* + .cache/*. "
                            "Напиши 'yes' для подтверждения: ").strip().lower()
            if confirm != "yes":
                print("❌ отменено\n")
                continue
            # файлы
            for p in (DRAFT_PATH, NOTES_PATH, PLAN_PATH, PLAN_JSON_PATH, SYNTHESIS_PATH, KB_PATH,
                      LESSONS_PATH, QUERYLOG_PATH, RUN_LOG_PATH):
                p.unlink(missing_ok=True)
            # поддиректории
            for d in (ARCHIVE_DIR, CACHE_DIR):
                if d.exists():
                    shutil.rmtree(d, ignore_errors=True)
            # пересоздаём пустые каталоги
            RESEARCH_DIR.mkdir(exist_ok=True)
            ARCHIVE_DIR.mkdir(exist_ok=True)
            print("💥  Полный ресет: research/*, archive/*, .cache/*, run.log стёрты\n")
            continue

        if q.startswith("/research"):
            q = q[len("/research"):].strip()
        if not q:
            print("⚠️  укажи тему\n")
            continue
        research_loop(q, depth=6, critic_rounds=2)
        print()


if __name__ == "__main__":
    main()
