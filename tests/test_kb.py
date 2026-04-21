"""Тесты структурированной базы знаний research/kb.jsonl."""
from __future__ import annotations


def _setup(tmp_path, monkeypatch):
    from lra import kb
    monkeypatch.setattr(kb, "KB_PATH", tmp_path / "kb.jsonl")
    monkeypatch.setattr(kb, "RESEARCH_DIR", tmp_path)
    return kb


def test_add_and_load_roundtrip(tmp_path, monkeypatch):
    kb = _setup(tmp_path, monkeypatch)
    kb.add(kb.Atom(id="2401.1234", kind="paper", topic="LoRA",
                   claim="LoRA снижает параметры в 100x", title="LoRA paper"))
    kb.add(kb.Atom(id="microsoft/LoRA", kind="repo", topic="LoRA",
                   claim="эталонная реализация", stars=8000, lang="Python"))
    atoms = kb.load()
    assert len(atoms) == 2
    kinds = {a["kind"] for a in atoms}
    assert kinds == {"paper", "repo"}


def test_load_deduplicates_by_kind_and_id(tmp_path, monkeypatch):
    kb = _setup(tmp_path, monkeypatch)
    kb.add(kb.Atom(id="2401.1234", kind="paper", topic="v1", claim="старая версия"))
    kb.add(kb.Atom(id="2401.1234", kind="paper", topic="v2", claim="уточнённая версия"))
    atoms = kb.load()
    assert len(atoms) == 1
    assert atoms[0]["claim"] == "уточнённая версия"


def test_search_finds_relevant_by_keyword(tmp_path, monkeypatch):
    kb = _setup(tmp_path, monkeypatch)
    kb.add(kb.Atom(id="2401.0001", kind="paper", topic="LoRA",
                   claim="LoRA уменьшает число обучаемых параметров через низкоранговые адаптеры",
                   title="LoRA: Low-Rank Adaptation"))
    kb.add(kb.Atom(id="2401.0002", kind="paper", topic="quant",
                   claim="квантизация в 4 бита сохраняет качество",
                   title="QLoRA quantization"))
    kb.add(kb.Atom(id="2401.0003", kind="paper", topic="rlhf",
                   claim="метод выравнивания через награды",
                   title="RLHF survey"))
    hits = kb.search("низкоранговая адаптация параметров", k=2)
    assert len(hits) >= 1
    # самый релевантный — LoRA
    assert hits[0]["id"] == "2401.0001"


def test_search_empty_kb_returns_empty(tmp_path, monkeypatch):
    kb = _setup(tmp_path, monkeypatch)
    assert kb.search("любой запрос") == []


def test_search_empty_query_returns_empty(tmp_path, monkeypatch):
    kb = _setup(tmp_path, monkeypatch)
    kb.add(kb.Atom(id="x", kind="paper", topic="t", claim="c"))
    assert kb.search("") == []
    assert kb.search("   ") == []


def test_format_atoms_paper_and_repo(tmp_path, monkeypatch):
    kb = _setup(tmp_path, monkeypatch)
    atoms = [
        {"id": "2401.0001", "kind": "paper", "title": "Test Paper",
         "claim": "важный факт"},
        {"id": "org/name", "kind": "repo", "stars": 42, "lang": "Rust",
         "claim": "интересная архитектура"},
    ]
    out = kb.format_atoms(atoms)
    assert "[2401.0001]" in out
    assert "Test Paper" in out
    assert "важный факт" in out
    assert "[repo: org/name ★42 Rust]" in out
    assert "интересная архитектура" in out


def test_format_atoms_empty():
    from lra import kb
    assert kb.format_atoms([]) == ""
