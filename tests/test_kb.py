"""Tests for the structured knowledge base research/kb.jsonl."""
from __future__ import annotations


def _setup(tmp_path, monkeypatch):
    from lra import kb
    monkeypatch.setattr(kb, "KB_PATH", tmp_path / "kb.jsonl")
    monkeypatch.setattr(kb, "RESEARCH_DIR", tmp_path)
    return kb


def test_add_and_load_roundtrip(tmp_path, monkeypatch):
    kb = _setup(tmp_path, monkeypatch)
    kb.add(kb.Atom(id="2401.1234", kind="paper", topic="LoRA",
                   claim="LoRA cuts parameters 100x", title="LoRA paper"))
    kb.add(kb.Atom(id="microsoft/LoRA", kind="repo", topic="LoRA",
                   claim="reference implementation", stars=8000, lang="Python"))
    atoms = kb.load()
    assert len(atoms) == 2
    kinds = {a["kind"] for a in atoms}
    assert kinds == {"paper", "repo"}


def test_load_deduplicates_by_kind_and_id(tmp_path, monkeypatch):
    kb = _setup(tmp_path, monkeypatch)
    kb.add(kb.Atom(id="2401.1234", kind="paper", topic="v1", claim="old version"))
    kb.add(kb.Atom(id="2401.1234", kind="paper", topic="v2", claim="refined version"))
    atoms = kb.load()
    assert len(atoms) == 1
    assert atoms[0]["claim"] == "refined version"


def test_search_finds_relevant_by_keyword(tmp_path, monkeypatch):
    kb = _setup(tmp_path, monkeypatch)
    kb.add(kb.Atom(id="2401.0001", kind="paper", topic="LoRA",
                   claim="LoRA reduces the number of trainable parameters via low-rank adapters",
                   title="LoRA: Low-Rank Adaptation"))
    kb.add(kb.Atom(id="2401.0002", kind="paper", topic="quant",
                   claim="4-bit quantization preserves quality",
                   title="QLoRA quantization"))
    kb.add(kb.Atom(id="2401.0003", kind="paper", topic="rlhf",
                   claim="alignment method via rewards",
                   title="RLHF survey"))
    hits = kb.search("low-rank parameter adaptation", k=2)
    assert len(hits) >= 1
    # most relevant — LoRA
    assert hits[0]["id"] == "2401.0001"


def test_search_empty_kb_returns_empty(tmp_path, monkeypatch):
    kb = _setup(tmp_path, monkeypatch)
    assert kb.search("any query") == []


def test_search_empty_query_returns_empty(tmp_path, monkeypatch):
    kb = _setup(tmp_path, monkeypatch)
    kb.add(kb.Atom(id="x", kind="paper", topic="t", claim="c"))
    assert kb.search("") == []
    assert kb.search("   ") == []


def test_format_atoms_paper_and_repo(tmp_path, monkeypatch):
    kb = _setup(tmp_path, monkeypatch)
    atoms = [
        {"id": "2401.0001", "kind": "paper", "title": "Test Paper",
         "claim": "important fact"},
        {"id": "org/name", "kind": "repo", "stars": 42, "lang": "Rust",
         "claim": "interesting architecture"},
    ]
    out = kb.format_atoms(atoms)
    assert "[2401.0001]" in out
    assert "Test Paper" in out
    assert "important fact" in out
    assert "[repo: org/name ★42 Rust]" in out
    assert "interesting architecture" in out


def test_format_atoms_empty():
    from lra import kb
    assert kb.format_atoms([]) == ""
