"""P8: kb collision detection по title."""
from lra import kb


def _patch(tmp_path, monkeypatch):
    monkeypatch.setattr(kb, "RESEARCH_DIR", tmp_path)
    monkeypatch.setattr(kb, "KB_PATH", tmp_path / "kb.jsonl")
    monkeypatch.setattr(kb, "KB_COLLISIONS_PATH", tmp_path / "kb_collisions.jsonl")
    return tmp_path


def test_no_collision_identical_title(tmp_path, monkeypatch):
    """Та же arxiv-id с тем же title → коллизии нет."""
    _patch(tmp_path, monkeypatch)
    kb.add(kb.Atom(id="2401.00001", kind="paper", topic="t",
                   title="State Space Models for Radar", claim="c1"))
    kb.add(kb.Atom(id="2401.00001", kind="paper", topic="t",
                   title="State Space Models for Radar", claim="c2"))
    assert not (tmp_path / "kb_collisions.jsonl").exists() or \
        (tmp_path / "kb_collisions.jsonl").read_text().strip() == ""


def test_collision_different_title_logged(tmp_path, monkeypatch):
    """Та же id с резко отличным title → запись в kb_collisions.jsonl."""
    _patch(tmp_path, monkeypatch)
    kb.add(kb.Atom(id="2602.10434", kind="paper", topic="t",
                   title="Benchmarking DL for Landmine Detection", claim="c1"))
    kb.add(kb.Atom(id="2602.10434", kind="paper", topic="t",
                   title="Engineering AI Agents Four Pillars", claim="c2"))
    coll = (tmp_path / "kb_collisions.jsonl").read_text(encoding="utf-8").strip()
    assert coll
    import json
    rec = json.loads(coll)
    assert rec["id"] == "2602.10434"
    assert "Landmine" in rec["prior_title"]
    assert "Agents" in rec["new_title"]


def test_empty_title_no_collision(tmp_path, monkeypatch):
    """Пустой title не триггерит коллизию."""
    _patch(tmp_path, monkeypatch)
    kb.add(kb.Atom(id="2401.00002", kind="paper", topic="t", title="", claim="c1"))
    kb.add(kb.Atom(id="2401.00002", kind="paper", topic="t", title="Some Paper", claim="c2"))
    # Нет prior title → коллизии не должно быть
    assert not (tmp_path / "kb_collisions.jsonl").exists()


def test_load_dedup_keeps_last(tmp_path, monkeypatch):
    """load() по-прежнему дедуплицирует — последняя запись побеждает."""
    _patch(tmp_path, monkeypatch)
    kb.add(kb.Atom(id="2401.00003", kind="paper", topic="t",
                   title="Title A", claim="claim A"))
    kb.add(kb.Atom(id="2401.00003", kind="paper", topic="t",
                   title="Title A", claim="claim B updated"))
    atoms = kb.load()
    assert len(atoms) == 1
    assert atoms[0]["claim"] == "claim B updated"
