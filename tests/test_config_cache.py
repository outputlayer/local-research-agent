"""Юниты для Settings (валидация конфига) и CLI-кеша."""
import json

import pytest

from lra import cache
from lra.config import Settings


def test_settings_valid(tmp_path):
    f = tmp_path / "c.json"
    f.write_text(json.dumps({"model": "m", "temperature": 0.5}))
    s = Settings.load(f)
    assert s.model == "m"
    assert s.temperature == 0.5
    assert s["temperature"] == 0.5  # dict-style
    assert s.get("missing", 42) == 42


def test_settings_rejects_bad_temperature(tmp_path):
    f = tmp_path / "c.json"
    f.write_text(json.dumps({"model": "m", "temperature": 5.0}))
    with pytest.raises(ValueError, match="temperature"):
        Settings.load(f)


def test_settings_rejects_empty_model(tmp_path):
    f = tmp_path / "c.json"
    f.write_text(json.dumps({"model": ""}))
    with pytest.raises(ValueError, match="model"):
        Settings.load(f)


def test_settings_captures_extra_keys(tmp_path):
    f = tmp_path / "c.json"
    f.write_text(json.dumps({"model": "m", "custom_field": "x"}))
    s = Settings.load(f)
    assert s.extra["custom_field"] == "x"
    assert s["custom_field"] == "x"


def test_cache_hit_miss(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    cmd = ["hf", "papers", "search", "foo"]
    assert cache.get(cmd) is None
    cache.put(cmd, "stdout-x", "", 0)
    hit = cache.get(cmd)
    assert hit is not None
    assert hit["stdout"] == "stdout-x"


def test_cache_ttl_expires(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    cmd = ["hf", "x"]
    cache.put(cmd, "old", "", 0)
    # ttl=0ч → любой результат считается протухшим
    assert cache.get(cmd, ttl_hours=0) is None


def test_cache_clear(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    cache.put(["a"], "1", "", 0)
    cache.put(["b"], "2", "", 0)
    n = cache.clear()
    assert n == 2
    assert cache.get(["a"]) is None
