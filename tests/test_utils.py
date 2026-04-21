"""Юниты для чистых утилит — без MLX, без сети."""
from lra.utils import extract_ids, jaccard, keyword_set, normalize_query, parse_args


class TestParseArgs:
    def test_passthrough_dict(self):
        assert parse_args({"a": 1}) == {"a": 1}

    def test_valid_json(self):
        assert parse_args('{"query": "hello"}') == {"query": "hello"}

    def test_literal_newline_inside_string(self):
        # JSON не допускает \n внутри строки — наш парсер должен починить
        raw = '{"code": "print(1)\nprint(2)"}'
        out = parse_args(raw)
        assert out["code"] == "print(1)\nprint(2)"

    def test_bare_string_wrapped(self):
        # LLM прислал просто строку вместо объекта
        out = parse_args("hello world")
        assert out == {"content": "hello world"}

    def test_non_string_non_dict_wrapped(self):
        out = parse_args(42)
        assert out == {"content": "42"}

    def test_regex_fallback_for_content(self):
        # Сломанный JSON, но regex-fallback должен вытащить content
        raw = '{"content": "abc}def"}'  # нелегальный, но поймаем regex'ом
        out = parse_args(raw)
        assert "content" in out


class TestNormalizeQuery:
    def test_lower_and_trim(self):
        assert normalize_query("  Hello   WORLD  ") == "hello world"

    def test_multi_whitespace_collapsed(self):
        assert normalize_query("a\t\n  b") == "a b"


class TestCountArxivIds:
    def test_extracts_ids(self):
        text = "See [2301.12345] and [2404.00001] for details."
        assert extract_ids(text) == {"2301.12345", "2404.00001"}

    def test_no_false_positives(self):
        # 3 цифры после точки — не arXiv
        assert extract_ids("version 1.23 and date 2024") == set()

    def test_dedup(self):
        text = "[2301.12345] is cited again as [2301.12345]"
        assert extract_ids(text) == {"2301.12345"}


class TestKeywordSet:
    def test_extracts_words_5plus_chars(self):
        kws = keyword_set("Deep learning model")
        assert "learning" in kws
        assert "model" in kws  # 5 chars
        assert "deep" not in kws  # 4 chars → отсечено

    def test_lowercased(self):
        assert "learning" in keyword_set("LEARNING")


class TestJaccard:
    def test_identical(self):
        assert jaccard({"a", "b"}, {"a", "b"}) == 1.0

    def test_disjoint(self):
        assert jaccard({"a"}, {"b"}) == 0.0

    def test_empty(self):
        assert jaccard(set(), {"a"}) == 0.0

    def test_partial(self):
        # |a∩b| / |a∪b| = 1/3
        assert abs(jaccard({"a", "b"}, {"b", "c"}) - 1 / 3) < 1e-9
