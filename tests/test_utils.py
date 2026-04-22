"""Юниты для чистых утилит — без MLX, без сети."""
from lra.utils import extract_ids, get_content, jaccard, keyword_set, normalize_query, parse_args


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

    def test_trailing_tool_call_tag_stripped(self):
        # LLM иногда лепит хвост tool-call wrapper'а внутрь arguments —
        # без pre-clean мы ловили это финальным fallback'ом и писали в файл
        # СЫРОЙ JSON-блоб. Теперь должны получить clean content.
        raw = '{"content": "hello world"}</arguments'
        out = parse_args(raw)
        assert out == {"content": "hello world"}

    def test_trailing_garbage_after_balanced_braces_stripped(self):
        raw = '{"content": "x"}garbage tail'
        out = parse_args(raw)
        assert out == {"content": "x"}

    def test_escaped_quotes_in_content_preserved(self):
        # Regex-fallback должен уважать \" — иначе ломались большие
        # draft append'ы с inline-цитатами.
        raw = r'{"content": "He said \"hi\" today"} extra'
        out = parse_args(raw)
        assert out["content"] == 'He said "hi" today'

    def test_double_encoded_json_unwrapped(self):
        # Qwen иногда оборачивает args в JSON string literal вместо объекта.
        # Без unwrap json5 бы распарсил это в python-строку и мы бы записали
        # сериализованный JSON-блоб в draft.md (виден был в одном прогоне).
        raw = r'"{\"content\": \"hello\"}"'
        out = parse_args(raw)
        assert out == {"content": "hello"}

    def test_triple_encoded_json_unwrapped(self):
        raw = r'"\"{\\\"content\\\": \\\"deep\\\"}\""'
        out = parse_args(raw)
        assert out == {"content": "deep"}


class TestNormalizeQuery:
    def test_lower_and_trim(self):
        assert normalize_query("  Hello   WORLD  ") == "hello world"

    def test_multi_whitespace_collapsed(self):
        assert normalize_query("a\t\n  b") == "a b"


class TestGetContent:
    def test_unwraps_nested_content_blob(self):
        raw = r'{"content": "{\"content\": \"# Clean markdown\"}"}'
        assert get_content(raw) == "# Clean markdown"

    def test_preserves_plain_braces_text(self):
        raw = {"content": "{not a json blob}"}
        assert get_content(raw) == "{not a json blob}"


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

    def test_decimal_in_prose_not_matched(self):
        """Короткие decimal не матчатся (<4 цифр seq)."""
        assert extract_ids("measured 2504.03 mm/s") == set()
        assert extract_ids("version 2.123") == set()
        # секвенс >5 цифр также отсекается
        assert extract_ids("factor 2301.123456 bias") == set()

    def test_invalid_month_not_matched(self):
        """MM=13 невалидный месяц, не arxiv."""
        assert extract_ids("[2013.12345]") == set()
        assert extract_ids("[2399.12345]") == set()

    def test_valid_edge_months(self):
        """MM=01 и MM=12 ok."""
        assert extract_ids("[2401.12345]") == {"2401.12345"}
        assert extract_ids("[2312.12345]") == {"2312.12345"}

    def test_leading_digit_blocks_match(self):
        """Не матчим середину более длинной цифры."""
        assert extract_ids("A12301.12345") == set()


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
