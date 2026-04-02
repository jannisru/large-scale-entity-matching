from large_scale_entity_matching.preprocessing import normalize_key, tokenize_normalized_name


class TestNormalizeKey:
    def test_lowercase_and_special_chars(self):
        assert normalize_key("Apple Inc.") == "apple_inc"

    def test_strips_surrounding_whitespace(self):
        assert normalize_key("  HELLO WORLD  ") == "hello_world"

    def test_none_returns_empty_string(self):
        assert normalize_key(None) == ""

    def test_empty_string_returns_empty_string(self):
        assert normalize_key("") == ""

    def test_collapses_repeated_separators(self):
        assert normalize_key("a--b__c") == "a_b_c"

    def test_preserves_numbers(self):
        assert normalize_key("Company 123") == "company_123"

    def test_leading_trailing_underscores_stripped(self):
        assert normalize_key("...apple...") == "apple"


class TestTokenizeNormalizedName:
    def test_two_tokens(self):
        assert tokenize_normalized_name("apple_inc") == ["apple", "inc"]

    def test_single_token(self):
        assert tokenize_normalized_name("single") == ["single"]

    def test_empty_string_returns_empty_list(self):
        assert tokenize_normalized_name("") == []

    def test_none_returns_empty_list(self):
        assert tokenize_normalized_name(None) == []

    def test_three_tokens(self):
        assert tokenize_normalized_name("john_doe_corp") == ["john", "doe", "corp"]
