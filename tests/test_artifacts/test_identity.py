from artifacts.identity import content_identity


def test_content_identity_is_order_independent_for_mappings() -> None:
    assert content_identity({"a": 1, "b": 2}) == content_identity({"b": 2, "a": 1})
