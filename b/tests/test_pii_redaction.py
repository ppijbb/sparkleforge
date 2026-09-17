from src.core.pii_redaction import PIIRedactor, _is_secret_key

def test_secret_key_matching():
    # True positives
    assert _is_secret_key("api_key") is True
    assert _is_secret_key("SECRET") is True
    assert _is_secret_key("nested.auth_token") is True
    assert _is_secret_key("password") is True

    # False positives that previously failed due to substring matching
    assert _is_secret_key("my_api_key_backup") is False
    assert _is_secret_key("tokenizer_config") is False
    assert _is_secret_key("unspecified_key") is False
    assert _is_secret_key("token_count") is False

    # Test actual redaction dictionary
    redactor = PIIRedactor()
    data = {"api_key": "secret123", "my_api_key_backup": "public123", "tokenizer_config": "abc"}
    redacted, _ = redactor.redact_session_data(data)
    assert redacted["api_key"] == "[REDACTED]"
    assert redacted["my_api_key_backup"] == "public123"
    assert redacted["tokenizer_config"] == "abc"
