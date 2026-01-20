"""
Unit Tests for SensitiveDataFilter (PII Guard)

Tests the PII/sensitive data filtering functionality in transparency_context.py.
This ensures that sensitive information like API keys, passwords, SSNs, credit cards,
and other PII is properly redacted before being logged or stored.

v0.2.9: Added comprehensive PII guard test coverage.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest


class TestSensitiveDataFilterText:
    """Tests for SensitiveDataFilter.filter_text()."""

    @pytest.fixture
    def filter_class(self):
        from services.transparency_context import SensitiveDataFilter
        return SensitiveDataFilter

    def test_filter_api_keys(self, filter_class):
        """Should redact API keys in various formats."""
        # api_key format
        text = 'api_key="sk_live_abcdefghij1234567890"'
        result = filter_class.filter_text(text)
        assert "sk_live_abcdefghij1234567890" not in result
        assert "[API_KEY_REDACTED]" in result

        # apikey format
        text = 'apikey: ABCD1234567890EFGH1234567890'
        result = filter_class.filter_text(text)
        assert "[API_KEY_REDACTED]" in result

    def test_filter_bearer_tokens(self, filter_class):
        """Should redact Bearer tokens."""
        text = 'Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test'
        result = filter_class.filter_text(text)
        assert "Bearer [REDACTED]" in result

    def test_filter_passwords(self, filter_class):
        """Should redact passwords."""
        test_cases = [
            'password="secret123"',
            'passwd: mypassword',
            'pwd=supersecret',
        ]
        for text in test_cases:
            result = filter_class.filter_text(text)
            assert "[PASSWORD_REDACTED]" in result

    def test_filter_aws_keys(self, filter_class):
        """Should redact AWS access keys."""
        text = 'AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE'
        result = filter_class.filter_text(text)
        assert "[AWS_ACCESS_KEY_REDACTED]" in result

    def test_filter_private_keys(self, filter_class):
        """Should redact private keys."""
        text = '''-----BEGIN RSA PRIVATE KEY-----
MIIEowIBAAKCAQEAtest
-----END RSA PRIVATE KEY-----'''
        result = filter_class.filter_text(text)
        assert "[PRIVATE_KEY_REDACTED]" in result

    def test_filter_credit_cards(self, filter_class):
        """Should redact credit card numbers."""
        test_cases = [
            "4111111111111111",  # Visa
            "5500-0000-0000-0004",  # Mastercard with dashes
            "4111 1111 1111 1111",  # Visa with spaces
        ]
        for text in test_cases:
            result = filter_class.filter_text(text)
            assert "[CREDIT_CARD_REDACTED]" in result

    def test_filter_ssn(self, filter_class):
        """Should redact Social Security Numbers."""
        text = "User SSN: 123-45-6789"
        result = filter_class.filter_text(text)
        assert "[SSN_REDACTED]" in result
        assert "123-45-6789" not in result

    def test_filter_jwt_tokens(self, filter_class):
        """Should redact JWT tokens (may be caught by TOKEN pattern first)."""
        jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        text = f"Token: {jwt}"
        result = filter_class.filter_text(text)
        # JWT tokens are redacted - may be caught by TOKEN, JWT, or AWS_SECRET pattern
        # The key check is that sensitive parts are redacted (contains "REDACTED")
        assert "REDACTED" in result
        # Original JWT should not be fully present
        assert jwt not in result

    def test_filter_database_urls(self, filter_class):
        """Should redact database connection strings."""
        test_cases = [
            "mongodb://user:pass@localhost:27017/db",
            "postgres://user:pass@host:5432/database",
            "mysql://root:secret@localhost/app",
            "redis://localhost:6379",
        ]
        for text in test_cases:
            result = filter_class.filter_text(text)
            assert "[DB_CONNECTION_REDACTED]" in result

    def test_preserves_non_sensitive_text(self, filter_class):
        """Should preserve text that doesn't contain sensitive data."""
        text = "User prefers dark mode and uses pytest for testing"
        result = filter_class.filter_text(text)
        assert result == text

    def test_handles_empty_text(self, filter_class):
        """Should handle empty or None input."""
        assert filter_class.filter_text("") == ""
        assert filter_class.filter_text(None) is None


class TestSensitiveDataFilterDict:
    """Tests for SensitiveDataFilter.filter_dict()."""

    @pytest.fixture
    def filter_class(self):
        from services.transparency_context import SensitiveDataFilter
        return SensitiveDataFilter

    def test_redacts_sensitive_keys(self, filter_class):
        """Should redact values for keys that suggest sensitive data."""
        data = {
            "password": "secret123",
            "api_key": "sk_live_abc123",
            "token": "eyJtoken",
            "auth_header": "Bearer xyz",
            "credential": "user:pass",
            "username": "john_doe"  # Not sensitive - doesn't contain sensitive terms
        }
        result = filter_class.filter_dict(data)

        assert result["password"] == "[REDACTED]"
        assert result["api_key"] == "[REDACTED]"
        assert result["token"] == "[REDACTED]"
        assert result["auth_header"] == "[REDACTED]"
        assert result["credential"] == "[REDACTED]"
        assert result["username"] == "john_doe"  # Should be preserved

    def test_filters_nested_dicts(self, filter_class):
        """Should recursively filter nested dictionaries."""
        data = {
            "config": {
                "database_url": "postgres://user:pass@localhost/db",
                "name": "myapp"
            }
        }
        result = filter_class.filter_dict(data)

        assert "[DB_CONNECTION_REDACTED]" in result["config"]["database_url"]
        assert result["config"]["name"] == "myapp"

    def test_filters_lists_in_dicts(self, filter_class):
        """Should filter string items in lists (for non-sensitive keys)."""
        data = {
            # Note: "tokens" key would be redacted because it contains "token"
            # Using "messages" which doesn't trigger key-based redaction
            "messages": ["Bearer token123", "normal_string"],
            "values": [1, 2, 3]  # Non-strings should be preserved
        }
        result = filter_class.filter_dict(data)

        # Bearer tokens in list items should be filtered
        assert "Bearer [REDACTED]" in result["messages"][0]
        assert result["messages"][1] == "normal_string"
        assert result["values"] == [1, 2, 3]

    def test_sensitive_key_redacts_entire_value(self, filter_class):
        """Keys containing sensitive terms should redact entire value, not filter content."""
        data = {
            "tokens": ["Bearer token123", "value2"],  # "tokens" contains "token"
            "api_keys": {"nested": "value"}  # "api_keys" contains "key"
        }
        result = filter_class.filter_dict(data)

        # Entire values are redacted for sensitive keys, not filtered
        assert result["tokens"] == "[REDACTED]"
        assert result["api_keys"] == "[REDACTED]"

    def test_handles_empty_dict(self, filter_class):
        """Should handle empty dict."""
        assert filter_class.filter_dict({}) == {}
        assert filter_class.filter_dict(None) is None


class TestSensitiveDataFilterIntegration:
    """Integration tests for SensitiveDataFilter in context."""

    @pytest.fixture
    def filter_class(self):
        from services.transparency_context import SensitiveDataFilter
        return SensitiveDataFilter

    def test_mixed_sensitive_data(self, filter_class):
        """Should handle text with multiple types of sensitive data."""
        text = """
        Database: postgres://admin:password123@db.example.com/prod
        API Key: api_key=sk_live_1234567890abcdef1234
        User SSN: 123-45-6789
        """
        result = filter_class.filter_text(text)

        assert "[DB_CONNECTION_REDACTED]" in result
        assert "[API_KEY_REDACTED]" in result
        assert "[SSN_REDACTED]" in result
        assert "password123" not in result
        assert "123-45-6789" not in result

    def test_case_insensitive_key_detection(self, filter_class):
        """Should detect sensitive keys regardless of case."""
        data = {
            "PASSWORD": "secret",
            "Api_Key": "abc123xyz456789012345",
            "TOKEN": "token123"
        }
        result = filter_class.filter_dict(data)

        assert result["PASSWORD"] == "[REDACTED]"
        assert result["Api_Key"] == "[REDACTED]"
        assert result["TOKEN"] == "[REDACTED]"

    def test_partial_sensitive_key_match(self, filter_class):
        """Should detect keys containing sensitive terms."""
        data = {
            "database_password": "secret",
            "user_api_key": "key123",
            "auth_token_header": "bearer_token"
        }
        result = filter_class.filter_dict(data)

        assert result["database_password"] == "[REDACTED]"
        assert result["user_api_key"] == "[REDACTED]"
        assert result["auth_token_header"] == "[REDACTED]"


class TestSensitiveDataFilterEdgeCases:
    """Edge case tests for SensitiveDataFilter."""

    @pytest.fixture
    def filter_class(self):
        from services.transparency_context import SensitiveDataFilter
        return SensitiveDataFilter

    def test_short_numbers_not_credit_cards(self, filter_class):
        """Short numbers should not be flagged as credit cards."""
        text = "Order #12345"
        result = filter_class.filter_text(text)
        assert result == text  # Should be unchanged

    def test_non_ssn_number_patterns(self, filter_class):
        """Non-SSN patterns with dashes should be preserved."""
        text = "Phone: 555-1234"  # Too short for SSN
        result = filter_class.filter_text(text)
        assert "555-1234" in result

    def test_preserves_code_snippets(self, filter_class):
        """Should preserve code that looks like but isn't sensitive."""
        text = "def api_handler(): return 'ok'"
        result = filter_class.filter_text(text)
        assert "api_handler" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
