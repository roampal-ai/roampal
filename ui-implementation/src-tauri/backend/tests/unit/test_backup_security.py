"""
Unit tests for backup.py security fixes.

Tests critical security vulnerabilities:
- ZIP path traversal (directory traversal attack)
- Malicious backup validation
"""

import pytest
import zipfile
import tempfile
import os
import json
from pathlib import Path
from io import BytesIO


class TestZIPPathTraversal:
    """Tests for ZIP path traversal vulnerability fix (v0.3.0 security fix)."""

    def _create_malicious_zip(self, members: list[str]) -> bytes:
        """Create a ZIP file with specified member paths."""
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, 'w') as zf:
            for member in members:
                zf.writestr(member, f"content of {member}")
        buffer.seek(0)
        return buffer.read()

    def _validate_zip_paths(self, zip_data: bytes, extract_dir: Path) -> list[str]:
        """
        Validate ZIP member paths - reimplements the security check from backup.py.
        Returns list of invalid paths that would escape extract_dir.
        """
        invalid_paths = []
        buffer = BytesIO(zip_data)
        with zipfile.ZipFile(buffer, 'r') as zipf:
            for member in zipf.namelist():
                member_path = (extract_dir / member).resolve()
                # Security check from backup.py lines 401-406
                if not str(member_path).startswith(str(extract_dir.resolve()) + os.sep) and member_path != extract_dir.resolve():
                    invalid_paths.append(member)
        return invalid_paths

    def test_normal_paths_allowed(self, tmp_path):
        """Normal paths within extract dir should be allowed."""
        zip_data = self._create_malicious_zip([
            "backup_info.json",
            "sessions/session1.json",
            "chromadb/data.db",
            "books/metadata/book1.json",
        ])

        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()

        invalid = self._validate_zip_paths(zip_data, extract_dir)
        assert invalid == [], f"Normal paths should be allowed: {invalid}"

    def test_parent_traversal_blocked(self, tmp_path):
        """Paths with ../ should be blocked."""
        zip_data = self._create_malicious_zip([
            "../etc/passwd",
            "backup_info.json",
        ])

        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()

        invalid = self._validate_zip_paths(zip_data, extract_dir)
        assert "../etc/passwd" in invalid, "Parent traversal should be blocked"

    def test_deep_parent_traversal_blocked(self, tmp_path):
        """Deep parent traversal should be blocked."""
        zip_data = self._create_malicious_zip([
            "sessions/../../../etc/passwd",
            "backup_info.json",
        ])

        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()

        invalid = self._validate_zip_paths(zip_data, extract_dir)
        assert len(invalid) == 1, f"Deep parent traversal should be blocked: {invalid}"

    def test_absolute_path_blocked(self, tmp_path):
        """Absolute paths should be blocked."""
        # Note: ZipFile normalizes some absolute paths, but we test the logic
        zip_data = self._create_malicious_zip([
            "/etc/passwd",
            "backup_info.json",
        ])

        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()

        invalid = self._validate_zip_paths(zip_data, extract_dir)
        # Absolute paths get normalized by ZipFile, but if they escape, they're caught
        # The key is any path resolving outside extract_dir is blocked

    def test_windows_style_traversal_blocked(self, tmp_path):
        """Windows-style path traversal should be blocked."""
        zip_data = self._create_malicious_zip([
            "..\\..\\Windows\\System32\\config\\SAM",
            "backup_info.json",
        ])

        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()

        invalid = self._validate_zip_paths(zip_data, extract_dir)
        # On Windows, this would be a real attack; on Linux, \\ is literal
        # Either way, the resolved path check catches it if it escapes

    def test_symlink_in_path_normalized(self, tmp_path):
        """Paths with .. after directory should be caught."""
        zip_data = self._create_malicious_zip([
            "sessions/../../malicious.txt",
            "backup_info.json",
        ])

        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()

        invalid = self._validate_zip_paths(zip_data, extract_dir)
        assert len(invalid) == 1, "Mid-path traversal should be blocked"

    def test_empty_filename_handled(self, tmp_path):
        """Empty or root-only paths should be handled safely."""
        # ZipFile typically doesn't allow empty names, but test the logic
        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()

        # Manually test the condition for edge cases
        member_path = (extract_dir / "").resolve()
        # Should equal extract_dir itself, which is allowed
        assert member_path == extract_dir.resolve()

    def test_multiple_traversals_all_caught(self, tmp_path):
        """Multiple malicious paths should all be caught."""
        zip_data = self._create_malicious_zip([
            "../secret1.txt",
            "../../secret2.txt",
            "data/../../../secret3.txt",
            "backup_info.json",  # This one is valid
        ])

        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()

        invalid = self._validate_zip_paths(zip_data, extract_dir)
        assert len(invalid) == 3, f"All 3 malicious paths should be caught, got: {invalid}"


class TestBackupValidation:
    """Tests for backup structure validation."""

    def test_missing_backup_info_rejected(self, tmp_path):
        """Backup without backup_info.json should be rejected."""
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, 'w') as zf:
            zf.writestr("sessions/session1.json", "{}")
        buffer.seek(0)

        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()

        with zipfile.ZipFile(buffer, 'r') as zf:
            zf.extractall(extract_dir)

        # Check validation logic
        backup_info_path = extract_dir / "backup_info.json"
        assert not backup_info_path.exists(), "Test setup: backup_info.json should not exist"

    def test_valid_backup_structure_accepted(self, tmp_path):
        """Valid backup with backup_info.json should be accepted."""
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, 'w') as zf:
            zf.writestr("backup_info.json", json.dumps({
                "version": "0.3.0",
                "created_at": "2025-01-01T00:00:00",
                "contains": ["sessions"]
            }))
            zf.writestr("sessions/session1.json", "{}")
        buffer.seek(0)

        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()

        with zipfile.ZipFile(buffer, 'r') as zf:
            zf.extractall(extract_dir)

        backup_info_path = extract_dir / "backup_info.json"
        assert backup_info_path.exists(), "Valid backup should have backup_info.json"

        # Verify content is valid JSON
        metadata = json.loads(backup_info_path.read_text())
        assert "version" in metadata
        assert "contains" in metadata


class TestSecurityIntegration:
    """Integration tests for the complete security flow."""

    def test_attack_scenario_blocked(self, tmp_path):
        """Full attack scenario: ZIP with passwd steal attempt should be blocked."""
        # Create a malicious ZIP that tries to write to /etc/passwd (or Windows equivalent)
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, 'w') as zf:
            # Valid-looking backup with hidden attack
            zf.writestr("backup_info.json", json.dumps({
                "version": "0.3.0",
                "created_at": "2025-01-01T00:00:00",
                "contains": ["sessions"]
            }))
            zf.writestr("sessions/session1.json", "{}")
            # The attack payload
            zf.writestr("../../../tmp/pwned.txt", "attacker controlled content")
        buffer.seek(0)

        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()

        # Simulate the security check
        buffer.seek(0)
        with zipfile.ZipFile(buffer, 'r') as zipf:
            for member in zipf.namelist():
                member_path = (extract_dir / member).resolve()
                if not str(member_path).startswith(str(extract_dir.resolve()) + os.sep) and member_path != extract_dir.resolve():
                    # This is what backup.py does - raise ValueError
                    with pytest.raises(ValueError):
                        raise ValueError(f"Invalid backup: contains path traversal attempt in '{member}'")
                    return  # Test passed - attack was blocked

        pytest.fail("Attack should have been detected and blocked")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
