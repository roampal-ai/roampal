"""Unit tests for atomic JSON write helper."""

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add backend to path (mirrors other test files)
backend_dir = Path(__file__).parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from utils.atomic_json import write_json_atomic


@pytest.fixture
def tmp_path_factory(tmp_path):
    """Return a temporary directory path for test isolation."""
    return tmp_path / "atomic_test"


class TestWriteJsonAtomic:

    def test_write_json_atomic_creates_file_on_fresh_path(self, tmp_path_factory):
        """Should create the file when it doesn't exist yet."""
        target = tmp_path_factory / "new.json"
        assert not target.exists()

        write_json_atomic(target, {"key": "value"})

        assert target.exists()
        data = json.loads(target.read_text())
        assert data == {"key": "value"}

    def test_write_json_atomic_replaces_existing_file(self, tmp_path_factory):
        """Should replace existing file contents."""
        target = tmp_path_factory / "existing.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"old": True}))

        write_json_atomic(target, {"new": True})

        data = json.loads(target.read_text())
        assert data == {"new": True}
        assert "old" not in data

    def test_write_json_atomic_leaves_no_tmp_files_on_success(self, tmp_path_factory):
        """Should clean up temp files after successful write."""
        target = tmp_path_factory / "clean.json"
        target.parent.mkdir(parents=True, exist_ok=True)

        write_json_atomic(target, {"data": 123})

        # No .tmp files should remain in the directory
        tmp_files = list(tmp_path_factory.glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_write_json_atomic_preserves_original_on_exception(self, tmp_path_factory):
        """If os.replace raises, original file must be byte-for-byte unchanged."""
        target = tmp_path_factory / "protected.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        original_content = json.dumps({"safe": True})
        target.write_text(original_content)

        # Monkeypatch os.replace to raise mid-operation
        with patch("os.replace", side_effect=OSError("simulated disk failure")):
            with pytest.raises(OSError, match="simulated disk failure"):
                write_json_atomic(target, {"corrupted": True})

        assert target.read_text() == original_content

    def test_write_json_atomic_creates_parent_dirs(self):
        """Should create parent directories that don't exist."""
        import tempfile
        base = Path(tempfile.gettempdir()) / "roampal_atomic_test_unique"
        target = base / "deeply" / "nested" / "data.json"

        try:
            write_json_atomic(target, {"created": True})

            assert target.exists()
            data = json.loads(target.read_text())
            assert data == {"created": True}
        finally:
            # Cleanup
            import shutil
            if base.exists():
                shutil.rmtree(base)
