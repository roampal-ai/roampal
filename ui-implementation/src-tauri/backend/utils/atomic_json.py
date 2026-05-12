"""Atomic JSON write with temp-file + rename. Crash-safe on Linux
and Windows NTFS. Mirrors roampal-core v0.5.3 Section 10's helper
of the same name; code duplicated intentionally to keep the two
codebases decoupled.
"""
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def write_json_atomic(path: Path, data: Any, *, indent: int | None = 2) -> None:
    """Write `data` as JSON to `path` atomically.

    Writes to a sibling .tmp file first, then os.replace()s into place.
    If any exception is raised during the write, the temp file is
    removed and the original `path` is left untouched.
    """
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(dir=str(parent), suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent)
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise
