"""Image attachment persistence — v0.3.3 Section 4 Defect 4 fix.

Stores user-attached image bytes as files in `<DATA_PATH>/attachments/`, keyed
by SHA-256 of the raw bytes. Session JSONL records only the filename; the
bytes endpoint at `/api/attachments/<filename>` streams them back to the
frontend on session reload.

Content-addressing gives automatic dedup (same image attached twice → one
file) and crash-safe writes via the atomic-JSON pattern.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


# Mapping from canonical image MIME types to filename extensions.
# Only image types are accepted by the save path — non-image data URLs raise.
_MIME_TO_EXT = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
}

_DATA_URL_RE = re.compile(r"^data:([\w./+-]+);base64,(.*)$", re.DOTALL)


class AttachmentError(ValueError):
    """Raised when a data URL can't be parsed or saved."""


def _parse_data_url(data_url: str) -> Tuple[str, bytes]:
    """Extract (mime, raw_bytes) from a `data:image/...;base64,...` URL.

    Raises AttachmentError on malformed input or non-image MIME.
    """
    if not isinstance(data_url, str):
        raise AttachmentError(f"Expected str, got {type(data_url).__name__}")
    match = _DATA_URL_RE.match(data_url)
    if not match:
        raise AttachmentError("Not a base64 data URL")
    mime = match.group(1).lower()
    if mime not in _MIME_TO_EXT:
        raise AttachmentError(f"Unsupported image MIME type: {mime}")
    try:
        raw = base64.b64decode(match.group(2), validate=False)
    except (ValueError, TypeError) as exc:
        raise AttachmentError(f"base64 decode failed: {exc}") from exc
    if not raw:
        raise AttachmentError("Empty image payload after decode")
    return mime, raw


def save_image_attachment(data_url: str, attachments_dir: Path) -> str:
    """Persist a base64 image data URL to disk; return the relative filename.

    Filename format: `<sha256_hex>.<ext>`. Identical inputs produce identical
    filenames (content-addressed dedup). Existing files are not overwritten.

    Args:
        data_url: A `data:image/...;base64,...` string from the frontend.
        attachments_dir: Target directory (typically `<DATA_PATH>/attachments`).

    Returns:
        The filename (no directory prefix) suitable for storage in the session
        JSONL and for the bytes-endpoint path.

    Raises:
        AttachmentError: on parse, decode, or write failure.
    """
    mime, raw = _parse_data_url(data_url)
    ext = _MIME_TO_EXT[mime]
    digest = hashlib.sha256(raw).hexdigest()
    filename = f"{digest}{ext}"

    attachments_dir.mkdir(parents=True, exist_ok=True)
    final_path = attachments_dir / filename

    if final_path.exists():
        # Content-addressed dedup: identical bytes already on disk.
        return filename

    # Atomic write: temp file + rename, mirrors utils/atomic_json.py pattern.
    fd, tmp_name = tempfile.mkstemp(
        dir=str(attachments_dir),
        prefix=f".{digest}",
        suffix=ext + ".tmp",
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
        os.replace(tmp_name, final_path)
    except Exception:
        # Best effort cleanup if the rename never happened.
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise

    logger.info(
        f"Saved image attachment {filename} ({len(raw)} bytes, {mime})"
    )
    return filename


def resolve_attachment_path(filename: str, attachments_dir: Path) -> Optional[Path]:
    """Validate a filename and return the absolute path if it exists.

    Rejects path traversal (`..`, separators) and unsupported extensions.
    Returns None if the file does not exist; raises AttachmentError on
    malformed input that should never have reached the endpoint.
    """
    if not isinstance(filename, str) or not filename:
        raise AttachmentError("Empty filename")
    # Path traversal guard — filename must be a flat hex hash + ext.
    if "/" in filename or "\\" in filename or ".." in filename or filename.startswith("."):
        raise AttachmentError(f"Invalid filename: {filename!r}")
    # Cheap shape check — sha256 hex is 64 chars; allow any of our extensions.
    stem, dot, ext = filename.rpartition(".")
    if not dot:
        raise AttachmentError("Filename missing extension")
    if f".{ext}" not in _MIME_TO_EXT.values():
        raise AttachmentError(f"Unsupported attachment extension: .{ext}")
    if not stem or not all(c in "0123456789abcdef" for c in stem.lower()):
        raise AttachmentError("Filename stem must be hex digest")

    candidate = (attachments_dir / filename).resolve()
    # Confirm resolved path is inside attachments_dir (defense in depth).
    try:
        candidate.relative_to(attachments_dir.resolve())
    except ValueError:
        raise AttachmentError("Resolved path escapes attachments directory")

    return candidate if candidate.exists() else None


def load_image_attachment_as_data_url(
    filename: str, attachments_dir: Path
) -> Optional[str]:
    """Read a saved image file and re-encode it as a base64 data URL.

    v0.3.3 Defect 12: session JSONL stores only `<hash>.<ext>` filenames
    (for crash-safe persistence — see save_image_attachment). The LLM
    history-replay path in modules/llm/ollama_client.py:702-743 expects
    `_h["images"]` entries to be full `data:image/...;base64,...` URLs
    so it can emit them in OpenAI content_blocks (LM Studio) or strip
    the prefix for Ollama. Without rehydration the replay path passes
    raw filenames to LM Studio which rejects with HTTP 400 "Invalid url."

    Returns None if the file is missing, the filename is malformed, or
    the read fails — the caller is expected to drop the entry from the
    `images` list rather than abort the entire conversation reload.
    """
    try:
        path = resolve_attachment_path(filename, attachments_dir)
    except AttachmentError as exc:
        logger.warning(f"Invalid attachment filename {filename!r}: {exc}")
        return None
    if path is None:
        logger.warning(f"Attachment file missing: {filename}")
        return None
    try:
        raw = path.read_bytes()
    except OSError as exc:
        logger.warning(f"Failed to read attachment {filename}: {exc}")
        return None
    mime = extension_to_mime(filename)
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def extension_to_mime(filename: str) -> str:
    """Return the image MIME type implied by a stored filename's extension."""
    _, dot, ext = filename.rpartition(".")
    if not dot:
        return "application/octet-stream"
    dotted = f".{ext.lower()}"
    for mime, mapped_ext in _MIME_TO_EXT.items():
        if mapped_ext == dotted:
            return mime
    return "application/octet-stream"
