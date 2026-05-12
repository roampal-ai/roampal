"""Tests for utils/attachments — v0.3.3 §4 Defect 4 image persistence helpers."""

import base64
import hashlib
from pathlib import Path

import pytest

from utils.attachments import (
    AttachmentError,
    extension_to_mime,
    resolve_attachment_path,
    save_image_attachment,
)


# A 1×1 transparent PNG as raw bytes (the smallest valid PNG).
_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgAAIAAAUAAeImBZsAAAAASUVORK5CYII="
)
_PNG_DATA_URL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgAAIAAAUAAeImBZsAAAAASUVORK5CYII="
_PNG_SHA256 = hashlib.sha256(_PNG_BYTES).hexdigest()

_JPEG_DATA_URL = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFAEBAAAAAAAAAAAAAAAAAAAAAP/EABQRAQAAAAAAAAAAAAAAAAAAAAD/2gAMAwEAAhEDEQA/AKpgAH//2Q=="


class TestSaveImageAttachment:

    def test_writes_file_with_hash_keyed_name(self, tmp_path):
        attachments = tmp_path / "attachments"
        name = save_image_attachment(_PNG_DATA_URL, attachments)
        assert name == f"{_PNG_SHA256}.png"
        on_disk = attachments / name
        assert on_disk.exists()
        assert on_disk.read_bytes() == _PNG_BYTES

    def test_dedup_identical_content(self, tmp_path):
        attachments = tmp_path / "attachments"
        a = save_image_attachment(_PNG_DATA_URL, attachments)
        b = save_image_attachment(_PNG_DATA_URL, attachments)
        # Same bytes → same filename; only one file on disk.
        assert a == b
        files = [p for p in attachments.iterdir() if not p.name.startswith(".")]
        assert len(files) == 1

    def test_creates_attachments_dir_if_missing(self, tmp_path):
        target = tmp_path / "nope" / "still-nope" / "attachments"
        assert not target.exists()
        name = save_image_attachment(_PNG_DATA_URL, target)
        assert (target / name).exists()

    def test_jpeg_data_url_maps_to_jpg_extension(self, tmp_path):
        attachments = tmp_path / "attachments"
        name = save_image_attachment(_JPEG_DATA_URL, attachments)
        assert name.endswith(".jpg")

    def test_rejects_non_image_mime(self, tmp_path):
        attachments = tmp_path / "attachments"
        with pytest.raises(AttachmentError):
            save_image_attachment("data:text/plain;base64,aGVsbG8=", attachments)

    def test_rejects_malformed_data_url(self, tmp_path):
        attachments = tmp_path / "attachments"
        with pytest.raises(AttachmentError):
            save_image_attachment("not a data url", attachments)

    def test_rejects_empty_payload(self, tmp_path):
        attachments = tmp_path / "attachments"
        with pytest.raises(AttachmentError):
            save_image_attachment("data:image/png;base64,", attachments)

    def test_rejects_non_string(self, tmp_path):
        with pytest.raises(AttachmentError):
            save_image_attachment(None, tmp_path / "attachments")  # type: ignore[arg-type]

    def test_does_not_leave_tmp_files_on_success(self, tmp_path):
        attachments = tmp_path / "attachments"
        save_image_attachment(_PNG_DATA_URL, attachments)
        stray = [p for p in attachments.iterdir() if p.name.startswith(".") or p.suffix == ".tmp"]
        assert stray == []


class TestResolveAttachmentPath:

    def test_returns_path_for_existing_file(self, tmp_path):
        attachments = tmp_path / "attachments"
        name = save_image_attachment(_PNG_DATA_URL, attachments)
        resolved = resolve_attachment_path(name, attachments)
        assert resolved is not None
        assert resolved.read_bytes() == _PNG_BYTES

    def test_returns_none_for_missing_file(self, tmp_path):
        attachments = tmp_path / "attachments"
        attachments.mkdir()
        # Valid filename shape but file doesn't exist.
        missing = f"{'a' * 64}.png"
        assert resolve_attachment_path(missing, attachments) is None

    def test_rejects_path_traversal_with_dotdot(self, tmp_path):
        attachments = tmp_path / "attachments"
        with pytest.raises(AttachmentError):
            resolve_attachment_path("../etc/passwd.png", attachments)

    def test_rejects_path_separator(self, tmp_path):
        attachments = tmp_path / "attachments"
        with pytest.raises(AttachmentError):
            resolve_attachment_path("subdir/file.png", attachments)

    def test_rejects_unsupported_extension(self, tmp_path):
        attachments = tmp_path / "attachments"
        with pytest.raises(AttachmentError):
            resolve_attachment_path(f"{'a' * 64}.exe", attachments)

    def test_rejects_non_hex_stem(self, tmp_path):
        attachments = tmp_path / "attachments"
        with pytest.raises(AttachmentError):
            resolve_attachment_path("notahash.png", attachments)

    def test_rejects_empty(self, tmp_path):
        attachments = tmp_path / "attachments"
        with pytest.raises(AttachmentError):
            resolve_attachment_path("", attachments)


class TestExtensionToMime:
    def test_png(self):
        assert extension_to_mime("abc.png") == "image/png"

    def test_jpg(self):
        assert extension_to_mime("abc.jpg") == "image/jpeg"

    def test_unknown_extension(self):
        assert extension_to_mime("abc.xyz") == "application/octet-stream"

    def test_no_extension(self):
        assert extension_to_mime("abc") == "application/octet-stream"
