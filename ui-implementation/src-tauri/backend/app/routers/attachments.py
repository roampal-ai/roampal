"""v0.3.3 Section 4 Defect 4: attachment bytes-serving endpoint.

The frontend stores image attachments by filename in the session JSONL; on
session reload it constructs URLs like `http://localhost:<port>/api/attachments/<filename>`
that the `<img>` tag fetches. This router serves those bytes back.

Filenames are content-addressed (`<sha256>.<ext>`) so there's no PII or auth
concern in serving them — but defensive path-traversal checks still apply
inside `resolve_attachment_path` to keep the endpoint robust.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from config.settings import DATA_PATH
from utils.attachments import (
    AttachmentError,
    extension_to_mime,
    resolve_attachment_path,
)

logger = logging.getLogger(__name__)

router = APIRouter()

_ATTACHMENTS_DIR = Path(DATA_PATH) / "attachments"


@router.get("/api/attachments/{filename}")
async def get_attachment(filename: str):
    """Serve a stored image attachment by filename.

    Returns 404 if the file does not exist (or never did).
    Returns 400 if the filename fails validation (path traversal, etc.).
    """
    try:
        resolved = resolve_attachment_path(filename, _ATTACHMENTS_DIR)
    except AttachmentError as exc:
        logger.warning(f"Attachment request rejected: {exc} (filename={filename!r})")
        raise HTTPException(status_code=400, detail=str(exc))

    if resolved is None:
        raise HTTPException(status_code=404, detail="Attachment not found")

    return FileResponse(
        path=resolved,
        media_type=extension_to_mime(filename),
        # Cache aggressively — content is immutable (hash-keyed).
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )
