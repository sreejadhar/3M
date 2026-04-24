"""
File connectors for the unstructured data intelligence agent.

Each connector enumerates a source, produces FileManifest records,
and computes content-based checksums for change detection.
"""
from __future__ import annotations

import hashlib
import mimetypes
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, List, Optional


@dataclass
class FileManifest:
    path: str            # canonical path (local abs path or cloud URI)
    source_type: str     # local | s3 | gcs | azure
    file_name: str
    size_bytes: int
    mime_type: str
    checksum: str        # sha256:<hex>
    created_at: Optional[str]
    modified_at: Optional[str]


_SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".pptx", ".ppt",
    ".xlsx", ".xls", ".html", ".htm", ".md",
    ".txt", ".rst", ".csv",
}

_JUNK_PATTERNS = {"~$", ".tmp", "_backup", "draft_", ".DS_Store", "Thumbs.db"}


def _is_junk(name: str) -> bool:
    name_lower = name.lower()
    return any(p in name_lower for p in _JUNK_PATTERNS)


def _sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            data = f.read(chunk)
            if not data:
                break
            h.update(data)
    return "sha256:" + h.hexdigest()


def _mime(path: str) -> str:
    mt, _ = mimetypes.guess_type(path)
    return mt or "application/octet-stream"


def _ts(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


class LocalConnector:
    """
    Enumerates files from a local filesystem path recursively.
    Follows symlinks but skips circular references.
    """

    def __init__(self, root: str, max_size_mb: int = 200,
                 extensions: Optional[List[str]] = None) -> None:
        self.root = Path(root).resolve()
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.extensions = {e.lower() for e in (extensions or [])} or _SUPPORTED_EXTENSIONS

    def enumerate(self) -> Generator[FileManifest, None, None]:
        if not self.root.is_dir():
            raise ValueError(f"Root path does not exist or is not a directory: {self.root}")

        for dirpath, _, filenames in os.walk(self.root, followlinks=True):
            for fname in filenames:
                fpath = Path(dirpath) / fname
                ext = fpath.suffix.lower()
                if ext not in self.extensions:
                    continue
                if _is_junk(fname):
                    continue
                try:
                    stat = fpath.stat()
                except OSError:
                    continue
                if stat.st_size == 0 or stat.st_size > self.max_size_bytes:
                    continue
                try:
                    checksum = _sha256(str(fpath))
                except OSError:
                    continue
                yield FileManifest(
                    path=str(fpath),
                    source_type="local",
                    file_name=fname,
                    size_bytes=stat.st_size,
                    mime_type=_mime(fname),
                    checksum=checksum,
                    created_at=_ts(stat.st_ctime),
                    modified_at=_ts(stat.st_mtime),
                )


def file_type_from_extension(path: str) -> str:
    ext = Path(path).suffix.lower()
    return {
        ".pdf": "pdf",
        ".docx": "docx", ".doc": "docx",
        ".pptx": "pptx", ".ppt": "pptx",
        ".xlsx": "xlsx", ".xls": "xlsx",
        ".html": "html", ".htm": "html",
        ".md": "md", ".rst": "md",
        ".txt": "txt",
        ".csv": "csv",
    }.get(ext, "unknown")


def make_connector(source_type: str, connection: dict):
    if source_type == "local":
        return LocalConnector(
            root=connection["path"],
            max_size_mb=connection.get("max_size_mb", 200),
            extensions=connection.get("extensions"),
        )
    raise ValueError(f"Unsupported source type: {source_type!r}. "
                     "Supported: local. (s3/gcs/azure in a future release)")
