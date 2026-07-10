"""
Connectors for the Document Intelligence agent.

Each connector enumerates files from a repository and yields FileManifest
records. Cloud SDKs (boto3, google-api-python-client, msal) are imported
lazily inside each connector so the service still starts and lists sources
even when a given SDK isn't installed — only the connectors that need it
fail, with a clear error message, when actually used.

Supported connectors:
  local      — local filesystem path
  s3         — Amazon S3 bucket/prefix
  gdrive     — Google Drive folder (OAuth2)
  sharepoint — SharePoint Online document library (Microsoft Graph API)
  onedrive   — OneDrive (Microsoft Graph API)
"""
from __future__ import annotations

import hashlib
import logging
import mimetypes
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Optional

logger = logging.getLogger(__name__)

CONNECTOR_TYPES = ("local", "s3", "gdrive", "sharepoint", "onedrive")

_SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".pptx", ".ppt",
    ".xlsx", ".xls", ".html", ".htm", ".md", ".txt", ".csv",
}


@dataclass
class FileManifest:
    remote_id: str        # connector-specific unique id (path, S3 key, Drive file id, Graph item id)
    file_name: str
    size_bytes: int
    mime_type: str
    modified_at: Optional[str]
    checksum: Optional[str] = None   # sha256 for local files; etag/version for cloud


def _mime(name: str) -> str:
    mt, _ = mimetypes.guess_type(name)
    return mt or "application/octet-stream"


def _is_supported(name: str) -> bool:
    return Path(name).suffix.lower() in _SUPPORTED_EXTENSIONS


class ConnectorError(RuntimeError):
    pass


class LocalConnector:
    """Enumerates files under a local filesystem path, recursively."""

    def __init__(self, config: dict):
        self.root = config.get("root_path", "")
        if not self.root:
            raise ConnectorError("local connector requires 'root_path'")

    def list_files(self) -> Generator[FileManifest, None, None]:
        root = Path(self.root)
        if not root.exists():
            raise ConnectorError(f"path does not exist: {self.root}")
        for path in root.rglob("*"):
            if not path.is_file() or not _is_supported(path.name):
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            yield FileManifest(
                remote_id=str(path),
                file_name=path.name,
                size_bytes=stat.st_size,
                mime_type=_mime(path.name),
                modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                checksum=self._sha256(path),
            )

    @staticmethod
    def _sha256(path: Path, chunk: int = 1 << 20) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                data = f.read(chunk)
                if not data:
                    break
                h.update(data)
        return "sha256:" + h.hexdigest()

    def read_bytes(self, remote_id: str) -> bytes:
        return Path(remote_id).read_bytes()


class S3Connector:
    """Enumerates objects under an S3 bucket/prefix."""

    def __init__(self, config: dict):
        self.bucket = config.get("bucket", "")
        self.prefix = config.get("prefix", "")
        self.region = config.get("region") or None
        self.access_key = config.get("aws_access_key_id")
        self.secret_key = config.get("aws_secret_access_key")
        if not self.bucket:
            raise ConnectorError("s3 connector requires 'bucket'")

    def _client(self):
        try:
            import boto3
        except ImportError:
            raise ConnectorError("boto3 is not installed — run `pip install boto3` to use the S3 connector")
        kwargs = {}
        if self.region:
            kwargs["region_name"] = self.region
        if self.access_key and self.secret_key:
            kwargs["aws_access_key_id"] = self.access_key
            kwargs["aws_secret_access_key"] = self.secret_key
        return boto3.client("s3", **kwargs)

    def list_files(self) -> Generator[FileManifest, None, None]:
        client = self._client()
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get("Contents", []):
                name = obj["Key"].rsplit("/", 1)[-1]
                if not name or not _is_supported(name):
                    continue
                yield FileManifest(
                    remote_id=obj["Key"],
                    file_name=name,
                    size_bytes=obj["Size"],
                    mime_type=_mime(name),
                    modified_at=obj["LastModified"].astimezone(timezone.utc).isoformat(),
                    checksum=obj.get("ETag", "").strip('"'),
                )

    def read_bytes(self, remote_id: str) -> bytes:
        client = self._client()
        return client.get_object(Bucket=self.bucket, Key=remote_id)["Body"].read()


class GoogleDriveConnector:
    """Enumerates files in a Google Drive folder via OAuth2 access token."""

    def __init__(self, config: dict):
        self.folder_id = config.get("folder_id", "")
        self.access_token = config.get("access_token", "")
        if not self.folder_id:
            raise ConnectorError("gdrive connector requires 'folder_id'")
        if not self.access_token:
            raise ConnectorError("gdrive connector requires 'access_token' (complete Google OAuth first)")

    def _client(self):
        try:
            import httpx
        except ImportError:
            raise ConnectorError("httpx is required for the Google Drive connector")
        return httpx.Client(
            base_url="https://www.googleapis.com/drive/v3",
            headers={"Authorization": f"Bearer {self.access_token}"},
            timeout=30.0,
        )

    def list_files(self) -> Generator[FileManifest, None, None]:
        with self._client() as client:
            page_token = None
            while True:
                params = {
                    "q": f"'{self.folder_id}' in parents and trashed = false",
                    "fields": "nextPageToken, files(id,name,size,mimeType,modifiedTime,md5Checksum)",
                    "pageSize": 200,
                }
                if page_token:
                    params["pageToken"] = page_token
                resp = client.get("/files", params=params)
                if resp.status_code != 200:
                    raise ConnectorError(f"Google Drive API error: {resp.status_code} {resp.text[:200]}")
                data = resp.json()
                for f in data.get("files", []):
                    if not _is_supported(f["name"]):
                        continue
                    yield FileManifest(
                        remote_id=f["id"],
                        file_name=f["name"],
                        size_bytes=int(f.get("size", 0) or 0),
                        mime_type=f.get("mimeType") or _mime(f["name"]),
                        modified_at=f.get("modifiedTime"),
                        checksum=f.get("md5Checksum"),
                    )
                page_token = data.get("nextPageToken")
                if not page_token:
                    break

    def read_bytes(self, remote_id: str) -> bytes:
        with self._client() as client:
            resp = client.get(f"/files/{remote_id}", params={"alt": "media"})
            resp.raise_for_status()
            return resp.content


class _GraphConnectorBase:
    """Shared Microsoft Graph API logic for SharePoint and OneDrive."""

    GRAPH_ROOT = "https://graph.microsoft.com/v1.0"

    def __init__(self, config: dict):
        self.access_token = config.get("access_token", "")
        if not self.access_token:
            raise ConnectorError("requires 'access_token' (complete Microsoft OAuth first)")

    def _client(self):
        try:
            import httpx
        except ImportError:
            raise ConnectorError("httpx is required for Microsoft Graph connectors")
        return httpx.Client(
            base_url=self.GRAPH_ROOT,
            headers={"Authorization": f"Bearer {self.access_token}"},
            timeout=30.0,
        )

    def _walk_drive_item(self, client, drive_id: str, item_id: str) -> Generator[FileManifest, None, None]:
        url = f"/drives/{drive_id}/items/{item_id}/children"
        while url:
            resp = client.get(url)
            if resp.status_code != 200:
                raise ConnectorError(f"Microsoft Graph API error: {resp.status_code} {resp.text[:200]}")
            data = resp.json()
            for item in data.get("value", []):
                if "folder" in item:
                    yield from self._walk_drive_item(client, drive_id, item["id"])
                    continue
                if "file" not in item or not _is_supported(item["name"]):
                    continue
                yield FileManifest(
                    remote_id=f"{drive_id}:{item['id']}",
                    file_name=item["name"],
                    size_bytes=item.get("size", 0),
                    mime_type=item.get("file", {}).get("mimeType") or _mime(item["name"]),
                    modified_at=item.get("lastModifiedDateTime"),
                    checksum=item.get("file", {}).get("hashes", {}).get("quickXorHash"),
                )
            next_link = data.get("@odata.nextLink")
            url = next_link.replace(self.GRAPH_ROOT, "") if next_link else None

    def read_bytes(self, remote_id: str) -> bytes:
        drive_id, item_id = remote_id.split(":", 1)
        with self._client() as client:
            resp = client.get(f"/drives/{drive_id}/items/{item_id}/content", follow_redirects=True)
            resp.raise_for_status()
            return resp.content


class SharePointConnector(_GraphConnectorBase):
    """Enumerates files in a SharePoint document library via Microsoft Graph."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.site_id = config.get("site_id", "")
        self.drive_id = config.get("drive_id", "")
        if not self.site_id and not self.drive_id:
            raise ConnectorError("sharepoint connector requires 'site_id' or 'drive_id'")

    def list_files(self) -> Generator[FileManifest, None, None]:
        with self._client() as client:
            drive_id = self.drive_id
            if not drive_id:
                resp = client.get(f"/sites/{self.site_id}/drive")
                resp.raise_for_status()
                drive_id = resp.json()["id"]
            resp = client.get(f"/drives/{drive_id}/root")
            resp.raise_for_status()
            root_id = resp.json()["id"]
            yield from self._walk_drive_item(client, drive_id, root_id)


class OneDriveConnector(_GraphConnectorBase):
    """Enumerates files in a user's OneDrive via Microsoft Graph."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.user_id = config.get("user_id", "me")

    def list_files(self) -> Generator[FileManifest, None, None]:
        with self._client() as client:
            base = "/me/drive" if self.user_id == "me" else f"/users/{self.user_id}/drive"
            resp = client.get(base)
            resp.raise_for_status()
            drive_id = resp.json()["id"]
            resp = client.get(f"/drives/{drive_id}/root")
            resp.raise_for_status()
            root_id = resp.json()["id"]
            yield from self._walk_drive_item(client, drive_id, root_id)


_CONNECTOR_CLASSES = {
    "local": LocalConnector,
    "s3": S3Connector,
    "gdrive": GoogleDriveConnector,
    "sharepoint": SharePointConnector,
    "onedrive": OneDriveConnector,
}


def make_connector(connector_type: str, config: dict):
    cls = _CONNECTOR_CLASSES.get(connector_type)
    if not cls:
        raise ConnectorError(f"unknown connector type: {connector_type!r} (expected one of {CONNECTOR_TYPES})")
    return cls(config or {})
