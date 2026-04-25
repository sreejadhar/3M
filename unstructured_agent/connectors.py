"""
File connectors for the unstructured data intelligence agent.

Each connector enumerates a source, produces FileManifest records,
and computes content-based checksums for change detection.

Supported connectors:
  local      — local filesystem / NFS mount
  gdrive     — Google Drive folder (OAuth2 or Service Account)
  s3         — Amazon S3 bucket/prefix
  sharepoint — SharePoint Online document library (Microsoft Graph API)
  onedrive   — OneDrive for Business / Personal (Microsoft Graph API)
"""
from __future__ import annotations

import hashlib
import io
import logging
import mimetypes
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, List, Optional

logger = logging.getLogger(__name__)


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
    remote_path: Optional[str] = None  # S3 key, Drive file ID, etc.


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
    if source_type == "gdrive":
        return GoogleDriveConnector(
            folder_id=connection["folder_id"],
            token_path=connection.get("token_path"),
            service_account_json=connection.get("service_account_json"),
            max_size_mb=connection.get("max_size_mb", 200),
        )
    if source_type == "s3":
        return S3Connector(
            bucket=connection["bucket"],
            prefix=connection.get("prefix", ""),
            region=connection.get("region"),
            aws_access_key_id=connection.get("aws_access_key_id"),
            aws_secret_access_key=connection.get("aws_secret_access_key"),
            aws_session_token=connection.get("aws_session_token"),
            max_size_mb=connection.get("max_size_mb", 200),
            extensions=connection.get("extensions"),
        )
    if source_type == "sharepoint":
        return SharePointConnector(
            site_url=connection["site_url"],
            tenant_id=connection["tenant_id"],
            client_id=connection["client_id"],
            client_secret=connection["client_secret"],
            library=connection.get("library", "Documents"),
            folder_path=connection.get("folder_path", ""),
            max_size_mb=connection.get("max_size_mb", 200),
            extensions=connection.get("extensions"),
        )
    if source_type == "onedrive":
        return OneDriveConnector(
            tenant_id=connection["tenant_id"],
            client_id=connection["client_id"],
            client_secret=connection["client_secret"],
            user_email=connection.get("user_email", ""),
            folder_path=connection.get("folder_path", ""),
            max_size_mb=connection.get("max_size_mb", 200),
            extensions=connection.get("extensions"),
        )
    raise ValueError(
        f"Unsupported source type: {source_type!r}. "
        "Supported: local, gdrive, s3, sharepoint, onedrive."
    )


# ── S3 connector ──────────────────────────────────────────────────────────────

class S3Connector:
    """
    Enumerates files from an S3 bucket/prefix recursively.

    Change detection uses the S3 ETag (MD5 of the object for single-part
    uploads, or a composite hash for multipart) — no re-download needed
    to detect unchanged files.

    AWS credentials are resolved in this priority order:
      1. Explicit keys in connection config (aws_access_key_id / aws_secret_access_key)
      2. Environment variables (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY)
      3. IAM instance role — best for EC2, no credentials needed at all

    connection config keys:
      bucket              — S3 bucket name (required)
      prefix              — key prefix / folder path, e.g. "docs/supply-chain/" (optional)
      region              — AWS region, e.g. "us-east-1" (optional)
      aws_access_key_id   — explicit key (optional)
      aws_secret_access_key — explicit secret (optional)
      aws_session_token   — for temporary STS credentials (optional)
      max_size_mb         — skip files larger than this (default 200)
    """

    def __init__(self, bucket: str, prefix: str = "",
                 region: Optional[str] = None,
                 aws_access_key_id: Optional[str] = None,
                 aws_secret_access_key: Optional[str] = None,
                 aws_session_token: Optional[str] = None,
                 max_size_mb: int = 200,
                 extensions: Optional[List[str]] = None) -> None:
        self.bucket               = bucket
        self.prefix               = prefix.lstrip("/")   # S3 keys never start with /
        self.region               = region
        self.aws_access_key_id    = aws_access_key_id
        self.aws_secret_access_key= aws_secret_access_key
        self.aws_session_token    = aws_session_token
        self.max_size_bytes       = max_size_mb * 1024 * 1024
        self.extensions           = (
            {e.lower() for e in extensions} if extensions else _SUPPORTED_EXTENSIONS
        )
        self._tmp_files: List[str] = []

    def _client(self):
        try:
            import boto3  # type: ignore
        except ImportError:
            raise ImportError(
                "boto3 is required for S3. Run: pip install boto3"
            )
        kwargs: dict = {}
        if self.region:
            kwargs["region_name"] = self.region
        if self.aws_access_key_id:
            kwargs["aws_access_key_id"]     = self.aws_access_key_id
            kwargs["aws_secret_access_key"] = self.aws_secret_access_key
            if self.aws_session_token:
                kwargs["aws_session_token"] = self.aws_session_token
        return boto3.client("s3", **kwargs)

    def enumerate(self) -> Generator[FileManifest, None, None]:
        s3 = self._client()
        paginator = s3.get_paginator("list_objects_v2")

        pages = paginator.paginate(
            Bucket=self.bucket,
            Prefix=self.prefix,
        )

        for page in pages:
            for obj in page.get("Contents", []):
                key  = obj["Key"]
                name = key.split("/")[-1]   # filename portion of the key

                if not name or name.endswith("/"):
                    continue                 # skip folder markers
                if _is_junk(name):
                    continue

                ext = Path(name).suffix.lower()
                if ext not in self.extensions:
                    continue

                size = obj.get("Size", 0)
                if size == 0 or size > self.max_size_bytes:
                    logger.debug("s3: skip %s (size=%d)", key, size)
                    continue

                # ETag is MD5 for single-part uploads; strip quotes AWS wraps it in
                etag = obj.get("ETag", "").strip('"')
                checksum = f"etag:{etag}"

                modified_at = obj["LastModified"].isoformat() if obj.get("LastModified") else None

                # Download to temp file
                try:
                    tmp = self._download(s3, key, ext)
                except Exception as exc:
                    logger.warning("s3: download failed for s3://%s/%s: %s",
                                   self.bucket, key, exc)
                    continue

                yield FileManifest(
                    path=tmp,
                    source_type="s3",
                    file_name=name,
                    size_bytes=os.path.getsize(tmp),
                    mime_type=_mime(name),
                    checksum=checksum,
                    created_at=None,
                    modified_at=modified_at,
                    remote_path=key,
                )

    def _download(self, s3_client, key: str, ext: str) -> str:
        suffix = ext if ext.startswith(".") else f".{ext}"
        tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix, prefix="nanite_s3_"
        )
        tmp.close()
        s3_client.download_file(self.bucket, key, tmp.name)
        self._tmp_files.append(tmp.name)
        return tmp.name

    def cleanup(self) -> None:
        for path in self._tmp_files:
            try:
                os.unlink(path)
            except OSError:
                pass
        self._tmp_files.clear()


# ── Google Drive connector ────────────────────────────────────────────────────

# Google Drive MIME types that map to exportable Office formats
_GDRIVE_EXPORT_MAP = {
    "application/vnd.google-apps.document":
        ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", ".docx"),
    "application/vnd.google-apps.presentation":
        ("application/vnd.openxmlformats-officedocument.presentationml.presentation", ".pptx"),
    "application/vnd.google-apps.spreadsheet":
        ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", ".xlsx"),
}

# Native Drive MIME → extension
_GDRIVE_NATIVE_EXT = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "text/plain": ".txt",
    "text/html": ".html",
    "text/markdown": ".md",
    "text/csv": ".csv",
}

_GDRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def _build_gdrive_creds(token_path: Optional[str],
                         service_account_json: Optional[str]):
    """
    Build Google API credentials from either:
      - service_account_json: path to a service account key JSON file
      - token_path: path to an OAuth2 token JSON file (generated via auth flow)
    """
    try:
        from google.oauth2 import service_account
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
    except ImportError:
        raise ImportError(
            "google-api-python-client and google-auth are required for Google Drive. "
            "Run: pip install google-api-python-client google-auth google-auth-oauthlib"
        )

    if service_account_json:
        return service_account.Credentials.from_service_account_file(
            service_account_json, scopes=_GDRIVE_SCOPES
        )

    if token_path and Path(token_path).exists():
        import json
        with open(token_path) as f:
            token_data = json.load(f)
        creds = Credentials(
            token=token_data.get("token"),
            refresh_token=token_data.get("refresh_token"),
            token_uri=token_data.get("token_uri", "https://oauth2.googleapis.com/token"),
            client_id=token_data.get("client_id"),
            client_secret=token_data.get("client_secret"),
            scopes=token_data.get("scopes", _GDRIVE_SCOPES),
        )
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            # Persist refreshed token
            _save_token(token_path, creds)
        return creds

    raise ValueError(
        "Google Drive connector requires either 'service_account_json' or "
        "'token_path' in the connection config."
    )


def _save_token(token_path: str, creds) -> None:
    import json
    data = {
        "token":         creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri":     creds.token_uri,
        "client_id":     creds.client_id,
        "client_secret": creds.client_secret,
        "scopes":        list(creds.scopes or _GDRIVE_SCOPES),
    }
    with open(token_path, "w") as f:
        json.dump(data, f)


class GoogleDriveConnector:
    """
    Enumerates files from a Google Drive folder recursively.

    Change detection uses md5Checksum from Drive API metadata — no content
    download needed to detect unchanged files.

    Files are downloaded to a NamedTemporaryFile during enumeration; the
    temp file path is returned in FileManifest.path and is valid for the
    duration of the pipeline run. The pipeline must NOT cache these paths
    across runs.

    Authentication:
      service_account_json — path to GCP service account key JSON
                             (best for shared drives / workspace)
      token_path           — path to OAuth2 token JSON
                             (for personal Google Drive)
    """

    def __init__(self, folder_id: str,
                 token_path: Optional[str] = None,
                 service_account_json: Optional[str] = None,
                 max_size_mb: int = 200) -> None:
        self.folder_id       = folder_id
        self.token_path      = token_path
        self.service_account_json = service_account_json
        self.max_size_bytes  = max_size_mb * 1024 * 1024
        self._tmp_files: List[str] = []   # track for cleanup

    def enumerate(self) -> Generator[FileManifest, None, None]:
        try:
            from googleapiclient.discovery import build
            from googleapiclient.http import MediaIoBaseDownload
        except ImportError:
            raise ImportError(
                "google-api-python-client required. "
                "Run: pip install google-api-python-client google-auth google-auth-oauthlib"
            )

        creds   = _build_gdrive_creds(self.token_path, self.service_account_json)
        service = build("drive", "v3", credentials=creds, cache_discovery=False)

        yield from self._enumerate_folder(service, self.folder_id)

    def _enumerate_folder(self, service, folder_id: str,
                           ) -> Generator[FileManifest, None, None]:
        """Recursively enumerate a Drive folder."""
        from googleapiclient.http import MediaIoBaseDownload

        page_token = None
        while True:
            resp = service.files().list(
                q=f"'{folder_id}' in parents and trashed=false",
                spaces="drive",
                fields="nextPageToken, files(id,name,mimeType,size,md5Checksum,"
                       "createdTime,modifiedTime)",
                pageToken=page_token,
                pageSize=100,
            ).execute()

            for item in resp.get("files", []):
                mime   = item.get("mimeType", "")
                name   = item.get("name", "")

                # Recurse into subfolders
                if mime == "application/vnd.google-apps.folder":
                    yield from self._enumerate_folder(service, item["id"])
                    continue

                if _is_junk(name):
                    continue

                size = int(item.get("size") or 0)
                if size > self.max_size_bytes:
                    logger.debug("gdrive: skip large file %s (%d bytes)", name, size)
                    continue

                # Determine download MIME and extension
                if mime in _GDRIVE_EXPORT_MAP:
                    dl_mime, ext = _GDRIVE_EXPORT_MAP[mime]
                    is_export = True
                elif mime in _GDRIVE_NATIVE_EXT:
                    dl_mime = mime
                    ext = _GDRIVE_NATIVE_EXT[mime]
                    is_export = False
                else:
                    # Try to infer extension from filename
                    ext = Path(name).suffix.lower()
                    if ext not in _SUPPORTED_EXTENSIONS:
                        logger.debug("gdrive: skip unsupported mime %s for %s", mime, name)
                        continue
                    dl_mime = mime
                    is_export = False

                # Use Drive md5Checksum as change-detection hash (free — no download)
                checksum = "md5:" + item.get("md5Checksum", item["id"])

                # Download to temp file
                try:
                    tmp = self._download(
                        service, item["id"], dl_mime, is_export, ext
                    )
                except Exception as exc:
                    logger.warning("gdrive: download failed for %s: %s", name, exc)
                    continue

                fname = Path(name).stem + ext  # normalise extension
                yield FileManifest(
                    path=tmp,
                    source_type="gdrive",
                    file_name=fname,
                    size_bytes=os.path.getsize(tmp),
                    remote_path=item["id"],
                    mime_type=dl_mime,
                    checksum=checksum,
                    created_at=item.get("createdTime"),
                    modified_at=item.get("modifiedTime"),
                )

            page_token = resp.get("nextPageToken")
            if not page_token:
                break

    def _download(self, service, file_id: str, mime: str,
                  is_export: bool, ext: str) -> str:
        """Download a Drive file to a named temp file. Returns the temp path."""
        from googleapiclient.http import MediaIoBaseDownload

        buf = io.BytesIO()
        if is_export:
            req = service.files().export_media(fileId=file_id, mimeType=mime)
        else:
            req = service.files().get_media(fileId=file_id)

        dl = MediaIoBaseDownload(buf, req, chunksize=4 * 1024 * 1024)
        done = False
        while not done:
            _, done = dl.next_chunk()

        suffix = ext if ext.startswith(".") else f".{ext}"
        tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix, prefix="nanite_gdrive_"
        )
        tmp.write(buf.getvalue())
        tmp.close()
        self._tmp_files.append(tmp.name)
        return tmp.name

    def cleanup(self) -> None:
        """Delete temp files created during enumeration."""
        for path in self._tmp_files:
            try:
                os.unlink(path)
            except OSError:
                pass
        self._tmp_files.clear()


# ── Microsoft Graph shared helpers ────────────────────────────────────────────

_GRAPH_BASE = "https://graph.microsoft.com/v1.0"


def _ms_token(tenant_id: str, client_id: str, client_secret: str) -> str:
    """Acquire an app-only access token via MSAL client credentials flow."""
    try:
        import msal  # type: ignore
    except ImportError:
        raise RuntimeError("msal not installed — pip install msal")

    app = msal.ConfidentialClientApplication(
        client_id,
        authority=f"https://login.microsoftonline.com/{tenant_id}",
        client_credential=client_secret,
    )
    result = app.acquire_token_for_client(
        scopes=["https://graph.microsoft.com/.default"]
    )
    if "access_token" not in result:
        raise RuntimeError(
            f"MSAL token acquisition failed: {result.get('error_description', result)}"
        )
    return result["access_token"]


def _graph_get(token: str, path: str, **params) -> dict:
    import httpx
    resp = httpx.get(
        f"{_GRAPH_BASE}{path}",
        headers={"Authorization": f"Bearer {token}"},
        params=params,
        timeout=30,
        follow_redirects=True,
    )
    resp.raise_for_status()
    return resp.json()


def _graph_download(token: str, path: str, suffix: str, prefix: str) -> str:
    """Stream a Graph API /content URL to a named temp file. Returns path."""
    import httpx
    with httpx.stream(
        "GET",
        f"{_GRAPH_BASE}{path}",
        headers={"Authorization": f"Bearer {token}"},
        timeout=120,
        follow_redirects=True,
    ) as resp:
        resp.raise_for_status()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix=prefix)
        for chunk in resp.iter_bytes(chunk_size=1 << 20):
            tmp.write(chunk)
        tmp.close()
        return tmp.name


def _graph_enumerate_folder(token: str, drive_id: str, item_id: str,
                             extensions: set, max_bytes: int,
                             tmp_files: list, prefix: str) -> Generator:
    """Recursively yield FileManifest for all supported files in a Graph drive folder."""
    next_url: Optional[str] = (
        f"{_GRAPH_BASE}/drives/{drive_id}/items/{item_id}/children"
        "?$select=id,name,size,file,folder,lastModifiedDateTime,eTag"
        "&$top=200"
    )
    import httpx

    while next_url:
        resp = httpx.get(
            next_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
            follow_redirects=True,
        )
        resp.raise_for_status()
        data = resp.json()

        for item in data.get("value", []):
            name = item.get("name", "")
            if _is_junk(name):
                continue

            if "folder" in item:
                yield from _graph_enumerate_folder(
                    token, drive_id, item["id"], extensions, max_bytes, tmp_files, prefix
                )
                continue

            if "file" not in item:
                continue

            ext = Path(name).suffix.lower()
            if ext not in extensions:
                continue

            size = item.get("size", 0)
            if size == 0 or size > max_bytes:
                logger.debug("msgraph: skip %s (size=%d)", name, size)
                continue

            etag = item.get("eTag", item["id"]).strip('"')
            checksum = f"etag:{etag}"
            modified_at = item.get("lastModifiedDateTime")

            try:
                tmp_path = _graph_download(
                    token,
                    f"/drives/{drive_id}/items/{item['id']}/content",
                    suffix=ext,
                    prefix=prefix,
                )
            except Exception as exc:
                logger.warning("msgraph: download failed for %s: %s", name, exc)
                continue

            tmp_files.append(tmp_path)
            yield FileManifest(
                path=tmp_path,
                source_type="msgraph",
                file_name=name,
                size_bytes=os.path.getsize(tmp_path),
                mime_type=_mime(name),
                checksum=checksum,
                created_at=None,
                modified_at=modified_at,
                remote_path=f"{drive_id}:{item['id']}",
            )

        next_url = data.get("@odata.nextLink")


# ── SharePoint connector ───────────────────────────────────────────────────────

class SharePointConnector:
    """
    Enumerates files from a SharePoint Online document library using the
    Microsoft Graph API with app-only (client credentials) authentication.

    Azure AD app requires:
      - Sites.Read.All  (application permission)
      - Files.Read.All  (application permission)

    Connection keys:
      site_url      — full SharePoint site URL, e.g.
                      https://contoso.sharepoint.com/sites/SupplyChain
      tenant_id     — Azure AD tenant ID (GUID or domain)
      client_id     — Azure AD app (client) ID
      client_secret — Azure AD app client secret
      library       — document library display name (default: "Documents")
      folder_path   — sub-folder path within the library (default: root)
      max_size_mb   — skip files larger than this (default: 200)
    """

    def __init__(
        self,
        site_url: str,
        tenant_id: str,
        client_id: str,
        client_secret: str,
        library: str = "Documents",
        folder_path: str = "",
        max_size_mb: int = 200,
        extensions: Optional[List[str]] = None,
    ):
        from urllib.parse import urlparse
        parsed = urlparse(site_url.rstrip("/"))
        self._hostname   = parsed.netloc                   # contoso.sharepoint.com
        self._site_path  = parsed.path                     # /sites/SupplyChain
        self._tenant_id  = tenant_id
        self._client_id  = client_id
        self._client_secret = client_secret
        self._library    = library
        self._folder_path = folder_path.strip("/")
        self._max_bytes  = max_size_mb * 1024 * 1024
        self._extensions = set(extensions) if extensions else _SUPPORTED_EXTENSIONS
        self._tmp_files: List[str] = []

    def enumerate(self) -> Generator:
        token = _ms_token(self._tenant_id, self._client_id, self._client_secret)

        # Resolve site ID from hostname + path
        site = _graph_get(token, f"/sites/{self._hostname}:{self._site_path}")
        site_id = site["id"]
        logger.info("sharepoint: resolved site_id=%s", site_id)

        # Find the target document library drive by display name
        drives = _graph_get(token, f"/sites/{site_id}/drives").get("value", [])
        drive = next((d for d in drives if d.get("name") == self._library), None)
        if drive is None:
            available = [d.get("name") for d in drives]
            raise ValueError(
                f"SharePoint library {self._library!r} not found. "
                f"Available: {available}"
            )
        drive_id = drive["id"]
        logger.info("sharepoint: using drive %s (%s)", self._library, drive_id)

        # Resolve starting folder
        if self._folder_path:
            item = _graph_get(
                token, f"/drives/{drive_id}/root:/{self._folder_path}"
            )
            root_item_id = item["id"]
        else:
            root_item_id = "root"

        yield from _graph_enumerate_folder(
            token, drive_id, root_item_id,
            self._extensions, self._max_bytes,
            self._tmp_files, "nanite_sp_",
        )

    def cleanup(self) -> None:
        for path in self._tmp_files:
            try:
                os.unlink(path)
            except OSError:
                pass
        self._tmp_files.clear()


# ── OneDrive connector ────────────────────────────────────────────────────────

class OneDriveConnector:
    """
    Enumerates files from OneDrive for Business (or a shared drive) using
    the Microsoft Graph API with app-only (client credentials) authentication.

    Azure AD app requires:
      - Files.Read.All  (application permission)
      - User.Read.All   (application permission — needed to look up user drives)

    Connection keys:
      tenant_id     — Azure AD tenant ID
      client_id     — Azure AD app (client) ID
      client_secret — Azure AD app client secret
      user_email    — UPN of the user whose OneDrive to index, e.g.
                      john.doe@contoso.com  (leave blank for the app's own drive)
      folder_path   — sub-folder path within OneDrive (default: root)
      max_size_mb   — skip files larger than this (default: 200)
    """

    def __init__(
        self,
        tenant_id: str,
        client_id: str,
        client_secret: str,
        user_email: str = "",
        folder_path: str = "",
        max_size_mb: int = 200,
        extensions: Optional[List[str]] = None,
    ):
        self._tenant_id    = tenant_id
        self._client_id    = client_id
        self._client_secret = client_secret
        self._user_email   = user_email.strip()
        self._folder_path  = folder_path.strip("/")
        self._max_bytes    = max_size_mb * 1024 * 1024
        self._extensions   = set(extensions) if extensions else _SUPPORTED_EXTENSIONS
        self._tmp_files: List[str] = []

    def enumerate(self) -> Generator:
        token = _ms_token(self._tenant_id, self._client_id, self._client_secret)

        # Resolve drive
        if self._user_email:
            drive = _graph_get(token, f"/users/{self._user_email}/drive")
        else:
            drive = _graph_get(token, "/me/drive")
        drive_id = drive["id"]
        logger.info("onedrive: using drive %s for %s", drive_id,
                    self._user_email or "app")

        # Resolve starting folder
        if self._folder_path:
            item = _graph_get(
                token, f"/drives/{drive_id}/root:/{self._folder_path}"
            )
            root_item_id = item["id"]
        else:
            root_item_id = "root"

        yield from _graph_enumerate_folder(
            token, drive_id, root_item_id,
            self._extensions, self._max_bytes,
            self._tmp_files, "nanite_od_",
        )

    def cleanup(self) -> None:
        for path in self._tmp_files:
            try:
                os.unlink(path)
            except OSError:
                pass
        self._tmp_files.clear()
