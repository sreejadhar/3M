"""
Tech Workforce UI Server
========================
Serves the tech_ui/ static files and proxies /api/* requests to the
orchestrator (chat-ui) service.  Runs on port 8006.

Environment variables
---------------------
ORCHESTRATOR_URL  Internal URL of the orchestrator/chat-ui service.
                  Default: http://chat-ui:8005
LOG_LEVEL         uvicorn log level.  Default: info
"""

import os
import logging
from pathlib import Path
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# ── Config ────────────────────────────────────────────────────────────────────
ORCHESTRATOR_URL = os.environ.get("ORCHESTRATOR_URL", "http://chat-ui:8005").rstrip("/")
STATIC_DIR = Path(__file__).parent / "tech_ui"

log = logging.getLogger("tech_ui_server")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="DataNanite Tech UI", docs_url=None, redoc_url=None)

# Serve static assets under /tech/  (CSS, JS, images)
app.mount("/tech", StaticFiles(directory=str(STATIC_DIR)), name="tech_static")


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "service": "tech-ui"}


# ── Root — serve index.html ───────────────────────────────────────────────────
@app.get("/")
async def root():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index), media_type="text/html")
    return HTMLResponse("<h1>tech_ui/index.html not found</h1>", status_code=404)


# ── Proxy helpers ─────────────────────────────────────────────────────────────
_PROXY_TIMEOUT = httpx.Timeout(connect=5.0, read=120.0, write=30.0, pool=5.0)
_SSE_TIMEOUT   = httpx.Timeout(connect=5.0, read=None,  write=30.0, pool=5.0)  # no read timeout for SSE

_HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "content-encoding",
}


def _req_headers(request: Request) -> dict:
    return {
        k: v for k, v in request.headers.items()
        if k.lower() not in _HOP_BY_HOP and k.lower() != "host"
    }


def _resp_headers(upstream_headers) -> dict:
    return {
        k: v for k, v in upstream_headers.items()
        if k.lower() not in _HOP_BY_HOP
    }


async def _proxy(request: Request, downstream_path: str) -> Response:
    """Buffered proxy for normal (non-streaming) requests."""
    url = f"{ORCHESTRATOR_URL}/{downstream_path}"
    if request.url.query:
        url = f"{url}?{request.url.query}"

    body = await request.body()

    async with httpx.AsyncClient(timeout=_PROXY_TIMEOUT) as client:
        upstream = await client.request(
            method=request.method,
            url=url,
            headers=_req_headers(request),
            content=body,
        )

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=_resp_headers(upstream.headers),
        media_type=upstream.headers.get("content-type"),
    )


# ── SSE streaming proxy — must be declared BEFORE the catch-all ───────────────
# The buffered _proxy() would hang forever on a text/event-stream because it
# waits for the response body to complete before returning.  SSE streams never
# end until the client disconnects, so we need a dedicated streaming route that
# pipes bytes as they arrive.

@app.get("/api/sources/{source_id}/index-events")
async def proxy_sse_index_events(request: Request, source_id: str):
    """Streaming SSE pass-through for indexing pipeline events."""
    url = f"{ORCHESTRATOR_URL}/sources/{source_id}/index-events"

    async def _stream() -> AsyncGenerator[bytes, None]:
        async with httpx.AsyncClient(timeout=_SSE_TIMEOUT) as client:
            async with client.stream("GET", url, headers=_req_headers(request)) as r:
                async for chunk in r.aiter_bytes():
                    yield chunk

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering if present
        },
    )


# ── Catch-all API proxy — forward /api/* → orchestrator /* ───────────────────
# The tech UI prefixes all calls with /api/ (const API = '/api').
# The orchestrator routes live at the root (e.g. GET /sources, not GET /api/sources).
# Strip the /api/ layer here so the paths align correctly.

@app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_api(request: Request, path: str):
    return await _proxy(request, path)   # "path" already has the api/ stripped by FastAPI
