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
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# ── Config ────────────────────────────────────────────────────────────────────
ORCHESTRATOR_URL = os.environ.get("ORCHESTRATOR_URL", "http://chat-ui:8005").rstrip("/")
STATIC_DIR = Path(__file__).parent / "tech_ui"
# React build (Vite) for the Engineer Workbench; served at "/" when present,
# with the legacy tech_ui/ kept as a fallback.
REACT_DIR = Path(__file__).parent / "frontend" / "dist"

_AUTH_VALIDATE_URL = f"{ORCHESTRATOR_URL}/auth/validate"


async def _is_authenticated(request: Request) -> bool:
    """Validate JWT against the orchestrator /auth/validate endpoint."""
    auth_header = request.headers.get("Authorization", "")
    token = auth_header.removeprefix("Bearer ").strip()
    if not token:
        token = request.cookies.get("auth_token", "")
    if not token:
        return False
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
            r = await client.get(_AUTH_VALIDATE_URL, headers={"Authorization": f"Bearer {token}"})
            return r.status_code == 200 and r.json().get("valid", False)
    except Exception:
        return False

log = logging.getLogger("tech_ui_server")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="DataNanite Tech UI", docs_url=None, redoc_url=None)


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    path = request.url.path
    # Always allow health check and auth passthrough
    if path in ("/", "/health") or path.startswith("/auth"):
        return await call_next(request)
    # Allow SSE index-events — EventSource cannot send Authorization headers
    if path.endswith("/index-events"):
        return await call_next(request)
    # Allow static JS/CSS (the login overlay needs them to render)
    if path.startswith("/tech/style") or path.startswith("/tech/app"):
        return await call_next(request)
    # Allow the React build's hashed assets + favicon (must load pre-auth)
    if path.startswith("/assets") or path == "/favicon.ico" or path == "/vite.svg":
        return await call_next(request)
    if not await _is_authenticated(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    return await call_next(request)


# Serve static assets under /tech/  (legacy tech_ui CSS, JS, images)
# html=False keeps the default; we set headers in middleware below
app.mount("/tech", StaticFiles(directory=str(STATIC_DIR)), name="tech_static")

# Serve the React build's hashed assets at /assets/ (index.html references them).
if (REACT_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(REACT_DIR / "assets")), name="react_assets")


# Disable browser caching for all static assets so CSS/JS changes are instant
from starlette.middleware.base import BaseHTTPMiddleware

class NoCacheMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path.startswith("/tech/") and (path.endswith(".css") or path.endswith(".js")):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
            response.headers["Pragma"]        = "no-cache"
            response.headers["Expires"]       = "0"
        return response

app.add_middleware(NoCacheMiddleware)


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "service": "tech-ui"}


# ── Auth proxy — forward /auth/* to orchestrator ──────────────────────────────
@app.api_route("/auth/{path:path}", methods=["GET", "POST"])
async def proxy_auth(request: Request, path: str):
    return await _proxy(request, f"auth/{path}")


# ── Root — serve index.html ───────────────────────────────────────────────────
@app.get("/")
async def root():
    # Prefer the React build; fall back to the legacy tech_ui.
    react_index = REACT_DIR / "index.html"
    if react_index.exists():
        return FileResponse(str(react_index), media_type="text/html",
                            headers={"Cache-Control": "no-store"})
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index), media_type="text/html")
    return HTMLResponse("<h1>UI not found — build frontend/ or ensure tech_ui/index.html exists</h1>", status_code=404)


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
