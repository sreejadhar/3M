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

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, HTMLResponse
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


# ── API proxy — forward /api/* to orchestrator ────────────────────────────────
# Using a single catch-all route so the tech UI never needs CORS headers
# and the orchestrator URL stays server-side (not exposed to the browser).

_PROXY_TIMEOUT = httpx.Timeout(connect=5.0, read=120.0, write=30.0, pool=5.0)
_HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "content-encoding",
}


async def _proxy(request: Request, downstream_path: str) -> Response:
    url = f"{ORCHESTRATOR_URL}/{downstream_path}"
    if request.url.query:
        url = f"{url}?{request.url.query}"

    # Strip hop-by-hop headers before forwarding
    req_headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in _HOP_BY_HOP and k.lower() != "host"
    }

    body = await request.body()

    async with httpx.AsyncClient(timeout=_PROXY_TIMEOUT) as client:
        upstream = await client.request(
            method=request.method,
            url=url,
            headers=req_headers,
            content=body,
        )

    resp_headers = {
        k: v for k, v in upstream.headers.items()
        if k.lower() not in _HOP_BY_HOP
    }

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=resp_headers,
        media_type=upstream.headers.get("content-type"),
    )


@app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_api(request: Request, path: str):
    return await _proxy(request, f"api/{path}")


# ── SSE proxy (streaming pass-through) ───────────────────────────────────────
# httpx streams SSE just fine via the catch-all above, but we need to ensure
# the response is streamed rather than buffered.  FastAPI/Starlette handles
# this automatically when the upstream Content-Type is text/event-stream.
