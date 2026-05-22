"""
Lightweight auth proxy that sits in front of the Streamlit UI.

Runs on port 8502 (exposed).  Streamlit runs on 127.0.0.1:8501 (internal only).

All requests are validated against the orchestrator's /auth/validate endpoint.
Invalid / missing tokens get a minimal login form that sets an auth_token cookie.
"""
import os

import requests as _req
from flask import Flask, Response, make_response, redirect, render_template_string, request

ORCHESTRATOR_URL = os.environ.get("ORCHESTRATOR_URL", "http://chat-ui-service:8005").rstrip("/")
STREAMLIT_URL    = "http://127.0.0.1:8501"
PORT             = int(os.environ.get("PROXY_PORT", 8502))

app = Flask(__name__)

_LOGIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>DataNanite — Sign in</title>
  <style>
    *{box-sizing:border-box;margin:0;padding:0}
    body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;display:flex;height:100vh}
    .left{flex:0 0 58%;background:linear-gradient(135deg,#0d1b3e 0%,#1a3a6b 50%,#0d2b5a 100%);
          display:flex;flex-direction:column;align-items:center;justify-content:center;padding:48px}
    .logo-box{width:64px;height:64px;background:linear-gradient(135deg,#4285F4,#9B72CB);border-radius:14px;
              display:flex;align-items:center;justify-content:center;font-size:1.5rem;font-weight:800;color:#fff;margin-bottom:20px}
    h1{color:#fff;font-size:2rem;font-weight:700;margin-bottom:12px}
    .sub{color:rgba(255,255,255,.65);font-size:1rem;line-height:1.6;text-align:center;max-width:320px}
    .right{flex:1;background:#f8f9fb;display:flex;align-items:center;justify-content:center;padding:48px}
    .card{width:100%;max-width:380px}
    .card h2{font-size:1.5rem;font-weight:700;color:#1a1a2e;margin-bottom:8px}
    .card p{color:#6b7280;font-size:.95rem;margin-bottom:28px}
    .err{background:#fef2f2;border:1px solid #fca5a5;color:#dc2626;padding:12px 16px;
         border-radius:8px;font-size:.9rem;margin-bottom:20px}
    label{display:block;font-size:.875rem;font-weight:500;color:#374151;margin-bottom:6px}
    .field{margin-bottom:20px}
    input[type=email],input[type=password]{
      width:100%;padding:11px 14px;border:1.5px solid #e5e7eb;border-radius:8px;
      font-size:.95rem;outline:none;background:#fff}
    input:focus{border-color:#4285F4}
    button{width:100%;padding:12px;background:linear-gradient(135deg,#4285F4,#9B72CB);
           color:#fff;border:none;border-radius:8px;font-size:1rem;font-weight:600;cursor:pointer}
  </style>
</head>
<body>
  <div class="left">
    <div class="logo-box">DN</div>
    <h1>DataNanite</h1>
    <p class="sub">Data Intelligence Platform</p>
  </div>
  <div class="right">
    <div class="card">
      <h2>Welcome back</h2>
      <p>Sign in to access DataNanite</p>
      {% if error %}<div class="err">{{ error }}</div>{% endif %}
      <form method="POST" action="/_login">
        <input type="hidden" name="next" value="{{ next }}"/>
        <div class="field">
          <label>Email address</label>
          <input type="email" name="email" placeholder="you@cognizant.com" required/>
        </div>
        <div class="field" style="margin-bottom:28px">
          <label>Password</label>
          <input type="password" name="password" placeholder="••••••••" required/>
        </div>
        <button type="submit">Sign in</button>
      </form>
    </div>
  </div>
</body>
</html>"""


def _validate_token(token: str) -> bool:
    if not token:
        return False
    try:
        r = _req.get(
            f"{ORCHESTRATOR_URL}/auth/validate",
            headers={"Authorization": f"Bearer {token}"},
            timeout=5,
        )
        return r.status_code == 200 and r.json().get("valid", False)
    except Exception:
        return False


def _get_token() -> str:
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:]
    return request.cookies.get("auth_token", "")


@app.route("/_login", methods=["POST"])
def do_login():
    email    = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")
    next_url = request.form.get("next", "/")
    try:
        r = _req.post(
            f"{ORCHESTRATOR_URL}/auth/login",
            json={"email": email, "password": password},
            timeout=5,
        )
        if r.status_code == 200:
            token = r.json()["access_token"]
            resp  = redirect(next_url)
            resp.set_cookie("auth_token", token, httponly=True, samesite="Lax")
            return resp
        error = r.json().get("detail", "Invalid email or password")
    except Exception as exc:
        error = f"Login service unavailable: {exc}"
    return render_template_string(_LOGIN_HTML, error=error, next=next_url)


@app.route("/health")
def health():
    return {"status": "ok", "service": "streamlit-proxy"}


@app.route("/", defaults={"path": ""}, methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
@app.route("/<path:path>",            methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
def proxy(path):
    token = _get_token()
    if not _validate_token(token):
        return render_template_string(_LOGIN_HTML, error=None, next=request.path), 200

    # Forward to Streamlit
    url = f"{STREAMLIT_URL}/{path}"
    if request.query_string:
        url += "?" + request.query_string.decode()

    headers = {k: v for k, v in request.headers if k.lower() not in ("host", "content-length")}

    try:
        resp = _req.request(
            method=request.method,
            url=url,
            headers=headers,
            data=request.get_data(),
            allow_redirects=False,
            timeout=60,
            stream=True,
        )
    except Exception as exc:
        return f"Streamlit unavailable: {exc}", 502

    excluded = {"transfer-encoding", "connection"}
    resp_headers = [(k, v) for k, v in resp.headers.items() if k.lower() not in excluded]
    return Response(resp.iter_content(chunk_size=8192), status=resp.status_code, headers=resp_headers)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, threaded=True)
