import { createContext, useContext, useEffect, useState, useCallback } from 'react';

// Auth model mirrors the existing HTML UIs (tech_ui/app.js):
//   token → sessionStorage.auth_token, email → localStorage.auth_email
//   POST /auth/login {email,password} -> {access_token,email,role}
//   GET  /auth/validate (Bearer)      -> {valid}
// Endpoints live on the orchestrator (proxied via /auth in vite.config.js).
const TOKEN_KEY = 'auth_token';
const EMAIL_KEY = 'auth_email';

// Capture the real fetch before we patch it, so auth calls themselves never
// recurse through the token injector.
const origFetch = window.fetch.bind(window);

export const getToken = () => sessionStorage.getItem(TOKEN_KEY) || '';
export const getEmail = () => localStorage.getItem(EMAIL_KEY) || '';

function setAuth(token, email) {
  sessionStorage.setItem(TOKEN_KEY, token);
  localStorage.setItem(EMAIL_KEY, email);
}
function clearAuth() {
  sessionStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(EMAIL_KEY);
}

// Inject the Bearer token into every same-origin request (matches tech_ui).
// The metadata microservices don't require it today, but this keeps the React
// app consistent with the rest of the stack and future-proof.
window.fetch = function patchedFetch(url, opts = {}) {
  const token = getToken();
  if (token && typeof url === 'string' && (url.startsWith('/') || url.startsWith(location.origin))) {
    return origFetch(url, {
      ...opts,
      headers: { Authorization: `Bearer ${token}`, ...(opts.headers || {}) },
    });
  }
  return origFetch(url, opts);
};

export async function login(email, password) {
  const r = await origFetch('/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  });
  if (!r.ok) {
    const d = await r.json().catch(() => ({}));
    throw new Error(d.detail || 'Invalid email or password');
  }
  const data = await r.json();
  setAuth(data.access_token, data.email);
  return data;
}

export async function validate() {
  const token = getToken();
  if (!token) return false;
  try {
    const r = await origFetch('/auth/validate', {
      headers: { Authorization: `Bearer ${token}` },
    });
    const d = await r.json();
    return Boolean(d.valid);
  } catch {
    return false;
  }
}

// ── React context ─────────────────────────────────────────────────────────────
const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [checking, setChecking] = useState(true);
  const [authed, setAuthed] = useState(false);
  const [email, setEmail] = useState(getEmail());

  useEffect(() => {
    let alive = true;
    validate().then((ok) => {
      if (!alive) return;
      setAuthed(ok);
      setChecking(false);
    });
    return () => {
      alive = false;
    };
  }, []);

  const doLogin = useCallback(async (em, pw) => {
    const data = await login(em, pw);
    setEmail(data.email);
    setAuthed(true);
    return data;
  }, []);

  const doLogout = useCallback(() => {
    clearAuth();
    setAuthed(false);
    setEmail('');
  }, []);

  return (
    <AuthContext.Provider value={{ checking, authed, email, doLogin, doLogout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
}
