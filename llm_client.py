"""
Central LLM client factory for DataNanite.
All services import get_client() from here — switching LLM providers
requires changes only in this file.

Provider selection:
  • If ANTHROPIC_API_KEY is set (local dev), use the direct Anthropic API.
    Vertex-style model ids (claude-sonnet-4-6, claude-haiku-4-5, claude-opus-4-x)
    are normalised by tier to valid public-API ids so the ~20 call sites that
    pass Vertex ids keep working unchanged.
  • Otherwise use GCP Vertex AI (AnthropicVertex) via Application Default
    Credentials — the production/GKE path (node service account has
    roles/aiplatform.user; no key needed at runtime).

Config:
  ANTHROPIC_API_KEY  — direct Anthropic API key (enables local dev path)
  GCP_PROJECT_ID     — GCP project ID (Vertex path)
  VERTEX_REGION      — Vertex AI region (default: us-east5)
"""
import os
import anthropic

# Map a Vertex-style / loose model id to a valid public Anthropic API id by tier.
# Targets are the current canonical ids for each tier.
_DIRECT_HAIKU = "claude-haiku-4-5-20251001"
_DIRECT_SONNET = "claude-sonnet-4-6"
_DIRECT_OPUS = "claude-opus-4-8"


def _normalize_model(model: str) -> str:
    s = (model or "").lower()
    if "haiku" in s:
        return _DIRECT_HAIKU
    if "opus" in s:
        return _DIRECT_OPUS
    if "sonnet" in s:
        return _DIRECT_SONNET
    return model


class _RemapMessages:
    """Proxy for client.messages that rewrites the `model` kwarg to a valid
    public-API id before delegating (covers create / stream / count_tokens)."""

    def __init__(self, inner):
        self._inner = inner

    def _remap(self, kwargs):
        if "model" in kwargs:
            kwargs["model"] = _normalize_model(kwargs["model"])
        # The pipeline was built for 1M-token-context models (the Vertex setup),
        # so enable the 1M context beta on the public API too — otherwise large
        # plan/synthesis prompts overflow the default 200k window.
        hdrs = dict(kwargs.get("extra_headers") or {})
        prev = hdrs.get("anthropic-beta", "")
        if "context-1m" not in prev:
            hdrs["anthropic-beta"] = (prev + "," if prev else "") + "context-1m-2025-08-07"
        kwargs["extra_headers"] = hdrs
        return kwargs

    def create(self, *args, **kwargs):
        return self._inner.create(*args, **self._remap(kwargs))

    def stream(self, *args, **kwargs):
        return self._inner.stream(*args, **self._remap(kwargs))

    def count_tokens(self, *args, **kwargs):
        return self._inner.count_tokens(*args, **self._remap(kwargs))

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _DirectClient:
    """Wraps anthropic.Anthropic so .messages remaps model ids; everything else
    passes through unchanged."""

    def __init__(self, inner):
        self._inner = inner
        self.messages = _RemapMessages(inner.messages)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def get_client():
    """Return an Anthropic-compatible client.

    Local dev: set ANTHROPIC_API_KEY in .env → direct Anthropic API.
    GKE/prod:  leave it unset → Vertex AI with ADC
               (run `gcloud auth application-default login` to use Vertex locally).
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if api_key:
        return _DirectClient(anthropic.Anthropic(api_key=api_key))

    project_id = os.environ.get("GCP_PROJECT_ID", "cog01k24f1ea555zdv7ynzthxanz5")
    vertex_region = os.environ.get("VERTEX_REGION", "us-east5")
    return anthropic.AnthropicVertex(region=vertex_region, project_id=project_id)
