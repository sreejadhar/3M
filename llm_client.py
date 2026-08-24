"""
Central LLM client factory for DataNanite.
All services import get_client() from here — switching LLM providers
requires changes only in this file.

Provider selection:
  • If ANTHROPIC_API_KEY is set (local dev), use the direct Anthropic API.
    Model-tier ids (claude-sonnet-4-6, claude-haiku-4-5, claude-opus-4-x) are
    normalised by tier to valid public-API ids so the ~20 call sites that pass
    those ids keep working unchanged.
  • If LLM_PROVIDER=vertex, use GCP Vertex AI (AnthropicVertex) via
    Application Default Credentials.
  • Otherwise (default/production, EC2 datananited01.mmm.com), use AWS
    Bedrock: the app assumes the cross-account role in aws_auth.py via STS,
    then calls Bedrock with those temporary credentials — no API key. The
    same tier-detection used for the direct-API path maps model-tier ids to
    the account's Bedrock application inference profile ARNs (required
    instead of raw model ids for cost tracking).

Config:
  ANTHROPIC_API_KEY  — direct Anthropic API key (enables local dev path)
  LLM_PROVIDER       — "vertex" to force the GCP path; default is Bedrock
  GCP_PROJECT_ID     — GCP project ID (Vertex path)
  VERTEX_REGION      — Vertex AI region (default: us-east5)
  AWS_ASSUME_ROLE_ARN, AWS_REGION — see aws_auth.py (Bedrock path)
"""
import logging
import os
import socket
import time

import anthropic

import aws_auth

logger = logging.getLogger(__name__)


def _configure_hf_offline_mode() -> None:
    """
    Unrelated to the Anthropic client below — this lives here because
    llm_client.py is the one module every service's Dockerfile bundles and
    every service imports on startup, making it the natural single place to
    run a check that must happen once, early, before any of the several
    dialog_agent / knowledge_graph_agent modules lazily import
    sentence-transformers.

    sentence-transformers (used for dialog schema retrieval and KG bridge
    inference) hits huggingface.co on every model load — even when the model
    is already cached — to check for a newer revision. On a host/pod with no
    outbound access to huggingface.co, that check fails via DNS resolution
    and huggingface_hub retries 5x with exponential backoff (up to ~30s)
    *per call*; multiplied across every dialog query and KG bridge-inference
    pass, that stalls the pipeline for minutes.

    Do one cheap, bounded reachability probe per process instead of paying
    that cost on every call. Unreachable -> force offline mode for the life
    of the process, so every subsequent load fails (or serves from cache)
    instantly. Reachable -> leave online mode untouched, so an environment
    with real internet access (e.g. a cloud deployment) can still download
    the model the first time it's needed, exactly as before this existed.

    Never overrides an explicit HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE already
    set in the environment (e.g. by a launcher script or a future Docker/K8s
    config) — this is a fallback for when no explicit choice was made.
    """
    if os.environ.get("HF_HUB_OFFLINE") or os.environ.get("TRANSFORMERS_OFFLINE"):
        return

    # socket.create_connection's timeout only bounds the TCP connect step —
    # DNS resolution (getaddrinfo) happens first and is NOT bounded by it. On
    # a host with no route to the internet, getaddrinfo itself can block for
    # 15+ seconds before failing (confirmed on this Windows machine), which
    # would make the probe itself as slow as the problem it's meant to avoid.
    # Run it in a worker thread and enforce the timeout via Future.result()
    # instead, which IS interruptible regardless of how long the underlying
    # blocking call takes. shutdown(wait=False) avoids blocking on that
    # worker if it's still stuck in getaddrinfo when we give up on it — it
    # simply finishes in the background and is dropped.
    import concurrent.futures

    def _probe() -> None:
        socket.create_connection(("huggingface.co", 443), timeout=1.5).close()

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        executor.submit(_probe).result(timeout=1.5)
    except Exception:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        logger.info(
            "huggingface.co unreachable (or too slow to resolve) — forcing "
            "HF_HUB_OFFLINE for this process so sentence-transformers model "
            "loads fail fast instead of retrying"
        )
    finally:
        executor.shutdown(wait=False)


_configure_hf_offline_mode()

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


# Map a model-tier id to this account's Bedrock application inference profile
# ARN — required instead of the raw model id so Bedrock usage is attributed
# per-application for cost tracking.
_BEDROCK_HAIKU = "arn:aws:bedrock:us-east-1:336756484937:application-inference-profile/wfd1mwndgpsn"
_BEDROCK_SONNET = "arn:aws:bedrock:us-east-1:336756484937:application-inference-profile/qp3hg66g81b3"
_BEDROCK_OPUS = "arn:aws:bedrock:us-east-1:336756484937:application-inference-profile/5lrrvuwa9oy0"


def _normalize_bedrock_model(model: str) -> str:
    s = (model or "").lower()
    if "haiku" in s:
        return _BEDROCK_HAIKU
    if "opus" in s:
        return _BEDROCK_OPUS
    if "sonnet" in s:
        return _BEDROCK_SONNET
    return model


class _TimedMessages:
    """Proxy for client.messages that logs wall-clock time for create/stream,
    optionally rewriting kwargs first via `_prepare` (identity by default).
    Timing is purely observational — call args/kwargs and return values pass
    through unchanged, so behavior is identical to calling `inner` directly."""

    def __init__(self, inner):
        self._inner = inner

    def _prepare(self, kwargs):
        return kwargs

    def create(self, *args, **kwargs):
        kwargs = self._prepare(kwargs)
        model = kwargs.get("model", "")
        t0 = time.perf_counter()
        try:
            return self._inner.create(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.info("llm_timing model=%s elapsed_ms=%.1f", model, elapsed_ms)

    def stream(self, *args, **kwargs):
        # Only times how long it takes to open the stream (setup), not the
        # full token-by-token duration, since that's consumed by the caller
        # after this returns.
        kwargs = self._prepare(kwargs)
        model = kwargs.get("model", "")
        t0 = time.perf_counter()
        try:
            return self._inner.stream(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.info("llm_timing_stream_open model=%s elapsed_ms=%.1f", model, elapsed_ms)

    def count_tokens(self, *args, **kwargs):
        return self._inner.count_tokens(*args, **self._prepare(kwargs))

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _RemapMessages(_TimedMessages):
    """Proxy for client.messages that rewrites the `model` kwarg to a valid
    public-API id before delegating (covers create / stream / count_tokens)."""

    def _prepare(self, kwargs):
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


class _DirectClient:
    """Wraps anthropic.Anthropic so .messages remaps model ids; everything else
    passes through unchanged."""

    def __init__(self, inner):
        self._inner = inner
        self.messages = _RemapMessages(inner.messages)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _VertexClient:
    """Wraps AnthropicVertex so .messages calls are timed; everything else
    passes through unchanged (no model-id remapping needed on this path)."""

    def __init__(self, inner):
        self._inner = inner
        self.messages = _TimedMessages(inner.messages)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _BedrockRemapMessages(_TimedMessages):
    """Proxy for client.messages that rewrites the `model` kwarg to the
    matching Bedrock application inference profile ARN before delegating."""

    def _prepare(self, kwargs):
        if "model" in kwargs:
            kwargs["model"] = _normalize_bedrock_model(kwargs["model"])
        return kwargs


class _BedrockClient:
    """Wraps AnthropicBedrock so .messages calls remap model ids to inference
    profile ARNs and are timed; everything else passes through unchanged."""

    def __init__(self, inner):
        self._inner = inner
        self.messages = _BedrockRemapMessages(inner.messages)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def get_client():
    """Return an Anthropic-compatible client.

    Local dev:      set ANTHROPIC_API_KEY in .env → direct Anthropic API.
    Vertex (opt-in): set LLM_PROVIDER=vertex → Vertex AI with ADC
                     (run `gcloud auth application-default login` locally).
    Default/prod:   AWS Bedrock, authenticated via the cross-account role
                     assumed in aws_auth.py (STS) — no key needed at runtime.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if api_key:
        return _DirectClient(anthropic.Anthropic(api_key=api_key))

    if os.environ.get("LLM_PROVIDER", "").strip().lower() == "vertex":
        project_id = os.environ.get("GCP_PROJECT_ID", "cog01k24f1ea555zdv7ynzthxanz5")
        vertex_region = os.environ.get("VERTEX_REGION", "us-east5")
        return _VertexClient(anthropic.AnthropicVertex(region=vertex_region, project_id=project_id))

    creds = aws_auth.get_frozen_credentials()
    return _BedrockClient(anthropic.AnthropicBedrock(
        aws_access_key=creds.access_key,
        aws_secret_key=creds.secret_key,
        aws_session_token=creds.token,
        aws_region=aws_auth.AWS_REGION,
    ))
