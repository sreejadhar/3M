"""
Central LLM client factory for DataNanite.
All services import get_client() from here — switching LLM providers
requires changes only in this file.

Provider selection:
  • If ANTHROPIC_API_KEY is set (local dev), use the direct Anthropic API.
    Loose model ids (claude-sonnet-4-6, claude-haiku-4-5, claude-opus-4-x)
    are normalised by tier to valid public-API ids so the ~20 call sites that
    pass those ids keep working unchanged.
  • Otherwise use AWS Bedrock (AnthropicBedrock) via a cross-account IAM role
    assumed through STS — the production path. The host (EC2 instance role,
    or whatever credentials boto3's default chain finds) must be allowed to
    sts:AssumeRole on BEDROCK_ROLE_ARN; no API key is needed at runtime.
    Model ids are mapped by tier to the application inference profile ARNs
    below (required by Bedrock for cost tracking — invoking the bare
    foundation-model id instead would bypass that tracking).

Config:
  ANTHROPIC_API_KEY         — direct Anthropic API key (enables local dev path)
  BEDROCK_ROLE_ARN          — cross-account role to assume for Bedrock access
  BEDROCK_REGION            — Bedrock region (default: us-east-1)
  BEDROCK_HAIKU_PROFILE_ARN  — application inference profile ARN for Haiku
  BEDROCK_SONNET_PROFILE_ARN — application inference profile ARN for Sonnet
  BEDROCK_OPUS_PROFILE_ARN   — application inference profile ARN for Opus
"""
import logging
import os
import socket
import threading
import time
from datetime import datetime, timedelta, timezone

import anthropic

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

# Map a loose tier-style model id to a valid public Anthropic API id by tier.
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
        # The pipeline was built for 1M-token-context models in production,
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


_BEDROCK_ROLE_ARN = os.environ.get(
    "BEDROCK_ROLE_ARN",
    "arn:aws:iam::336756484937:role/datananite-dev-execution-role",
)
_BEDROCK_REGION = os.environ.get("BEDROCK_REGION", "us-east-1")

# Application inference profile ARNs — Bedrock requires invoking these rather
# than the bare foundation-model id so usage/cost is attributed correctly.
_BEDROCK_PROFILE_ARNS = {
    "haiku": os.environ.get(
        "BEDROCK_HAIKU_PROFILE_ARN",
        "arn:aws:bedrock:us-east-1:336756484937:application-inference-profile/wfd1mwndgpsn",
    ),
    "sonnet": os.environ.get(
        "BEDROCK_SONNET_PROFILE_ARN",
        "arn:aws:bedrock:us-east-1:336756484937:application-inference-profile/qp3hg66g81b3",
    ),
    "opus": os.environ.get(
        "BEDROCK_OPUS_PROFILE_ARN",
        "arn:aws:bedrock:us-east-1:336756484937:application-inference-profile/5lrrvuwa9oy0",
    ),
}


def _normalize_bedrock_model(model: str) -> str:
    s = (model or "").lower()
    if "haiku" in s:
        return _BEDROCK_PROFILE_ARNS["haiku"]
    if "opus" in s:
        return _BEDROCK_PROFILE_ARNS["opus"]
    if "sonnet" in s:
        return _BEDROCK_PROFILE_ARNS["sonnet"]
    return model


class _RemapBedrockMessages(_TimedMessages):
    """Proxy for client.messages that rewrites the `model` kwarg from a loose
    tier-style id to the matching application inference profile ARN."""

    def _prepare(self, kwargs):
        if "model" in kwargs:
            kwargs["model"] = _normalize_bedrock_model(kwargs["model"])
        return kwargs


class _BedrockClient:
    """Wraps AnthropicBedrock so .messages remaps model ids to inference
    profile ARNs; everything else passes through unchanged."""

    def __init__(self, inner):
        self._inner = inner
        self.messages = _RemapBedrockMessages(inner.messages)

    def __getattr__(self, name):
        return getattr(self._inner, name)


# Assumed-role credentials expire (default STS session is 1h) — cache the
# client and re-assume the role shortly before expiry instead of on every
# call, and guard the refresh with a lock since multiple request threads can
# race to refresh at once in a multi-worker/multi-thread service.
_bedrock_lock = threading.Lock()
_bedrock_client = None
_bedrock_expiry = None


def _assume_bedrock_role():
    import boto3

    sts = boto3.client("sts", region_name=_BEDROCK_REGION)
    resp = sts.assume_role(
        RoleArn=_BEDROCK_ROLE_ARN,
        RoleSessionName="datananite-bedrock",
    )
    return resp["Credentials"]


def _get_bedrock_client():
    global _bedrock_client, _bedrock_expiry
    now = datetime.now(timezone.utc)
    with _bedrock_lock:
        if _bedrock_client is None or _bedrock_expiry is None or now >= _bedrock_expiry - timedelta(minutes=5):
            creds = _assume_bedrock_role()
            inner = anthropic.AnthropicBedrock(
                aws_access_key=creds["AccessKeyId"],
                aws_secret_key=creds["SecretAccessKey"],
                aws_session_token=creds["SessionToken"],
                aws_region=_BEDROCK_REGION,
            )
            _bedrock_client = _BedrockClient(inner)
            _bedrock_expiry = creds["Expiration"]
            logger.info(
                "llm_client: assumed Bedrock role %s, credentials valid until %s",
                _BEDROCK_ROLE_ARN, _bedrock_expiry,
            )
    return _bedrock_client


def get_client():
    """Return an Anthropic-compatible client.

    Local dev: set ANTHROPIC_API_KEY in .env → direct Anthropic API.
    Prod:      leave it unset → AWS Bedrock via an assumed cross-account role
               (boto3's default credential chain must be able to reach
               BEDROCK_ROLE_ARN — e.g. the EC2 instance role in production).
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if api_key:
        return _DirectClient(anthropic.Anthropic(api_key=api_key))

    return _get_bedrock_client()
