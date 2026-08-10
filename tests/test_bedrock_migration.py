"""
Tests to verify AWS Bedrock migration is working correctly.
Run: python -m pytest tests/test_bedrock_migration.py -v
"""
import os
import pytest


def test_get_client_returns_bedrock():
    """Factory must return AnthropicBedrock instance, not anthropic.Anthropic."""
    from llm_client import get_client
    import anthropic
    client = get_client()
    assert isinstance(client._inner, anthropic.AnthropicBedrock), (
        f"Expected AnthropicBedrock, got {type(client._inner)}"
    )


def test_bedrock_client_live_call():
    """Client must successfully reach Bedrock and get a response."""
    from llm_client import get_client
    client = get_client()
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=10,
        messages=[{"role": "user", "content": "Say OK"}]
    )
    assert msg.content[0].text is not None
    assert len(msg.content[0].text) > 0


def test_sonnet_accessible():
    """claude-sonnet-4-6 must be accessible via its Bedrock inference profile."""
    from llm_client import get_client
    client = get_client()
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=5,
        messages=[{"role": "user", "content": "hi"}]
    )
    assert msg.stop_reason in ("end_turn", "max_tokens")


def test_haiku_accessible():
    """claude-haiku-4-5 must be accessible via its Bedrock inference profile."""
    from llm_client import get_client
    client = get_client()
    msg = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=5,
        messages=[{"role": "user", "content": "hi"}]
    )
    assert msg.content[0].text is not None


def test_model_ids_map_to_inference_profile_arns():
    """Loose tier-style model ids must be rewritten to the configured
    application inference profile ARNs before hitting Bedrock."""
    from llm_client import _normalize_bedrock_model, _BEDROCK_PROFILE_ARNS
    assert _normalize_bedrock_model("claude-haiku-4-5") == _BEDROCK_PROFILE_ARNS["haiku"]
    assert _normalize_bedrock_model("claude-sonnet-4-6") == _BEDROCK_PROFILE_ARNS["sonnet"]
    assert _normalize_bedrock_model("claude-opus-4-8") == _BEDROCK_PROFILE_ARNS["opus"]


def test_no_anthropic_key_needed():
    """App must work without ANTHROPIC_API_KEY — Bedrock role assumption only."""
    saved = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        from llm_client import get_client
        client = get_client()
        assert client is not None
    finally:
        if saved:
            os.environ["ANTHROPIC_API_KEY"] = saved


def test_environment_variables_set():
    """Required env vars must be present in the environment."""
    assert os.environ.get("BEDROCK_ROLE_ARN") or os.environ.get("ANTHROPIC_API_KEY"), \
        "Either BEDROCK_ROLE_ARN (prod) or ANTHROPIC_API_KEY (local dev) must be set"
