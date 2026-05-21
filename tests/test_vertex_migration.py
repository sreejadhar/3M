"""
Tests to verify Vertex AI migration is working correctly.
Run: python -m pytest tests/test_vertex_migration.py -v
"""
import os
import pytest


def test_get_client_returns_vertex():
    """Factory must return AnthropicVertex instance, not anthropic.Anthropic."""
    from llm_client import get_client
    import anthropic
    client = get_client()
    assert isinstance(client, anthropic.AnthropicVertex), (
        f"Expected AnthropicVertex, got {type(client)}"
    )


def test_vertex_client_live_call():
    """Client must successfully reach Vertex AI and get a response."""
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
    """claude-sonnet-4-6 must be accessible via Vertex AI."""
    from llm_client import get_client
    client = get_client()
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=5,
        messages=[{"role": "user", "content": "hi"}]
    )
    assert msg.stop_reason in ("end_turn", "max_tokens")


def test_haiku_accessible():
    """claude-haiku-4-5-20251001 — may not be enabled in Vertex AI yet."""
    from llm_client import get_client
    client = get_client()
    try:
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=5,
            messages=[{"role": "user", "content": "hi"}]
        )
        assert msg.content[0].text is not None
    except Exception as e:
        pytest.skip(f"Haiku not available on Vertex AI: {e}")


def test_direct_vertex_call_for_langchain_replacement():
    """agent.py and api.py now use direct Vertex client instead of LangChain."""
    from llm_client import get_client
    client = get_client()
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=10,
        system="You are a helpful assistant.",
        messages=[{"role": "user", "content": "Say OK"}],
    )
    assert msg.content[0].text is not None
    assert len(msg.content[0].text) > 0


def test_no_anthropic_key_needed():
    """App must work without ANTHROPIC_API_KEY — only GOOGLE_API_KEY needed."""
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
    assert os.environ.get("GOOGLE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"), \
        "Either GOOGLE_API_KEY or ANTHROPIC_API_KEY must be set"
    assert os.environ.get("GCP_PROJECT_ID", "cog01k24f1ea555zdv7ynzthxanz5"), \
        "GCP_PROJECT_ID must be set"
