"""
Central LLM client factory for DataNanite.
All services import get_client() from here — switching LLM providers
requires changes only in this file.

Provider:  GCP Vertex AI (AnthropicVertex)
Auth:      ANTHROPIC_API_KEY env var holds the GCP API key.
           Injected by CI/CD from GitHub Actions secret ANTHROPIC_API_KEY.
Config:    GCP_PROJECT_ID and VERTEX_REGION injected from GitHub Actions secrets
           into K8s secret datananite-secrets → available as pod env vars.
"""
import os
import anthropic


def get_client() -> anthropic.AnthropicVertex:
    """
    Returns an AnthropicVertex client authenticated via GCP API key.

    All config is read from environment variables injected by CI/CD:
      ANTHROPIC_API_KEY  — GCP API key (from GitHub Secret)
      GCP_PROJECT_ID     — GCP project ID (from GitHub Secret)
      VERTEX_REGION      — Vertex AI region (from GitHub Secret)
    """
    gcp_api_key   = os.environ.get("ANTHROPIC_API_KEY", "")
    project_id    = os.environ.get("GCP_PROJECT_ID", "")
    vertex_region = os.environ.get("VERTEX_REGION", "us-east5")

    if not project_id:
        raise ValueError(
            "GCP_PROJECT_ID env var is not set. "
            "Ensure it is configured in GitHub Actions secrets and injected into the K8s secret."
        )

    if gcp_api_key:
        os.environ["GOOGLE_API_KEY"] = gcp_api_key

    return anthropic.AnthropicVertex(
        region=vertex_region,
        project_id=project_id,
    )
