"""
Central LLM client factory for DataNanite.
All services import get_client() from here — switching LLM providers
requires changes only in this file.

Provider:  GCP Vertex AI (AnthropicVertex)
Auth:      Application Default Credentials (ADC) — inside GKE pods this
           automatically uses the node service account which has
           roles/aiplatform.user. No API key needed at runtime.
Config:    GCP_PROJECT_ID and VERTEX_REGION injected from GitHub Actions
           secrets into K8s secret datananite-secrets → pod env vars.
"""
import os
import anthropic


def get_client() -> anthropic.AnthropicVertex:
    """
    Returns an AnthropicVertex client using Application Default Credentials.

    Inside GKE pods: uses node service account (has aiplatform.user) — no key needed.
    Locally: run `gcloud auth application-default login` first.

    Config from env vars (injected by CI/CD from GitHub secrets):
      GCP_PROJECT_ID  — GCP project ID
      VERTEX_REGION   — Vertex AI region (default: us-east5)
    """
    project_id    = os.environ.get("GCP_PROJECT_ID", "cog01k24f1ea555zdv7ynzthxanz5")
    vertex_region = os.environ.get("VERTEX_REGION", "us-east5")

    return anthropic.AnthropicVertex(
        region=vertex_region,
        project_id=project_id,
    )
