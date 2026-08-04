"""
One-time helper: download the all-MiniLM-L6-v2 sentence-transformers model
and save it as a plain local folder, for uploading to the shared PVC that
dialog-api/orchestrator-api/chat-ui mount at /data — no huggingface.co
access needed at build time or at container runtime once it's there.

Run this ONCE, from a machine that has internet access to huggingface.co
(NOT the CI build runner, NOT a deployed pod — both are network-restricted
in this environment):

    pip install sentence-transformers
    python scripts/download_embedding_model.py

This writes to models/all-MiniLM-L6-v2/ (repo root, gitignored — it's a
one-time transfer artifact, not something to commit). See
scripts/bootstrap_embedding_model_pvc.md for how to get it onto the PVC.
"""
from __future__ import annotations

import pathlib

from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "models" / MODEL_NAME


def main() -> None:
    print(f"Downloading {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)
    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(OUTPUT_DIR))
    print(f"Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
