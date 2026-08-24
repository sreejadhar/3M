"""
Shared AWS cross-account role assumption for DataNanite's AWS deployment (EC2
app server datananited01.mmm.com). Both Bedrock (llm_client.py) and Neptune
(neptune_store.py) authenticate with the temporary credentials this module
vends — no static API keys or DB passwords.

Config:
  AWS_ASSUME_ROLE_ARN — cross-account execution role to assume
                         (default: datananite-dev-execution-role)
  AWS_REGION           — region for STS/Bedrock/Neptune (default: us-east-1)
"""
from __future__ import annotations

import os
import threading
import time

import boto3

_ASSUME_ROLE_ARN = os.environ.get(
    "AWS_ASSUME_ROLE_ARN",
    "arn:aws:iam::336756484937:role/datananite-dev-execution-role",
)
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

_lock = threading.Lock()
_cached_session: "boto3.Session | None" = None
_cached_expiry = 0.0


def get_session() -> boto3.Session:
    """Return a boto3 Session backed by temporary credentials from the assumed
    cross-account role, refreshing ~5 minutes before they expire."""
    global _cached_session, _cached_expiry
    with _lock:
        if _cached_session is not None and time.time() < _cached_expiry - 300:
            return _cached_session

        sts = boto3.client("sts", region_name=AWS_REGION)
        creds = sts.assume_role(
            RoleArn=_ASSUME_ROLE_ARN,
            RoleSessionName="datananite-app",
        )["Credentials"]

        _cached_session = boto3.Session(
            aws_access_key_id=creds["AccessKeyId"],
            aws_secret_access_key=creds["SecretAccessKey"],
            aws_session_token=creds["SessionToken"],
            region_name=AWS_REGION,
        )
        _cached_expiry = creds["Expiration"].timestamp()
        return _cached_session


def get_frozen_credentials():
    """Convenience helper: resolved (access_key, secret_key, token) from the
    assumed-role session, for SDKs (e.g. AnthropicBedrock) that want raw
    credentials rather than a boto3 Session/Credentials object."""
    return get_session().get_credentials().get_frozen_credentials()
