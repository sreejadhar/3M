"""
Shared AWS cross-account role assumption helper.

Used by kg_store.py/neptune_store.py (Amazon Neptune) so every AWS-backed
service in this app can share one assumed role and one credential
cache/refresh implementation instead of each duplicating STS logic. (The
Bedrock LLM path in llm_client.py currently manages its own — see that file
if unifying the two later.)

Config:
  AWS_ROLE_ARN    — cross-account role to assume (defaults to the shared
                    datananite-dev-execution-role)
  AWS_ROLE_REGION — region for the STS client (default: us-east-1)
"""
from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_DEFAULT_ROLE_ARN = "arn:aws:iam::336756484937:role/datananite-dev-execution-role"

_lock = threading.Lock()
_cache: Dict[str, tuple] = {}  # role_arn -> (credentials_dict, expiry)


def assume_role(role_arn: Optional[str] = None, region: Optional[str] = None,
                 session_name: str = "datananite") -> Dict:
    """
    Return temporary credentials for role_arn, re-assuming via STS only when
    the cached ones are missing or within 5 minutes of expiring. Thread-safe.

    Returns the raw STS Credentials dict:
    {"AccessKeyId", "SecretAccessKey", "SessionToken", "Expiration"}.
    """
    role_arn = role_arn or os.environ.get("AWS_ROLE_ARN", _DEFAULT_ROLE_ARN)
    region = region or os.environ.get("AWS_ROLE_REGION", "us-east-1")
    now = datetime.now(timezone.utc)
    with _lock:
        cached = _cache.get(role_arn)
        if cached:
            creds, expiry = cached
            if now < expiry - timedelta(minutes=5):
                return creds

        import boto3
        sts = boto3.client("sts", region_name=region)
        resp = sts.assume_role(RoleArn=role_arn, RoleSessionName=session_name)
        creds = resp["Credentials"]
        _cache[role_arn] = (creds, creds["Expiration"])
        logger.info("aws_auth: assumed role %s, credentials valid until %s", role_arn, creds["Expiration"])
        return creds
