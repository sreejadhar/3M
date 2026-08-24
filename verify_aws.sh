#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# verify_aws.sh — smoke-test the AWS setup DataNanite depends on before/after
# deploying to EC2 (datananited01.mmm.com):
#   1. Assume the cross-account execution role (Bedrock + Neptune + RDS auth)
#   2. Confirm network reachability to the Neptune cluster (writer + reader)
#   3. Invoke a Bedrock model through one of the application inference profiles
#   4. Fetch DB credentials from Secrets Manager and confirm RDS reachability
#
# Usage:
#   ./verify_aws.sh
#
# Requires: aws CLI (configured with credentials/instance role that can
# sts:AssumeRole the execution role below), nc (netcat).
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${BLUE}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*" >&2; }
header()  { echo -e "\n${BOLD}$*${RESET}\n"; }

AWS_REGION="${AWS_REGION:-us-east-1}"
ASSUME_ROLE_ARN="${AWS_ASSUME_ROLE_ARN:-arn:aws:iam::336756484937:role/datananite-dev-execution-role}"
NEPTUNE_WRITER_ENDPOINT="${NEPTUNE_WRITER_ENDPOINT:-datananite-dev-neptune.cluster-c676y6esoazm.us-east-1.neptune.amazonaws.com}"
NEPTUNE_READER_ENDPOINT="${NEPTUNE_READER_ENDPOINT:-datananite-dev-neptune.cluster-ro-c676y6esoazm.us-east-1.neptune.amazonaws.com}"
NEPTUNE_PORT="${NEPTUNE_PORT:-8182}"

PG_SECRET_ID="${PG_SECRET_ID:-datananite/rds/app-user}"
PG_HOST="${PG_HOST:-pg-rds-datananite-dev-001a.cuepp5apko9u.us-east-1.rds.amazonaws.com}"
PG_PORT="${PG_PORT:-5432}"

# One profile ARN per tier — verified in step 3 below.
BEDROCK_HAIKU_PROFILE="arn:aws:bedrock:us-east-1:336756484937:application-inference-profile/wfd1mwndgpsn"
BEDROCK_SONNET_PROFILE="arn:aws:bedrock:us-east-1:336756484937:application-inference-profile/qp3hg66g81b3"
BEDROCK_OPUS_PROFILE="arn:aws:bedrock:us-east-1:336756484937:application-inference-profile/5lrrvuwa9oy0"
BEDROCK_PROFILE_ARN="${1:-$BEDROCK_HAIKU_PROFILE}"

FAIL=0

command -v aws &>/dev/null || { error "aws CLI not found — install it first."; exit 1; }
command -v nc  &>/dev/null || { error "nc (netcat) not found — install it first."; exit 1; }

# ── 1. Assume the execution role ─────────────────────────────────────────────
header "1. Assuming execution role"
info "Role: ${ASSUME_ROLE_ARN}"

ASSUME_OUT=$(aws sts assume-role \
    --role-arn "$ASSUME_ROLE_ARN" \
    --role-session-name "datananite-verify" \
    --region "$AWS_REGION" \
    --query 'Credentials.[AccessKeyId,SecretAccessKey,SessionToken,Expiration]' \
    --output text 2>&1) || {
    error "sts assume-role failed:"
    echo "$ASSUME_OUT" >&2
    exit 1
}

read -r ROLE_AKID ROLE_SECRET ROLE_TOKEN ROLE_EXPIRY <<< "$ASSUME_OUT"
success "Assumed role — temporary credentials expire at ${ROLE_EXPIRY}."

# Use the assumed-role credentials for the rest of this script, so Bedrock
# access is verified through the exact same path the app uses at runtime.
export AWS_ACCESS_KEY_ID="$ROLE_AKID"
export AWS_SECRET_ACCESS_KEY="$ROLE_SECRET"
export AWS_SESSION_TOKEN="$ROLE_TOKEN"

# ── 2. Neptune reachability ──────────────────────────────────────────────────
header "2. Testing Neptune reachability (port ${NEPTUNE_PORT})"

for endpoint in "$NEPTUNE_WRITER_ENDPOINT" "$NEPTUNE_READER_ENDPOINT"; do
    if nc -zv -w 5 "$endpoint" "$NEPTUNE_PORT" 2>&1 | tee /dev/stderr | grep -qi "succeeded\|open"; then
        success "Reachable: ${endpoint}:${NEPTUNE_PORT}"
    else
        error "NOT reachable: ${endpoint}:${NEPTUNE_PORT}"
        error "Check security groups / VPC routing between this host and the Neptune cluster."
        FAIL=1
    fi
done

# ── 3. Invoke a Bedrock model ─────────────────────────────────────────────────
header "3. Invoking Bedrock via inference profile"
info "Profile: ${BEDROCK_PROFILE_ARN}"

BODY='{"anthropic_version":"bedrock-2023-05-31","max_tokens":16,"messages":[{"role":"user","content":"Reply with exactly one word: OK"}]}'
OUT_FILE="$(mktemp)"

if aws bedrock-runtime invoke-model \
        --region "$AWS_REGION" \
        --model-id "$BEDROCK_PROFILE_ARN" \
        --body "$BODY" \
        --cli-binary-format raw-in-base64-out \
        "$OUT_FILE" >/dev/null 2>"${OUT_FILE}.err"; then
    success "Bedrock responded:"
    cat "$OUT_FILE"
    echo ""
else
    error "Bedrock invoke-model failed:"
    cat "${OUT_FILE}.err" >&2
    FAIL=1
fi
rm -f "$OUT_FILE" "${OUT_FILE}.err"

# ── 4. RDS PostgreSQL: Secrets Manager fetch + reachability ──────────────────
header "4. Fetching DB credentials from Secrets Manager and testing RDS reachability"
info "Secret: ${PG_SECRET_ID}"

SECRET_JSON=$(aws secretsmanager get-secret-value \
    --secret-id "$PG_SECRET_ID" \
    --region "$AWS_REGION" \
    --query 'SecretString' \
    --output text 2>&1) || {
    error "secretsmanager get-secret-value failed:"
    echo "$SECRET_JSON" >&2
    FAIL=1
}

if [[ -z "${SECRET_JSON:-}" ]]; then
    :  # already recorded as a failure above
elif ! echo "$SECRET_JSON" | python3 -c "import json,sys; d=json.load(sys.stdin); assert 'username' in d and 'password' in d" 2>/dev/null; then
    error "Secret ${PG_SECRET_ID} does not contain both 'username' and 'password' keys."
    FAIL=1
else
    success "Secret fetched and has the expected username/password shape."
fi

if nc -zv -w 5 "$PG_HOST" "$PG_PORT" 2>&1 | tee /dev/stderr | grep -qi "succeeded\|open"; then
    success "Reachable: ${PG_HOST}:${PG_PORT}"
else
    error "NOT reachable: ${PG_HOST}:${PG_PORT}"
    error "Check security groups / VPC routing between this host and the RDS instance."
    FAIL=1
fi

# ── Summary ────────────────────────────────────────────────────────────────────
header "Summary"
if [[ $FAIL -eq 0 ]]; then
    success "All checks passed — Bedrock, Neptune, and RDS are reachable via the assumed role."
else
    error "One or more checks failed — see above."
    exit 1
fi
