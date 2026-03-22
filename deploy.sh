#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# deploy.sh — Build and deploy all three containers:
#               agent-api      (metadata extraction  — port 8000)
#               ontology-api   (OWL/RDF generation   — port 8001)
#               ui             (Streamlit interface  — port 8501)
#
# Usage:
#   ./deploy.sh               Build images and start all containers
#   ./deploy.sh --build-only  Build images without starting
#   ./deploy.sh --start       Start pre-built images (no rebuild)
#   ./deploy.sh --restart     Restart running containers
#   ./deploy.sh --stop        Stop containers (keep volumes)
#   ./deploy.sh --down        Stop containers and remove volumes
#   ./deploy.sh --logs        Tail logs from all containers
#   ./deploy.sh --status      Show container status
#   ./deploy.sh --help        Show this help message
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Colours ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${BLUE}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*" >&2; }
header()  { echo -e "\n${BOLD}${CYAN}$*${RESET}\n"; }

# ── Config ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yml"
ENV_FILE="${SCRIPT_DIR}/.env"

AGENT_IMAGE="metadata-agent-api:latest"
ONTOLOGY_IMAGE="metadata-ontology-api:latest"
KG_IMAGE="metadata-kg-api:latest"
DIALOG_IMAGE="metadata-dialog-api:latest"
CHAT_IMAGE="metadata-chat-ui:latest"
UI_IMAGE="metadata-agent-ui:latest"

AGENT_PORT="${AGENT_PORT:-8000}"
ONTOLOGY_PORT="${ONTOLOGY_PORT:-8001}"
KG_PORT="${KG_PORT:-8002}"
DIALOG_PORT="${DIALOG_PORT:-8003}"
CHAT_PORT="${CHAT_PORT:-8005}"
UI_PORT="${UI_PORT:-8501}"

HEALTH_TIMEOUT=90   # seconds to wait for healthy status
HEALTH_INTERVAL=3   # seconds between health checks

# ── Helpers ────────────────────────────────────────────────────────────────────
require_binary() {
    if ! command -v "$1" &>/dev/null; then
        error "'$1' is not installed or not on PATH."
        exit 1
    fi
}

print_banner() {
    echo -e "${BOLD}"
    echo "  ██████████████████████████████████████████"
    echo "  ██  Metadata Agent Deployment Script    ██"
    echo "  ██████████████████████████████████████████"
    echo -e "${RESET}"
    echo -e "  Agent API    → ${CYAN}http://localhost:${AGENT_PORT}${RESET}"
    echo -e "  Ontology API → ${CYAN}http://localhost:${ONTOLOGY_PORT}${RESET}"
    echo -e "  KG API       → ${CYAN}http://localhost:${KG_PORT}${RESET}"
    echo -e "  Dialog API   → ${CYAN}http://localhost:${DIALOG_PORT}${RESET}"
    echo -e "  DataChat UI  → ${CYAN}http://localhost:${CHAT_PORT}${RESET}  ← business user interface"
    echo -e "  UI           → ${CYAN}http://localhost:${UI_PORT}${RESET}"
    echo ""
}

check_env() {
    # Load .env if present
    if [[ -f "$ENV_FILE" ]]; then
        info "Loading environment from ${ENV_FILE}"
        set -a
        # shellcheck disable=SC1090
        source "$ENV_FILE"
        set +a
    fi

    if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
        warn "ANTHROPIC_API_KEY is not set. LLM features will be unavailable."
        warn "Set it in .env or export it: export ANTHROPIC_API_KEY=sk-ant-..."
    else
        success "ANTHROPIC_API_KEY detected."
    fi
}

wait_healthy() {
    local service="$1"
    local url="$2"
    local elapsed=0

    info "Waiting for ${service} to become healthy (timeout: ${HEALTH_TIMEOUT}s)…"
    while [[ $elapsed -lt $HEALTH_TIMEOUT ]]; do
        if curl -sf "${url}" &>/dev/null; then
            success "${service} is healthy."
            return 0
        fi
        sleep "${HEALTH_INTERVAL}"
        elapsed=$((elapsed + HEALTH_INTERVAL))
        echo -ne "  ${YELLOW}⟳${RESET} ${elapsed}s elapsed\r"
    done

    error "${service} did not become healthy within ${HEALTH_TIMEOUT}s."
    error "Check logs with:  ./deploy.sh --logs"
    return 1
}

show_status() {
    header "Container Status"
    docker compose -f "$COMPOSE_FILE" ps
}

# ── Actions ────────────────────────────────────────────────────────────────────
cmd_build() {
    header "Building Docker Images"

    info "Building agent-api image…"
    docker build \
        --file "${SCRIPT_DIR}/Dockerfile.agent" \
        --tag  "${AGENT_IMAGE}" \
        "${SCRIPT_DIR}"
    success "Built ${AGENT_IMAGE}"

    info "Building ontology-api image…"
    docker build \
        --file "${SCRIPT_DIR}/Dockerfile.ontology" \
        --tag  "${ONTOLOGY_IMAGE}" \
        "${SCRIPT_DIR}"
    success "Built ${ONTOLOGY_IMAGE}"

    info "Building kg-api image…"
    docker build \
        --file "${SCRIPT_DIR}/Dockerfile.kg" \
        --tag  "${KG_IMAGE}" \
        "${SCRIPT_DIR}"
    success "Built ${KG_IMAGE}"

    info "Building dialog-api image…"
    docker build \
        --file "${SCRIPT_DIR}/Dockerfile.dialog" \
        --tag  "${DIALOG_IMAGE}" \
        "${SCRIPT_DIR}"
    success "Built ${DIALOG_IMAGE}"

    info "Building chat-ui (orchestrator) image…"
    docker build \
        --file "${SCRIPT_DIR}/Dockerfile.chat" \
        --tag  "${CHAT_IMAGE}" \
        "${SCRIPT_DIR}"
    success "Built ${CHAT_IMAGE}"

    info "Building ui image…"
    docker build \
        --file "${SCRIPT_DIR}/Dockerfile.ui" \
        --tag  "${UI_IMAGE}" \
        "${SCRIPT_DIR}"
    success "Built ${UI_IMAGE}"
}

cmd_start() {
    header "Starting Containers"
    docker compose -f "$COMPOSE_FILE" up --detach --remove-orphans

    echo ""
    wait_healthy "agent-api"    "http://localhost:${AGENT_PORT}/health"
    wait_healthy "ontology-api" "http://localhost:${ONTOLOGY_PORT}/health"
    wait_healthy "kg-api"       "http://localhost:${KG_PORT}/health"
    wait_healthy "dialog-api"   "http://localhost:${DIALOG_PORT}/health"
    wait_healthy "chat-ui"      "http://localhost:${CHAT_PORT}/health"
    wait_healthy "ui"           "http://localhost:${UI_PORT}/_stcore/health"

    echo ""
    success "All services are up!"
    echo ""
    echo -e "  ${BOLD}Agent API    ${RESET} →  ${CYAN}http://localhost:${AGENT_PORT}${RESET}"
    echo -e "  ${BOLD}Agent Docs   ${RESET} →  ${CYAN}http://localhost:${AGENT_PORT}/docs${RESET}"
    echo -e "  ${BOLD}Ontology API ${RESET} →  ${CYAN}http://localhost:${ONTOLOGY_PORT}${RESET}"
    echo -e "  ${BOLD}Ontology Docs${RESET} →  ${CYAN}http://localhost:${ONTOLOGY_PORT}/docs${RESET}"
    echo -e "  ${BOLD}KG API       ${RESET} →  ${CYAN}http://localhost:${KG_PORT}${RESET}"
    echo -e "  ${BOLD}KG Docs      ${RESET} →  ${CYAN}http://localhost:${KG_PORT}/docs${RESET}"
    echo -e "  ${BOLD}Dialog API   ${RESET} →  ${CYAN}http://localhost:${DIALOG_PORT}${RESET}"
    echo -e "  ${BOLD}Dialog Docs  ${RESET} →  ${CYAN}http://localhost:${DIALOG_PORT}/docs${RESET}"
    echo -e "  ${BOLD}DataChat UI  ${RESET} →  ${CYAN}http://localhost:${CHAT_PORT}${RESET}  ← open this in your browser"
    echo -e "  ${BOLD}UI           ${RESET} →  ${CYAN}http://localhost:${UI_PORT}${RESET}"
    echo ""
}

cmd_stop() {
    header "Stopping Containers"
    docker compose -f "$COMPOSE_FILE" stop
    success "Containers stopped (volumes preserved)."
}

cmd_down() {
    header "Removing Containers & Volumes"
    warn "This will delete all saved reports!"
    read -r -p "Are you sure? [y/N] " confirm
    if [[ "${confirm}" =~ ^[Yy]$ ]]; then
        docker compose -f "$COMPOSE_FILE" down --volumes --remove-orphans
        success "Containers and volumes removed."
    else
        info "Aborted."
    fi
}

cmd_restart() {
    header "Restarting Containers"
    docker compose -f "$COMPOSE_FILE" restart
    echo ""
    wait_healthy "agent-api"    "http://localhost:${AGENT_PORT}/health"
    wait_healthy "ontology-api" "http://localhost:${ONTOLOGY_PORT}/health"
    wait_healthy "kg-api"       "http://localhost:${KG_PORT}/health"
    wait_healthy "dialog-api"   "http://localhost:${DIALOG_PORT}/health"
    wait_healthy "chat-ui"      "http://localhost:${CHAT_PORT}/health"
    success "Services restarted."
}

cmd_logs() {
    header "Container Logs"
    info "Press Ctrl+C to stop following logs."
    docker compose -f "$COMPOSE_FILE" logs --follow --tail=100
}

cmd_help() {
    echo ""
    echo -e "${BOLD}Usage:${RESET} ./deploy.sh [OPTION]"
    echo ""
    echo "  (no option)      Build images and start both containers"
    echo "  --build-only     Build images without starting"
    echo "  --start          Start pre-built images (no rebuild)"
    echo "  --restart        Restart running containers"
    echo "  --stop           Stop containers (keep volumes)"
    echo "  --down           Stop containers AND delete volumes"
    echo "  --logs           Tail logs from all containers"
    echo "  --status         Show container status"
    echo "  --help           Show this help message"
    echo ""
    echo -e "${BOLD}Environment variables (set in .env or export):${RESET}"
    echo "  ANTHROPIC_API_KEY   Required for LLM Q&A features"
    echo "  AGENT_PORT          Agent API port      (default: 8000)"
    echo "  ONTOLOGY_PORT       Ontology API port   (default: 8001)"
    echo "  KG_PORT             KG API port         (default: 8002)"
    echo "  DIALOG_PORT         Dialog API port     (default: 8003)"
    echo "  CHAT_PORT           DataChat UI port    (default: 8005)"
    echo "  UI_PORT             Streamlit UI port   (default: 8501)"
    echo "  LOG_LEVEL           debug | info | warning (default: info)"
    echo ""
}

# ── Pre-flight checks ──────────────────────────────────────────────────────────
require_binary docker
require_binary curl

# Check docker compose (v2 plugin or standalone)
if ! docker compose version &>/dev/null 2>&1; then
    require_binary docker-compose   # fallback to v1
    COMPOSE_CMD="docker-compose"
else
    COMPOSE_CMD="docker compose"
fi

# ── Argument dispatch ──────────────────────────────────────────────────────────
print_banner
check_env

ARG="${1:-}"

case "$ARG" in
    --build-only)
        cmd_build
        success "Build complete. Run './deploy.sh --start' to launch."
        ;;
    --start)
        cmd_start
        ;;
    --restart)
        cmd_restart
        ;;
    --stop)
        cmd_stop
        ;;
    --down)
        cmd_down
        ;;
    --logs)
        cmd_logs
        ;;
    --status)
        show_status
        ;;
    --help|-h)
        cmd_help
        ;;
    "")
        # Default: build then start
        cmd_build
        cmd_start
        ;;
    *)
        error "Unknown option: $ARG"
        cmd_help
        exit 1
        ;;
esac
