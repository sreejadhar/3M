#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# prerequisite.sh — EC2 bootstrap for the Metadata Agent + DataChat solution
#
# Run this ONCE on a fresh EC2 instance before running deploy.sh.
#
# What it does:
#   1. Detects the Linux distro (Amazon Linux 2/2023, Ubuntu, Debian, RHEL/CentOS)
#   2. Installs Docker Engine + Docker Compose v2 plugin
#   3. Starts & enables Docker, adds the current user to the docker group
#   4. Installs curl and git (used by deploy.sh and cloning the repo)
#   5. Configures Docker daemon for log rotation (prevents disk bloat)
#   6. Adds a swap file on low-memory instances (< 4 GB RAM)
#   7. Checks free disk space (Docker image builds need ≥ 10 GB)
#   8. Creates a .env template if one does not already exist
#   9. Makes deploy.sh executable
#  10. Prints a verification summary and next-steps guide
#
# Supported AMIs:
#   Amazon Linux 2          (al2)
#   Amazon Linux 2023       (al2023)
#   Ubuntu 20.04 / 22.04 / 24.04
#   Debian 11 / 12
#   CentOS / RHEL 7 (yum) and 8+ (dnf)
#
# Usage:
#   chmod +x prerequisite.sh && sudo ./prerequisite.sh
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${BLUE}[INFO]${RESET}    $*"; }
success() { echo -e "${GREEN}[OK]${RESET}      $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}    $*"; }
error()   { echo -e "${RED}[ERROR]${RESET}   $*" >&2; }
header()  { echo -e "\n${BOLD}${CYAN}── $* ──${RESET}\n"; }
step()    { echo -e "${BOLD}▸ $*${RESET}"; }

# ── Must run as root ──────────────────────────────────────────────────────────
if [[ $EUID -ne 0 ]]; then
    error "This script must be run as root: sudo ./prerequisite.sh"
    exit 1
fi

# ── Script directory (repo root) ──────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Banner ────────────────────────────────────────────────────────────────────
echo -e "${BOLD}"
echo "  ╔══════════════════════════════════════════════════════╗"
echo "  ║    Metadata Agent + DataChat — EC2 Prerequisites     ║"
echo "  ╚══════════════════════════════════════════════════════╝"
echo -e "${RESET}"

# ── Detect OS ─────────────────────────────────────────────────────────────────
header "Detecting Operating System"

OS_ID=""
OS_VERSION_ID=""
if [[ -f /etc/os-release ]]; then
    # shellcheck disable=SC1091
    source /etc/os-release
    OS_ID="${ID:-}"
    OS_VERSION_ID="${VERSION_ID:-}"
fi

# Normalise Amazon Linux
if [[ "$OS_ID" == "amzn" ]]; then
    if [[ "$OS_VERSION_ID" == "2" ]]; then
        OS_FLAVOUR="al2"
    else
        OS_FLAVOUR="al2023"   # Amazon Linux 2023 reports VERSION_ID="2023"
    fi
elif [[ "$OS_ID" == "ubuntu" ]]; then
    OS_FLAVOUR="ubuntu"
elif [[ "$OS_ID" == "debian" ]]; then
    OS_FLAVOUR="debian"
elif [[ "$OS_ID" =~ ^(centos|rhel|rocky|almalinux)$ ]]; then
    if [[ "${OS_VERSION_ID%%.*}" -ge 8 ]]; then
        OS_FLAVOUR="rhel8"
    else
        OS_FLAVOUR="rhel7"
    fi
else
    warn "Unrecognised distro '${OS_ID}'. Will attempt Ubuntu/Debian path."
    OS_FLAVOUR="ubuntu"
fi

success "Detected: ${OS_ID} ${OS_VERSION_ID} → flavour=${OS_FLAVOUR}"

# ── Capture the login user (non-root) ────────────────────────────────────────
# When invoked with sudo, SUDO_USER is the real user; fall back to current user.
LOGIN_USER="${SUDO_USER:-$(logname 2>/dev/null || echo "$USER")}"
if [[ "$LOGIN_USER" == "root" ]]; then
    warn "Could not detect a non-root login user; docker group step will be skipped."
    LOGIN_USER=""
fi

# ── 1. System update ──────────────────────────────────────────────────────────
header "Updating System Packages"
case "$OS_FLAVOUR" in
    al2|rhel7)
        yum update -y -q
        ;;
    al2023|rhel8)
        dnf update -y -q
        ;;
    ubuntu|debian)
        apt-get update -qq
        DEBIAN_FRONTEND=noninteractive apt-get upgrade -y -qq
        ;;
esac
success "System packages updated."

# ── 2. Install base tools (curl, git) ────────────────────────────────────────
header "Installing Base Tools"

install_pkg() {
    local pkg="$1"
    case "$OS_FLAVOUR" in
        al2|rhel7)       yum install -y -q "$pkg" ;;
        al2023|rhel8)    dnf install -y -q "$pkg" ;;
        ubuntu|debian)   DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "$pkg" ;;
    esac
}

for tool in curl git ca-certificates; do
    if command -v "$tool" &>/dev/null; then
        success "${tool} already installed."
    else
        step "Installing ${tool}…"
        install_pkg "$tool"
        success "${tool} installed."
    fi
done

# ── 3. Install Docker Engine ──────────────────────────────────────────────────
header "Installing Docker Engine"

if command -v docker &>/dev/null; then
    DOCKER_VER=$(docker --version | awk '{print $3}' | tr -d ',')
    success "Docker already installed: ${DOCKER_VER}"
else
    step "Installing Docker Engine for ${OS_FLAVOUR}…"
    case "$OS_FLAVOUR" in
        al2)
            amazon-linux-extras install docker -y
            ;;
        al2023)
            dnf install -y docker
            ;;
        rhel7)
            yum install -y yum-utils
            yum-config-manager --add-repo \
                https://download.docker.com/linux/centos/docker-ce.repo
            yum install -y docker-ce docker-ce-cli containerd.io
            ;;
        rhel8)
            dnf install -y yum-utils
            yum-config-manager --add-repo \
                https://download.docker.com/linux/centos/docker-ce.repo
            dnf install -y docker-ce docker-ce-cli containerd.io \
                docker-buildx-plugin docker-compose-plugin
            ;;
        ubuntu)
            install_pkg "ca-certificates gnupg lsb-release"
            install -m 0755 -d /etc/apt/keyrings
            curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
                | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
            chmod a+r /etc/apt/keyrings/docker.gpg
            echo \
                "deb [arch=$(dpkg --print-architecture) \
                signed-by=/etc/apt/keyrings/docker.gpg] \
                https://download.docker.com/linux/ubuntu \
                $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
                | tee /etc/apt/sources.list.d/docker.list > /dev/null
            apt-get update -qq
            DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
                docker-ce docker-ce-cli containerd.io \
                docker-buildx-plugin docker-compose-plugin
            ;;
        debian)
            install_pkg "ca-certificates gnupg lsb-release"
            install -m 0755 -d /etc/apt/keyrings
            curl -fsSL https://download.docker.com/linux/debian/gpg \
                | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
            chmod a+r /etc/apt/keyrings/docker.gpg
            echo \
                "deb [arch=$(dpkg --print-architecture) \
                signed-by=/etc/apt/keyrings/docker.gpg] \
                https://download.docker.com/linux/debian \
                $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
                | tee /etc/apt/sources.list.d/docker.list > /dev/null
            apt-get update -qq
            DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
                docker-ce docker-ce-cli containerd.io \
                docker-buildx-plugin docker-compose-plugin
            ;;
    esac
    success "Docker Engine installed."
fi

# ── 4. Install Docker Compose v2 plugin (if not bundled) ─────────────────────
header "Installing Docker Compose v2"

if docker compose version &>/dev/null 2>&1; then
    COMPOSE_VER=$(docker compose version --short 2>/dev/null || echo "unknown")
    success "Docker Compose v2 already available: ${COMPOSE_VER}"
else
    step "Docker Compose plugin not found — installing via GitHub release…"
    COMPOSE_ARCH="$(uname -m)"
    # Normalise ARM arch names
    [[ "$COMPOSE_ARCH" == "aarch64" ]] && COMPOSE_ARCH="aarch64"
    [[ "$COMPOSE_ARCH" == "x86_64"  ]] && COMPOSE_ARCH="x86_64"

    COMPOSE_DEST="/usr/local/lib/docker/cli-plugins"
    mkdir -p "$COMPOSE_DEST"

    COMPOSE_URL="https://github.com/docker/compose/releases/latest/download/docker-compose-linux-${COMPOSE_ARCH}"
    curl -SL "$COMPOSE_URL" -o "${COMPOSE_DEST}/docker-compose"
    chmod +x "${COMPOSE_DEST}/docker-compose"

    if docker compose version &>/dev/null 2>&1; then
        success "Docker Compose v2 plugin installed."
    else
        error "Docker Compose v2 installation failed. Check connectivity and try again."
        exit 1
    fi
fi

# ── 5. Start & enable Docker ──────────────────────────────────────────────────
header "Starting Docker Service"

systemctl start docker
systemctl enable docker
success "Docker service started and enabled on boot."

# ── 6. Add login user to docker group ────────────────────────────────────────
header "Docker Group Membership"

if [[ -n "$LOGIN_USER" ]]; then
    if groups "$LOGIN_USER" | grep -qw docker; then
        success "${LOGIN_USER} is already in the docker group."
    else
        usermod -aG docker "$LOGIN_USER"
        success "Added ${LOGIN_USER} to the docker group."
        warn "You must log out and back in (or run 'newgrp docker') for"
        warn "group membership to take effect before running deploy.sh."
    fi
else
    warn "No non-root login user detected — skipping docker group step."
    warn "Run deploy.sh as root, or manually: usermod -aG docker <your-user>"
fi

# ── 7. Configure Docker daemon (log rotation + resource limits) ───────────────
header "Configuring Docker Daemon"

DOCKER_DAEMON_CFG="/etc/docker/daemon.json"

if [[ -f "$DOCKER_DAEMON_CFG" ]]; then
    info "Docker daemon config already exists at ${DOCKER_DAEMON_CFG} — skipping."
    info "Verify it contains log-driver and log-opts for log rotation."
else
    cat > "$DOCKER_DAEMON_CFG" <<'EOF'
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "50m",
    "max-file": "3"
  },
  "storage-driver": "overlay2"
}
EOF
    systemctl reload docker || systemctl restart docker
    success "Docker daemon configured: log rotation 50 MB × 3 files, overlay2 storage."
fi

# ── 8. Swap file for low-memory instances ────────────────────────────────────
header "Checking Memory & Swap"

TOTAL_MEM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
TOTAL_MEM_GB=$(( TOTAL_MEM_KB / 1024 / 1024 ))
SWAP_TOTAL_KB=$(grep SwapTotal /proc/meminfo | awk '{print $2}')

info "RAM detected: ${TOTAL_MEM_GB} GB"
info "Swap currently: $(( SWAP_TOTAL_KB / 1024 )) MB"

if [[ $SWAP_TOTAL_KB -gt 0 ]]; then
    success "Swap already configured — skipping."
elif [[ $TOTAL_MEM_GB -ge 4 ]]; then
    success "Sufficient RAM (≥ 4 GB) — swap not required."
else
    # Small instance (t2/t3.micro, t3.small): add 4 GB swap to survive Docker builds
    SWAP_FILE="/swapfile"
    SWAP_SIZE_GB=4
    warn "Low RAM (${TOTAL_MEM_GB} GB) — adding ${SWAP_SIZE_GB} GB swap file at ${SWAP_FILE}."
    step "Allocating swap (this may take 30–60 s)…"
    fallocate -l "${SWAP_SIZE_GB}G" "$SWAP_FILE" 2>/dev/null \
        || dd if=/dev/zero of="$SWAP_FILE" bs=1M count=$(( SWAP_SIZE_GB * 1024 )) status=none
    chmod 600 "$SWAP_FILE"
    mkswap "$SWAP_FILE"
    swapon "$SWAP_FILE"
    # Persist across reboots
    if ! grep -q "$SWAP_FILE" /etc/fstab; then
        echo "${SWAP_FILE}  none  swap  sw  0  0" >> /etc/fstab
    fi
    success "${SWAP_SIZE_GB} GB swap file created and activated."
fi

# ── 9. Disk space check ───────────────────────────────────────────────────────
header "Checking Disk Space"

# Check free space on the filesystem that holds /var/lib/docker
DOCKER_ROOT="/var/lib/docker"
[[ -d "$DOCKER_ROOT" ]] || DOCKER_ROOT="/"
FREE_KB=$(df -k "$DOCKER_ROOT" | tail -1 | awk '{print $4}')
FREE_GB=$(( FREE_KB / 1024 / 1024 ))
REQUIRED_GB=10

info "Free disk space on ${DOCKER_ROOT}: ${FREE_GB} GB"

if [[ $FREE_GB -lt $REQUIRED_GB ]]; then
    warn "Only ${FREE_GB} GB free. Building all Docker images requires ≥ ${REQUIRED_GB} GB."
    warn "Consider resizing your EBS volume before running deploy.sh."
    warn "  AWS Console → EC2 → Volumes → Modify → increase size, then:"
    warn "  sudo growpart /dev/xvda 1 && sudo resize2fs /dev/xvda1"
else
    success "Disk space OK: ${FREE_GB} GB free (minimum ${REQUIRED_GB} GB required)."
fi

# ── 10. Create .env template ──────────────────────────────────────────────────
header "Environment File (.env)"

ENV_FILE="${SCRIPT_DIR}/.env"

if [[ -f "$ENV_FILE" ]]; then
    success ".env already exists at ${ENV_FILE} — not overwriting."
    info "Edit it to verify ANTHROPIC_API_KEY is set."
else
    cat > "$ENV_FILE" <<'EOF'
# ── Anthropic API key (REQUIRED) ─────────────────────────────────────────────
# Get yours at https://console.anthropic.com/
ANTHROPIC_API_KEY=sk-ant-REPLACE_ME

# ── Service ports (change only if the defaults conflict with other services) ──
AGENT_PORT=8000
ONTOLOGY_PORT=8001
KG_PORT=8002
DIALOG_PORT=8003
CONFORMITY_PORT=8004
CHAT_PORT=8005
UI_PORT=8501

# ── Log verbosity: debug | info | warning ─────────────────────────────────────
LOG_LEVEL=info
EOF
    chmod 600 "$ENV_FILE"
    success ".env template created at ${ENV_FILE}"
    warn "IMPORTANT: Edit .env and replace ANTHROPIC_API_KEY before running deploy.sh."
fi

# ── 11. Make deploy.sh executable ────────────────────────────────────────────
header "Setting File Permissions"

if [[ -f "${SCRIPT_DIR}/deploy.sh" ]]; then
    chmod +x "${SCRIPT_DIR}/deploy.sh"
    success "deploy.sh is now executable."
else
    warn "deploy.sh not found in ${SCRIPT_DIR} — it will be set executable once present."
fi

# ── 12. Verify all prerequisites ─────────────────────────────────────────────
header "Verification"

PASS=0; FAIL=0

check() {
    local label="$1"; shift
    if "$@" &>/dev/null; then
        success "${label}"
        (( PASS++ )) || true
    else
        error "${label} — FAILED"
        (( FAIL++ )) || true
    fi
}

check "docker binary present"          command -v docker
check "docker daemon running"          docker info
check "docker compose v2 available"    docker compose version
check "curl binary present"            command -v curl
check "git binary present"             command -v git
check ".env file exists"               test -f "${SCRIPT_DIR}/.env"
check "deploy.sh is executable"        test -x "${SCRIPT_DIR}/deploy.sh"

echo ""
if [[ $FAIL -eq 0 ]]; then
    echo -e "${GREEN}${BOLD}All ${PASS} checks passed.${RESET}"
else
    echo -e "${RED}${BOLD}${FAIL} check(s) failed — resolve the errors above before proceeding.${RESET}"
fi

# ── 13. Security group reminder ───────────────────────────────────────────────
header "AWS Security Group — Required Inbound Rules"

echo -e "  Open the following ports in your EC2 Security Group"
echo -e "  (EC2 Console → Security Groups → Inbound Rules → Edit):\n"
printf "  %-12s %-10s %-40s\n" "Port"      "Protocol" "Purpose"
printf "  %-12s %-10s %-40s\n" "────────"  "────────" "──────────────────────────────────"
printf "  %-12s %-10s %-40s\n" "22"        "TCP"      "SSH (your IP only)"
printf "  %-12s %-10s %-40s\n" "8005"      "TCP"      "DataChat UI  ← business user URL"
printf "  %-12s %-10s %-40s\n" "8501"      "TCP"      "Streamlit UI (technical)"
printf "  %-12s %-10s %-40s\n" "8000-8004" "TCP"      "API services (internal / optional)"
echo ""
warn "Never open all ports (0-65535) to 0.0.0.0/0 — restrict to your office IP range."

# ── 14. Instance size recommendation ─────────────────────────────────────────
header "EC2 Instance Size Recommendation"

echo -e "  Building 7 Docker images and running them concurrently requires:\n"
printf "  %-20s %-10s %-10s %-30s\n" "Instance Type"  "vCPU"  "RAM"    "Verdict"
printf "  %-20s %-10s %-10s %-30s\n" "────────────"   "────"  "───"    "───────"
printf "  %-20s %-10s %-10s %-30s\n" "t3.micro"       "2"     "1 GB"   "⚠  Marginal (swap added)"
printf "  %-20s %-10s %-10s %-30s\n" "t3.small"       "2"     "2 GB"   "⚠  Tight — builds may be slow"
printf "  %-20s %-10s %-10s %-30s\n" "t3.medium"      "2"     "4 GB"   "✓  Minimum recommended"
printf "  %-20s %-10s %-10s %-30s\n" "t3.large"       "2"     "8 GB"   "✓✓ Comfortable for production"
printf "  %-20s %-10s %-10s %-30s\n" "t3.xlarge"      "4"     "16 GB"  "✓✓ Recommended for heavy load"
echo ""

# ── 15. Next steps ────────────────────────────────────────────────────────────
header "Next Steps"

EC2_IP=$(curl -sf --max-time 3 http://169.254.169.254/latest/meta-data/public-ipv4 || echo "<your-ec2-ip>")

echo -e "  ${BOLD}1.${RESET} Edit your API key:"
echo -e "     ${CYAN}nano ${ENV_FILE}${RESET}"
echo -e "     Replace  ANTHROPIC_API_KEY=sk-ant-REPLACE_ME  with your real key.\n"

echo -e "  ${BOLD}2.${RESET} Log out and back in so docker group takes effect:"
echo -e "     ${CYAN}exit${RESET}  then reconnect with SSH\n"

if [[ -n "$LOGIN_USER" ]]; then
    echo -e "  ${BOLD}3.${RESET} Run the deployment (as ${LOGIN_USER}, not root):"
else
    echo -e "  ${BOLD}3.${RESET} Run the deployment:"
fi
echo -e "     ${CYAN}cd ${SCRIPT_DIR} && ./deploy.sh${RESET}\n"

echo -e "  ${BOLD}4.${RESET} Open DataChat in your browser:"
echo -e "     ${CYAN}http://${EC2_IP}:8005${RESET}  ← upload your Excel/CSV and start chatting\n"

echo -e "  ${BOLD}Other commands:${RESET}"
echo -e "     ${CYAN}./deploy.sh --status${RESET}   — check container health"
echo -e "     ${CYAN}./deploy.sh --logs${RESET}     — tail all service logs"
echo -e "     ${CYAN}./deploy.sh --stop${RESET}    — stop all containers"
echo -e "     ${CYAN}./deploy.sh --restart${RESET} — restart all containers\n"
