#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# attach_ebs.sh — Create, attach, format, and mount a new EBS volume,
#                 then move Docker's data directory to it.
#
# Run this ON the EC2 instance with sudo:
#   chmod +x attach_ebs.sh
#   sudo ./attach_ebs.sh
#
# Requirements:
#   - AWS CLI v2 installed and configured (or instance has an IAM role with
#     ec2:CreateVolume, ec2:AttachVolume, ec2:DescribeVolumes permissions)
#   - Docker installed
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

# ── Config (edit these if needed) ─────────────────────────────────────────────
VOLUME_SIZE="${VOLUME_SIZE:-30}"          # GB
VOLUME_TYPE="${VOLUME_TYPE:-gp3}"
MOUNT_POINT="${MOUNT_POINT:-/mnt/docker-data}"
DEVICE_NAME="${DEVICE_NAME:-/dev/sdf}"   # AWS device name (logical)
DOCKER_DATA_ROOT="${DOCKER_DATA_ROOT:-/var/lib/docker}"

# ── Pre-flight ────────────────────────────────────────────────────────────────
if [[ $EUID -ne 0 ]]; then
    error "This script must be run as root (sudo ./attach_ebs.sh)"
    exit 1
fi

for bin in aws curl lsblk mkfs.ext4 rsync; do
    if ! command -v "$bin" &>/dev/null; then
        error "'$bin' is required but not installed."
        [[ "$bin" == "rsync" ]] && error "  Install with: apt-get install -y rsync  OR  yum install -y rsync"
        exit 1
    fi
done

# ── Detect instance metadata ───────────────────────────────────────────────────
header "Detecting instance metadata"

IMDS_TOKEN=$(curl -sf -X PUT "http://169.254.169.254/latest/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 60" 2>/dev/null || true)

if [[ -n "$IMDS_TOKEN" ]]; then
    INSTANCE_ID=$(curl -sf -H "X-aws-ec2-metadata-token: $IMDS_TOKEN" \
        http://169.254.169.254/latest/meta-data/instance-id)
    AZ=$(curl -sf -H "X-aws-ec2-metadata-token: $IMDS_TOKEN" \
        http://169.254.169.254/latest/meta-data/placement/availability-zone)
    REGION=$(curl -sf -H "X-aws-ec2-metadata-token: $IMDS_TOKEN" \
        http://169.254.169.254/latest/meta-data/placement/region)
else
    # IMDSv1 fallback
    INSTANCE_ID=$(curl -sf http://169.254.169.254/latest/meta-data/instance-id)
    AZ=$(curl -sf http://169.254.169.254/latest/meta-data/placement/availability-zone)
    REGION=$(curl -sf http://169.254.169.254/latest/meta-data/placement/region)
fi

info "Instance ID : $INSTANCE_ID"
info "Region      : $REGION"
info "AZ          : $AZ"
info "Volume size : ${VOLUME_SIZE} GB (${VOLUME_TYPE})"
info "Mount point : $MOUNT_POINT"

# ── Create EBS volume ─────────────────────────────────────────────────────────
header "Creating EBS volume"

VOLUME_ID=$(aws ec2 create-volume \
    --region          "$REGION" \
    --availability-zone "$AZ" \
    --size            "$VOLUME_SIZE" \
    --volume-type     "$VOLUME_TYPE" \
    --tag-specifications "ResourceType=volume,Tags=[{Key=Name,Value=docker-data},{Key=Purpose,Value=docker}]" \
    --query "VolumeId" \
    --output text)

success "Created volume: $VOLUME_ID"

# ── Wait for volume to be available ───────────────────────────────────────────
info "Waiting for volume to become available…"
aws ec2 wait volume-available --region "$REGION" --volume-ids "$VOLUME_ID"
success "Volume is available"

# ── Attach volume ─────────────────────────────────────────────────────────────
header "Attaching volume to instance"

aws ec2 attach-volume \
    --region      "$REGION" \
    --volume-id   "$VOLUME_ID" \
    --instance-id "$INSTANCE_ID" \
    --device      "$DEVICE_NAME" \
    --output      text > /dev/null

info "Waiting for volume to attach…"
aws ec2 wait volume-in-use --region "$REGION" --volume-ids "$VOLUME_ID"
success "Volume attached as $DEVICE_NAME"

# ── Discover the actual kernel device name ────────────────────────────────────
# AWS maps /dev/sdf → /dev/xvdf on older instances, or /dev/nvme1n1 on Nitro
header "Discovering kernel device"

sleep 5   # give the kernel a moment to register the device

KERNEL_DEVICE=""
# Try NVMe first (Nitro instances: m5, c5, t3, t4g, etc.)
for dev in /dev/nvme1n1 /dev/nvme2n1 /dev/nvme3n1; do
    if [[ -b "$dev" ]]; then
        # Make sure it's the new unformatted volume (no filesystem)
        TYPE=$(blkid -o value -s TYPE "$dev" 2>/dev/null || true)
        if [[ -z "$TYPE" ]]; then
            KERNEL_DEVICE="$dev"
            break
        fi
    fi
done

# Fallback: xvdf style (older instances)
if [[ -z "$KERNEL_DEVICE" ]]; then
    MAPPED="${DEVICE_NAME/\/dev\/sd//dev/xvd}"
    if [[ -b "$MAPPED" ]]; then
        KERNEL_DEVICE="$MAPPED"
    fi
fi

if [[ -z "$KERNEL_DEVICE" ]]; then
    error "Could not find attached device. Run 'lsblk' and re-run with:"
    error "  DEVICE_NAME=/dev/your-device sudo ./attach_ebs.sh"
    exit 1
fi

success "Kernel device: $KERNEL_DEVICE"
lsblk "$KERNEL_DEVICE"

# ── Format the volume ─────────────────────────────────────────────────────────
header "Formatting volume (ext4)"

mkfs.ext4 -F "$KERNEL_DEVICE"
success "Formatted $KERNEL_DEVICE as ext4"

# ── Mount the volume ──────────────────────────────────────────────────────────
header "Mounting volume at $MOUNT_POINT"

mkdir -p "$MOUNT_POINT"
mount "$KERNEL_DEVICE" "$MOUNT_POINT"
success "Mounted at $MOUNT_POINT"

# Add to /etc/fstab for persistence across reboots
UUID=$(blkid -o value -s UUID "$KERNEL_DEVICE")
FSTAB_ENTRY="UUID=$UUID  $MOUNT_POINT  ext4  defaults,nofail  0  2"

if grep -q "$UUID" /etc/fstab; then
    info "fstab entry already present — skipping"
else
    echo "$FSTAB_ENTRY" >> /etc/fstab
    success "Added to /etc/fstab (UUID=$UUID)"
fi

# ── Move Docker data to new volume ────────────────────────────────────────────
header "Moving Docker data to new volume"

info "Stopping Docker…"
systemctl stop docker

if [[ -d "$DOCKER_DATA_ROOT" ]]; then
    info "Syncing $DOCKER_DATA_ROOT → $MOUNT_POINT …"
    rsync -aP "$DOCKER_DATA_ROOT/" "$MOUNT_POINT/"
    success "Data synced"

    info "Backing up old Docker directory…"
    mv "$DOCKER_DATA_ROOT" "${DOCKER_DATA_ROOT}.bak"
    success "Old directory moved to ${DOCKER_DATA_ROOT}.bak"
else
    warn "Docker data directory $DOCKER_DATA_ROOT not found — nothing to migrate"
fi

# ── Configure Docker daemon ───────────────────────────────────────────────────
header "Configuring Docker daemon"

DAEMON_JSON="/etc/docker/daemon.json"
mkdir -p /etc/docker

if [[ -f "$DAEMON_JSON" ]]; then
    cp "$DAEMON_JSON" "${DAEMON_JSON}.bak"
    info "Backed up existing daemon.json to ${DAEMON_JSON}.bak"
fi

cat > "$DAEMON_JSON" <<EOF
{
  "data-root": "$MOUNT_POINT",
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "50m",
    "max-file": "3"
  },
  "storage-driver": "overlay2"
}
EOF

success "daemon.json written"

# ── Restart Docker ─────────────────────────────────────────────────────────────
header "Restarting Docker"

systemctl start docker
sleep 3

DOCKER_ROOT=$(docker info 2>/dev/null | grep "Docker Root Dir" | awk '{print $NF}')
if [[ "$DOCKER_ROOT" == "$MOUNT_POINT" ]]; then
    success "Docker is using new root: $DOCKER_ROOT"
else
    error "Docker root is still $DOCKER_ROOT — check daemon.json"
    exit 1
fi

# ── Disk usage summary ────────────────────────────────────────────────────────
header "Disk usage summary"
df -h "$MOUNT_POINT" /
docker system df

# ── Cleanup instructions ──────────────────────────────────────────────────────
echo ""
success "EBS volume setup complete!"
echo ""
echo -e "  Volume ID   : ${CYAN}${VOLUME_ID}${RESET}"
echo -e "  Device      : ${CYAN}${KERNEL_DEVICE}${RESET}"
echo -e "  Mount point : ${CYAN}${MOUNT_POINT}${RESET}"
echo -e "  Size        : ${CYAN}${VOLUME_SIZE} GB (${VOLUME_TYPE})${RESET}"
echo ""
echo -e "${YELLOW}Once you've verified everything works, remove the old Docker backup:${RESET}"
echo -e "  sudo rm -rf ${DOCKER_DATA_ROOT}.bak"
echo ""
echo -e "${YELLOW}Now rebuild and start your containers:${RESET}"
echo -e "  cd ~/metadata_agent && ./deploy.sh"
echo ""
