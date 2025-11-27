#!/bin/bash
# BinaryLane VPS Bootstrap Script for MCP Memory Service
# Optimized for 1GB RAM constraint
#
# Usage (as root): curl -sSL https://raw.githubusercontent.com/.../bootstrap.sh | bash
# Or: scp bootstrap.sh root@vps: && ssh root@vps 'bash bootstrap.sh'
#
# Creates non-root user 'fish' with sudo access for running services
# IDEMPOTENT: Safe to run multiple times

set -euo pipefail

echo "=== MCP Memory Service - BinaryLane Bootstrap ==="
echo "Optimized for 1GB RAM VPS"
echo ""

# Must run as root initially
if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: This script must be run as root"
    echo "Run: sudo bash bootstrap.sh"
    exit 1
fi

# Check we have enough memory
TOTAL_MEM=$(free -m | awk '/^Mem:/{print $2}')
echo "Detected RAM: ${TOTAL_MEM}MB"

if [ "$TOTAL_MEM" -lt 900 ]; then
    echo "WARNING: Less than 1GB RAM detected. This may not work well."
fi

# 0. Create non-root user 'fish'
echo ""
echo "=== Step 0: Creating user 'fish' ==="
USERNAME="fish"

if id "$USERNAME" &>/dev/null; then
    echo "User '$USERNAME' already exists"
else
    useradd -m -s /bin/bash "$USERNAME"
    echo "Created user '$USERNAME'"
fi

# Add to sudo group (passwordless sudo for deployment)
usermod -aG sudo "$USERNAME"
echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME
chmod 440 /etc/sudoers.d/$USERNAME
echo "Granted '$USERNAME' passwordless sudo"

# Set up SSH key access (copy from root if fish doesn't have keys yet)
mkdir -p /home/$USERNAME/.ssh
chown $USERNAME:$USERNAME /home/$USERNAME/.ssh
chmod 700 /home/$USERNAME/.ssh

if [ -f /home/$USERNAME/.ssh/authorized_keys ]; then
    # Fish already has keys - don't overwrite, but merge any new keys from root
    if [ -f /root/.ssh/authorized_keys ]; then
        # Add any keys from root that aren't already in fish's file
        while IFS= read -r key; do
            if [ -n "$key" ] && ! grep -qF "$key" /home/$USERNAME/.ssh/authorized_keys 2>/dev/null; then
                echo "$key" >> /home/$USERNAME/.ssh/authorized_keys
                echo "Added new key to '$USERNAME' authorized_keys"
            fi
        done < /root/.ssh/authorized_keys
    fi
    echo "Preserved existing SSH keys for '$USERNAME'"
elif [ -f /root/.ssh/authorized_keys ]; then
    # Fish has no keys - copy from root
    cp /root/.ssh/authorized_keys /home/$USERNAME/.ssh/
    echo "Copied SSH authorized_keys from root to '$USERNAME'"
else
    echo "WARNING: No SSH keys found for root or '$USERNAME'"
    echo "You'll need to set up SSH keys for '$USERNAME' manually"
fi

chown $USERNAME:$USERNAME /home/$USERNAME/.ssh/authorized_keys 2>/dev/null || true
chmod 600 /home/$USERNAME/.ssh/authorized_keys 2>/dev/null || true

# 1. System updates
echo ""
echo "=== Step 1: System Updates ==="
apt-get update
apt-get upgrade -y

# 2. Create swap (CRITICAL for 1GB RAM)
echo ""
echo "=== Step 2: Creating 2GB Swap ==="
if [ ! -f /swapfile ]; then
    fallocate -l 2G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    echo "Swap file created and enabled"
else
    echo "Swap file already exists"
    # Ensure it's enabled
    swapon /swapfile 2>/dev/null || true
fi

# Add fstab entry (idempotent)
if ! grep -q '/swapfile' /etc/fstab; then
    echo '/swapfile none swap sw 0 0' >> /etc/fstab
    echo "Added swap to fstab"
else
    echo "Swap already in fstab"
fi

# Optimize swappiness (idempotent)
if ! grep -q 'vm.swappiness=10' /etc/sysctl.conf; then
    echo 'vm.swappiness=10' >> /etc/sysctl.conf
    echo "Added swappiness setting"
else
    echo "Swappiness already configured"
fi

if ! grep -q 'vm.vfs_cache_pressure=50' /etc/sysctl.conf; then
    echo 'vm.vfs_cache_pressure=50' >> /etc/sysctl.conf
    echo "Added vfs_cache_pressure setting"
else
    echo "vfs_cache_pressure already configured"
fi

sysctl -p

# 3. Install Docker
echo ""
echo "=== Step 3: Installing Docker ==="
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com | sh
    echo "Docker installed"
else
    echo "Docker already installed"
fi

# Add fish to docker group (idempotent)
usermod -aG docker $USERNAME
echo "Ensured '$USERNAME' in docker group"

# 4. Install Docker Compose
echo ""
echo "=== Step 4: Installing Docker Compose ==="
if ! docker compose version &> /dev/null; then
    apt-get install -y docker-compose-plugin
    echo "Docker Compose plugin installed"
else
    echo "Docker Compose already installed"
fi

# Install standalone docker-compose if missing
if ! command -v docker-compose &> /dev/null; then
    curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
    echo "Standalone docker-compose installed"
fi

docker compose version

# 5. Install Tailscale (for private network access)
echo ""
echo "=== Step 5: Installing Tailscale ==="
if ! command -v tailscale &> /dev/null; then
    curl -fsSL https://tailscale.com/install.sh | sh
    echo "Tailscale installed"
    echo ""
    echo ">>> Run 'tailscale up --ssh' to join your tailnet <<<"
    echo ""
else
    echo "Tailscale already installed"
    tailscale status || echo "(Not connected - run 'tailscale up --ssh')"
fi

# 6. Security Hardening
echo ""
echo "=== Step 6: Security Hardening ==="

# 6a. Install security packages
echo "Installing security packages..."
apt-get install -y ufw fail2ban unattended-upgrades apt-listchanges

# 6b. Configure UFW firewall
echo "Configuring firewall (UFW)..."
ufw --force reset  # Idempotent reset

# Default policies
ufw default deny incoming
ufw default allow outgoing

# Allow SSH (before enabling!)
ufw allow ssh

# Allow Tailscale interface (if exists)
if ip link show tailscale0 &>/dev/null; then
    ufw allow in on tailscale0
    echo "Allowed all traffic on Tailscale interface"
fi

# Allow Tailscale UDP port
ufw allow 41641/udp comment 'Tailscale'

# Enable firewall
ufw --force enable
echo "UFW firewall enabled"

# 6c. SSH Hardening
echo "Hardening SSH..."
SSHD_CONFIG="/etc/ssh/sshd_config"

# Backup original (once)
if [ ! -f "${SSHD_CONFIG}.original" ]; then
    cp "$SSHD_CONFIG" "${SSHD_CONFIG}.original"
fi

# Create hardened sshd_config.d drop-in (idempotent)
cat > /etc/ssh/sshd_config.d/99-hardening.conf << 'EOF'
# Security hardening - managed by bootstrap.sh
PermitRootLogin no
PasswordAuthentication no
PubkeyAuthentication yes
PermitEmptyPasswords no
X11Forwarding no
MaxAuthTries 3
ClientAliveInterval 300
ClientAliveCountMax 2
EOF

# Restart SSH to apply (service name varies: ssh on Debian/Ubuntu, sshd on RHEL/CentOS)
if systemctl list-units --type=service | grep -q 'sshd.service'; then
    systemctl restart sshd
elif systemctl list-units --type=service | grep -q 'ssh.service'; then
    systemctl restart ssh
else
    echo "WARNING: Could not find SSH service to restart"
fi
echo "SSH hardened (root login disabled, password auth disabled)"

# 6d. Configure Fail2ban
echo "Configuring Fail2ban..."
cat > /etc/fail2ban/jail.local << 'EOF'
[DEFAULT]
bantime = 1h
findtime = 10m
maxretry = 5
ignoreip = 127.0.0.1/8 ::1 100.64.0.0/10

[sshd]
enabled = true
port = ssh
filter = sshd
logpath = /var/log/auth.log
maxretry = 3
bantime = 24h
EOF

systemctl enable fail2ban
systemctl restart fail2ban
echo "Fail2ban configured and enabled"

# 6e. Enable automatic security updates
echo "Enabling automatic security updates..."
cat > /etc/apt/apt.conf.d/50unattended-upgrades << 'EOF'
Unattended-Upgrade::Allowed-Origins {
    "${distro_id}:${distro_codename}-security";
    "${distro_id}ESMApps:${distro_codename}-apps-security";
    "${distro_id}ESM:${distro_codename}-infra-security";
};
Unattended-Upgrade::AutoFixInterruptedDpkg "true";
Unattended-Upgrade::MinimalSteps "true";
Unattended-Upgrade::Remove-Unused-Dependencies "true";
Unattended-Upgrade::Automatic-Reboot "false";
EOF

cat > /etc/apt/apt.conf.d/20auto-upgrades << 'EOF'
APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Unattended-Upgrade "1";
APT::Periodic::AutocleanInterval "7";
EOF

systemctl enable unattended-upgrades
echo "Automatic security updates enabled"

# 6f. Kernel security hardening (sysctl)
echo "Applying kernel security settings..."

# Idempotent sysctl security settings
declare -A SYSCTL_SECURITY=(
    ["net.ipv4.conf.all.rp_filter"]="1"
    ["net.ipv4.conf.default.rp_filter"]="1"
    ["net.ipv4.icmp_echo_ignore_broadcasts"]="1"
    ["net.ipv4.conf.all.accept_redirects"]="0"
    ["net.ipv4.conf.default.accept_redirects"]="0"
    ["net.ipv4.conf.all.send_redirects"]="0"
    ["net.ipv4.conf.default.send_redirects"]="0"
    ["net.ipv4.conf.all.accept_source_route"]="0"
    ["net.ipv4.conf.default.accept_source_route"]="0"
    ["net.ipv4.tcp_syncookies"]="1"
    ["kernel.randomize_va_space"]="2"
)

for key in "${!SYSCTL_SECURITY[@]}"; do
    value="${SYSCTL_SECURITY[$key]}"
    if ! grep -q "^${key}" /etc/sysctl.conf; then
        echo "${key}=${value}" >> /etc/sysctl.conf
    fi
done

sysctl -p
echo "Kernel security settings applied"

echo "Security hardening complete"

# 7. Create application directory (owned by fish)
echo ""
echo "=== Step 7: Setting up application directory ==="
mkdir -p /opt/mcp-memory
chown $USERNAME:$USERNAME /opt/mcp-memory

# Create data directories as fish user
su - $USERNAME -c "mkdir -p /opt/mcp-memory/data/qdrant /opt/mcp-memory/data/sqlite /opt/mcp-memory/backups"

# 8. Create environment file template
echo ""
echo "=== Step 8: Creating environment template ==="
cat > /opt/mcp-memory/.env.template << 'EOF'
# MCP Memory Service Configuration
# Copy to .env and customize

# Storage backend: sqlite_vec (recommended for 1GB RAM) or qdrant
MCP_MEMORY_STORAGE_BACKEND=qdrant

# Embedding model (smaller = less RAM)
# Options: intfloat/e5-small-v2 (384-dim, ~400MB RAM)
#          intfloat/e5-base-v2 (768-dim, ~600MB RAM)
MCP_MEMORY_EMBEDDING_MODEL=intfloat/e5-small-v2

# Server settings
MCP_HTTP_ENABLED=true
MCP_API_PORT=8000
MCP_SERVER_HOST=0.0.0.0
MCP_TRANSPORT_MODE=streamable-http
MCP_SERVER_PORT=8001

# Security (disable for local, enable for production)
MCP_OAUTH_ENABLED=false
MCP_ALLOW_ANONYMOUS_ACCESS=true

# Logging
MCP_LOG_LEVEL=INFO

# Qdrant settings (if using qdrant backend)
MCP_QDRANT_URL=http://qdrant:6333
MCP_QDRANT_COLLECTION=memories
EOF

if [ ! -f /opt/mcp-memory/.env ]; then
    cp /opt/mcp-memory/.env.template /opt/mcp-memory/.env
    echo "Created .env from template"
else
    echo ".env already exists (not overwriting)"
fi

# Set ownership
chown -R $USERNAME:$USERNAME /opt/mcp-memory

# 9. Summary
echo ""
echo "========================================="
echo "=== Bootstrap Complete ==="
echo "========================================="
echo ""
echo "User '$USERNAME' created with:"
echo "  - Passwordless sudo access"
echo "  - Docker group membership"
echo "  - SSH keys (if root had them)"
echo ""
echo "Security hardening applied:"
echo "  - UFW firewall (SSH + Tailscale only)"
echo "  - SSH hardened (no root, no passwords)"
echo "  - Fail2ban (brute force protection)"
echo "  - Automatic security updates"
echo "  - Kernel security settings"
echo ""
echo "WARNING: Root SSH login is now DISABLED."
echo "         Use 'ssh $USERNAME@<ip>' or Tailscale SSH."
echo ""
echo "Tailscale setup:"
echo "  sudo tailscale up --ssh"
echo ""
echo "Next steps (as user '$USERNAME'):"
echo ""
echo "  ssh $USERNAME@$(hostname -I | awk '{print $1}')"
echo "  # Or via Tailscale: ssh $USERNAME@<tailscale-hostname>"
echo ""
echo "  cd /opt/mcp-memory"
echo "  curl -O https://raw.githubusercontent.com/27Bslash6/mcp-memory-service/main/deployment/binarylane/docker-compose.qdrant.yml"
echo "  docker compose -f docker-compose.qdrant.yml up -d"
echo ""
echo "Memory check:"
free -h
echo ""
echo "Swap check:"
swapon --show
echo ""
echo "Firewall status:"
ufw status
