# Ubuntu Setup Guide

This guide provides comprehensive instructions for setting up MCP Memory Service on Ubuntu systems, covering both desktop and server environments.

## Prerequisites

- **Ubuntu 20.04 LTS or later** (22.04 LTS recommended)
- **Python 3.10+** (Python 3.11 recommended)
- **Git** for repository cloning
- **Build essentials** for compiling dependencies

## Quick Installation

### Automatic Installation (Recommended)

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Clone repository
git clone https://github.com/doobidoo/mcp-memory-service.git
cd mcp-memory-service

# Run Ubuntu-specific installer
python install.py --ubuntu
```

The installer automatically:
- Detects Ubuntu version and architecture
- Installs system dependencies
- Configures optimal storage backend
- Sets up CUDA support (if available)

## Manual Installation

### 1. System Dependencies

```bash
# Update package lists
sudo apt update

# Install essential build tools
sudo apt install -y build-essential git curl wget

# Install Python and development headers
sudo apt install -y python3.11 python3.11-dev python3.11-venv python3-pip

# Install additional dependencies
sudo apt install -y libsqlite3-dev libffi-dev libssl-dev
```

### 2. CUDA Support (Optional)

For GPU acceleration on systems with NVIDIA GPUs:

```bash
# Check for NVIDIA GPU
lspci | grep -i nvidia

# Install NVIDIA drivers (if not already installed)
sudo apt install -y nvidia-driver-535

# Install CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-1

# Reboot to load drivers
sudo reboot
```

### 3. Python Environment Setup

```bash
# Navigate to project directory
cd mcp-memory-service

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### 4. Install Dependencies

#### For CUDA Systems

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install -e .
pip install chromadb sentence-transformers
```

#### For CPU-only Systems

```bash
# Install CPU-only PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install with SQLite-vec backend (recommended for servers)
pip install -e .
pip install sentence-transformers sqlite-vec
```

## Configuration

### Environment Variables

#### For CUDA Systems

```bash
# Add to ~/.bashrc or ~/.profile
export MCP_MEMORY_STORAGE_BACKEND=chromadb
export MCP_MEMORY_USE_CUDA=true
export MCP_MEMORY_CHROMA_PATH="$HOME/.mcp_memory_chroma"

# CUDA-specific settings
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
```

#### For CPU-only Systems

```bash
# Add to ~/.bashrc or ~/.profile
export MCP_MEMORY_STORAGE_BACKEND=sqlite_vec
export MCP_MEMORY_SQLITE_VEC_PATH="$HOME/.mcp_memory_sqlite"
export MCP_MEMORY_CPU_ONLY=true

# CPU optimization
export OMP_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)
```

#### For Server Deployments

```bash
# Server-specific settings
export MCP_MEMORY_HTTP_HOST=0.0.0.0
export MCP_MEMORY_HTTP_PORT=8000
export MCP_MEMORY_LOG_LEVEL=INFO
export MCP_MEMORY_SERVER_MODE=true
```

### Reload Environment

```bash
# Reload shell configuration
source ~/.bashrc

# Or restart shell session
exec bash
```

## Service Integration

### Systemd Service

Create a systemd service for automatic startup:

```bash
# Create service file
sudo tee /etc/systemd/system/mcp-memory.service > /dev/null <<EOF
[Unit]
Description=MCP Memory Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/path/to/mcp-memory-service
ExecStart=/path/to/mcp-memory-service/venv/bin/python src/mcp_memory_service/server.py
Restart=always
RestartSec=10
Environment=MCP_MEMORY_STORAGE_BACKEND=sqlite_vec
Environment=MCP_MEMORY_SERVER_MODE=true

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable mcp-memory.service
sudo systemctl start mcp-memory.service

# Check service status
sudo systemctl status mcp-memory.service
```

### Docker Integration

#### Using Docker Compose

```yaml
# docker-compose.yml
version: '3.8'
services:
  mcp-memory-service:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MCP_MEMORY_STORAGE_BACKEND=sqlite_vec
      - MCP_MEMORY_HTTP_HOST=0.0.0.0
      - MCP_MEMORY_HTTP_PORT=8000
    volumes:
      - ./data:/app/data
      - /etc/localtime:/etc/localtime:ro
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
```

```bash
# Deploy with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f mcp-memory-service
```

## Claude Desktop Integration on Ubuntu

### Installation

```bash
# Install Claude Desktop (if not already installed)
# Download from https://claude.ai/download or use snap
sudo snap install claude-desktop
```

### Configuration

Claude Desktop configuration location: `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "memory": {
      "command": "python",
      "args": ["/path/to/mcp-memory-service/src/mcp_memory_service/server.py"],
      "env": {
        "MCP_MEMORY_STORAGE_BACKEND": "sqlite_vec",
        "PATH": "/path/to/mcp-memory-service/venv/bin:/usr/bin:/bin"
      }
    }
  }
}
```

## VS Code Integration

### Installation

```bash
# Install VS Code
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
sudo install -o root -g root -m 644 packages.microsoft.gpg /etc/apt/trusted.gpg.d/
sudo sh -c 'echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/trusted.gpg.d/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list'
sudo apt update
sudo apt install -y code
```

### MCP Extension Configuration

1. Install the MCP extension in VS Code
2. Configure in VS Code settings:

```json
{
  "mcp.servers": {
    "memory": {
      "command": "python",
      "args": ["/path/to/mcp-memory-service/src/mcp_memory_service/server.py"],
      "env": {
        "MCP_MEMORY_STORAGE_BACKEND": "sqlite_vec"
      }
    }
  }
}
```

## Server Deployment

### Nginx Reverse Proxy

```bash
# Install Nginx
sudo apt install -y nginx

# Create Nginx configuration
sudo tee /etc/nginx/sites-available/mcp-memory > /dev/null <<EOF
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # SSE support
        proxy_buffering off;
        proxy_cache off;
        proxy_set_header Connection '';
        proxy_http_version 1.1;
        chunked_transfer_encoding off;
    }
}
EOF

# Enable site
sudo ln -s /etc/nginx/sites-available/mcp-memory /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### SSL with Let's Encrypt

```bash
# Install Certbot
sudo apt install -y certbot python3-certbot-nginx

# Obtain SSL certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal is configured automatically
```

### Firewall Configuration

```bash
# Configure UFW firewall
sudo ufw enable
sudo ufw allow ssh
sudo ufw allow 'Nginx Full'
sudo ufw allow 8000  # If accessing directly

# Check status
sudo ufw status
```

## Performance Optimization

### CPU Optimization

```bash
# Set CPU governor to performance mode
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Install CPU frequency utilities
sudo apt install -y cpufrequtils
sudo cpufreq-set -r -g performance
```

### Memory Optimization

```bash
# Increase shared memory (for large datasets)
echo "kernel.shmmax = 68719476736" | sudo tee -a /etc/sysctl.conf
echo "kernel.shmall = 4294967296" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

### SSD Optimization

```bash
# Enable TRIM for SSDs
sudo systemctl enable fstrim.timer
sudo systemctl start fstrim.timer

# Check TRIM status
sudo fstrim -v /
```

## Monitoring and Logging

### System Monitoring

```bash
# Install monitoring tools
sudo apt install -y htop iotop nethogs

# Monitor MCP Memory Service
htop -p $(pgrep -f "mcp_memory_service")

# Monitor GPU usage (if CUDA)
watch -n 1 nvidia-smi
```

### Log Management

```bash
# Configure log rotation
sudo tee /etc/logrotate.d/mcp-memory > /dev/null <<EOF
/path/to/mcp-memory-service/logs/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    copytruncate
}
EOF

# Test log rotation
sudo logrotate -d /etc/logrotate.d/mcp-memory
```

## Troubleshooting

### Common Ubuntu Issues

#### 1. Python Version Issues

**Symptom**: Wrong Python version or missing python3.11

**Solution**:
```bash
# Add deadsnakes PPA for newer Python versions
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-dev python3.11-venv

# Set python3.11 as default (optional)
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
```

#### 2. Permission Issues

**Symptom**: Permission denied errors

**Solution**:
```bash
# Fix file permissions
chmod +x scripts/*.sh
chmod +x scripts/install_ubuntu.sh

# Add user to required groups
sudo usermod -a -G dialout,plugdev $USER

# Re-login or run:
exec bash
```

#### 3. CUDA Driver Issues

**Symptom**: CUDA not detected or driver conflicts

**Solution**:
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Reinstall NVIDIA drivers if needed
sudo apt purge nvidia-*
sudo apt autoremove
sudo apt install -y nvidia-driver-535
sudo reboot

# Verify installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### 4. SQLite-vec Installation Issues

**Symptom**: SQLite-vec fails to install or import

**Solution**:
```bash
# Install system SQLite development headers
sudo apt install -y libsqlite3-dev

# Reinstall sqlite-vec
pip uninstall -y sqlite-vec
pip install sqlite-vec --no-cache-dir

# Test installation
python -c "import sqlite_vec; print('SQLite-vec installed successfully')"
```

#### 5. Service Startup Issues

**Symptom**: Systemd service fails to start

**Solution**:
```bash
# Check service status and logs
sudo systemctl status mcp-memory.service
sudo journalctl -u mcp-memory.service -f

# Check configuration
sudo systemctl edit mcp-memory.service

# Restart service
sudo systemctl restart mcp-memory.service
```

### Diagnostic Commands

#### System Information

```bash
# Check Ubuntu version
lsb_release -a

# Check system resources
free -h
df -h
lscpu

# Check GPU information
lspci | grep -i vga
nvidia-smi  # If NVIDIA GPU present
```

#### Environment Verification

```bash
# Check virtual environment
echo $VIRTUAL_ENV
which python
python --version

# Check installed packages
pip list | grep -E "(torch|sentence|chroma|sqlite)"

# Test imports
python -c "
import torch
import sentence_transformers
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
```

#### Service Health

```bash
# Test server startup
python scripts/verify_environment.py

# Test HTTP server (if configured)
curl http://localhost:8000/health

# Check database access
python -c "
from src.mcp_memory_service.storage.sqlite_vec import SqliteVecStorage
storage = SqliteVecStorage()
print('Database connection: OK')
"
```

## Development Environment

### Setting up for Development

```bash
# Clone repository
git clone https://github.com/doobidoo/mcp-memory-service.git
cd mcp-memory-service

# Create development environment
python3.11 -m venv venv-dev
source venv-dev/bin/activate

# Install in development mode
pip install -e .
pip install pytest black isort mypy pre-commit

# Set up pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v
```

### Ubuntu-Specific Testing

```bash
# Run Ubuntu-specific tests
pytest tests/platform/test_ubuntu.py -v

# Test CUDA functionality (if available)
pytest tests/cuda/ -v

# Test server deployment
pytest tests/integration/test_server.py -v
```

## Related Documentation

- [Installation Guide](../installation/master-guide.md) - General installation instructions
- [Multi-Client Setup](../integration/multi-client.md) - Multi-client configuration
- [Docker Deployment](../deployment/docker.md) - Docker setup details
- [Troubleshooting](../troubleshooting/general.md) - Ubuntu-specific troubleshooting
- [Performance Tuning](../implementation/performance.md) - Performance optimization