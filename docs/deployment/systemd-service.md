# Systemd Service Setup for Linux

This guide explains how to set up the MCP Memory HTTP server as a systemd service on Linux for automatic startup and management.

## Overview

The systemd service provides:
- ✅ **Automatic startup** on user login
- ✅ **Persistent operation** even when logged out (with linger enabled)
- ✅ **Automatic restarts** on failure
- ✅ **Centralized logging** via journald
- ✅ **Easy management** via systemctl commands

## Installation

### Quick Install

```bash
# Run the installation script
cd /path/to/mcp-memory-service
bash scripts/service/install_http_service.sh
```

The script will:
1. Check prerequisites (.env file, venv)
2. Ask whether to install as user or system service
3. Copy service file to appropriate location
4. Reload systemd configuration
5. Show next steps

### Manual Installation

If you prefer manual installation:

**1. User Service (Recommended - No sudo required):**

```bash
# Create directory
mkdir -p ~/.config/systemd/user

# Copy service file
cp scripts/service/mcp-memory-http.service ~/.config/systemd/user/

# Reload systemd
systemctl --user daemon-reload

# Start service
systemctl --user start mcp-memory-http.service

# Enable auto-start
systemctl --user enable mcp-memory-http.service

# Enable linger (runs even when logged out)
loginctl enable-linger $USER
```

**2. System Service (Requires sudo):**

```bash
# Copy service file
sudo cp scripts/service/mcp-memory-http.service /etc/systemd/system/

# Edit to ensure paths are correct
sudo nano /etc/systemd/system/mcp-memory-http.service

# Reload systemd
sudo systemctl daemon-reload

# Start service
sudo systemctl start mcp-memory-http.service

# Enable auto-start
sudo systemctl enable mcp-memory-http.service
```

## Service Management

### Basic Commands

```bash
# Start service
systemctl --user start mcp-memory-http.service

# Stop service
systemctl --user stop mcp-memory-http.service

# Restart service
systemctl --user restart mcp-memory-http.service

# Check status
systemctl --user status mcp-memory-http.service

# Enable auto-start on login
systemctl --user enable mcp-memory-http.service

# Disable auto-start
systemctl --user disable mcp-memory-http.service
```

### Viewing Logs

```bash
# Live logs (follow mode)
journalctl --user -u mcp-memory-http.service -f

# Last 50 lines
journalctl --user -u mcp-memory-http.service -n 50

# Logs since boot
journalctl --user -u mcp-memory-http.service -b

# Logs for specific time range
journalctl --user -u mcp-memory-http.service --since "2 hours ago"

# Logs with priority filter (only errors and above)
journalctl --user -u mcp-memory-http.service -p err
```

## Configuration

The service file is located at:
- User service: `~/.config/systemd/user/mcp-memory-http.service`
- System service: `/etc/systemd/system/mcp-memory-http.service`

### Service File Structure

```ini
[Unit]
Description=MCP Memory Service HTTP Server (Hybrid Backend)
Documentation=https://github.com/doobidoo/mcp-memory-service
After=network.target network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=%h/repositories/mcp-memory-service
Environment=PATH=%h/repositories/mcp-memory-service/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
Environment=PYTHONPATH=%h/repositories/mcp-memory-service/src
EnvironmentFile=%h/repositories/mcp-memory-service/.env
ExecStart=%h/repositories/mcp-memory-service/venv/bin/python %h/repositories/mcp-memory-service/scripts/server/run_http_server.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=mcp-memory-http

# Security hardening
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=default.target
```

### Important Configuration Points

1. **User Service vs System Service:**
   - User services run as your user (recommended)
   - System services run at boot (before user login)
   - User services can't have `User=` and `Group=` directives
   - User services use `WantedBy=default.target` not `multi-user.target`

2. **Environment Loading:**
   - Service loads `.env` file via `EnvironmentFile` directive
   - All environment variables are available to the service
   - Changes to `.env` require service restart

3. **Working Directory:**
   - Service runs from project root
   - Relative paths in code work correctly
   - Database paths should be absolute or relative to working directory

## Troubleshooting

### Service Won't Start

**Check status for errors:**
```bash
systemctl --user status mcp-memory-http.service
```

**Common Issues:**

1. **GROUP error (status=216/GROUP):**
   - Remove `User=` and `Group=` directives from user service file
   - These are only for system services

2. **Permission denied:**
   - Check that `.env` file is readable by your user
   - Check that venv and scripts are accessible
   - For system services, ensure files are owned by service user

3. **Port already in use:**
   ```bash
   lsof -i :8000
   # Kill existing process or change port in .env
   ```

4. **Missing dependencies:**
   ```bash
   # Verify venv is set up
   ls -la venv/bin/python

   # Reinstall if needed
   python -m venv venv
   source venv/bin/activate
   pip install -e .
   ```

### Service Fails to Enable

**Error:** "Unit is added as a dependency to a non-existent unit"

**Solution:** For user services, change `WantedBy=` target:
```bash
# Edit service file
nano ~/.config/systemd/user/mcp-memory-http.service

# Change this:
[Install]
WantedBy=multi-user.target

# To this:
[Install]
WantedBy=default.target

# Reload and reenable
systemctl --user daemon-reload
systemctl --user reenable mcp-memory-http.service
```

### Logs Show Configuration Errors

**Check environment loading:**
```bash
# View effective environment
systemctl --user show-environment

# Test service startup manually
cd /path/to/mcp-memory-service
source .env
venv/bin/python scripts/server/run_http_server.py
```

### Service Stops After Logout

**Enable linger to keep user services running:**
```bash
loginctl enable-linger $USER

# Verify
loginctl show-user $USER | grep Linger
# Should show: Linger=yes
```

## Performance Monitoring

```bash
# Check memory usage
systemctl --user status mcp-memory-http.service | grep Memory

# Check CPU usage
systemctl --user status mcp-memory-http.service | grep CPU

# Monitor in real-time
watch -n 2 'systemctl --user status mcp-memory-http.service | grep -E "Memory|CPU"'

# Detailed resource usage
systemd-cgtop --user
```

## Security Considerations

The service includes basic security hardening:
- `NoNewPrivileges=true` - Prevents privilege escalation
- `PrivateTmp=true` - Isolated /tmp directory
- User services run with user permissions (no root access)

For system services, consider additional hardening:
- `ProtectSystem=strict` - Read-only access to system directories
- `ProtectHome=read-only` - Limited home directory access
- `ReadWritePaths=` - Explicitly allow write access to database paths

**Note:** Some security directives may conflict with application requirements. Test thoroughly when adding restrictions.

### Network exposure hardening

If you run the service on one machine and connect to it from other devices on your LAN (common setup for sharing a single memory database across Claude Desktop, Claude Code, Codex CLI, etc.), the following hardening is recommended.

**1. Bind to a specific interface, not `0.0.0.0`.**

Setting `--sse-host 0.0.0.0` (or `MCP_SSE_HOST=0.0.0.0`) exposes the service on every network interface, including any VPN / guest / IoT network your host joins. Prefer binding to the single LAN interface IP:

```ini
[Service]
Environment=MCP_SSE_HOST=192.168.1.42
Environment=MCP_SSE_PORT=8765
ExecStart=%h/repositories/mcp-memory-service/venv/bin/python %h/repositories/mcp-memory-service/scripts/server/run_http_server.py
```

**2. Firewall allow-list specific client IPs, not the whole subnet.**

A `/24` subnet rule trusts every device on the LAN. Prefer explicit per-device rules:

```bash
# Example: allow only your two workstation IPs
sudo ufw allow from 192.168.1.10 to any port 8765 proto tcp
sudo ufw allow from 192.168.1.11 to any port 8765 proto tcp
```

**3. Restrict database file permissions.**

The SQLite database may contain personal or project-sensitive memories. The default `0644` lets any local user on the host read it:

```bash
chmod 700 ~/.local/share/mcp-memory/
```

**4. Never expose this service to the public internet without TLS + auth.**

The streamable-HTTP / SSE transport speaks plain HTTP with no built-in authentication. For anything beyond a trusted LAN, put it behind:

- a WireGuard / Tailscale overlay (easiest), or
- a reverse proxy terminating TLS and enforcing auth (e.g. Caddy with basic-auth or OAuth2).

**5. Consider the client config symlink surface.**

If your MCP client config (e.g. `~/.claude.json`) lives on shared storage (D: drive symlink between WSL and Windows, NFS home on a mount, etc.), any host that reads the config will try to connect to the same URL. Make sure all such hosts are on the allow-list, or narrow the scope.

## Uninstallation

```bash
# Stop and disable service
systemctl --user stop mcp-memory-http.service
systemctl --user disable mcp-memory-http.service

# Remove service file
rm ~/.config/systemd/user/mcp-memory-http.service

# Reload systemd
systemctl --user daemon-reload

# Optional: Disable linger if no other user services needed
loginctl disable-linger $USER
```

## See Also

- [HTTP Server Management](../http-server-management.md) - General server management
- [Troubleshooting Guide](https://github.com/doobidoo/mcp-memory-service/wiki/07-TROUBLESHOOTING) - Common issues
- [Claude Code Hooks Configuration](../../CLAUDE.md#claude-code-hooks-configuration-) - Hooks setup
- [systemd.service(5)](https://www.freedesktop.org/software/systemd/man/systemd.service.html) - systemd documentation

---

**Last Updated**: 2025-10-13
**Version**: 8.5.4
**Tested On**: Ubuntu 22.04, Debian 12, Fedora 38
