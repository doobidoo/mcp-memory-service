# Cross-Platform Service Installation Guide

This guide provides instructions for installing MCP Memory Service as a native service on Windows, macOS, and Linux systems. The service will automatically start when your system boots or when you log in.

## Overview

The MCP Memory Service can be installed as a system service to:
- Start automatically on boot/login
- Run in the background without a terminal window
- Restart automatically if it crashes
- Integrate with system service management tools

## Quick Start

### Universal Installation

Use the cross-platform installer that automatically detects your operating system:

```bash
# Clone the repository
git clone https://github.com/doobidoo/mcp-memory-service.git
cd mcp-memory-service

# Install dependencies (if not already done)
pip install -e .

# Install as a service
python install_service.py
```

The installer will:
1. Detect your operating system
2. Check dependencies
3. Generate a secure API key
4. Install the appropriate service type
5. Provide platform-specific management commands

### Service Management

After installation, you can manage the service using:

```bash
# Check service status
python install_service.py --status

# Start the service
python install_service.py --start

# Stop the service
python install_service.py --stop

# Uninstall the service
python install_service.py --uninstall
```

## Platform-Specific Instructions

### Windows

#### Installation

```powershell
# Run as Administrator
python install_service.py

# Or install directly
python scripts/install_windows_service.py
```

This creates a Windows Service that:
- Runs under the current user account
- Starts automatically on system boot
- Can be managed via Services console or `net` commands

#### Management Commands

```powershell
# Using Windows commands
net start MCPMemoryService    # Start service
net stop MCPMemoryService     # Stop service
sc query MCPMemoryService     # Check status

# Using convenience scripts
.\scripts\windows\start_service.bat
.\scripts\windows\stop_service.bat
.\scripts\windows\service_status.bat
```

#### Requirements

- Administrator privileges for installation
- Python 3.10 or newer
- `pywin32` package (auto-installed if missing)

### macOS

#### Installation

```bash
# Install as user LaunchAgent (default)
python install_service.py

# Install as system LaunchDaemon (requires sudo)
sudo python install_service.py --system

# Or install directly
python scripts/install_macos_service.py
```

This creates a LaunchAgent/LaunchDaemon that:
- Runs on login (user) or boot (system)
- Restarts automatically if it crashes
- Integrates with macOS launchd system

#### Management Commands

```bash
# Using launchctl
launchctl load ~/Library/LaunchAgents/com.mcp.memory-service.plist
launchctl unload ~/Library/LaunchAgents/com.mcp.memory-service.plist
launchctl list | grep com.mcp.memory-service

# Using convenience scripts
./scripts/macos/start_service.sh
./scripts/macos/stop_service.sh
./scripts/macos/service_status.sh
```

#### Viewing Logs

- Check Console.app for service logs
- Or tail the log files directly:
  ```bash
  tail -f ~/.mcp_memory_service/logs/mcp-memory-service.log
  ```

### Linux

#### Installation

```bash
# Install as user service (default)
python install_service.py

# Install as system service (requires sudo)
sudo python install_service.py --system

# Or install directly
python scripts/install_linux_service.py
```

This creates a systemd service that:
- Runs on login (user) or boot (system)
- Integrates with systemd and journald
- Supports automatic restart and resource limits

#### Management Commands

```bash
# For user service
systemctl --user start mcp-memory
systemctl --user stop mcp-memory
systemctl --user status mcp-memory
journalctl --user -u mcp-memory -f

# For system service
sudo systemctl start mcp-memory
sudo systemctl stop mcp-memory
sudo systemctl status mcp-memory
sudo journalctl -u mcp-memory -f

# Using convenience scripts
./scripts/linux/start_service.sh
./scripts/linux/stop_service.sh
./scripts/linux/service_status.sh
./scripts/linux/view_logs.sh
```

## Configuration

### Service Configuration

All platforms store configuration in:
- **Config directory**: `~/.mcp_memory_service/`
- **Config file**: `~/.mcp_memory_service/service_config.json`
- **Log directory**: `~/.mcp_memory_service/logs/`

### Environment Variables

The service inherits these environment variables:
- `MCP_MEMORY_STORAGE_BACKEND`: Storage backend (default: `sqlite_vec`)
- `MCP_HTTP_ENABLED`: Enable HTTP interface (default: `true`)
- `MCP_HTTP_PORT`: HTTP port (default: `8000`)
- `MCP_HTTPS_ENABLED`: Enable HTTPS (default: `true`)
- `MCP_MDNS_ENABLED`: Enable mDNS discovery (default: `true`)
- `MCP_CONSOLIDATION_ENABLED`: Enable memory consolidation (default: `true`)
- `MCP_API_KEY`: API key for HTTP authentication (optional, auto-generated during install)

### API Key

The installer automatically generates a secure API key for HTTP authentication. This key protects your memory service from unauthorized access when running as a service.

#### Finding Your API Key

You can find your API key in several ways:

1. **Installation Output**: The key is displayed during installation
2. **Config File**: Located in `~/.mcp_memory_service/service_config.json`
3. **Status Command**: Run `python install_service.py --status`
4. **Service File**: Check the systemd/service configuration file directly

#### Generating a New API Key

To generate a new secure API key:

```bash
# Generate a 32-byte base64 encoded key
openssl rand -base64 32

# Or generate a hex key
openssl rand -hex 32
```

#### Updating the API Key

To change your API key after installation:

1. **Stop the service**:
   ```bash
   python install_service.py --stop
   ```

2. **Edit the service configuration**:
   - **Linux**: Edit `/etc/systemd/system/mcp-memory.service` or `~/.config/systemd/user/mcp-memory.service`
   - **macOS**: Edit `~/Library/LaunchAgents/com.mcp.memory-service.plist`
   - **Windows**: Use `sc config` or Services Manager

3. **Update the environment variable**:
   ```bash
   # Find the line with MCP_API_KEY and replace the value
   Environment=MCP_API_KEY=your-new-api-key-here
   ```

4. **Reload and restart the service**:
   ```bash
   # Linux (system service)
   sudo systemctl daemon-reload
   sudo systemctl restart mcp-memory
   
   # Linux (user service)
   systemctl --user daemon-reload
   systemctl --user restart mcp-memory
   
   # macOS
   launchctl unload ~/Library/LaunchAgents/com.mcp.memory-service.plist
   launchctl load ~/Library/LaunchAgents/com.mcp.memory-service.plist
   
   # Or use the installer
   python install_service.py --restart
   ```

#### Security Best Practices

- **Keep it Secret**: Never share your API key in logs, emails, or version control
- **Regular Rotation**: Consider rotating your API key periodically
- **Secure Storage**: Ensure proper file permissions on configuration files
- **Environment-Specific Keys**: Use different keys for development, staging, and production

## User vs System Services

### User Services
- **Pros**: No admin privileges required, runs in user context
- **Cons**: Only runs when user is logged in
- **Best for**: Desktop systems, development

### System Services
- **Pros**: Runs independently of user login, available to all users
- **Cons**: Requires admin privileges, runs as specific user
- **Best for**: Servers, shared systems

## Troubleshooting

### Service Won't Start

1. **Check dependencies**:
   ```bash
   python scripts/verify_environment.py
   ```

2. **Check logs**:
   - Windows: Event Viewer → Windows Logs → Application
   - macOS: Console.app or `~/.mcp_memory_service/logs/`
   - Linux: `journalctl -u mcp-memory` or `journalctl --user -u mcp-memory`

3. **Verify configuration**:
   ```bash
   cat ~/.mcp_memory_service/service_config.json
   ```

### Permission Errors

- **Windows**: Run as Administrator
- **macOS/Linux**: Use `sudo` for system services
- Check file ownership in `~/.mcp_memory_service/`

### Port Already in Use

If port 8000 is already in use:
1. Change the port in environment variables
2. Reinstall the service
3. Or stop the conflicting service

### Service Not Found

- Ensure the service was installed successfully
- Check the correct service name:
  - Windows: `MCPMemoryService`
  - macOS: `com.mcp.memory-service`
  - Linux: `mcp-memory`

## Uninstalling

To completely remove the service:

```bash
# Uninstall service
python install_service.py --uninstall

# Remove configuration (optional)
rm -rf ~/.mcp_memory_service/

# Remove the repository (optional)
cd ..
rm -rf mcp-memory-service/
```

## Advanced Usage

### Custom Service Names

For multiple instances, you can modify the service name in the platform-specific installer scripts before installation.

### Custom Startup Commands

Edit the service configuration after installation:
1. Stop the service
2. Edit `~/.mcp_memory_service/service_config.json`
3. Modify the `command` array
4. Restart the service

### Integration with Claude Desktop

After service installation, update your Claude Desktop configuration:

```json
{
  "mcpServers": {
    "memory": {
      "url": "http://localhost:8000/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_API_KEY_HERE"
      }
    }
  }
}
```

Replace `YOUR_API_KEY_HERE` with the API key from the installation.

## Security Considerations

- The API key is stored in plain text in the configuration file
- Ensure proper file permissions on configuration files
- Use system services with caution on shared systems
- Consider firewall rules if exposing the service beyond localhost

## Getting Help

If you encounter issues:
1. Check the troubleshooting section above
2. Review platform-specific documentation in `docs/platforms/`
3. Check logs for detailed error messages
4. Open an issue on GitHub with:
   - Your operating system and version
   - Python version
   - Error messages from logs
   - Steps to reproduce the issue