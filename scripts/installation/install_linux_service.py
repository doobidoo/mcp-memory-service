#!/usr/bin/env python3
"""
Linux systemd service installer for MCP Memory Service.
Creates and manages systemd service files for automatic service startup.
"""

import argparse
import grp
import os
import pwd
import subprocess
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from scripts.service_utils import (
        check_dependencies,
        generate_api_key,
        get_project_root,
        get_service_command,
        get_service_environment,
        get_service_paths,
        load_service_config,
        print_service_info,
        require_admin,
        save_service_config,
    )
except ImportError as e:
    print(f"Error importing service utilities: {e}")
    print("Please ensure you're running this from the project directory")
    sys.exit(1)


SERVICE_NAME = "mcp-memory"
SERVICE_DISPLAY_NAME = "MCP Memory Service"
SERVICE_DESCRIPTION = "MCP Memory Service with Consolidation and mDNS"


def get_systemd_paths(user_level=True):
    """Get the paths for systemd service files."""
    if user_level:
        # User-level systemd service
        service_dir = Path.home() / ".config" / "systemd" / "user"
        service_file = service_dir / f"{SERVICE_NAME}.service"
        systemctl_cmd = "systemctl --user"
    else:
        # System-level systemd service
        service_dir = Path("/etc/systemd/system")
        service_file = service_dir / f"{SERVICE_NAME}.service"
        systemctl_cmd = "sudo systemctl"

    return service_dir, service_file, systemctl_cmd


def create_systemd_service(api_key, user_level=True):
    """Create the systemd service unit file."""
    paths = get_service_paths()
    command = get_service_command()
    environment = get_service_environment()
    environment["MCP_API_KEY"] = api_key

    # Get current user info
    current_user = pwd.getpwuid(os.getuid())
    username = current_user.pw_name
    groupname = grp.getgrgid(current_user.pw_gid).gr_name

    # Build environment lines
    env_lines = []
    for key, value in environment.items():
        env_lines.append(f"Environment={key}={value}")

    # Create service content
    service_content = f"""[Unit]
Description={SERVICE_DESCRIPTION}
Documentation=https://github.com/doobidoo/mcp-memory-service
After=network.target network-online.target
Wants=network-online.target

[Service]
Type=simple
"""

    # Add user/group for system-level service
    if not user_level:
        service_content += f"""User={username}
Group={groupname}
"""

    service_content += f"""WorkingDirectory={paths["project_root"]}
ExecStart={" ".join(command)}
{chr(10).join(env_lines)}
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier={SERVICE_NAME}
"""

    # Add capabilities for binding to privileged ports (if using HTTPS on 443)
    if not user_level and environment.get("MCP_HTTP_PORT") == "443":
        service_content += """AmbientCapabilities=CAP_NET_BIND_SERVICE
CapabilityBoundingSet=CAP_NET_BIND_SERVICE
"""

    service_content += """
[Install]
WantedBy="""

    if user_level:
        service_content += "default.target"
    else:
        service_content += "multi-user.target"

    return service_content


def create_shell_scripts():
    """Create convenient shell scripts for service management."""
    paths = get_service_paths()
    scripts_dir = paths["scripts_dir"] / "linux"
    scripts_dir.mkdir(exist_ok=True)

    # Determine if user or system service based on existing installation
    user_service_file = Path.home() / ".config" / "systemd" / "user" / f"{SERVICE_NAME}.service"
    system_service_file = Path(f"/etc/systemd/system/{SERVICE_NAME}.service")

    if user_service_file.exists():
        systemctl = "systemctl --user"
        sudo = ""
    elif system_service_file.exists():
        systemctl = "systemctl"
        sudo = "sudo "
    else:
        # Default to user
        systemctl = "systemctl --user"
        sudo = ""

    # Start script
    start_script = scripts_dir / "start_service.sh"
    with open(start_script, "w") as f:
        f.write(f"""#!/bin/bash
echo "Starting {SERVICE_DISPLAY_NAME}..."
{sudo}{systemctl} start {SERVICE_NAME}
if [ $? -eq 0 ]; then
    echo "✅ Service started successfully!"
else
    echo "❌ Failed to start service"
fi
""")
    start_script.chmod(0o755)

    # Stop script
    stop_script = scripts_dir / "stop_service.sh"
    with open(stop_script, "w") as f:
        f.write(f"""#!/bin/bash
echo "Stopping {SERVICE_DISPLAY_NAME}..."
{sudo}{systemctl} stop {SERVICE_NAME}
if [ $? -eq 0 ]; then
    echo "✅ Service stopped successfully!"
else
    echo "❌ Failed to stop service"
fi
""")
    stop_script.chmod(0o755)

    # Status script
    status_script = scripts_dir / "service_status.sh"
    with open(status_script, "w") as f:
        f.write(f"""#!/bin/bash
echo "{SERVICE_DISPLAY_NAME} Status:"
echo "-" | tr '-' '='
{sudo}{systemctl} status {SERVICE_NAME}
""")
    status_script.chmod(0o755)

    # Logs script
    logs_script = scripts_dir / "view_logs.sh"
    with open(logs_script, "w") as f:
        f.write(f"""#!/bin/bash
echo "Viewing {SERVICE_DISPLAY_NAME} logs (press Ctrl+C to exit)..."
{sudo}journalctl -u {SERVICE_NAME} -f
""")
    logs_script.chmod(0o755)

    # Uninstall script
    uninstall_script = scripts_dir / "uninstall_service.sh"
    with open(uninstall_script, "w") as f:
        f.write(f"""#!/bin/bash
echo "This will uninstall {SERVICE_DISPLAY_NAME}."
read -p "Are you sure? (y/N): " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    exit 0
fi

echo "Stopping service..."
{sudo}{systemctl} stop {SERVICE_NAME} 2>/dev/null
{sudo}{systemctl} disable {SERVICE_NAME} 2>/dev/null

echo "Removing service files..."
if [ -f "$HOME/.config/systemd/user/{SERVICE_NAME}.service" ]; then
    rm -f "$HOME/.config/systemd/user/{SERVICE_NAME}.service"
    systemctl --user daemon-reload
else
    sudo rm -f /etc/systemd/system/{SERVICE_NAME}.service
    sudo systemctl daemon-reload
fi

echo "✅ Service uninstalled"
""")
    uninstall_script.chmod(0o755)

    return scripts_dir


def install_service(user_level=True):
    """Install the Linux systemd service."""
    service_type = "user service" if user_level else "system service"

    # Check for root if system-level
    if not user_level:
        require_admin("System-level service installation requires root privileges")

    print("\n🔍 Checking dependencies...")
    deps_ok, deps_msg = check_dependencies()
    if not deps_ok:
        print(f"❌ {deps_msg}")
        sys.exit(1)
    print(f"✅ {deps_msg}")

    # Generate API key
    api_key = generate_api_key()
    print(f"\n🔑 Generated API key: {api_key}")

    # Create service configuration
    config = {
        "service_name": SERVICE_NAME,
        "api_key": api_key,
        "command": get_service_command(),
        "environment": get_service_environment(),
        "user_level": user_level,
    }

    # Save configuration
    config_file = save_service_config(config)
    print(f"💾 Saved configuration to: {config_file}")

    # Get systemd paths
    service_dir, service_file, systemctl_cmd = get_systemd_paths(user_level)

    # Create service directory if it doesn't exist
    service_dir.mkdir(parents=True, exist_ok=True)

    # Create service file
    print(f"\n📝 Creating systemd {service_type} file...")
    service_content = create_systemd_service(api_key, user_level)

    # Write service file
    if user_level:
        with open(service_file, "w") as f:
            f.write(service_content)
        os.chmod(service_file, 0o644)
    else:
        # Use sudo to write system service file
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
            tmp.write(service_content)
            tmp_path = tmp.name

        subprocess.run(["sudo", "cp", tmp_path, str(service_file)], check=True)
        subprocess.run(["sudo", "chmod", "644", str(service_file)], check=True)
        os.unlink(tmp_path)

    print(f"✅ Created service file at: {service_file}")

    # Reload systemd
    print("\n🔄 Reloading systemd daemon...")
    if user_level:
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    else:
        subprocess.run(["sudo", "systemctl", "daemon-reload"], check=True)

    # Enable the service
    print(f"\n🚀 Enabling {service_type}...")
    cmd = systemctl_cmd.split() + ["enable", SERVICE_NAME]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ Failed to enable service: {result.stderr}")
        sys.exit(1)

    print("✅ Service enabled for automatic startup!")

    # Create convenience scripts
    scripts_dir = create_shell_scripts()
    print(f"\n📁 Created management scripts in: {scripts_dir}")

    # Print service information
    platform_info = {
        "Start Service": f"{systemctl_cmd} start {SERVICE_NAME}",
        "Stop Service": f"{systemctl_cmd} stop {SERVICE_NAME}",
        "Service Status": f"{systemctl_cmd} status {SERVICE_NAME}",
        "View Logs": f"{'sudo ' if not user_level else ''}journalctl {'--user ' if user_level else ''}-u {SERVICE_NAME} -f",
        "Uninstall": f'python "{Path(__file__)}" --uninstall',
    }

    print_service_info(api_key, platform_info)

    # Additional Linux-specific tips
    print("\n📌 Linux Tips:")
    print(f"  • Service will start automatically on {'login' if user_level else 'boot'}")
    print("  • Use journalctl to view detailed logs")
    print(f"  • {'User services require you to be logged in' if user_level else 'System service runs independently'}")

    # Offer to start the service
    print("\n▶️  To start the service now, run:")
    print(f"  {systemctl_cmd} start {SERVICE_NAME}")

    return True


def uninstall_service(user_level=None):
    """Uninstall the Linux systemd service."""
    # Auto-detect installation type if not specified
    if user_level is None:
        user_service_file = Path.home() / ".config" / "systemd" / "user" / f"{SERVICE_NAME}.service"
        system_service_file = Path(f"/etc/systemd/system/{SERVICE_NAME}.service")

        if user_service_file.exists():
            user_level = True
        elif system_service_file.exists():
            user_level = False
        else:
            print("❌ Service is not installed")
            return

    service_type = "user service" if user_level else "system service"

    # Check for root if system-level
    if not user_level:
        require_admin("System-level service removal requires root privileges")

    print(f"\n🗑️  Uninstalling {SERVICE_DISPLAY_NAME} {service_type}...")

    # Get systemd paths
    service_dir, service_file, systemctl_cmd = get_systemd_paths(user_level)

    if service_file.exists() or (not user_level and Path(f"/etc/systemd/system/{SERVICE_NAME}.service").exists()):
        # Stop the service
        print("⏹️  Stopping service...")
        cmd = systemctl_cmd.split() + ["stop", SERVICE_NAME]
        subprocess.run(cmd, capture_output=True)

        # Disable the service
        print("🔌 Disabling service...")
        cmd = systemctl_cmd.split() + ["disable", SERVICE_NAME]
        subprocess.run(cmd, capture_output=True)

        # Remove service file
        print("🗑️  Removing service file...")
        if user_level:
            service_file.unlink()
        else:
            subprocess.run(["sudo", "rm", "-f", str(service_file)], check=True)

        # Reload systemd
        print("🔄 Reloading systemd daemon...")
        if user_level:
            subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        else:
            subprocess.run(["sudo", "systemctl", "daemon-reload"], check=True)

        print(f"✅ {service_type} uninstalled successfully!")
    else:
        print(f"ℹ️  {service_type} is not installed")

    # Clean up configuration
    config = load_service_config()
    if config and config.get("service_name") == SERVICE_NAME:
        print("🧹 Cleaning up configuration...")
        config_file = get_service_paths()["config_dir"] / "service_config.json"
        config_file.unlink()


def start_service(user_level=None):
    """Start the Linux service."""
    # Auto-detect if not specified
    if user_level is None:
        user_service_file = Path.home() / ".config" / "systemd" / "user" / f"{SERVICE_NAME}.service"
        user_level = user_service_file.exists()

    service_dir, service_file, systemctl_cmd = get_systemd_paths(user_level)

    print(f"\n▶️  Starting {SERVICE_DISPLAY_NAME}...")

    cmd = systemctl_cmd.split() + ["start", SERVICE_NAME]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ Service started successfully!")
    else:
        print(f"❌ Failed to start service: {result.stderr}")
        print(f"\n💡 Check logs with: {systemctl_cmd} status {SERVICE_NAME}")


def stop_service(user_level=None):
    """Stop the Linux service."""
    # Auto-detect if not specified
    if user_level is None:
        user_service_file = Path.home() / ".config" / "systemd" / "user" / f"{SERVICE_NAME}.service"
        user_level = user_service_file.exists()

    service_dir, service_file, systemctl_cmd = get_systemd_paths(user_level)

    print(f"\n⏹️  Stopping {SERVICE_DISPLAY_NAME}...")

    cmd = systemctl_cmd.split() + ["stop", SERVICE_NAME]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ Service stopped successfully!")
    else:
        print(f"ℹ️  Service may not be running: {result.stderr}")


def service_status(user_level=None):
    """Check the Linux service status."""
    # Auto-detect if not specified
    if user_level is None:
        user_service_file = Path.home() / ".config" / "systemd" / "user" / f"{SERVICE_NAME}.service"
        system_service_file = Path(f"/etc/systemd/system/{SERVICE_NAME}.service")

        if user_service_file.exists():
            user_level = True
        elif system_service_file.exists():
            user_level = False
        else:
            print(f"\n❌ {SERVICE_DISPLAY_NAME} is not installed")
            return

    service_dir, service_file, systemctl_cmd = get_systemd_paths(user_level)

    print(f"\n📊 {SERVICE_DISPLAY_NAME} Status:")
    print("-" * 60)

    # Get detailed status
    cmd = systemctl_cmd.split() + ["status", SERVICE_NAME, "--no-pager"]
    subprocess.run(cmd)

    # Show configuration
    config = load_service_config()
    if config:
        print("\n📋 Configuration:")
        print(f"  Service Name: {SERVICE_NAME}")
        print(f"  API Key: {config.get('api_key', 'Not set')}")
        print(f"  Type: {'User Service' if user_level else 'System Service'}")
        print(f"  Service File: {service_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Linux systemd service installer for MCP Memory Service")

    # Service level
    parser.add_argument("--user", action="store_true", help="Install as user service (default)")
    parser.add_argument("--system", action="store_true", help="Install as system service (requires sudo)")

    # Actions
    parser.add_argument("--uninstall", action="store_true", help="Uninstall the service")
    parser.add_argument("--start", action="store_true", help="Start the service")
    parser.add_argument("--stop", action="store_true", help="Stop the service")
    parser.add_argument("--status", action="store_true", help="Check service status")
    parser.add_argument("--restart", action="store_true", help="Restart the service")

    args = parser.parse_args()

    # Determine service level
    if args.system and args.user:
        print("❌ Cannot specify both --user and --system")
        sys.exit(1)

    user_level = None  # Auto-detect for status/start/stop
    if args.system:
        user_level = False
    elif args.user or not any([args.uninstall, args.start, args.stop, args.status, args.restart]):
        user_level = True  # Default to user for installation

    if args.uninstall:
        uninstall_service(user_level)
    elif args.start:
        start_service(user_level)
    elif args.stop:
        stop_service(user_level)
    elif args.status:
        service_status(user_level)
    elif args.restart:
        stop_service(user_level)
        start_service(user_level)
    else:
        # Default action is to install
        install_service(user_level)


if __name__ == "__main__":
    main()
