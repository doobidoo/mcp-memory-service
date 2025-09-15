#!/usr/bin/env python3
"""
macOS LaunchAgent installer for MCP Memory Service.
Creates and manages LaunchAgent plist files for automatic service startup.
"""
import os
import sys
import json
import plistlib
import argparse
import subprocess
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from scripts.service_utils import (
        get_project_root, get_service_paths, get_service_environment,
        generate_api_key, save_service_config, load_service_config,
        check_dependencies, get_service_command, print_service_info
    )
except ImportError as e:
    print(f"Error importing service utilities: {e}")
    print("Please ensure you're running this from the project directory")
    sys.exit(1)


SERVICE_LABEL = "com.mcp.memory-service"
SERVICE_NAME = "MCP Memory Service"


def get_launchd_paths(user_level=True):
    """Get the paths for LaunchAgent/LaunchDaemon files."""
    if user_level:
        # User-level LaunchAgent
        plist_dir = Path.home() / "Library" / "LaunchAgents"
        plist_file = plist_dir / f"{SERVICE_LABEL}.plist"
    else:
        # System-level LaunchDaemon (requires root)
        plist_dir = Path("/Library/LaunchDaemons")
        plist_file = plist_dir / f"{SERVICE_LABEL}.plist"
    
    return plist_dir, plist_file


def create_plist(api_key, user_level=True):
    """Create the LaunchAgent/LaunchDaemon plist configuration."""
    paths = get_service_paths()
    command = get_service_command()
    environment = get_service_environment()
    environment['MCP_API_KEY'] = api_key
    
    # Create plist dictionary
    plist_dict = {
        'Label': SERVICE_LABEL,
        'ProgramArguments': command,
        'EnvironmentVariables': environment,
        'WorkingDirectory': str(paths['project_root']),
        'RunAtLoad': True,
        'KeepAlive': {
            'SuccessfulExit': False,
            'Crashed': True
        },
        'StandardOutPath': str(paths['log_dir'] / 'mcp-memory-service.log'),
        'StandardErrorPath': str(paths['log_dir'] / 'mcp-memory-service.error.log'),
        'ProcessType': 'Interactive' if user_level else 'Background',
    }
    
    # Add user/group for system-level daemon
    if not user_level:
        plist_dict['UserName'] = os.environ.get('USER', 'nobody')
        plist_dict['GroupName'] = 'staff'
    
    return plist_dict


def create_shell_scripts():
    """Create convenient shell scripts for service management."""
    paths = get_service_paths()
    scripts_dir = paths['scripts_dir'] / 'macos'
    scripts_dir.mkdir(exist_ok=True)
    
    # Start script
    start_script = scripts_dir / 'start_service.sh'
    with open(start_script, 'w') as f:
        f.write(f'''#!/bin/bash
echo "Starting {SERVICE_NAME}..."
launchctl load ~/Library/LaunchAgents/{SERVICE_LABEL}.plist
if [ $? -eq 0 ]; then
    echo "✅ Service started successfully!"
else
    echo "❌ Failed to start service"
fi
''')
    start_script.chmod(0o755)
    
    # Stop script
    stop_script = scripts_dir / 'stop_service.sh'
    with open(stop_script, 'w') as f:
        f.write(f'''#!/bin/bash
echo "Stopping {SERVICE_NAME}..."
launchctl unload ~/Library/LaunchAgents/{SERVICE_LABEL}.plist
if [ $? -eq 0 ]; then
    echo "✅ Service stopped successfully!"
else
    echo "❌ Failed to stop service"
fi
''')
    stop_script.chmod(0o755)
    
    # Status script
    status_script = scripts_dir / 'service_status.sh'
    with open(status_script, 'w') as f:
        f.write(f'''#!/bin/bash
echo "{SERVICE_NAME} Status:"
echo "-" | tr '-' '='
launchctl list | grep {SERVICE_LABEL}
if [ $? -eq 0 ]; then
    echo ""
    echo "Service is loaded. PID shown above (- means not running)"
else
    echo "Service is not loaded"
fi
''')
    status_script.chmod(0o755)
    
    # Uninstall script
    uninstall_script = scripts_dir / 'uninstall_service.sh'
    with open(uninstall_script, 'w') as f:
        f.write(f'''#!/bin/bash
echo "This will uninstall {SERVICE_NAME}."
read -p "Are you sure? (y/N): " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    exit 0
fi

echo "Stopping service..."
launchctl unload ~/Library/LaunchAgents/{SERVICE_LABEL}.plist 2>/dev/null

echo "Removing service files..."
rm -f ~/Library/LaunchAgents/{SERVICE_LABEL}.plist

echo "✅ Service uninstalled"
''')
    uninstall_script.chmod(0o755)
    
    return scripts_dir


def install_service(user_level=True):
    """Install the macOS LaunchAgent/LaunchDaemon."""
    service_type = "LaunchAgent" if user_level else "LaunchDaemon"
    
    # Check for root if system-level
    if not user_level and os.geteuid() != 0:
        print("\n❌ ERROR: System-level LaunchDaemon requires root privileges")
        print("Please run with sudo or use --user for user-level installation")
        sys.exit(1)
    
    print(f"\n🔍 Checking dependencies...")
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
        'service_label': SERVICE_LABEL,
        'api_key': api_key,
        'command': get_service_command(),
        'environment': get_service_environment(),
        'user_level': user_level
    }
    
    # Save configuration
    config_file = save_service_config(config)
    print(f"💾 Saved configuration to: {config_file}")
    
    # Get plist paths
    plist_dir, plist_file = get_launchd_paths(user_level)
    
    # Create plist directory if it doesn't exist
    plist_dir.mkdir(parents=True, exist_ok=True)
    
    # Create plist
    print(f"\n📝 Creating {service_type} plist...")
    plist_dict = create_plist(api_key, user_level)
    
    # Write plist file
    with open(plist_file, 'wb') as f:
        plistlib.dump(plist_dict, f)
    
    # Set proper permissions
    if user_level:
        os.chmod(plist_file, 0o644)
    else:
        os.chmod(plist_file, 0o644)
        os.chown(plist_file, 0, 0)  # root:wheel
    
    print(f"✅ Created plist at: {plist_file}")
    
    # Load the service
    print(f"\n🚀 Loading {service_type}...")
    result = subprocess.run([
        'launchctl', 'load', '-w', str(plist_file)
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        if "already loaded" in result.stderr:
            print("ℹ️  Service was already loaded, reloading...")
            # Unload first
            subprocess.run(['launchctl', 'unload', str(plist_file)], capture_output=True)
            # Load again
            subprocess.run(['launchctl', 'load', '-w', str(plist_file)], capture_output=True)
        else:
            print(f"❌ Failed to load service: {result.stderr}")
            print("\n💡 Try checking Console.app for detailed error messages")
            sys.exit(1)
    
    print(f"✅ {service_type} loaded successfully!")
    
    # Create convenience scripts
    if user_level:
        scripts_dir = create_shell_scripts()
        print(f"\n📁 Created management scripts in: {scripts_dir}")
    
    # Print service information
    paths = get_service_paths()
    platform_info = {
        'Start Service': f'launchctl load -w {plist_file}',
        'Stop Service': f'launchctl unload {plist_file}',
        'Service Status': f'launchctl list | grep {SERVICE_LABEL}',
        'View Logs': f'tail -f {paths["log_dir"] / "mcp-memory-service.log"}',
        'Uninstall': f'python "{Path(__file__)}" --uninstall'
    }
    
    print_service_info(api_key, platform_info)
    
    # Additional macOS-specific tips
    print("\n📌 macOS Tips:")
    print("  • Check Console.app for detailed service logs")
    print("  • Service will start automatically on login/boot")
    print("  • Use Activity Monitor to verify the process is running")
    
    return True


def uninstall_service(user_level=True):
    """Uninstall the macOS LaunchAgent/LaunchDaemon."""
    service_type = "LaunchAgent" if user_level else "LaunchDaemon"
    
    # Check for root if system-level
    if not user_level and os.geteuid() != 0:
        print("\n❌ ERROR: System-level LaunchDaemon requires root privileges")
        print("Please run with sudo")
        sys.exit(1)
    
    print(f"\n🗑️  Uninstalling {SERVICE_NAME} {service_type}...")
    
    # Get plist paths
    plist_dir, plist_file = get_launchd_paths(user_level)
    
    if plist_file.exists():
        # Unload the service
        print("⏹️  Stopping service...")
        subprocess.run([
            'launchctl', 'unload', str(plist_file)
        ], capture_output=True)
        
        # Remove the plist file
        print("🗑️  Removing plist file...")
        plist_file.unlink()
        
        print(f"✅ {service_type} uninstalled successfully!")
    else:
        print(f"ℹ️  {service_type} is not installed")
    
    # Clean up configuration
    config = load_service_config()
    if config and config.get('service_label') == SERVICE_LABEL:
        print("🧹 Cleaning up configuration...")
        config_file = get_service_paths()['config_dir'] / 'service_config.json'
        config_file.unlink()


def start_service(user_level=True):
    """Start the macOS service."""
    plist_dir, plist_file = get_launchd_paths(user_level)
    
    if not plist_file.exists():
        print(f"❌ Service is not installed. Run without --start to install first.")
        sys.exit(1)
    
    print(f"\n▶️  Starting {SERVICE_NAME}...")
    
    result = subprocess.run([
        'launchctl', 'load', str(plist_file)
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Service started successfully!")
    else:
        if "already loaded" in result.stderr:
            print("ℹ️  Service is already running")
        else:
            print(f"❌ Failed to start service: {result.stderr}")


def stop_service(user_level=True):
    """Stop the macOS service."""
    plist_dir, plist_file = get_launchd_paths(user_level)
    
    print(f"\n⏹️  Stopping {SERVICE_NAME}...")
    
    result = subprocess.run([
        'launchctl', 'unload', str(plist_file)
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Service stopped successfully!")
    else:
        print(f"ℹ️  Service may not be running: {result.stderr}")


def service_status(user_level=True):
    """Check the macOS service status."""
    print(f"\n📊 {SERVICE_NAME} Status:")
    print("-" * 40)
    
    # Check if plist exists
    plist_dir, plist_file = get_launchd_paths(user_level)
    if not plist_file.exists():
        print("❌ Service is not installed")
        return
    
    # Check launchctl list
    result = subprocess.run([
        'launchctl', 'list'
    ], capture_output=True, text=True)
    
    service_found = False
    for line in result.stdout.splitlines():
        if SERVICE_LABEL in line:
            service_found = True
            parts = line.split()
            if len(parts) >= 3:
                pid = parts[0]
                status = parts[1]
                if pid != '-':
                    print(f"✅ Service is RUNNING (PID: {pid})")
                else:
                    print(f"⏹️  Service is STOPPED (last exit: {status})")
            break
    
    if not service_found:
        print("⏹️  Service is not loaded")
    
    # Show configuration
    config = load_service_config()
    if config:
        print(f"\n📋 Configuration:")
        print(f"  Service Label: {SERVICE_LABEL}")
        print(f"  API Key: {config.get('api_key', 'Not set')}")
        print(f"  Type: {'User LaunchAgent' if user_level else 'System LaunchDaemon'}")
        print(f"  Plist: {plist_file}")
    
    # Show recent logs
    paths = get_service_paths()
    log_file = paths['log_dir'] / 'mcp-memory-service.log'
    if log_file.exists():
        print(f"\n📜 Recent logs from {log_file}:")
        result = subprocess.run([
            'tail', '-n', '10', str(log_file)
        ], capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="macOS LaunchAgent installer for MCP Memory Service"
    )
    
    # Service level
    parser.add_argument('--user', action='store_true', default=True,
                        help='Install as user LaunchAgent (default)')
    parser.add_argument('--system', action='store_true',
                        help='Install as system LaunchDaemon (requires sudo)')
    
    # Actions
    parser.add_argument('--uninstall', action='store_true', help='Uninstall the service')
    parser.add_argument('--start', action='store_true', help='Start the service')
    parser.add_argument('--stop', action='store_true', help='Stop the service')
    parser.add_argument('--status', action='store_true', help='Check service status')
    parser.add_argument('--restart', action='store_true', help='Restart the service')
    
    args = parser.parse_args()
    
    # Determine service level
    user_level = not args.system
    
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


if __name__ == '__main__':
    main()