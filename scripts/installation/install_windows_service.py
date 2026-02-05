#!/usr/bin/env python3
"""
Windows Service installer for MCP Memory Service.
Installs the service using Python's Windows service capabilities.
"""

import argparse
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


SERVICE_NAME = "MCPMemoryService"
SERVICE_DISPLAY_NAME = "MCP Memory Service"
SERVICE_DESCRIPTION = "Semantic memory and persistent storage service for Claude Desktop"


def create_windows_service_script():
    """Create the Windows service wrapper script."""
    paths = get_service_paths()
    service_script = paths["scripts_dir"] / "mcp_memory_windows_service.py"

    script_content = '''#!/usr/bin/env python3
"""
Windows Service wrapper for MCP Memory Service.
This script runs as a Windows service and manages the MCP Memory server process.
"""
import os
import sys
import time
import subprocess
import win32serviceutil
import win32service
import win32event
import servicemanager
import socket
import json
from pathlib import Path


class MCPMemoryService(win32serviceutil.ServiceFramework):
    _svc_name_ = "MCPMemoryService"
    _svc_display_name_ = "MCP Memory Service"
    _svc_description_ = "Semantic memory and persistent storage service for Claude Desktop"

    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)
        socket.setdefaulttimeout(60)
        self.is_running = True
        self.process = None

    def SvcStop(self):
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.hWaitStop)
        self.is_running = False

        # Stop the subprocess
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()

    def SvcDoRun(self):
        servicemanager.LogMsg(
            servicemanager.EVENTLOG_INFORMATION_TYPE,
            servicemanager.PYS_SERVICE_STARTED,
            (self._svc_name_, '')
        )
        self.main()

    def main(self):
        # Load service configuration
        config_dir = Path.home() / '.mcp_memory_service'
        config_file = config_dir / 'service_config.json'

        if not config_file.exists():
            servicemanager.LogErrorMsg("Service configuration not found")
            return

        with open(config_file, 'r') as f:
            config = json.load(f)

        # Set up environment
        env = os.environ.copy()
        env.update(config['environment'])

        # Start the service process
        try:
            self.process = subprocess.Popen(
                config['command'],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            servicemanager.LogMsg(
                servicemanager.EVENTLOG_INFORMATION_TYPE,
                0,
                "MCP Memory Service process started"
            )

            # Monitor the process
            while self.is_running:
                if self.process.poll() is not None:
                    # Process died, log error and restart
                    stdout, stderr = self.process.communicate()
                    servicemanager.LogErrorMsg(
                        f"Service process died unexpectedly. stderr: {stderr}"
                    )

                    # Wait a bit before restarting
                    time.sleep(5)

                    # Restart the process
                    self.process = subprocess.Popen(
                        config['command'],
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )

                # Check if we should stop
                if win32event.WaitForSingleObject(self.hWaitStop, 1000) == win32event.WAIT_OBJECT_0:
                    break

        except Exception as e:
            servicemanager.LogErrorMsg(f"Error in service: {str(e)}")


if __name__ == '__main__':
    if len(sys.argv) == 1:
        servicemanager.Initialize()
        servicemanager.PrepareToHostSingle(MCPMemoryService)
        servicemanager.StartServiceCtrlDispatcher()
    else:
        win32serviceutil.HandleCommandLine(MCPMemoryService)
'''

    with open(service_script, "w") as f:
        f.write(script_content)

    return service_script


def create_batch_scripts():
    """Create convenient batch scripts for service management."""
    paths = get_service_paths()
    scripts_dir = paths["scripts_dir"] / "windows"
    scripts_dir.mkdir(exist_ok=True)

    # Start service batch file
    start_script = scripts_dir / "start_service.bat"
    with open(start_script, "w") as f:
        f.write(f"""@echo off
echo Starting {SERVICE_DISPLAY_NAME}...
net start {SERVICE_NAME}
if %ERRORLEVEL% == 0 (
    echo Service started successfully!
) else (
    echo Failed to start service. Run as Administrator if needed.
)
pause
""")

    # Stop service batch file
    stop_script = scripts_dir / "stop_service.bat"
    with open(stop_script, "w") as f:
        f.write(f"""@echo off
echo Stopping {SERVICE_DISPLAY_NAME}...
net stop {SERVICE_NAME}
if %ERRORLEVEL% == 0 (
    echo Service stopped successfully!
) else (
    echo Failed to stop service. Run as Administrator if needed.
)
pause
""")

    # Status batch file
    status_script = scripts_dir / "service_status.bat"
    with open(status_script, "w") as f:
        f.write(f"""@echo off
echo Checking {SERVICE_DISPLAY_NAME} status...
sc query {SERVICE_NAME}
pause
""")

    # Uninstall batch file
    uninstall_script = scripts_dir / "uninstall_service.bat"
    with open(uninstall_script, "w") as f:
        f.write(f"""@echo off
echo This will uninstall {SERVICE_DISPLAY_NAME}.
echo.
set /p confirm="Are you sure? (Y/N): "
if /i "%confirm%" neq "Y" exit /b

echo Stopping service...
net stop {SERVICE_NAME} 2>nul

echo Uninstalling service...
python "{paths["scripts_dir"] / "install_windows_service.py"}" --uninstall

pause
""")

    return scripts_dir


def install_service():
    """Install the Windows service."""
    # Check if pywin32 is installed
    try:
        import win32service
        import win32serviceutil
    except ImportError:
        print("\n❌ ERROR: pywin32 is required for Windows service installation")
        print("Please install it with: pip install pywin32")
        sys.exit(1)

    # Require administrator privileges
    require_admin("Administrator privileges are required to install Windows services")

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
    }
    config["environment"]["MCP_API_KEY"] = api_key

    # Save configuration
    config_file = save_service_config(config)
    print(f"💾 Saved configuration to: {config_file}")

    # Create service wrapper script
    print("\n📝 Creating service wrapper...")
    service_script = create_windows_service_script()

    # Install the service using the wrapper
    print(f"\n🚀 Installing {SERVICE_DISPLAY_NAME}...")

    try:
        # First, try to stop and remove existing service
        subprocess.run([sys.executable, str(service_script), "stop"], capture_output=True)
        subprocess.run([sys.executable, str(service_script), "remove"], capture_output=True)

        # Install the service
        result = subprocess.run([sys.executable, str(service_script), "install"], capture_output=True, text=True)

        if result.returncode != 0:
            print(f"❌ Failed to install service: {result.stderr}")
            sys.exit(1)

        # Configure service for automatic startup
        subprocess.run(["sc", "config", SERVICE_NAME, "start=", "auto"], capture_output=True)

        # Set service description
        subprocess.run(["sc", "description", SERVICE_NAME, SERVICE_DESCRIPTION], capture_output=True)

        print("✅ Service installed successfully!")

    except Exception as e:
        print(f"❌ Error installing service: {e}")
        sys.exit(1)

    # Create batch scripts
    scripts_dir = create_batch_scripts()
    print(f"\n📁 Created management scripts in: {scripts_dir}")

    # Print service information
    platform_info = {
        "Start Service": f"net start {SERVICE_NAME}",
        "Stop Service": f"net stop {SERVICE_NAME}",
        "Service Status": f"sc query {SERVICE_NAME}",
        "Uninstall": f'python "{Path(__file__)}" --uninstall',
    }

    print_service_info(api_key, platform_info)

    return True


def uninstall_service():
    """Uninstall the Windows service."""
    require_admin("Administrator privileges are required to uninstall Windows services")

    print(f"\n🗑️  Uninstalling {SERVICE_DISPLAY_NAME}...")

    paths = get_service_paths()
    service_script = paths["scripts_dir"] / "mcp_memory_windows_service.py"

    if not service_script.exists():
        # Try using sc command directly
        result = subprocess.run(["sc", "delete", SERVICE_NAME], capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ Service uninstalled successfully!")
        else:
            print(f"❌ Failed to uninstall service: {result.stderr}")
    else:
        # Stop the service first
        subprocess.run([sys.executable, str(service_script), "stop"], capture_output=True)

        # Remove the service
        result = subprocess.run([sys.executable, str(service_script), "remove"], capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ Service uninstalled successfully!")
        else:
            print(f"❌ Failed to uninstall service: {result.stderr}")


def start_service():
    """Start the Windows service."""
    print(f"\n▶️  Starting {SERVICE_DISPLAY_NAME}...")

    result = subprocess.run(["net", "start", SERVICE_NAME], capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ Service started successfully!")
    else:
        if "already been started" in result.stderr:
            print("ℹ️  Service is already running")
        else:
            print(f"❌ Failed to start service: {result.stderr}")
            print("\n💡 Try running as Administrator if you see access denied errors")


def stop_service():
    """Stop the Windows service."""
    print(f"\n⏹️  Stopping {SERVICE_DISPLAY_NAME}...")

    result = subprocess.run(["net", "stop", SERVICE_NAME], capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ Service stopped successfully!")
    else:
        if "is not started" in result.stderr:
            print("ℹ️  Service is not running")
        else:
            print(f"❌ Failed to stop service: {result.stderr}")


def service_status():
    """Check the Windows service status."""
    print(f"\n📊 {SERVICE_DISPLAY_NAME} Status:")
    print("-" * 40)

    result = subprocess.run(["sc", "query", SERVICE_NAME], capture_output=True, text=True)

    if result.returncode == 0:
        # Parse the output
        for line in result.stdout.splitlines():
            if "STATE" in line:
                if "RUNNING" in line:
                    print("✅ Service is RUNNING")
                elif "STOPPED" in line:
                    print("⏹️  Service is STOPPED")
                else:
                    print(f"ℹ️  {line.strip()}")
            elif "SERVICE_NAME:" in line:
                print(f"Service Name: {SERVICE_NAME}")
    else:
        print("❌ Service is not installed")

    # Show configuration if available
    config = load_service_config()
    if config:
        print("\n📋 Configuration:")
        print(f"  API Key: {config.get('api_key', 'Not set')}")
        print(f"  Config File: {get_service_paths()['config_dir'] / 'service_config.json'}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Windows Service installer for MCP Memory Service")

    parser.add_argument("--uninstall", action="store_true", help="Uninstall the service")
    parser.add_argument("--start", action="store_true", help="Start the service")
    parser.add_argument("--stop", action="store_true", help="Stop the service")
    parser.add_argument("--status", action="store_true", help="Check service status")
    parser.add_argument("--restart", action="store_true", help="Restart the service")

    args = parser.parse_args()

    if args.uninstall:
        uninstall_service()
    elif args.start:
        start_service()
    elif args.stop:
        stop_service()
    elif args.status:
        service_status()
    elif args.restart:
        stop_service()
        start_service()
    else:
        # Default action is to install
        install_service()


if __name__ == "__main__":
    main()
