#!/usr/bin/env python3
"""
MCP Memory Service Wrapper with Orphan Process Cleanup

This wrapper cleans up orphaned MCP memory processes before starting the server.
Orphaned processes occur when Claude Desktop/Code crashes without properly
terminating child processes, causing SQLite "database locked" errors.

Works on: macOS, Linux, Windows

Usage in MCP config:
{
  "memory": {
    "command": "python",
    "args": ["/path/to/mcp-memory-service/scripts/run/memory_wrapper_cleanup.py"],
    "env": { ... }
  }
}
"""
import os
import sys
import signal
import subprocess
import platform
from pathlib import Path

def get_script_dir():
    """Get the directory containing this script."""
    return Path(__file__).parent.absolute()

def get_project_dir():
    """Get the project root directory."""
    return get_script_dir().parent.parent

def print_stderr(msg):
    """Print to stderr (visible in MCP logs)."""
    print(f"[mcp-memory-wrapper] {msg}", file=sys.stderr, flush=True)

def find_orphaned_processes_unix():
    """Find orphaned MCP memory processes on Unix-like systems (macOS/Linux).
    
    Orphaned processes have ppid=1 (adopted by init/launchd).
    """
    orphans = []
    try:
        # Find all mcp-memory-service processes
        result = subprocess.run(
            ["pgrep", "-f", "mcp-memory-service"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            return orphans
        
        pids = [int(p.strip()) for p in result.stdout.strip().split('\n') if p.strip()]
        
        for pid in pids:
            # Skip our own process
            if pid == os.getpid():
                continue
            
            try:
                # Get parent PID
                ppid_result = subprocess.run(
                    ["ps", "-o", "ppid=", "-p", str(pid)],
                    capture_output=True, text=True
                )
                if ppid_result.returncode == 0:
                    ppid = int(ppid_result.stdout.strip())
                    # ppid=1 means orphaned (parent is init/launchd)
                    if ppid == 1:
                        orphans.append(pid)
            except (ValueError, subprocess.SubprocessError):
                continue
                
    except FileNotFoundError:
        # pgrep not available
        pass
    except Exception as e:
        print_stderr(f"Error finding orphans: {e}")
    
    return orphans

def find_orphaned_processes_windows():
    """Find orphaned MCP memory processes on Windows.
    
    On Windows, we check if the parent process still exists.
    """
    orphans = []
    try:
        import ctypes
        from ctypes import wintypes
        
        # Get list of Python processes running mcp-memory-service
        result = subprocess.run(
            ["wmic", "process", "where",
             "commandline like '%mcp-memory-service%' and name='python.exe'",
             "get", "processid,parentprocessid"],
            capture_output=True, text=True
        )
        
        if result.returncode != 0:
            return orphans
        
        lines = result.stdout.strip().split('\n')[1:]  # Skip header
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    ppid = int(parts[0])
                    pid = int(parts[1])
                    
                    # Skip our own process
                    if pid == os.getpid():
                        continue
                    
                    # Check if parent process exists
                    parent_check = subprocess.run(
                        ["tasklist", "/FI", f"PID eq {ppid}"],
                        capture_output=True, text=True
                    )
                    
                    # If parent doesn't exist in task list, it's orphaned
                    if str(ppid) not in parent_check.stdout:
                        orphans.append(pid)
                        
                except (ValueError, IndexError):
                    continue
                    
    except Exception as e:
        print_stderr(f"Error finding orphans on Windows: {e}")
    
    return orphans

def kill_process(pid, force=False):
    """Kill a process by PID."""
    system = platform.system()
    
    try:
        if system == "Windows":
            cmd = ["taskkill", "/PID", str(pid)]
            if force:
                cmd.insert(1, "/F")
            subprocess.run(cmd, capture_output=True)
        else:
            sig = signal.SIGKILL if force else signal.SIGTERM
            os.kill(pid, sig)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False

def cleanup_orphans():
    """Find and terminate orphaned MCP memory processes."""
    system = platform.system()
    
    if system == "Windows":
        orphans = find_orphaned_processes_windows()
    else:
        orphans = find_orphaned_processes_unix()
    
    if orphans:
        print_stderr(f"Found {len(orphans)} orphaned process(es): {orphans}")
        for pid in orphans:
            if kill_process(pid):
                print_stderr(f"Terminated orphaned process: {pid}")
            else:
                # Try force kill
                if kill_process(pid, force=True):
                    print_stderr(f"Force-terminated orphaned process: {pid}")
                else:
                    print_stderr(f"Failed to terminate process: {pid}")
    else:
        print_stderr("No orphaned processes found")

def run_memory_server():
    """Run the MCP memory server."""
    project_dir = get_project_dir()
    
    # Change to project directory
    os.chdir(project_dir)
    
    # Determine how to run the server
    system = platform.system()
    
    # Try uv first (preferred method)
    uv_path = "uv"
    if system == "Windows":
        # Check common Windows locations
        local_uv = Path.home() / ".local" / "bin" / "uv.exe"
        if local_uv.exists():
            uv_path = str(local_uv)
    else:
        local_uv = Path.home() / ".local" / "bin" / "uv"
        if local_uv.exists():
            uv_path = str(local_uv)
    
    # Build command
    cmd = [uv_path, "run", "memory"]
    
    # Add any passed arguments
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])
    
    print_stderr(f"Starting server: {' '.join(cmd)}")
    
    # Use exec on Unix to replace this process (clean signal handling)
    # On Windows, use subprocess
    if system != "Windows":
        os.execvp(cmd[0], cmd)
    else:
        # On Windows, run as subprocess and forward exit code
        result = subprocess.run(cmd)
        sys.exit(result.returncode)

def main():
    """Main entry point."""
    print_stderr("MCP Memory Wrapper starting...")
    print_stderr(f"Platform: {platform.system()} {platform.release()}")
    
    # Step 1: Cleanup orphaned processes
    cleanup_orphans()
    
    # Step 2: Start the server (replaces this process on Unix)
    run_memory_server()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_stderr("Interrupted")
        sys.exit(0)
    except Exception as e:
        print_stderr(f"Fatal error: {e}")
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
