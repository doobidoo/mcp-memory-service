#!/usr/bin/env python3
"""
Backward compatibility redirect to new location (v6.17.0+).

This stub ensures existing Claude Desktop configurations continue working
after the v6.17.0 script reorganization. The actual script has moved to
scripts/server/run_memory_server.py.

For best stability, consider using one of these approaches instead:
1. python -m mcp_memory_service.server (recommended)
2. uv run memory server
3. scripts/server/run_memory_server.py (direct path)
"""

import os
import sys

# Add informational notice (not a warning to avoid alarming users)
print("[INFO] Note: scripts/run_memory_server.py has moved to scripts/server/run_memory_server.py", file=sys.stderr)
print("[INFO] Consider using 'python -m mcp_memory_service.server' for better stability", file=sys.stderr)
print("[INFO] See https://github.com/doobidoo/mcp-memory-service for migration guide", file=sys.stderr)

# Execute the relocated script
script_dir = os.path.dirname(os.path.abspath(__file__))
new_script = os.path.join(script_dir, "server", "run_memory_server.py")

if os.path.exists(new_script):
    # Preserve the original __file__ context for the new script
    global_vars = {"__file__": new_script, "__name__": "__main__", "sys": sys, "os": os}

    with open(new_script, encoding="utf-8") as f:
        exec(compile(f.read(), new_script, "exec"), global_vars)
else:
    print(f"[ERROR] Could not find {new_script}", file=sys.stderr)
    print("[ERROR] Please ensure you have the complete mcp-memory-service repository", file=sys.stderr)
    sys.exit(1)
