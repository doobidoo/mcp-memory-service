#!/bin/bash
# Git merge driver for uv.lock files
# Automatically resolves conflicts and regenerates the lock file

# Arguments from git:
# %O = ancestor's version
# %A = current version
# %B = other version
# %L = conflict marker length
# %P = path to file

ANCESTOR="$1"
CURRENT="$2"
OTHER="$3"
MARKER_LENGTH="$4"
PATH="$5"

echo "Auto-resolving uv.lock conflict by regenerating lock file..."

# Accept the incoming version first (this resolves the conflict)
cp "$OTHER" "$PATH"

# Check if uv is available
if command -v uv >/dev/null 2>&1; then
    echo "Running uv sync to regenerate lock file..."
    # Regenerate the lock file based on pyproject.toml
    uv sync --quiet
    if [ $? -eq 0 ]; then
        echo "✓ uv.lock regenerated successfully"
        exit 0
    else
        echo "⚠ Warning: uv sync failed, using incoming version"
        exit 0
    fi
else
    echo "⚠ Warning: uv not found, using incoming version"
    exit 0
fi
