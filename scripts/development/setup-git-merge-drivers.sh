#!/bin/bash
# Setup script for git merge drivers
# Run this once after cloning the repository

echo "Setting up git merge drivers for uv.lock..."

# Configure the uv.lock merge driver
git config merge.uv-lock-merge.driver './scripts/uv-lock-merge.sh %O %A %B %L %P'
git config merge.uv-lock-merge.name 'UV lock file merge driver'

# Make the merge script executable
chmod +x scripts/uv-lock-merge.sh

echo "✓ Git merge drivers configured successfully!"
echo "  uv.lock conflicts will now be resolved automatically"
