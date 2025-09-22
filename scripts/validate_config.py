#!/usr/bin/env python3
"""
Configuration Validation Script for MCP Memory Service

This script validates that Claude Code is properly configured for Cloudflare backend
and identifies any conflicting configurations that could cause issues.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

def load_json_safe(file_path: Path) -> Optional[Dict]:
    """Load JSON file safely, return None if not found or invalid."""
    try:
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError, PermissionError) as e:
        print(f"⚠️  Warning: Could not load {file_path}: {e}")
    return None

def check_environment_conflicts() -> List[str]:
    """Check for conflicting environment configurations."""
    issues = []

    # Check for conflicting .env files
    project_root = Path(__file__).parent.parent
    env_files = list(project_root.glob('.env*'))

    # Exclude .env.sqlite as it's the intended backup configuration, not a conflict
    conflicting_files = [f for f in env_files if f.name.endswith('.sqlite') and not f.name.endswith('.backup') and f.name != '.env.sqlite']
    if conflicting_files:
        issues.append(f"❌ Conflicting SQLite environment files found: {[str(f) for f in conflicting_files]}")

    # Check if .env has proper Cloudflare config using regex
    env_file = project_root / '.env'
    if env_file.exists():
        with open(env_file, 'r', encoding='utf-8') as f:
            env_content = f.read()
            # Use regex to match the exact configuration line
            cloudflare_pattern = r'^MCP_MEMORY_STORAGE_BACKEND\s*=\s*cloudflare\s*$'
            if re.search(cloudflare_pattern, env_content, re.MULTILINE):
                issues.append("✅ .env file has Cloudflare backend configured")
            else:
                issues.append("❌ .env file does not specify Cloudflare backend")
    else:
        issues.append("⚠️  No .env file found")

    return issues

def check_claude_config() -> Tuple[bool, List[str]]:
    """Check Claude Code configuration for memory service."""
    issues = []
    claude_config_path = Path.home() / '.claude.json'

    config = load_json_safe(claude_config_path)
    if not config:
        issues.append("❌ Could not load ~/.claude.json")
        return False, issues

    # Check for memory server configuration in projects
    memory_configs = []
    projects = config.get('projects', {})

    for project_path, project_config in projects.items():
        mcp_servers = project_config.get('mcpServers', {})
        if 'memory' in mcp_servers:
            memory_config = mcp_servers['memory']
            backend = memory_config.get('env', {}).get('MCP_MEMORY_STORAGE_BACKEND', 'unknown')
            memory_configs.append((project_path, backend))

    if not memory_configs:
        issues.append("❌ No memory server configuration found in Claude config")
        return False, issues

    # Check consistency
    cloudflare_configs = [cfg for cfg in memory_configs if cfg[1] == 'cloudflare']
    non_cloudflare_configs = [cfg for cfg in memory_configs if cfg[1] != 'cloudflare']

    if cloudflare_configs:
        issues.append(f"✅ Found {len(cloudflare_configs)} Cloudflare memory configurations")
        for path, backend in cloudflare_configs[:3]:  # Show first 3
            issues.append(f"   • {path}: {backend}")

    if non_cloudflare_configs:
        issues.append(f"⚠️  Found {len(non_cloudflare_configs)} non-Cloudflare memory configurations:")
        for path, backend in non_cloudflare_configs:
            issues.append(f"   • {path}: {backend}")

    return len(cloudflare_configs) > 0, issues

def check_project_mcp_configs() -> List[str]:
    """Check for conflicting project-level .mcp.json files."""
    issues = []

    # Check current project
    current_mcp = Path('.mcp.json')
    if current_mcp.exists():
        config = load_json_safe(current_mcp)
        if config and 'memory' in config.get('mcpServers', {}):
            issues.append("❌ Local .mcp.json contains memory server configuration (conflicts with global)")
        else:
            issues.append("✅ Local .mcp.json does not conflict with global memory configuration")
    else:
        issues.append("✅ No local .mcp.json found (good - using global configuration)")

    return issues

def check_cloudflare_credentials() -> List[str]:
    """Validate Cloudflare credentials are present."""
    issues = []

    # Check environment variables
    required_vars = [
        'CLOUDFLARE_API_TOKEN',
        'CLOUDFLARE_ACCOUNT_ID',
        'CLOUDFLARE_D1_DATABASE_ID',
        'CLOUDFLARE_VECTORIZE_INDEX'
    ]

    # Check .env file
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file, 'r', encoding='utf-8') as f:
            env_content = f.read()

        missing_vars = []
        for var in required_vars:
            if f"{var}=" not in env_content:
                missing_vars.append(var)

        if not missing_vars:
            issues.append("✅ All required Cloudflare environment variables found in .env")
        else:
            issues.append(f"❌ Missing Cloudflare variables in .env: {missing_vars}")
    else:
        issues.append("❌ No .env file found - Cloudflare credentials not configured")

    return issues

def main():
    """Run complete configuration validation."""
    print("🔍 MCP Memory Service Configuration Validation")
    print("=" * 50)

    all_issues = []

    print("\n1. Environment Configuration Check:")
    env_issues = check_environment_conflicts()
    all_issues.extend(env_issues)
    for issue in env_issues:
        print(f"   {issue}")

    print("\n2. Claude Code Global Configuration Check:")
    has_cloudflare, claude_issues = check_claude_config()
    all_issues.extend(claude_issues)
    for issue in claude_issues:
        print(f"   {issue}")

    print("\n3. Project-Level Configuration Check:")
    project_issues = check_project_mcp_configs()
    all_issues.extend(project_issues)
    for issue in project_issues:
        print(f"   {issue}")

    print("\n4. Cloudflare Credentials Check:")
    cred_issues = check_cloudflare_credentials()
    all_issues.extend(cred_issues)
    for issue in cred_issues:
        print(f"   {issue}")

    # Summary
    print("\n" + "=" * 50)
    error_count = len([i for i in all_issues if i.startswith("❌")])
    warning_count = len([i for i in all_issues if i.startswith("⚠️")])
    success_count = len([i for i in all_issues if i.startswith("✅")])

    if error_count == 0:
        print("🎉 Configuration validation PASSED!")
        print(f"   ✅ {success_count} checks passed")
        if warning_count > 0:
            print(f"   ⚠️  {warning_count} warnings (non-critical)")
        print("\n💡 Your Claude Code should be ready to use Cloudflare backend.")
        return 0
    else:
        print("❌ Configuration validation FAILED!")
        print(f"   ❌ {error_count} errors found")
        print(f"   ⚠️  {warning_count} warnings")
        print(f"   ✅ {success_count} checks passed")
        print("\n🔧 Please fix the errors above before using the memory service.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
