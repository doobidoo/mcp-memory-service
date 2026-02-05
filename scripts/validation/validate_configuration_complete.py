#!/usr/bin/env python3
"""
Comprehensive Configuration Validation Script for MCP Memory Service

This unified script validates all configuration aspects:
- Claude Code global configuration (~/.claude.json)
- Claude Desktop configuration (claude_desktop_config.json)
- Project .env file configuration
- Cross-configuration consistency
- API token validation
- Cloudflare credentials validation

Consolidates functionality from validate_config.py and validate_configuration.py
"""

import json
import logging
import os
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class ComprehensiveConfigValidator:
    """Unified configuration validator for all MCP Memory Service configurations."""

    def __init__(self):
        """Initialize validator with all configuration paths and requirements."""
        self.project_root = Path(__file__).parent.parent.parent
        self.env_file = self.project_root / ".env"

        # Platform-specific Claude Desktop config paths
        if os.name == "nt":  # Windows
            self.claude_desktop_config_file = Path.home() / "AppData" / "Roaming" / "Claude" / "claude_desktop_config.json"
        else:  # macOS/Linux
            self.claude_desktop_config_file = Path.home() / ".config" / "claude" / "claude_desktop_config.json"

        # Claude Code global config (different from Claude Desktop)
        self.claude_code_config_file = Path.home() / ".claude.json"

        # Local project MCP config (should usually not exist for memory service)
        self.local_mcp_config_file = self.project_root / ".mcp.json"

        # Required environment variables for Cloudflare backend
        self.required_vars = [
            "MCP_MEMORY_STORAGE_BACKEND",
            "CLOUDFLARE_API_TOKEN",
            "CLOUDFLARE_ACCOUNT_ID",
            "CLOUDFLARE_D1_DATABASE_ID",
            "CLOUDFLARE_VECTORIZE_INDEX",
        ]

        # Optional but commonly used variables
        self.optional_vars = ["MCP_MEMORY_BACKUPS_PATH", "MCP_MEMORY_SQLITE_PATH"]

        # Results tracking
        self.issues = []
        self.error_count = 0
        self.warning_count = 0
        self.success_count = 0

    def load_json_safe(self, file_path: Path) -> dict | None:
        """Load JSON file safely, return None if not found or invalid."""
        try:
            if file_path.exists():
                with open(file_path, encoding="utf-8") as f:
                    return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError, PermissionError) as e:
            self.add_warning(f"Could not load {file_path}: {e}")
        return None

    def load_env_file(self) -> dict[str, str]:
        """Load environment variables from .env file."""
        env_vars = {}

        if not self.env_file.exists():
            self.add_warning(f"No .env file found at {self.env_file}")
            return env_vars

        try:
            with open(self.env_file, encoding="utf-8") as f:
                for _line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        env_vars[key.strip()] = value.strip()
        except Exception as e:
            self.add_error(f"Failed to load .env file: {e}")

        return env_vars

    def add_error(self, message: str):
        """Add error message and increment counter."""
        self.issues.append(f"ERROR: {message}")
        self.error_count += 1

    def add_warning(self, message: str):
        """Add warning message and increment counter."""
        self.issues.append(f"WARNING: {message}")
        self.warning_count += 1

    def add_success(self, message: str):
        """Add success message and increment counter."""
        self.issues.append(f"SUCCESS: {message}")
        self.success_count += 1

    def validate_env_file(self) -> dict[str, str]:
        """Validate .env file configuration."""
        env_vars = self.load_env_file()

        # Check for required variables
        missing_vars = []
        for var in self.required_vars:
            if var not in env_vars or not env_vars[var].strip():
                missing_vars.append(var)

        if missing_vars:
            self.add_error(f"Missing required variables in .env file: {missing_vars}")
        else:
            self.add_success("All required variables present in .env file")

        # Check backend setting
        backend = env_vars.get("MCP_MEMORY_STORAGE_BACKEND", "").lower()
        if backend == "cloudflare":
            self.add_success(".env file configured for Cloudflare backend")
        elif backend:
            self.add_warning(f".env file configured for '{backend}' backend (not Cloudflare)")
        else:
            self.add_error("MCP_MEMORY_STORAGE_BACKEND not set in .env file")

        return env_vars

    def validate_claude_desktop_config(self) -> dict[str, str] | None:
        """Validate Claude Desktop configuration."""
        config = self.load_json_safe(self.claude_desktop_config_file)

        if not config:
            self.add_error(f"Could not load Claude Desktop config from {self.claude_desktop_config_file}")
            return None

        # Extract memory server configuration
        mcp_servers = config.get("mcpServers", {})
        memory_server = mcp_servers.get("memory", {})

        if not memory_server:
            self.add_error("Memory server not found in Claude Desktop configuration")
            return None

        self.add_success("Memory server found in Claude Desktop configuration")

        # Get environment variables from memory server config
        memory_env = memory_server.get("env", {})

        # Check required variables
        missing_vars = []
        for var in self.required_vars:
            if var not in memory_env or not str(memory_env[var]).strip():
                missing_vars.append(var)

        if missing_vars:
            self.add_error(f"Missing required variables in Claude Desktop config: {missing_vars}")
        else:
            self.add_success("All required variables present in Claude Desktop config")

        return memory_env

    def validate_claude_code_config(self):
        """Validate Claude Code global configuration (different from Claude Desktop)."""
        config = self.load_json_safe(self.claude_code_config_file)

        if not config:
            self.add_warning(f"Claude Code config not found at {self.claude_code_config_file} (this is optional)")
            return

        # Check for memory server configurations in projects
        memory_configs = []
        projects = config.get("projects", {})

        for project_path, project_config in projects.items():
            mcp_servers = project_config.get("mcpServers", {})
            if "memory" in mcp_servers:
                memory_config = mcp_servers["memory"]
                backend = memory_config.get("env", {}).get("MCP_MEMORY_STORAGE_BACKEND", "unknown")
                memory_configs.append((project_path, backend))

        if memory_configs:
            cloudflare_configs = [cfg for cfg in memory_configs if cfg[1] == "cloudflare"]
            non_cloudflare_configs = [cfg for cfg in memory_configs if cfg[1] != "cloudflare"]

            if cloudflare_configs:
                self.add_success(f"Found {len(cloudflare_configs)} Cloudflare memory configurations in Claude Code")

            if non_cloudflare_configs:
                self.add_warning(f"Found {len(non_cloudflare_configs)} non-Cloudflare memory configurations in Claude Code")
        else:
            self.add_warning("No memory server configurations found in Claude Code (this is optional)")

    def validate_local_mcp_config(self):
        """Check for conflicting local .mcp.json files."""
        if self.local_mcp_config_file.exists():
            config = self.load_json_safe(self.local_mcp_config_file)
            if config and "memory" in config.get("mcpServers", {}):
                self.add_error("Local .mcp.json contains memory server configuration (conflicts with global config)")
            else:
                self.add_success("Local .mcp.json exists but does not conflict with memory configuration")
        else:
            self.add_success("No local .mcp.json found (using global configuration)")

    def compare_configurations(self, env_config: dict[str, str], claude_desktop_config: dict[str, str] | None):
        """Compare configurations between .env and Claude Desktop config."""
        if not claude_desktop_config:
            self.add_error("Cannot compare configurations - Claude Desktop config not available")
            return

        # Compare each required variable
        differences = []
        for var in self.required_vars:
            env_value = env_config.get(var, "<MISSING>")
            claude_value = str(claude_desktop_config.get(var, "<MISSING>"))

            if env_value != claude_value:
                differences.append((var, env_value, claude_value))

        if differences:
            self.add_warning(f"Found {len(differences)} configuration differences between .env and Claude Desktop config:")
            for var, env_val, claude_val in differences:
                self.add_warning(f"  {var}: .env='{env_val[:50]}...' vs Claude='{claude_val[:50]}...'")
        else:
            self.add_success("All configurations match between .env and Claude Desktop config")

    def validate_api_token_format(self, token: str) -> tuple[bool, str]:
        """Validate API token format and detect known invalid tokens."""
        if not token or token == "<MISSING>":
            return False, "Token is missing"

        if len(token) < 20:
            return False, "Token appears too short"

        if not any(c.isalnum() for c in token):
            return False, "Token should contain alphanumeric characters"

        # Check for known placeholder/invalid tokens
        invalid_tokens = [
            "your_token_here",
            "replace_with_token",
            "mkdXbb-iplcHNBRQ5tfqV3Sh_7eALYBpO4e3Di1m",  # Known invalid token
        ]
        if token in invalid_tokens:
            return False, "Token appears to be a placeholder or known invalid token"

        return True, "Token format appears valid"

    def validate_api_tokens(self, env_config: dict[str, str], claude_desktop_config: dict[str, str] | None):
        """Validate API tokens in both configurations."""
        # Check .env token
        env_token = env_config.get("CLOUDFLARE_API_TOKEN", "")
        is_valid, message = self.validate_api_token_format(env_token)

        if is_valid:
            self.add_success(f".env API token format: {message}")
        else:
            self.add_error(f".env API token: {message}")

        # Check Claude Desktop token
        if claude_desktop_config:
            claude_token = str(claude_desktop_config.get("CLOUDFLARE_API_TOKEN", ""))
            is_valid, message = self.validate_api_token_format(claude_token)

            if is_valid:
                self.add_success(f"Claude Desktop API token format: {message}")
            else:
                self.add_error(f"Claude Desktop API token: {message}")

    def check_environment_conflicts(self):
        """Check for conflicting environment configurations."""
        # Check for conflicting .env files
        env_files = list(self.project_root.glob(".env*"))

        # Exclude legitimate backup files
        conflicting_files = [
            f for f in env_files if f.name.endswith(".sqlite") and not f.name.endswith(".backup") and f.name != ".env.sqlite"
        ]

        if conflicting_files:
            self.add_warning(f"Potentially conflicting environment files found: {[str(f.name) for f in conflicting_files]}")
        else:
            self.add_success("No conflicting environment files detected")

    def run_comprehensive_validation(self) -> bool:
        """Run complete configuration validation across all sources."""
        print("Comprehensive MCP Memory Service Configuration Validation")
        print("=" * 70)

        # 1. Environment file validation
        print("\n1. Environment File (.env) Validation:")
        env_config = self.validate_env_file()
        self._print_section_results()

        # 2. Claude Desktop configuration validation
        print("\n2. Claude Desktop Configuration Validation:")
        claude_desktop_config = self.validate_claude_desktop_config()
        self._print_section_results()

        # 3. Claude Code configuration validation (optional)
        print("\n3. Claude Code Global Configuration Check:")
        self.validate_claude_code_config()
        self._print_section_results()

        # 4. Local MCP configuration check
        print("\n4. Local Project Configuration Check:")
        self.validate_local_mcp_config()
        self._print_section_results()

        # 5. Cross-configuration comparison
        print("\n5. Cross-Configuration Consistency Check:")
        self.compare_configurations(env_config, claude_desktop_config)
        self._print_section_results()

        # 6. API token validation
        print("\n6. API Token Validation:")
        self.validate_api_tokens(env_config, claude_desktop_config)
        self._print_section_results()

        # 7. Environment conflicts check
        print("\n7. Environment Conflicts Check:")
        self.check_environment_conflicts()
        self._print_section_results()

        # Final summary
        self._print_final_summary()

        return self.error_count == 0

    def _print_section_results(self):
        """Print results for the current section."""
        # Print only new issues since last call
        current_total = len(self.issues)
        if hasattr(self, "_last_printed_index"):
            start_index = self._last_printed_index
        else:
            start_index = 0

        for issue in self.issues[start_index:]:
            print(f"   {issue}")

        self._last_printed_index = current_total

    def _print_final_summary(self):
        """Print comprehensive final summary."""
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)

        if self.error_count == 0:
            print("CONFIGURATION VALIDATION PASSED!")
            print(f"   SUCCESS: {self.success_count} checks passed")
            if self.warning_count > 0:
                print(f"   WARNING: {self.warning_count} warnings (non-critical)")
            print("\nYour MCP Memory Service configuration appears to be correct.")
            print("You should be able to use the memory service with Cloudflare backend.")
        else:
            print("CONFIGURATION VALIDATION FAILED!")
            print(f"   ERROR: {self.error_count} critical errors found")
            print(f"   WARNING: {self.warning_count} warnings")
            print(f"   SUCCESS: {self.success_count} checks passed")
            print("\nPlease fix the critical errors above before using the memory service.")

        print("\nConfiguration files checked:")
        print(f"   • .env file: {self.env_file}")
        print(f"   • Claude Desktop config: {self.claude_desktop_config_file}")
        print(f"   • Claude Code config: {self.claude_code_config_file}")
        print(f"   • Local MCP config: {self.local_mcp_config_file}")


def main():
    """Main validation function."""
    validator = ComprehensiveConfigValidator()
    success = validator.run_comprehensive_validation()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
