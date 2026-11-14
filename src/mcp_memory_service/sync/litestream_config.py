# Copyright 2024 Heinrich Krupp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Litestream configuration management for database synchronization.
"""

import yaml
import logging
import platform
from pathlib import Path
from typing import Dict, Any, Optional, List

from ..compat import _sanitize_log_value

logger = logging.getLogger(__name__)


class LitestreamManager:
    """
    Manages Litestream configuration for SQLite database replication.
    
    Provides utilities to generate configuration files for different
    deployment scenarios and machine types.
    """
    
    def __init__(self):
        """Initialize the Litestream manager."""
        self.platform = platform.system().lower()
    
    def generate_master_config(
        self,
        db_path: Path,
        replica_endpoint: str,
        backup_path: Optional[Path] = None,
        checkpoint_interval: str = "30s",
        wal_retention: str = "10m"
    ) -> Dict[str, Any]:
        """
        Generate Litestream configuration for master database.
        
        Args:
            db_path: Path to the SQLite database
            replica_endpoint: Endpoint where replicas can access the stream
            backup_path: Optional local backup path
            checkpoint_interval: How often to checkpoint
            wal_retention: How long to retain WAL entries
            
        Returns:
            Litestream configuration dictionary
        """
        config = {
            "dbs": [{
                "path": str(db_path),
                "replicas": [],
                "checkpoint-interval": checkpoint_interval,
                "wal-retention": wal_retention
            }]
        }
        
        db_config = config["dbs"][0]
        
        # Add HTTP replica endpoint
        if replica_endpoint:
            db_config["replicas"].append({
                "type": "file",
                "path": replica_endpoint,
                "sync-interval": "10s"
            })
        
        # Add local backup if specified
        if backup_path:
            db_config["replicas"].append({
                "type": "file",
                "path": str(backup_path),
                "sync-interval": "1m"
            })
        
        return config
    
    def generate_replica_config(
        self,
        db_path: Path,
        upstream_url: str,
        sync_interval: str = "10s"
    ) -> Dict[str, Any]:
        """
        Generate Litestream configuration for replica database.
        
        Args:
            db_path: Local path for the replicated database
            upstream_url: URL of the master database stream
            sync_interval: How often to sync from upstream
            
        Returns:
            Litestream configuration dictionary
        """
        config = {
            "dbs": [{
                "path": str(db_path),
                "replicas": [{
                    "type": "file",
                    "url": upstream_url,
                    "sync-interval": sync_interval
                }]
            }]
        }
        
        return config
    
    def generate_s3_config(
        self,
        db_path: Path,
        s3_endpoint: str,
        bucket: str,
        path: str,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        is_master: bool = True
    ) -> Dict[str, Any]:
        """
        Generate Litestream configuration for S3-compatible storage.
        
        Args:
            db_path: Path to the SQLite database
            s3_endpoint: S3-compatible endpoint URL
            bucket: S3 bucket name
            path: Path within the bucket
            access_key: S3 access key (optional, can use env vars)
            secret_key: S3 secret key (optional, can use env vars)
            is_master: Whether this is the master or replica
            
        Returns:
            Litestream configuration dictionary
        """
        replica_config = {
            "type": "s3",
            "endpoint": s3_endpoint,
            "bucket": bucket,
            "path": path
        }
        
        # Add credentials if provided
        if access_key and secret_key:
            replica_config.update({
                "access-key-id": access_key,
                "secret-access-key": secret_key
            })
        
        if is_master:
            config = {
                "dbs": [{
                    "path": str(db_path),
                    "replicas": [replica_config],
                    "checkpoint-interval": "30s",
                    "wal-retention": "10m"
                }]
            }
        else:
            config = {
                "dbs": [{
                    "path": str(db_path),
                    "replicas": [replica_config]
                }]
            }
        
        return config
    
    def get_default_config_path(self) -> Path:
        """Get the default Litestream configuration file path for this platform."""
        if self.platform == "windows":
            return Path("C:/ProgramData/litestream/litestream.yml")
        elif self.platform == "darwin":  # macOS
            return Path("/usr/local/etc/litestream.yml")
        else:  # Linux
            return Path("/etc/litestream.yml")
    
    def write_config(self, config: Dict[str, Any], config_path: Optional[Path] = None) -> Path:
        """
        Write Litestream configuration to file.
        
        Args:
            config: Configuration dictionary
            config_path: Path to write config file (uses default if not provided)
            
        Returns:
            Path where configuration was written
        """
        if config_path is None:
            config_path = self.get_default_config_path()
        
        # Create parent directory if needed
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write YAML configuration
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        logger.info("Litestream configuration written to %s", _sanitize_log_value(config_path))
        return config_path
    
    def generate_systemd_service(self, config_path: Path) -> str:
        """
        Generate systemd service file content for Litestream.
        
        Args:
            config_path: Path to the Litestream configuration file
            
        Returns:
            Systemd service file content
        """
        service_content = f"""[Unit]
Description=Litestream replication service
After=network.target
StartLimitIntervalSec=0

[Service]
Type=simple
Restart=always
RestartSec=1
User=root
ExecStart=/usr/local/bin/litestream replicate -config {config_path}

[Install]
WantedBy=multi-user.target
"""
        return service_content
    
    def generate_launchd_plist(self, config_path: Path) -> str:
        """
        Generate macOS LaunchDaemon plist for Litestream.
        
        Args:
            config_path: Path to the Litestream configuration file
            
        Returns:
            LaunchDaemon plist content
        """
        plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>io.litestream.replication</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/litestream</string>
        <string>replicate</string>
        <string>-config</string>
        <string>{config_path}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/var/log/litestream.log</string>
    <key>StandardErrorPath</key>
    <string>/var/log/litestream.log</string>
</dict>
</plist>
"""
        return plist_content
    
    def get_installation_commands(self) -> List[str]:
        """
        Get platform-specific Litestream installation commands.
        
        Returns:
            List of commands to install Litestream
        """
        if self.platform == "windows":
            return [
                "# Download and install Litestream for Windows",
                "# Visit: https://github.com/benbjohnson/litestream/releases",
                "# Extract litestream.exe to C:\\Program Files\\Litestream\\",
                "# Add to PATH environment variable"
            ]
        elif self.platform == "darwin":  # macOS
            return [
                "brew install benbjohnson/litestream/litestream"
            ]
        else:  # Linux
            return [
                "curl -LsS https://github.com/benbjohnson/litestream/releases/latest/download/litestream-linux-amd64.tar.gz | tar -xzf -",
                "sudo mv litestream /usr/local/bin/",
                "sudo chmod +x /usr/local/bin/litestream"
            ]
    
    def generate_deployment_script(
        self,
        role: str,  # "master" or "replica"
        db_path: Path,
        replica_endpoint: Optional[str] = None,
        upstream_url: Optional[str] = None
    ) -> str:
        """
        Generate a deployment script for setting up Litestream.
        
        Args:
            role: Whether this is a "master" or "replica"
            db_path: Path to the SQLite database
            replica_endpoint: Endpoint for serving replicas (master only)
            upstream_url: URL of master stream (replica only)
            
        Returns:
            Shell script content for deployment
        """
        install_commands = self.get_installation_commands()
        
        script_lines = [
            "#!/bin/bash",
            "# Litestream deployment script",
            f"# Role: {role}",
            "",
            "set -e",
            "",
            "echo 'Installing Litestream...'",
        ]
        
        script_lines.extend(install_commands)
        script_lines.extend([
            "",
            "echo 'Generating configuration...'",
        ])
        
        if role == "master":
            script_lines.extend([
                f"# Master configuration for {db_path}",
                f"# Serving replicas at: {replica_endpoint}",
            ])
        else:
            script_lines.extend([
                f"# Replica configuration for {db_path}",
                f"# Syncing from: {upstream_url}",
            ])
        
        script_lines.extend([
            "",
            "echo 'Starting Litestream service...'",
        ])
        
        if self.platform == "linux":
            script_lines.extend([
                "sudo systemctl enable litestream",
                "sudo systemctl start litestream",
                "sudo systemctl status litestream",
            ])
        elif self.platform == "darwin":
            script_lines.extend([
                "sudo launchctl load /Library/LaunchDaemons/io.litestream.replication.plist",
                "sudo launchctl start io.litestream.replication",
            ])
        
        script_lines.extend([
            "",
            "echo 'Litestream deployment completed!'",
            ""
        ])
        
        return "\n".join(script_lines)