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
mDNS service advertisement and discovery for MCP Memory Service.

This module provides classes to advertise the MCP Memory Service on the local
network using mDNS (Multicast DNS) and discover other MCP Memory Service instances.
"""

import asyncio
import logging
import socket
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from zeroconf import Zeroconf, ServiceInfo, ServiceListener
from zeroconf.asyncio import AsyncZeroconf, AsyncServiceBrowser

from ..compat import _sanitize_log_value
from ..config import (
    MDNS_SERVICE_NAME,
    MDNS_SERVICE_TYPE,
    MDNS_DISCOVERY_TIMEOUT,
    HTTP_HOST,
    HTTP_PORT,
    HTTPS_ENABLED,
    API_KEY,
    SERVER_VERSION
)

logger = logging.getLogger(__name__)


@dataclass
class ServiceDetails:
    """Details of a discovered MCP Memory Service."""
    name: str
    host: str
    port: int
    https: bool
    api_version: str
    requires_auth: bool
    service_info: ServiceInfo
    
    @property
    def url(self) -> str:
        """Get the service URL."""
        protocol = "https" if self.https else "http"
        return f"{protocol}://{self.host}:{self.port}"
    
    @property
    def api_url(self) -> str:
        """Get the API base URL."""
        return f"{self.url}/api"


class ServiceAdvertiser:
    """Advertises MCP Memory Service via mDNS."""
    
    def __init__(
        self,
        service_name: str = MDNS_SERVICE_NAME,
        service_type: str = MDNS_SERVICE_TYPE,
        host: str = HTTP_HOST,
        port: int = HTTP_PORT,
        https_enabled: bool = HTTPS_ENABLED,
        api_key_required: bool = bool(API_KEY)
    ):
        self.service_name = service_name
        self.service_type = service_type
        self.host = host
        self.port = port
        self.https_enabled = https_enabled
        self.api_key_required = api_key_required
        
        self._zeroconf: Optional[AsyncZeroconf] = None
        self._service_info: Optional[ServiceInfo] = None
        self._registered = False
    
    def _get_local_ip(self) -> str:
        """Get the local IP address for service advertisement."""
        try:
            # Connect to a remote address to determine the local IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
                return local_ip
        except Exception:
            # Fallback to localhost if unable to determine IP
            return "127.0.0.1"
    
    def _create_service_info(self) -> ServiceInfo:
        """Create ServiceInfo for mDNS advertisement."""
        # Get local IP address
        local_ip = self._get_local_ip()
        
        # Create service properties
        properties = {
            'api_version': SERVER_VERSION.encode('utf-8'),
            'https': str(self.https_enabled).encode('utf-8'),
            'auth_required': str(self.api_key_required).encode('utf-8'),
            'api_path': b'/api',
            'sse_path': b'/api/events',
            'docs_path': b'/api/docs'
        }
        
        # Create unique service name
        full_service_name = f"{self.service_name}.{self.service_type}"
        
        service_info = ServiceInfo(
            type_=self.service_type,
            name=full_service_name,
            addresses=[socket.inet_aton(local_ip)],
            port=self.port,
            properties=properties,
            server=f"{self.service_name.replace(' ', '-').lower()}.local."
        )
        
        logger.info("Created service info: %s at %s:%s", full_service_name, local_ip, self.port)
        return service_info
    
    async def start(self) -> bool:
        """Start advertising the service via mDNS."""
        if self._registered:
            logger.warning("Service is already being advertised")
            return True
        
        try:
            self._zeroconf = AsyncZeroconf()
            self._service_info = self._create_service_info()
            
            await self._zeroconf.async_register_service(self._service_info)
            self._registered = True
            
            logger.info("mDNS service advertisement started for %s", self.service_name)
            return True
            
        except Exception as e:
            logger.error("Failed to start mDNS service advertisement: %s", _sanitize_log_value(e))
            return False
    
    async def stop(self) -> None:
        """Stop advertising the service."""
        if not self._registered:
            return
        
        try:
            if self._zeroconf and self._service_info:
                await self._zeroconf.async_unregister_service(self._service_info)
                await self._zeroconf.async_close()
            
            self._registered = False
            self._zeroconf = None
            self._service_info = None
            
            logger.info("mDNS service advertisement stopped for %s", self.service_name)
            
        except Exception as e:
            logger.error("Error stopping mDNS service advertisement: %s", _sanitize_log_value(e))
    
    def __del__(self):
        """Cleanup on deletion."""
        if self._registered and self._zeroconf:
            # Note: This should be handled by explicit stop() calls
            logger.warning("ServiceAdvertiser being deleted while still registered")


class DiscoveryListener(ServiceListener):
    """Listener for MCP Memory Service discoveries."""
    
    def __init__(self, callback: Optional[Callable[[ServiceDetails], None]] = None):
        self.callback = callback
        self.services: Dict[str, ServiceDetails] = {}
    
    def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        """Called when a service is discovered."""
        info = zc.get_service_info(type_, name)
        if info:
            try:
                service_details = self._parse_service_info(info)
                self.services[name] = service_details
                
                logger.info("Discovered MCP Memory Service: %s at %s", _sanitize_log_value(service_details.name), _sanitize_log_value(service_details.url))
                
                if self.callback:
                    self.callback(service_details)
                    
            except Exception as e:
                logger.error("Error parsing discovered service %s: %s", _sanitize_log_value(name), _sanitize_log_value(e))
    
    def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        """Called when a service is removed."""
        if name in self.services:
            service_details = self.services.pop(name)
            logger.info("MCP Memory Service removed: %s", _sanitize_log_value(service_details.name))
    
    def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        """Called when a service is updated."""
        info = zc.get_service_info(type_, name)
        if info:
            try:
                service_details = self._parse_service_info(info)
                self.services[name] = service_details
                
                logger.info("MCP Memory Service updated: %s", _sanitize_log_value(service_details.name))
                
                if self.callback:
                    self.callback(service_details)
                    
            except Exception as e:
                logger.error("Error parsing updated service %s: %s", _sanitize_log_value(name), _sanitize_log_value(e))
    
    def _parse_service_info(self, info: ServiceInfo) -> ServiceDetails:
        """Parse ServiceInfo into ServiceDetails."""
        # Get host address
        host = socket.inet_ntoa(info.addresses[0]) if info.addresses else "localhost"
        
        # Parse properties
        properties = info.properties or {}
        https = properties.get(b'https', b'false').decode('utf-8').lower() == 'true'
        api_version = properties.get(b'api_version', b'unknown').decode('utf-8')
        requires_auth = properties.get(b'auth_required', b'false').decode('utf-8').lower() == 'true'
        
        # Extract service name from full service name
        service_name = info.name.replace(f".{info.type}", "")
        
        return ServiceDetails(
            name=service_name,
            host=host,
            port=info.port,
            https=https,
            api_version=api_version,
            requires_auth=requires_auth,
            service_info=info
        )


class ServiceDiscovery:
    """Discovers MCP Memory Services on the local network."""
    
    def __init__(
        self,
        service_type: str = MDNS_SERVICE_TYPE,
        discovery_timeout: int = MDNS_DISCOVERY_TIMEOUT
    ):
        self.service_type = service_type
        self.discovery_timeout = discovery_timeout
        
        self._zeroconf: Optional[AsyncZeroconf] = None
        self._browser: Optional[AsyncServiceBrowser] = None
        self._listener: Optional[DiscoveryListener] = None
        self._discovering = False
    
    async def discover_services(
        self,
        callback: Optional[Callable[[ServiceDetails], None]] = None
    ) -> List[ServiceDetails]:
        """Discover MCP Memory Services on the network."""
        if self._discovering:
            logger.warning("Discovery is already in progress")
            return list(self._listener.services.values()) if self._listener else []
        
        services = []
        try:
            self._zeroconf = AsyncZeroconf()
            self._listener = DiscoveryListener(callback)
            
            self._browser = AsyncServiceBrowser(
                self._zeroconf.zeroconf,
                self.service_type,
                handlers=[self._listener]
            )
            
            self._discovering = True
            logger.info("Starting mDNS discovery for %s", self.service_type)
            
            # Wait for discovery timeout
            await asyncio.sleep(self.discovery_timeout)
            
            services = list(self._listener.services.values())
            logger.info("Discovered %d MCP Memory Services", len(services))
            
        except Exception as e:
            logger.error("Error during service discovery: %s", _sanitize_log_value(e))
        
        finally:
            await self.stop_discovery()
        
        return services
    
    async def start_continuous_discovery(
        self,
        callback: Callable[[ServiceDetails], None]
    ) -> bool:
        """Start continuous service discovery."""
        if self._discovering:
            logger.warning("Discovery is already in progress")
            return False
        
        try:
            self._zeroconf = AsyncZeroconf()
            self._listener = DiscoveryListener(callback)
            
            self._browser = AsyncServiceBrowser(
                self._zeroconf.zeroconf,
                self.service_type,
                handlers=[self._listener]
            )
            
            self._discovering = True
            logger.info("Started continuous mDNS discovery for %s", self.service_type)
            return True
            
        except Exception as e:
            logger.error("Error starting continuous service discovery: %s", _sanitize_log_value(e))
            return False
    
    async def stop_discovery(self) -> None:
        """Stop service discovery."""
        if not self._discovering:
            return
        
        try:
            if self._browser:
                await self._browser.async_cancel()
            
            if self._zeroconf:
                await self._zeroconf.async_close()
            
            self._discovering = False
            self._browser = None
            self._zeroconf = None
            self._listener = None
            
            logger.info("mDNS service discovery stopped")
            
        except Exception as e:
            logger.error("Error stopping service discovery: %s", _sanitize_log_value(e))
    
    def get_discovered_services(self) -> List[ServiceDetails]:
        """Get currently discovered services."""
        if self._listener:
            return list(self._listener.services.values())
        return []
    
    def __del__(self):
        """Cleanup on deletion."""
        if self._discovering:
            logger.warning("ServiceDiscovery being deleted while still discovering")