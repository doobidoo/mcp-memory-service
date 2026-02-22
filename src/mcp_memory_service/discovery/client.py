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
Discovery client for MCP Memory Service.

This module provides a high-level client for discovering and connecting to
MCP Memory Service instances on the local network.
"""

import asyncio
import logging
import aiohttp
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from .mdns_service import ServiceDiscovery, ServiceDetails
from ..config import MDNS_DISCOVERY_TIMEOUT

logger = logging.getLogger(__name__)


@dataclass
class HealthStatus:
    """Health status of a discovered service."""
    healthy: bool
    status: str
    backend: str
    statistics: Dict[str, Any]
    response_time_ms: float
    error: Optional[str] = None


class DiscoveryClient:
    """High-level client for discovering and validating MCP Memory Services."""
    
    def __init__(self, discovery_timeout: int = MDNS_DISCOVERY_TIMEOUT):
        self.discovery_timeout = discovery_timeout
        self._discovery = ServiceDiscovery(discovery_timeout=discovery_timeout)
    
    async def find_best_service(
        self,
        prefer_https: bool = True,
        require_auth: Optional[bool] = None,
        validate_health: bool = True
    ) -> Optional[ServiceDetails]:
        """
        Find the best MCP Memory Service on the network.
        
        Args:
            prefer_https: Prefer HTTPS services over HTTP
            require_auth: Require (True) or reject (False) services with auth, None for any
            validate_health: Validate service health before returning
            
        Returns:
            Best service found, or None if no suitable service
        """
        services = await self.discover_services()
        if not services:
            logger.info("No MCP Memory Services found on the network")
            return None
        
        # Filter services based on requirements
        filtered_services = []
        for service in services:
            # Check auth requirement
            if require_auth is not None and service.requires_auth != require_auth:
                continue
            
            filtered_services.append(service)
        
        if not filtered_services:
            logger.info("No services match the specified requirements")
            return None
        
        # Sort services by preference (HTTPS first if preferred)
        def service_priority(service: ServiceDetails) -> tuple:
            https_score = 1 if service.https else 0
            if not prefer_https:
                https_score = 1 - https_score  # Invert preference
            
            return (https_score, service.port)  # Secondary sort by port for consistency
        
        filtered_services.sort(key=service_priority, reverse=True)
        
        # Validate health if requested
        if validate_health:
            for service in filtered_services:
                health = await self.check_service_health(service)
                if health and health.healthy:
                    logger.info(f"Selected healthy service: {service.name} at {service.url}")
                    return service
                else:
                    logger.warning(f"Service {service.name} failed health check: {health.error if health else 'Unknown error'}")
            
            logger.warning("No healthy services found")
            return None
        else:
            # Return first service without health validation
            best_service = filtered_services[0]
            logger.info(f"Selected service: {best_service.name} at {best_service.url}")
            return best_service
    
    async def discover_services(self) -> List[ServiceDetails]:
        """Discover all MCP Memory Services on the network."""
        logger.info("Discovering MCP Memory Services on the network...")
        services = await self._discovery.discover_services()
        
        if services:
            logger.info(f"Found {len(services)} MCP Memory Services:")
            for service in services:
                logger.info(f"  - {service.name} at {service.url} (Auth: {service.requires_auth})")
        else:
            logger.info("No MCP Memory Services found")
        
        return services
    
    async def check_service_health(
        self,
        service: ServiceDetails,
        timeout: float = 5.0
    ) -> Optional[HealthStatus]:
        """
        Check the health of a discovered service.
        
        Args:
            service: Service to check
            timeout: Request timeout in seconds
            
        Returns:
            HealthStatus if check succeeded, None if failed
        """
        health_url = f"{service.api_url}/health"
        
        try:
            import time
            start_time = time.time()
            
            timeout_config = aiohttp.ClientTimeout(total=timeout)
            connector = aiohttp.TCPConnector(verify_ssl=False)  # Allow self-signed certs
            
            async with aiohttp.ClientSession(
                timeout=timeout_config,
                connector=connector
            ) as session:
                async with session.get(health_url) as response:
                    response_time = (time.time() - start_time) * 1000  # Convert to ms
                    
                    if response.status == 200:
                        data = await response.json()
                        return HealthStatus(
                            healthy=True,
                            status=data.get('status', 'unknown'),
                            backend=data.get('storage_type', 'unknown'),
                            statistics=data.get('statistics', {}),
                            response_time_ms=response_time
                        )
                    else:
                        return HealthStatus(
                            healthy=False,
                            status='error',
                            backend='unknown',
                            statistics={},
                            response_time_ms=response_time,
                            error=f"HTTP {response.status}"
                        )
        
        except asyncio.TimeoutError:
            return HealthStatus(
                healthy=False,
                status='timeout',
                backend='unknown',
                statistics={},
                response_time_ms=timeout * 1000,
                error="Request timeout"
            )
        except Exception as e:
            return HealthStatus(
                healthy=False,
                status='error',
                backend='unknown',
                statistics={},
                response_time_ms=0,
                error=str(e)
            )
    
    async def get_service_capabilities(
        self,
        service: ServiceDetails,
        api_key: Optional[str] = None,
        timeout: float = 5.0
    ) -> Optional[Dict[str, Any]]:
        """
        Get detailed capabilities of a service.
        
        Args:
            service: Service to query
            api_key: API key if required
            timeout: Request timeout
            
        Returns:
            Service capabilities or None if failed
        """
        try:
            headers = {}
            if api_key and service.requires_auth:
                headers['Authorization'] = f'Bearer {api_key}'
            
            timeout_config = aiohttp.ClientTimeout(total=timeout)
            connector = aiohttp.TCPConnector(verify_ssl=False)
            
            async with aiohttp.ClientSession(
                timeout=timeout_config,
                connector=connector
            ) as session:
                # Try to get OpenAPI spec
                openapi_url = f"{service.api_url}/openapi.json"
                async with session.get(openapi_url, headers=headers) as response:
                    if response.status == 200:
                        return await response.json()
                    
        except Exception as e:
            logger.error(f"Failed to get service capabilities: {e}")
        
        return None
    
    async def find_services_with_health(
        self,
        prefer_https: bool = True,
        require_auth: Optional[bool] = None
    ) -> List[tuple[ServiceDetails, HealthStatus]]:
        """
        Find all services and their health status.
        
        Returns:
            List of (service, health_status) tuples, sorted by preference
        """
        services = await self.discover_services()
        if not services:
            return []
        
        # Filter by auth requirement
        if require_auth is not None:
            services = [s for s in services if s.requires_auth == require_auth]
        
        # Check health for all services concurrently
        health_tasks = [self.check_service_health(service) for service in services]
        health_results = await asyncio.gather(*health_tasks, return_exceptions=True)
        
        # Combine services with health status
        service_health_pairs = []
        for service, health_result in zip(services, health_results):
            if isinstance(health_result, Exception):
                health = HealthStatus(
                    healthy=False,
                    status='error',
                    backend='unknown',
                    statistics={},
                    response_time_ms=0,
                    error=str(health_result)
                )
            else:
                health = health_result or HealthStatus(
                    healthy=False,
                    status='unknown',
                    backend='unknown',
                    statistics={},
                    response_time_ms=0,
                    error="No response"
                )
            
            service_health_pairs.append((service, health))
        
        # Sort by preference: healthy first, then HTTPS if preferred, then response time
        def sort_key(pair: tuple[ServiceDetails, HealthStatus]) -> tuple:
            service, health = pair
            healthy_score = 1 if health.healthy else 0
            https_score = 1 if service.https else 0
            if not prefer_https:
                https_score = 1 - https_score
            response_time = health.response_time_ms if health.healthy else float('inf')
            
            return (healthy_score, https_score, -response_time)  # Negative for ascending order
        
        service_health_pairs.sort(key=sort_key, reverse=True)
        return service_health_pairs
    
    async def stop(self) -> None:
        """Stop the discovery client."""
        await self._discovery.stop_discovery()