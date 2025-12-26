#!/usr/bin/env python3
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
Simple test script for mDNS functionality without external test frameworks.
"""

import asyncio
import sys
import os
import traceback
import pytest
from unittest.mock import Mock, AsyncMock, patch

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'src'))

def run_test(test_func, test_name):
    """Run a single test function and handle exceptions."""
    try:
        if asyncio.iscoroutinefunction(test_func):
            asyncio.run(test_func())
        else:
            test_func()
        print(f"✅ {test_name}")
        return True
    except Exception as e:
        print(f"❌ {test_name}: {e}")
        tb_lines = traceback.format_exc().split('\n')
        print(f"   {tb_lines[-3].strip()}")
        return False

def test_imports():
    """Test that mDNS modules can be imported."""
    from mcp_memory_service.discovery.mdns_service import (
        ServiceAdvertiser, ServiceDiscovery, DiscoveryListener, ServiceDetails
    )
    from mcp_memory_service.discovery.client import DiscoveryClient, HealthStatus
    
    # Test ServiceDetails creation
    service_info = Mock()
    details = ServiceDetails(
        name="Test Service",
        host="192.168.1.100",
        port=8000,
        https=False,
        api_version="2.1.0",
        requires_auth=True,
        service_info=service_info
    )
    
    assert details.url == "http://192.168.1.100:8000"
    assert details.api_url == "http://192.168.1.100:8000/api"

def test_service_advertiser_init():
    """Test ServiceAdvertiser initialization."""
    from mcp_memory_service.discovery.mdns_service import ServiceAdvertiser
    
    # Test default initialization
    advertiser = ServiceAdvertiser()
    assert advertiser.service_name == "MCP Memory Service"
    assert advertiser.service_type == "_mcp-memory._tcp.local."
    assert advertiser.port == 8000
    assert advertiser._registered is False
    
    # Test custom initialization
    custom_advertiser = ServiceAdvertiser(
        service_name="Custom Service",
        port=8443,
        https_enabled=True
    )
    assert custom_advertiser.service_name == "Custom Service"
    assert custom_advertiser.port == 8443
    assert custom_advertiser.https_enabled is True

@pytest.mark.asyncio
async def test_service_advertiser_start_stop():
    """Test ServiceAdvertiser start/stop with mocks."""
    from mcp_memory_service.discovery.mdns_service import ServiceAdvertiser

    with patch('mcp_memory_service.discovery.mdns_service.AsyncZeroconf') as mock_zeroconf_class:
        # Setup AsyncZeroconf mock with async context manager support
        mock_zeroconf = AsyncMock()
        mock_zeroconf.__aenter__ = AsyncMock(return_value=mock_zeroconf)
        mock_zeroconf.__aexit__ = AsyncMock(return_value=None)
        mock_zeroconf.async_close = AsyncMock()
        mock_zeroconf.async_unregister_service = AsyncMock()
        mock_zeroconf_class.return_value = mock_zeroconf

        advertiser = ServiceAdvertiser()

        with patch.object(advertiser, '_create_service_info') as mock_create_info:
            mock_service_info = Mock()
            mock_create_info.return_value = mock_service_info

            # Test start
            result = await advertiser.start()
            assert result is True
            assert advertiser._registered is True

            # Test stop
            await advertiser.stop()
            assert advertiser._registered is False

def test_service_discovery_init():
    """Test ServiceDiscovery initialization."""
    from mcp_memory_service.discovery.mdns_service import ServiceDiscovery
    
    discovery = ServiceDiscovery()
    assert discovery.service_type == "_mcp-memory._tcp.local."
    assert discovery.discovery_timeout == 5
    assert discovery._discovering is False

@pytest.mark.asyncio
async def test_service_discovery_operations():
    """Test ServiceDiscovery operations with mocks."""
    from mcp_memory_service.discovery.mdns_service import ServiceDiscovery, ServiceDetails

    with patch('mcp_memory_service.discovery.mdns_service.AsyncZeroconf') as mock_zeroconf_class, \
         patch('mcp_memory_service.discovery.mdns_service.AsyncServiceBrowser') as mock_browser_class, \
         patch('mcp_memory_service.discovery.mdns_service.DiscoveryListener') as mock_listener_class:

        # Setup AsyncZeroconf mock with async context manager support
        mock_zeroconf = AsyncMock()
        mock_zeroconf.__aenter__ = AsyncMock(return_value=mock_zeroconf)
        mock_zeroconf.__aexit__ = AsyncMock(return_value=None)
        mock_zeroconf.async_close = AsyncMock()
        mock_zeroconf.zeroconf = Mock()
        mock_zeroconf_class.return_value = mock_zeroconf

        # Setup AsyncServiceBrowser mock with async cancel method
        mock_browser = Mock()
        mock_browser.async_cancel = AsyncMock()
        mock_browser_class.return_value = mock_browser

        discovery = ServiceDiscovery(discovery_timeout=1)

        # Test get_discovered_services with no listener
        services = discovery.get_discovered_services()
        assert len(services) == 0

        # Test with mock listener
        mock_service = ServiceDetails(
            name="Test Service",
            host="192.168.1.100",
            port=8000,
            https=False,
            api_version="2.1.0",
            requires_auth=False,
            service_info=Mock()
        )
        mock_listener = Mock()
        mock_listener.services = {"test": mock_service}
        mock_listener_class.return_value = mock_listener
        discovery._listener = mock_listener

        services = discovery.get_discovered_services()
        assert len(services) == 1
        assert services[0] == mock_service

def test_discovery_listener():
    """Test DiscoveryListener functionality."""
    from mcp_memory_service.discovery.mdns_service import DiscoveryListener
    
    # Test initialization
    listener = DiscoveryListener()
    assert listener.callback is None
    assert len(listener.services) == 0
    
    # Test with callback
    callback = Mock()
    listener_with_callback = DiscoveryListener(callback)
    assert listener_with_callback.callback == callback

def test_discovery_client_init():
    """Test DiscoveryClient initialization."""
    from mcp_memory_service.discovery.client import DiscoveryClient
    
    client = DiscoveryClient()
    assert client.discovery_timeout == 5
    
    custom_client = DiscoveryClient(discovery_timeout=10)
    assert custom_client.discovery_timeout == 10

@pytest.mark.asyncio
async def test_discovery_client_operations():
    """Test DiscoveryClient operations with mocks."""
    from mcp_memory_service.discovery.client import DiscoveryClient, HealthStatus
    from mcp_memory_service.discovery.mdns_service import ServiceDetails

    with patch('mcp_memory_service.discovery.mdns_service.AsyncZeroconf') as mock_zeroconf_class, \
         patch('mcp_memory_service.discovery.mdns_service.AsyncServiceBrowser') as mock_browser_class, \
         patch('mcp_memory_service.discovery.mdns_service.DiscoveryListener') as mock_listener_class:

        # Setup AsyncZeroconf mock with async context manager support
        mock_zeroconf = AsyncMock()
        mock_zeroconf.__aenter__ = AsyncMock(return_value=mock_zeroconf)
        mock_zeroconf.__aexit__ = AsyncMock(return_value=None)
        mock_zeroconf.async_close = AsyncMock()
        mock_zeroconf.zeroconf = Mock()
        mock_zeroconf_class.return_value = mock_zeroconf

        # Setup AsyncServiceBrowser mock with async cancel method
        mock_browser = Mock()
        mock_browser.async_cancel = AsyncMock()
        mock_browser_class.return_value = mock_browser

        # Setup listener mock
        mock_listener = Mock()
        mock_listener_class.return_value = mock_listener

        client = DiscoveryClient()

        # Test discover_services
        mock_service = ServiceDetails(
            name="Test Service",
            host="192.168.1.100",
            port=8000,
            https=False,
            api_version="2.1.0",
            requires_auth=False,
            service_info=Mock()
        )

        with patch.object(client._discovery, 'discover_services', return_value=[mock_service]):
            services = await client.discover_services()
            assert len(services) == 1
            assert services[0] == mock_service

def test_health_status():
    """Test HealthStatus dataclass."""
    from mcp_memory_service.discovery.client import HealthStatus
    
    health = HealthStatus(
        healthy=True,
        status='ok',
        backend='sqlite_vec',
        statistics={'memory_count': 100},
        response_time_ms=50.0
    )
    
    assert health.healthy is True
    assert health.status == 'ok'
    assert health.backend == 'sqlite_vec'
    assert health.response_time_ms == 50.0

def test_service_details_properties():
    """Test ServiceDetails URL properties."""
    from mcp_memory_service.discovery.mdns_service import ServiceDetails
    
    # Test HTTP service
    http_service = ServiceDetails(
        name="HTTP Service",
        host="192.168.1.100",
        port=8000,
        https=False,
        api_version="2.1.0",
        requires_auth=False,
        service_info=Mock()
    )
    
    assert http_service.url == "http://192.168.1.100:8000"
    assert http_service.api_url == "http://192.168.1.100:8000/api"
    
    # Test HTTPS service
    https_service = ServiceDetails(
        name="HTTPS Service",
        host="192.168.1.100",
        port=8443,
        https=True,
        api_version="2.1.0",
        requires_auth=True,
        service_info=Mock()
    )
    
    assert https_service.url == "https://192.168.1.100:8443"
    assert https_service.api_url == "https://192.168.1.100:8443/api"

def main():
    """Run all tests."""
    print("🔧 MCP Memory Service - mDNS Unit Tests")
    print("=" * 50)
    
    tests = [
        (test_imports, "Import mDNS modules"),
        (test_service_advertiser_init, "ServiceAdvertiser initialization"),
        (test_service_advertiser_start_stop, "ServiceAdvertiser start/stop"),
        (test_service_discovery_init, "ServiceDiscovery initialization"),
        (test_service_discovery_operations, "ServiceDiscovery operations"),
        (test_discovery_listener, "DiscoveryListener functionality"),
        (test_discovery_client_init, "DiscoveryClient initialization"),
        (test_discovery_client_operations, "DiscoveryClient operations"),
        (test_health_status, "HealthStatus dataclass"),
        (test_service_details_properties, "ServiceDetails properties"),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func, test_name in tests:
        if run_test(test_func, test_name):
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All mDNS unit tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())