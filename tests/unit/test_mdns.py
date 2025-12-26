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
Unit tests for mDNS service discovery functionality.
"""

import pytest
import asyncio
import socket
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from zeroconf import ServiceInfo, Zeroconf

# Import the modules under test
from mcp_memory_service.discovery.mdns_service import (
    ServiceAdvertiser, 
    ServiceDiscovery, 
    DiscoveryListener,
    ServiceDetails
)
from mcp_memory_service.discovery.client import DiscoveryClient, HealthStatus


class TestServiceDetails:
    """Test ServiceDetails dataclass."""
    
    def test_service_details_creation(self):
        """Test ServiceDetails creation with basic parameters."""
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
        
        assert details.name == "Test Service"
        assert details.host == "192.168.1.100"
        assert details.port == 8000
        assert details.https is False
        assert details.api_version == "2.1.0"
        assert details.requires_auth is True
        assert details.service_info == service_info
    
    def test_service_details_url_http(self):
        """Test URL generation for HTTP service."""
        details = ServiceDetails(
            name="Test Service",
            host="192.168.1.100",
            port=8000,
            https=False,
            api_version="2.1.0",
            requires_auth=False,
            service_info=Mock()
        )
        
        assert details.url == "http://192.168.1.100:8000"
        assert details.api_url == "http://192.168.1.100:8000/api"
    
    def test_service_details_url_https(self):
        """Test URL generation for HTTPS service."""
        details = ServiceDetails(
            name="Test Service",
            host="192.168.1.100",
            port=8443,
            https=True,
            api_version="2.1.0",
            requires_auth=True,
            service_info=Mock()
        )
        
        assert details.url == "https://192.168.1.100:8443"
        assert details.api_url == "https://192.168.1.100:8443/api"


class TestServiceAdvertiser:
    """Test ServiceAdvertiser class."""
    
    def test_init_default_parameters(self):
        """Test ServiceAdvertiser initialization with default parameters."""
        advertiser = ServiceAdvertiser()
        
        assert advertiser.service_name == "MCP Memory Service"
        assert advertiser.service_type == "_mcp-memory._tcp.local."
        assert advertiser.host == "0.0.0.0"
        assert advertiser.port == 8000
        assert advertiser.https_enabled is False
        assert advertiser.api_key_required is False
        assert advertiser._registered is False
    
    def test_init_custom_parameters(self):
        """Test ServiceAdvertiser initialization with custom parameters."""
        advertiser = ServiceAdvertiser(
            service_name="Custom Service",
            service_type="_custom._tcp.local.",
            host="192.168.1.100",
            port=8443,
            https_enabled=True,
            api_key_required=True
        )
        
        assert advertiser.service_name == "Custom Service"
        assert advertiser.service_type == "_custom._tcp.local."
        assert advertiser.host == "192.168.1.100"
        assert advertiser.port == 8443
        assert advertiser.https_enabled is True
        assert advertiser.api_key_required is True
    
    @patch('socket.socket')
    def test_get_local_ip(self, mock_socket):
        """Test local IP address detection."""
        mock_sock_instance = Mock()
        mock_sock_instance.getsockname.return_value = ("192.168.1.100", 12345)
        mock_socket.return_value.__enter__.return_value = mock_sock_instance
        
        advertiser = ServiceAdvertiser()
        ip = advertiser._get_local_ip()
        
        assert ip == "192.168.1.100"
        mock_sock_instance.connect.assert_called_once_with(("8.8.8.8", 80))
    
    @patch('socket.socket')
    def test_get_local_ip_fallback(self, mock_socket):
        """Test local IP address detection fallback."""
        mock_socket.side_effect = Exception("Network error")
        
        advertiser = ServiceAdvertiser()
        ip = advertiser._get_local_ip()
        
        assert ip == "127.0.0.1"
    
    @patch('socket.inet_aton')
    @patch.object(ServiceAdvertiser, '_get_local_ip')
    def test_create_service_info(self, mock_get_ip, mock_inet_aton):
        """Test ServiceInfo creation."""
        mock_get_ip.return_value = "192.168.1.100"
        mock_inet_aton.return_value = b'\xc0\xa8\x01\x64'  # 192.168.1.100
        
        advertiser = ServiceAdvertiser(
            service_name="Test Service",
            https_enabled=True,
            api_key_required=True
        )
        
        service_info = advertiser._create_service_info()
        
        assert service_info.type == "_mcp-memory._tcp.local."
        assert service_info.name == "Test Service._mcp-memory._tcp.local."
        assert service_info.port == 8000
        assert service_info.server == "test-service.local."
        
        # Check properties
        properties = service_info.properties
        assert properties[b'https'] == b'True'
        assert properties[b'auth_required'] == b'True'
        assert properties[b'api_path'] == b'/api'
    
    @pytest.mark.asyncio
    async def test_start_success(self):
        """Test successful service advertisement start."""
        with patch('mcp_memory_service.discovery.mdns_service.AsyncZeroconf') as mock_zeroconf_class:
            mock_zeroconf = AsyncMock()
            mock_zeroconf_class.return_value = mock_zeroconf
            
            advertiser = ServiceAdvertiser()
            
            with patch.object(advertiser, '_create_service_info') as mock_create_info:
                mock_service_info = Mock()
                mock_create_info.return_value = mock_service_info
                
                result = await advertiser.start()
                
                assert result is True
                assert advertiser._registered is True
                mock_zeroconf.async_register_service.assert_called_once_with(mock_service_info)
    
    @pytest.mark.asyncio
    async def test_start_already_registered(self):
        """Test starting advertisement when already registered."""
        advertiser = ServiceAdvertiser()
        advertiser._registered = True
        
        result = await advertiser.start()
        
        assert result is True  # Should return True but log warning
    
    @pytest.mark.asyncio
    async def test_start_failure(self):
        """Test service advertisement start failure."""
        with patch('mcp_memory_service.discovery.mdns_service.AsyncZeroconf') as mock_zeroconf_class:
            mock_zeroconf = AsyncMock()
            mock_zeroconf.async_register_service.side_effect = Exception("Registration failed")
            mock_zeroconf_class.return_value = mock_zeroconf
            
            advertiser = ServiceAdvertiser()
            
            with patch.object(advertiser, '_create_service_info'):
                result = await advertiser.start()
                
                assert result is False
                assert advertiser._registered is False
    
    @pytest.mark.asyncio
    async def test_stop_success(self):
        """Test successful service advertisement stop."""
        with patch('mcp_memory_service.discovery.mdns_service.AsyncZeroconf') as mock_zeroconf_class:
            mock_zeroconf = AsyncMock()
            mock_zeroconf_class.return_value = mock_zeroconf
            
            advertiser = ServiceAdvertiser()
            advertiser._registered = True
            advertiser._zeroconf = mock_zeroconf
            advertiser._service_info = Mock()
            
            await advertiser.stop()
            
            assert advertiser._registered is False
            mock_zeroconf.async_unregister_service.assert_called_once()
            mock_zeroconf.async_close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_stop_not_registered(self):
        """Test stopping advertisement when not registered."""
        advertiser = ServiceAdvertiser()
        
        # Should not raise exception
        await advertiser.stop()


class TestDiscoveryListener:
    """Test DiscoveryListener class."""
    
    def test_init_no_callback(self):
        """Test DiscoveryListener initialization without callback."""
        listener = DiscoveryListener()
        
        assert listener.callback is None
        assert len(listener.services) == 0
    
    def test_init_with_callback(self):
        """Test DiscoveryListener initialization with callback."""
        callback = Mock()
        listener = DiscoveryListener(callback)
        
        assert listener.callback == callback
    
    @patch('socket.inet_ntoa')
    def test_parse_service_info(self, mock_inet_ntoa):
        """Test parsing ServiceInfo into ServiceDetails."""
        mock_inet_ntoa.return_value = "192.168.1.100"
        
        # Create mock ServiceInfo
        service_info = Mock()
        service_info.name = "Test Service._mcp-memory._tcp.local."
        service_info.type = "_mcp-memory._tcp.local."
        service_info.port = 8000
        service_info.addresses = [b'\xc0\xa8\x01\x64']  # 192.168.1.100
        service_info.properties = {
            b'https': b'true',
            b'api_version': b'2.1.0',
            b'auth_required': b'false'
        }
        
        listener = DiscoveryListener()
        details = listener._parse_service_info(service_info)
        
        assert details.name == "Test Service"
        assert details.host == "192.168.1.100"
        assert details.port == 8000
        assert details.https is True
        assert details.api_version == "2.1.0"
        assert details.requires_auth is False
    
    @patch('socket.inet_ntoa')
    def test_parse_service_info_no_addresses(self, mock_inet_ntoa):
        """Test parsing ServiceInfo with no addresses."""
        service_info = Mock()
        service_info.name = "Test Service._mcp-memory._tcp.local."
        service_info.type = "_mcp-memory._tcp.local."
        service_info.port = 8000
        service_info.addresses = []
        service_info.properties = {}
        
        listener = DiscoveryListener()
        details = listener._parse_service_info(service_info)
        
        assert details.host == "localhost"
    
    def test_add_service_success(self):
        """Test successful service addition."""
        callback = Mock()
        listener = DiscoveryListener(callback)
        
        # Mock zeroconf and service info
        mock_zc = Mock()
        mock_service_info = Mock()
        mock_zc.get_service_info.return_value = mock_service_info
        
        with patch.object(listener, '_parse_service_info') as mock_parse:
            mock_details = Mock()
            mock_details.name = "Test Service"
            mock_parse.return_value = mock_details
            
            listener.add_service(mock_zc, "_mcp-memory._tcp.local.", "test-service")
            
            assert "test-service" in listener.services
            callback.assert_called_once_with(mock_details)
    
    def test_add_service_no_info(self):
        """Test service addition when no service info available."""
        listener = DiscoveryListener()
        
        mock_zc = Mock()
        mock_zc.get_service_info.return_value = None
        
        listener.add_service(mock_zc, "_mcp-memory._tcp.local.", "test-service")
        
        assert "test-service" not in listener.services
    
    def test_remove_service(self):
        """Test service removal."""
        listener = DiscoveryListener()
        mock_details = Mock()
        mock_details.name = "Test Service"
        listener.services["test-service"] = mock_details
        
        listener.remove_service(Mock(), "_mcp-memory._tcp.local.", "test-service")
        
        assert "test-service" not in listener.services


class TestServiceDiscovery:
    """Test ServiceDiscovery class."""
    
    def test_init_default_parameters(self):
        """Test ServiceDiscovery initialization with defaults."""
        discovery = ServiceDiscovery()
        
        assert discovery.service_type == "_mcp-memory._tcp.local."
        assert discovery.discovery_timeout == 5
        assert discovery._discovering is False
    
    def test_init_custom_parameters(self):
        """Test ServiceDiscovery initialization with custom parameters."""
        discovery = ServiceDiscovery(
            service_type="_custom._tcp.local.",
            discovery_timeout=10
        )
        
        assert discovery.service_type == "_custom._tcp.local."
        assert discovery.discovery_timeout == 10
    
    @pytest.mark.asyncio
    async def test_discover_services_success(self):
        """Test successful service discovery."""
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

            # Mock discovered services via listener
            mock_service = Mock()
            mock_service.name = "Test Service"
            mock_listener = Mock()
            mock_listener.services = {"test-service": mock_service}
            mock_listener_class.return_value = mock_listener

            discovery = ServiceDiscovery(discovery_timeout=0.1)  # Very short timeout for testing

            services = await discovery.discover_services()

            assert len(services) == 1
            assert services[0] == mock_service
    
    @pytest.mark.asyncio
    async def test_discover_services_already_discovering(self):
        """Test discovery when already in progress."""
        discovery = ServiceDiscovery()
        discovery._discovering = True
        
        # Mock existing services
        mock_listener = Mock()
        mock_service = Mock()
        mock_listener.services = {"test-service": mock_service}
        discovery._listener = mock_listener
        
        services = await discovery.discover_services()
        
        assert len(services) == 1
        assert services[0] == mock_service
    
    @pytest.mark.asyncio
    async def test_start_continuous_discovery(self):
        """Test starting continuous service discovery."""
        callback = Mock()

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

            discovery = ServiceDiscovery()

            result = await discovery.start_continuous_discovery(callback)

            assert result is True
            assert discovery._discovering is True
    
    @pytest.mark.asyncio
    async def test_start_continuous_discovery_already_started(self):
        """Test starting continuous discovery when already started."""
        discovery = ServiceDiscovery()
        discovery._discovering = True
        
        result = await discovery.start_continuous_discovery(Mock())
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_stop_discovery(self):
        """Test stopping service discovery."""
        with patch('mcp_memory_service.discovery.mdns_service.AsyncZeroconf') as mock_zeroconf_class, \
             patch('mcp_memory_service.discovery.mdns_service.AsyncServiceBrowser') as mock_browser_class:

            # Setup AsyncZeroconf mock with async methods
            mock_zeroconf = AsyncMock()
            mock_zeroconf.async_close = AsyncMock()

            # Setup AsyncServiceBrowser mock with async cancel method
            mock_browser = Mock()
            mock_browser.async_cancel = AsyncMock()

            discovery = ServiceDiscovery()
            discovery._discovering = True
            discovery._zeroconf = mock_zeroconf
            discovery._browser = mock_browser

            await discovery.stop_discovery()

            assert discovery._discovering is False
            mock_browser.async_cancel.assert_called_once()
            mock_zeroconf.async_close.assert_called_once()
    
    def test_get_discovered_services(self):
        """Test getting discovered services."""
        discovery = ServiceDiscovery()
        
        # No listener
        services = discovery.get_discovered_services()
        assert len(services) == 0
        
        # With listener
        mock_listener = Mock()
        mock_service = Mock()
        mock_listener.services = {"test": mock_service}
        discovery._listener = mock_listener
        
        services = discovery.get_discovered_services()
        assert len(services) == 1
        assert services[0] == mock_service


class TestDiscoveryClient:
    """Test DiscoveryClient class."""
    
    def test_init_default_timeout(self):
        """Test DiscoveryClient initialization with default timeout."""
        client = DiscoveryClient()
        
        assert client.discovery_timeout == 5
    
    def test_init_custom_timeout(self):
        """Test DiscoveryClient initialization with custom timeout."""
        client = DiscoveryClient(discovery_timeout=10)
        
        assert client.discovery_timeout == 10
    
    @pytest.mark.asyncio
    async def test_discover_services(self):
        """Test service discovery."""
        client = DiscoveryClient()
        
        mock_service = Mock()
        mock_service.name = "Test Service"
        mock_service.url = "http://192.168.1.100:8000"
        mock_service.requires_auth = False
        
        with patch.object(client._discovery, 'discover_services', return_value=[mock_service]):
            services = await client.discover_services()
            
            assert len(services) == 1
            assert services[0] == mock_service
    
    @pytest.mark.asyncio
    async def test_find_best_service_no_services(self):
        """Test finding best service when no services available."""
        client = DiscoveryClient()
        
        with patch.object(client, 'discover_services', return_value=[]):
            service = await client.find_best_service()
            
            assert service is None
    
    @pytest.mark.asyncio
    async def test_find_best_service_with_validation(self):
        """Test finding best service with health validation."""
        client = DiscoveryClient()
        
        # Create mock services
        http_service = Mock()
        http_service.https = False
        http_service.requires_auth = False
        http_service.port = 8000
        http_service.name = "HTTP Service"
        http_service.url = "http://192.168.1.100:8000"
        
        https_service = Mock()
        https_service.https = True
        https_service.requires_auth = False
        https_service.port = 8443
        https_service.name = "HTTPS Service"
        https_service.url = "https://192.168.1.100:8443"
        
        with patch.object(client, 'discover_services', return_value=[http_service, https_service]), \
             patch.object(client, 'check_service_health') as mock_health:
            
            # Mock health check results
            def health_side_effect(service):
                if service.https:
                    return HealthStatus(
                        healthy=True, status='ok', backend='sqlite_vec',
                        statistics={}, response_time_ms=50.0
                    )
                else:
                    return HealthStatus(
                        healthy=False, status='error', backend='unknown',
                        statistics={}, response_time_ms=0, error='Connection failed'
                    )
            
            mock_health.side_effect = health_side_effect
            
            service = await client.find_best_service(prefer_https=True)
            
            assert service == https_service
    
    @pytest.mark.asyncio
    async def test_check_service_health_success(self):
        """Test successful service health check."""
        client = DiscoveryClient()

        mock_service = Mock()
        mock_service.api_url = "http://192.168.1.100:8000/api"

        # Setup response mock with async context manager support
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            'status': 'healthy',
            'storage_type': 'sqlite_vec',
            'statistics': {'memory_count': 100}
        })
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        with patch('aiohttp.ClientSession') as mock_session_class:
            # Setup session mock with async context manager support
            mock_session = Mock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            # Setup get method to return response with async context manager
            mock_get_result = Mock()
            mock_get_result.__aenter__ = AsyncMock(return_value=mock_response)
            mock_get_result.__aexit__ = AsyncMock(return_value=None)
            mock_session.get = Mock(return_value=mock_get_result)

            mock_session_class.return_value = mock_session

            health = await client.check_service_health(mock_service)

            assert health is not None
            assert health.healthy is True
            assert health.status == 'healthy'
            assert health.backend == 'sqlite_vec'
            assert health.statistics == {'memory_count': 100}
    
    @pytest.mark.asyncio
    async def test_check_service_health_failure(self):
        """Test service health check failure."""
        client = DiscoveryClient()
        
        mock_service = Mock()
        mock_service.api_url = "http://192.168.1.100:8000/api"
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session_class.side_effect = Exception("Connection failed")
            
            health = await client.check_service_health(mock_service)
            
            assert health is not None
            assert health.healthy is False
            assert health.error == "Connection failed"
    
    @pytest.mark.asyncio
    async def test_find_services_with_health(self):
        """Test finding services with health status."""
        client = DiscoveryClient()
        
        # Create mock services
        service1 = Mock()
        service1.https = True
        service1.requires_auth = False
        
        service2 = Mock()
        service2.https = False
        service2.requires_auth = False
        
        health1 = HealthStatus(
            healthy=True, status='ok', backend='sqlite_vec',
            statistics={}, response_time_ms=50.0
        )
        
        health2 = HealthStatus(
            healthy=False, status='error', backend='unknown',
            statistics={}, response_time_ms=0, error='Connection failed'
        )
        
        with patch.object(client, 'discover_services', return_value=[service1, service2]), \
             patch.object(client, 'check_service_health', side_effect=[health1, health2]):
            
            services_with_health = await client.find_services_with_health()
            
            assert len(services_with_health) == 2
            # Should be sorted with healthy services first
            assert services_with_health[0][1].healthy is True
            assert services_with_health[1][1].healthy is False
    
    @pytest.mark.asyncio
    async def test_stop(self):
        """Test stopping the discovery client."""
        client = DiscoveryClient()
        
        with patch.object(client._discovery, 'stop_discovery') as mock_stop:
            await client.stop()
            mock_stop.assert_called_once()


class TestHealthStatus:
    """Test HealthStatus dataclass."""
    
    def test_health_status_creation(self):
        """Test HealthStatus creation."""
        health = HealthStatus(
            healthy=True,
            status='ok',
            backend='sqlite_vec',
            statistics={'memory_count': 100},
            response_time_ms=50.0,
            error=None
        )
        
        assert health.healthy is True
        assert health.status == 'ok'
        assert health.backend == 'sqlite_vec'
        assert health.statistics == {'memory_count': 100}
        assert health.response_time_ms == 50.0
        assert health.error is None
    
    def test_health_status_with_error(self):
        """Test HealthStatus creation with error."""
        health = HealthStatus(
            healthy=False,
            status='error',
            backend='unknown',
            statistics={},
            response_time_ms=0,
            error='Connection timeout'
        )
        
        assert health.healthy is False
        assert health.error == 'Connection timeout'


# Integration tests that can run without actual network
class TestMDNSIntegration:
    """Integration tests for mDNS functionality."""
    
    @pytest.mark.asyncio
    async def test_advertiser_discovery_integration(self):
        """Test integration between advertiser and discovery (mocked)."""
        # This test uses mocks to simulate the integration without actual network traffic

        with patch('mcp_memory_service.discovery.mdns_service.AsyncZeroconf') as mock_zeroconf_class, \
             patch('mcp_memory_service.discovery.mdns_service.AsyncServiceBrowser') as mock_browser_class, \
             patch('mcp_memory_service.discovery.mdns_service.DiscoveryListener') as mock_listener_class:

            # Setup AsyncZeroconf mock with async context manager support
            mock_zeroconf = AsyncMock()
            mock_zeroconf.__aenter__ = AsyncMock(return_value=mock_zeroconf)
            mock_zeroconf.__aexit__ = AsyncMock(return_value=None)
            mock_zeroconf.async_close = AsyncMock()
            mock_zeroconf.async_unregister_service = AsyncMock()
            mock_zeroconf.zeroconf = Mock()
            mock_zeroconf_class.return_value = mock_zeroconf

            # Setup AsyncServiceBrowser mock with async cancel method
            mock_browser = Mock()
            mock_browser.async_cancel = AsyncMock()
            mock_browser_class.return_value = mock_browser

            # Mock discovered service via listener
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
            mock_listener.services = {"test-service": mock_service}
            mock_listener_class.return_value = mock_listener

            # Start advertiser
            advertiser = ServiceAdvertiser(service_name="Test Service")

            with patch.object(advertiser, '_create_service_info'):
                await advertiser.start()
                assert advertiser._registered is True

            # Start discovery
            discovery = ServiceDiscovery(discovery_timeout=0.1)

            services = await discovery.discover_services()

            assert len(services) == 1
            assert services[0].name == "Test Service"

            # Clean up
            await advertiser.stop()
            await discovery.stop_discovery()


if __name__ == '__main__':
    pytest.main([__file__])