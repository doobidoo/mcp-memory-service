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
OAuth 2.1 Dynamic Client Registration implementation for MCP Memory Service.

Provides OAuth 2.1 DCR endpoints to enable Claude Code HTTP transport integration.

This module implements:
- RFC 8414: OAuth 2.0 Authorization Server Metadata
- RFC 7591: OAuth 2.0 Dynamic Client Registration Protocol
- OAuth 2.1 security requirements and best practices

Key features:
- Dynamic client registration for automated OAuth client setup
- JWT-based access tokens with proper validation
- Authorization code flow with PKCE support
- Client credentials flow for server-to-server authentication
- Comprehensive scope-based authorization
- Backward compatibility with existing API key authentication
"""

__all__ = [
    "discovery",
    "models",
    "registration",
    "authorization",
    "middleware",
    "storage"
]