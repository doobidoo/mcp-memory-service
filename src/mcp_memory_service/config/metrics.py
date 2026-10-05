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

"""Prometheus ``/metrics`` endpoint configuration (issue #1097).

The endpoint is opt-in and OFF by default, mirroring ``OAUTH_ENABLED``. When
disabled the route is never registered (so ``GET /metrics`` is a 404), which
keeps the attack surface and output exactly as it was for every existing
deployment.
"""

import logging

from .base import safe_get_bool_env

logger = logging.getLogger(__name__)

# Public surface of this submodule. Keeping ``__all__`` explicit stops a
# ``from .metrics import *`` from leaking helpers (e.g. ``safe_get_bool_env``,
# ``logging``) into the ``config`` package namespace.
__all__ = ['METRICS_ENABLED']

# Opt-in flag for the Prometheus text-exposition endpoint. Default False:
# unset or an explicit false value leaves the route unregistered.
METRICS_ENABLED = safe_get_bool_env('MCP_METRICS_ENABLED', False)

if METRICS_ENABLED:
    logger.info("Prometheus /metrics endpoint enabled (MCP_METRICS_ENABLED)")
