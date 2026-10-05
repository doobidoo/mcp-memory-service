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
Dream-inspired memory consolidation system.

This module implements autonomous memory consolidation inspired by human cognitive 
processes during sleep cycles, featuring exponential decay scoring, creative 
association discovery, semantic compression, and controlled forgetting.
"""

from .base import ConsolidationBase
from .decay import ExponentialDecayCalculator
from .associations import CreativeAssociationEngine
from .clustering import SemanticClusteringEngine
from .compression import SemanticCompressionEngine
from .forgetting import ControlledForgettingEngine
from .consolidator import DreamInspiredConsolidator
from .scheduler import ConsolidationScheduler
from .health import ConsolidationHealthMonitor, HealthStatus, HealthMetric, HealthAlert
from .belief import derive_confidence, should_promote, should_supersede
from .belief_service import BeliefService
from .pair_allowlist import *  # noqa: F401,F403  apply issue #1458 pair rules

__all__ = [
    'ConsolidationBase',
    'ExponentialDecayCalculator',
    'CreativeAssociationEngine', 
    'SemanticClusteringEngine',
    'SemanticCompressionEngine',
    'ControlledForgettingEngine',
    'DreamInspiredConsolidator',
    'ConsolidationScheduler',
    'ConsolidationHealthMonitor',
    'HealthStatus',
    'HealthMetric',
    'HealthAlert',
    'derive_confidence',
    'should_promote',
    'should_supersede',
    'BeliefService',
]
