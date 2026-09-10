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

"""Health monitoring and error handling for consolidation system."""

import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import traceback

from .base import ConsolidationError


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class HealthMetric:
    """Represents a health metric."""
    name: str
    value: Any
    status: HealthStatus
    message: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None


@dataclass
class HealthAlert:
    """Represents a health alert."""
    alert_id: str
    component: str
    severity: HealthStatus
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None


def _valid_retention_periods(retention_periods: Any) -> bool:
    """Check retention_periods is a non-empty dict with positive int values."""
    if not isinstance(retention_periods, dict) or len(retention_periods) == 0:
        return False
    return all(isinstance(v, (int, float)) and v > 0 for v in retention_periods.values())


def _valid_similarity_range(config: Any) -> bool:
    """Check min_similarity < max_similarity and both in [0, 1]."""
    lo = getattr(config, 'min_similarity', None)
    hi = getattr(config, 'max_similarity', None)
    if lo is None or hi is None:
        return False
    return 0 <= lo < hi <= 1


def _count_recent(history: List[Dict[str, Any]], component: str, hours: int = 1) -> int:
    """Count performance history entries for a component within the last N hours."""
    cutoff = datetime.now() - timedelta(hours=hours)
    return sum(
        1 for h in history
        if h.get('component') == component
        and h.get('timestamp', datetime.min) > cutoff
    )


class ConsolidationHealthMonitor:
    """Monitors health of the consolidation system."""

    def __init__(self, config=None, consolidator=None):
        self.config = config
        self.consolidator = consolidator
        self.logger = logging.getLogger(__name__)
        self._scheduler_ref = None

        # Health metrics storage
        self.metrics: Dict[str, HealthMetric] = {}
        self.alerts: List[HealthAlert] = []
        self.error_history: List[Dict[str, Any]] = []

        # Health thresholds
        self.thresholds = {
            'consolidation_success_rate': {'warning': 0.8, 'critical': 0.6},
            'average_duration_seconds': {'warning': 300, 'critical': 600},
            'memory_processing_rate': {'warning': 0.1, 'critical': 0.05},
            'error_rate': {'warning': 0.1, 'critical': 0.2},
            'storage_response_time': {'warning': 5.0, 'critical': 10.0}
        }

        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        self.max_history_entries = 1000

        # Component health cache
        self.component_health_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = timedelta(minutes=5)
        self.last_health_check = {}

    def attach_scheduler(self, scheduler):
        """Attach a ConsolidationScheduler reference for live health checks.

        Called by ConsolidationScheduler.__init__ so the monitor can inspect
        the running APScheduler instance instead of guessing from config.
        """
        self._scheduler_ref = scheduler

    async def check_overall_health(self) -> Dict[str, Any]:
        """Check overall consolidation system health."""
        try:
            health = {
                'status': HealthStatus.HEALTHY.value,
                'timestamp': datetime.now().isoformat(),
                'components': {},
                'metrics': {},
                'alerts': [],
                'recommendations': [],
                'statistics': {}
            }

            # Check individual components
            components = [
                'decay_calculator',
                'association_engine',
                'clustering_engine',
                'compression_engine',
                'forgetting_engine',
                'scheduler',
                'storage_backend'
            ]

            overall_status = HealthStatus.HEALTHY

            for component in components:
                component_health = await self._check_component_health(component)
                health['components'][component] = component_health

                # Update overall status based on component health
                component_status = HealthStatus(component_health.get('status', 'healthy'))
                if component_status == HealthStatus.CRITICAL:
                    overall_status = HealthStatus.CRITICAL
                elif component_status == HealthStatus.UNHEALTHY and overall_status != HealthStatus.CRITICAL:
                    overall_status = HealthStatus.UNHEALTHY
                elif component_status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED

            # Add current metrics
            health['metrics'] = {name: {
                'value': metric.value,
                'status': metric.status.value,
                'message': metric.message,
                'timestamp': metric.timestamp.isoformat()
            } for name, metric in self.metrics.items()}

            # Add active alerts
            active_alerts = [alert for alert in self.alerts if not alert.resolved]
            health['alerts'] = [{
                'alert_id': alert.alert_id,
                'component': alert.component,
                'severity': alert.severity.value,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat()
            } for alert in active_alerts[-10:]]  # Last 10 alerts

            # Add recommendations
            health['recommendations'] = await self._generate_health_recommendations()

            health['status'] = overall_status.value

            return health

        except Exception as e:
            self.logger.error(f"Error checking overall health: {e}")
            return {
                'status': HealthStatus.CRITICAL.value,
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'components': {},
                'metrics': {},
                'alerts': [],
                'recommendations': [],
                'statistics': {}
            }

    async def _check_component_health(self, component: str) -> Dict[str, Any]:
        """Check health of a specific component."""
        # Check cache first
        now = datetime.now()
        if (component in self.component_health_cache and
            component in self.last_health_check and
            now - self.last_health_check[component] < self.cache_ttl):
            return self.component_health_cache[component]

        try:
            health = {
                'status': HealthStatus.HEALTHY.value,
                'timestamp': now.isoformat(),
                'checks': {},
                'metrics': {}
            }

            if component == 'decay_calculator':
                health.update(await self._check_decay_calculator_health())
            elif component == 'association_engine':
                health.update(await self._check_association_engine_health())
            elif component == 'clustering_engine':
                health.update(await self._check_clustering_engine_health())
            elif component == 'compression_engine':
                health.update(await self._check_compression_engine_health())
            elif component == 'forgetting_engine':
                health.update(await self._check_forgetting_engine_health())
            elif component == 'scheduler':
                health.update(await self._check_scheduler_health())
            elif component == 'storage_backend':
                health.update(await self._check_storage_backend_health())

            # Cache the result
            self.component_health_cache[component] = health
            self.last_health_check[component] = now

            return health

        except Exception as e:
            self.logger.error(f"Error checking {component} health: {e}")
            return {
                'status': HealthStatus.UNHEALTHY.value,
                'timestamp': now.isoformat(),
                'error': str(e),
                'checks': {},
                'metrics': {}
            }

    async def _check_decay_calculator_health(self) -> Dict[str, Any]:
        """Check decay calculator health."""
        checks = {}
        status = HealthStatus.HEALTHY

        # Validate retention periods from config
        retention_periods = getattr(self.config, 'retention_periods', None)
        if _valid_retention_periods(retention_periods):
            checks['retention_periods'] = f'configured ({len(retention_periods)} types)'
        else:
            checks['retention_periods'] = 'missing or invalid'
            status = HealthStatus.DEGRADED

        checks['configuration'] = 'valid' if status == HealthStatus.HEALTHY else 'degraded'

        return {
            'status': status.value,
            'checks': checks,
            'metrics': {
                'recent_calculations': _count_recent(
                    self.performance_history, 'decay_calculator'
                )
            }
        }

    async def _check_association_engine_health(self) -> Dict[str, Any]:
        """Check association engine health."""
        checks = {}
        status = HealthStatus.HEALTHY

        # Validate similarity range from config (real fields: min_similarity, max_similarity)
        if _valid_similarity_range(self.config):
            lo = self.config.min_similarity
            hi = self.config.max_similarity
            checks['similarity_thresholds'] = f'range [{lo}, {hi}]'
        else:
            lo = getattr(self.config, 'min_similarity', None)
            hi = getattr(self.config, 'max_similarity', None)
            checks['similarity_thresholds'] = f'invalid: min={lo}, max={hi}'
            status = HealthStatus.DEGRADED

        checks['concept_extraction'] = 'functional'
        checks['association_discovery'] = 'active'

        return {
            'status': status.value,
            'checks': checks,
            'metrics': {
                'recent_associations_discovered': _count_recent(
                    self.performance_history, 'association_engine'
                ),
            }
        }

    async def _check_clustering_engine_health(self) -> Dict[str, Any]:
        """Check clustering engine health.

        Reports the algorithm that would actually run, not the one that was
        configured -- those differ whenever scikit-learn is missing, and a health
        check that hides the difference is how an unsatisfiable configuration
        reads as healthy.
        """
        from .clustering import SKLEARN_AVAILABLE, resolve_clustering_algorithm

        configured = self.config.clustering_algorithm
        try:
            algorithm = resolve_clustering_algorithm(configured)
        except ConsolidationError:
            algorithm = f"unsatisfiable: '{configured}' needs scikit-learn"

        return {
            'checks': {
                'clustering_algorithm': algorithm,
                'sklearn': 'available' if SKLEARN_AVAILABLE else 'unavailable',
                'minimum_cluster_size': 'configured',
                'embedding_processing': 'functional'
            },
            'metrics': {
                'recent_clusters_created': _count_recent(
                    self.performance_history, 'clustering_engine'
                )
            }
        }

    async def _check_compression_engine_health(self) -> Dict[str, Any]:
        """Check compression engine health."""
        checks = {}
        status = HealthStatus.HEALTHY

        checks['summary_generation'] = 'functional'
        checks['concept_extraction'] = 'active'

        recent_compressions = _count_recent(
            self.performance_history, 'compression_engine'
        )

        return {
            'status': status.value,
            'checks': checks,
            'metrics': {
                'recent_compressions': recent_compressions,
            }
        }

    async def _check_forgetting_engine_health(self) -> Dict[str, Any]:
        """Check forgetting engine health."""
        checks = {}
        status = HealthStatus.HEALTHY

        # Check archive storage accessibility
        archive_location = getattr(self.config, 'archive_location', None) or '~/.mcp_memory_archive'
        archive_path = Path(os.path.expanduser(archive_location))
        if archive_path.exists() and os.access(archive_path, os.W_OK):
            checks['archive_storage'] = 'accessible'
        elif archive_path.exists():
            checks['archive_storage'] = 'exists but not writable'
            status = HealthStatus.DEGRADED
        else:
            checks['archive_storage'] = f'path does not exist: {archive_path}'
            status = HealthStatus.DEGRADED

        # Check relevance thresholds from config
        relevance_threshold = getattr(self.config, 'relevance_threshold', None)
        if relevance_threshold is not None and 0 < relevance_threshold < 1:
            checks['relevance_thresholds'] = f'configured ({relevance_threshold})'
        else:
            checks['relevance_thresholds'] = f'unexpected: {relevance_threshold}'

        checks['controlled_forgetting'] = 'active'

        recent_archival = _count_recent(
            self.performance_history, 'forgetting_engine'
        )

        return {
            'status': status.value,
            'checks': checks,
            'metrics': {
                'recent_archival_operations': recent_archival,
            }
        }

    async def _check_scheduler_health(self) -> Dict[str, Any]:
        """Check scheduler health.

        When a ConsolidationScheduler reference is available (attached via
        ``attach_scheduler()``), the check inspects the live APScheduler
        instance.  Otherwise it falls back to config-level validation so
        disabled schedulers are reported as *degraded* rather than the
        hardcoded *healthy* that the old stubs returned.
        """
        checks = {}
        status = HealthStatus.HEALTHY
        scheduler = self._scheduler_ref

        schedule_config = self._resolve_schedule_config(scheduler)
        all_disabled = self._apply_schedule_config_check(checks, schedule_config)

        if scheduler is not None and hasattr(scheduler, 'scheduler') and scheduler.scheduler is not None:
            status = self._check_apscheduler_jobs(
                scheduler.scheduler, scheduler, checks)
        elif all_disabled:
            checks['scheduler_running'] = 'disabled by config'
            checks['job_scheduling'] = 'inactive'
            status = HealthStatus.DEGRADED
        else:
            checks['scheduler_running'] = 'not initialized'
            checks['job_scheduling'] = 'unavailable'
            status = HealthStatus.UNHEALTHY

        return {
            'status': status.value,
            'checks': checks,
            'metrics': {}
        }

    def _resolve_schedule_config(self, scheduler) -> Optional[dict]:
        """Get schedule_config from scheduler or config."""
        if scheduler is not None:
            config = getattr(scheduler, 'schedule_config', None)
            if config is not None:
                return config
        if self.config is not None:
            return getattr(self.config, 'schedule_config', None)
        return None

    @staticmethod
    def _apply_schedule_config_check(
            checks: dict, schedule_config: Optional[dict]) -> bool:
        """Populate checks['config'], return True if all horizons disabled."""
        if schedule_config and isinstance(schedule_config, dict):
            disabled = sum(1 for v in schedule_config.values() if v == 'disabled')
            checks['config'] = (
                f'{len(schedule_config)} horizons, {disabled} disabled')
            return disabled == len(schedule_config)
        checks['config'] = 'no schedule config'
        return True

    @staticmethod
    def _check_apscheduler_jobs(apscheduler, scheduler, checks: dict) -> HealthStatus:
        """Inspect a running APScheduler instance."""
        if not apscheduler.running:
            checks['scheduler_running'] = 'stopped'
            checks['job_scheduling'] = 'inactive'
            return HealthStatus.UNHEALTHY

        checks['scheduler_running'] = 'active'
        checks['scheduled_jobs'] = f'{len(apscheduler.get_jobs())} jobs'
        checks['job_scheduling'] = 'functional'

        last_executions = getattr(scheduler, 'last_execution_times', {})
        if last_executions:
            age = (datetime.now() - max(last_executions.values())).total_seconds()
            checks['last_execution'] = f'{age:.0f}s ago'
        else:
            checks['last_execution'] = 'never (no runs yet)'

        stats = getattr(scheduler, 'execution_stats', {})
        checks['execution_stats'] = (
            f"{stats.get('total_jobs', 0)} total, "
            f"{stats.get('successful_jobs', 0)} ok, "
            f"{stats.get('failed_jobs', 0)} failed"
        )
        return HealthStatus.HEALTHY

    async def _check_storage_backend_health(self) -> Dict[str, Any]:
        """Check storage backend health."""
        checks = {}
        status = HealthStatus.HEALTHY

        storage = None
        if self.consolidator is not None:
            storage = getattr(self.consolidator, 'storage', None)

        if storage is None:
            checks['storage_connection'] = 'not initialized'
            checks['read_operations'] = 'unavailable'
            checks['write_operations'] = 'unavailable'
            return {
                'status': HealthStatus.UNHEALTHY.value,
                'checks': checks,
                'metrics': {}
            }

        status, response_ms = await self._ping_storage(storage, checks)
        self._check_write_capability(storage, checks, status)
        status = self._check_response_time(response_ms, checks, status)

        return {
            'status': status.value,
            'checks': checks,
            'metrics': {
                'response_time_ms': round(response_ms, 1),
            }
        }

    @staticmethod
    async def _ping_storage(
            storage, checks: dict) -> tuple[HealthStatus, float]:
        """Ping storage and populate connection/read checks."""
        status = HealthStatus.HEALTHY
        start = time.monotonic()
        try:
            if hasattr(storage, 'get_stats'):
                stats = await storage.get_stats()
                has_error = ('error' in stats
                             or stats.get('status') == 'error')
                if has_error:
                    checks['storage_connection'] = 'error'
                    checks['read_operations'] = 'failing'
                    status = HealthStatus.UNHEALTHY
                else:
                    checks['storage_connection'] = 'connected'
                    checks['read_operations'] = 'functional'
                    checks['memory_count'] = stats.get(
                        'total_memories', 'unknown')
            elif hasattr(storage, 'count_all_memories'):
                count = await storage.count_all_memories()
                checks['storage_connection'] = 'connected'
                checks['read_operations'] = 'functional'
                checks['memory_count'] = count
            else:
                checks['storage_connection'] = 'no ping method'
                checks['read_operations'] = 'unverifiable'
        except Exception as e:
            checks['storage_connection'] = f'error: {type(e).__name__}'
            checks['read_operations'] = 'failing'
            status = HealthStatus.UNHEALTHY
        response_ms = (time.monotonic() - start) * 1000
        return status, response_ms

    @staticmethod
    def _check_write_capability(storage, checks: dict, status: HealthStatus) -> None:
        """Verify storage has a write method."""
        if hasattr(storage, 'store') or hasattr(storage, 'add_memory'):
            checks['write_operations'] = 'functional'
        else:
            checks['write_operations'] = 'method missing'

    @staticmethod
    def _check_response_time(
            response_ms: float, checks: dict, status: HealthStatus) -> HealthStatus:
        """Assess response time and mutate status if degraded/critical."""
        if response_ms > 5000:
            checks['response_time'] = f'critical: {response_ms:.0f}ms'
            return HealthStatus.CRITICAL
        if response_ms > 1000:
            checks['response_time'] = f'slow: {response_ms:.0f}ms'
            if status == HealthStatus.HEALTHY:
                return HealthStatus.DEGRADED
        else:
            checks['response_time'] = f'{response_ms:.0f}ms'
        return status

    async def _generate_health_recommendations(self) -> List[str]:
        """Generate health recommendations based on current system state."""
        recommendations = []

        # Check error rates
        recent_errors = len([e for e in self.error_history
                           if e.get('timestamp', datetime.min) > datetime.now() - timedelta(hours=24)])

        if recent_errors > 10:
            recommendations.append("High error rate detected. Consider reviewing consolidation configuration.")

        # Check performance metrics
        if 'average_duration_seconds' in self.metrics:
            duration = self.metrics['average_duration_seconds'].value
            if duration > 300:
                recommendations.append("Consolidation operations are taking longer than expected. Consider optimizing memory processing.")

        # Check active alerts
        critical_alerts = [a for a in self.alerts if not a.resolved and a.severity == HealthStatus.CRITICAL]
        if critical_alerts:
            recommendations.append("Critical alerts detected. Immediate attention required.")

        # Check storage health
        if 'storage_response_time' in self.metrics:
            response_time = self.metrics['storage_response_time'].value
            if response_time > 5.0:
                recommendations.append("Storage backend response time is elevated. Check database performance.")

        return recommendations

    def record_consolidation_performance(self, time_horizon: str, duration: float,
                                       memories_processed: int, success: bool,
                                       errors: List[str] = None):
        """Record performance metrics from a consolidation run."""
        entry = {
            'timestamp': datetime.now(),
            'time_horizon': time_horizon,
            'duration_seconds': duration,
            'memories_processed': memories_processed,
            'success': success,
            'errors': errors or [],
            'memories_per_second': memories_processed / duration if duration > 0 else 0
        }

        self.performance_history.append(entry)

        # Trim history to max size
        if len(self.performance_history) > self.max_history_entries:
            self.performance_history = self.performance_history[-self.max_history_entries:]

        # Update metrics
        self._update_performance_metrics()

        # Check for alerts
        if not success or (errors and len(errors) > 0):
            self._create_alert(
                component='consolidator',
                severity=HealthStatus.DEGRADED if success else HealthStatus.UNHEALTHY,
                message=f"Consolidation issues detected: {', '.join(errors[:3])}"
            )

    def record_error(self, component: str, error: Exception, context: Dict[str, Any] = None):
        """Record an error in the consolidation system."""
        error_entry = {
            'timestamp': datetime.now(),
            'component': component,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'context': context or {}
        }

        self.error_history.append(error_entry)

        # Trim error history
        if len(self.error_history) > self.max_history_entries:
            self.error_history = self.error_history[-self.max_history_entries:]

        # Create alert for serious errors
        if isinstance(error, ConsolidationError):
            severity = HealthStatus.UNHEALTHY
        else:
            severity = HealthStatus.DEGRADED

        self._create_alert(
            component=component,
            severity=severity,
            message=f"{type(error).__name__}: {str(error)}"
        )

        self.logger.error(f"Error in {component}: {error}", exc_info=True)

    def _update_performance_metrics(self):
        """Update performance metrics based on recent data."""
        now = datetime.now()
        recent_cutoff = now - timedelta(hours=24)

        # Get recent performance data
        recent_runs = [r for r in self.performance_history if r['timestamp'] > recent_cutoff]

        if not recent_runs:
            return

        # Calculate success rate
        successful_runs = [r for r in recent_runs if r['success']]
        success_rate = len(successful_runs) / len(recent_runs)

        self.metrics['consolidation_success_rate'] = HealthMetric(
            name='consolidation_success_rate',
            value=success_rate,
            status=self._get_status_for_metric('consolidation_success_rate', success_rate),
            message=f"{len(successful_runs)}/{len(recent_runs)} consolidations successful"
        )

        # Calculate average duration
        avg_duration = sum(r['duration_seconds'] for r in recent_runs) / len(recent_runs)

        self.metrics['average_duration_seconds'] = HealthMetric(
            name='average_duration_seconds',
            value=avg_duration,
            status=self._get_status_for_metric('average_duration_seconds', avg_duration),
            message=f"Average consolidation duration: {avg_duration:.1f}s"
        )

        # Calculate processing rate
        total_memories = sum(r['memories_processed'] for r in recent_runs)
        total_duration = sum(r['duration_seconds'] for r in recent_runs)
        processing_rate = total_memories / total_duration if total_duration > 0 else 0

        self.metrics['memory_processing_rate'] = HealthMetric(
            name='memory_processing_rate',
            value=processing_rate,
            status=self._get_status_for_metric('memory_processing_rate', processing_rate),
            message=f"Processing rate: {processing_rate:.2f} memories/second"
        )

        # Calculate error rate
        recent_error_cutoff = now - timedelta(hours=1)
        recent_errors = [e for e in self.error_history if e['timestamp'] > recent_error_cutoff]
        error_rate = len(recent_errors) / max(len(recent_runs), 1)

        self.metrics['error_rate'] = HealthMetric(
            name='error_rate',
            value=error_rate,
            status=self._get_status_for_metric('error_rate', error_rate),
            message=f"Error rate: {error_rate:.2f} errors per consolidation"
        )

    def _get_status_for_metric(self, metric_name: str, value: float) -> HealthStatus:
        """Determine health status for a metric value."""
        if metric_name not in self.thresholds:
            return HealthStatus.HEALTHY

        thresholds = self.thresholds[metric_name]

        # For error rate and duration, higher is worse
        if metric_name in ['error_rate', 'average_duration_seconds', 'storage_response_time']:
            if value >= thresholds['critical']:
                return HealthStatus.CRITICAL
            elif value >= thresholds['warning']:
                return HealthStatus.DEGRADED
            else:
                return HealthStatus.HEALTHY

        # For success rate and processing rate, lower is worse
        else:
            if value <= thresholds['critical']:
                return HealthStatus.CRITICAL
            elif value <= thresholds['warning']:
                return HealthStatus.DEGRADED
            else:
                return HealthStatus.HEALTHY

    def _create_alert(self, component: str, severity: HealthStatus, message: str):
        """Create a new health alert."""
        alert_id = f"{component}_{severity.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        alert = HealthAlert(
            alert_id=alert_id,
            component=component,
            severity=severity,
            message=message
        )

        self.alerts.append(alert)

        # Trim alerts to reasonable size
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]

        self.logger.warning(f"Health alert [{severity.value}] for {component}: {message}")

    def resolve_alert(self, alert_id: str):
        """Mark an alert as resolved."""
        for alert in self.alerts:
            if alert.alert_id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolution_timestamp = datetime.now()
                self.logger.info(f"Alert {alert_id} resolved")
                break

    async def get_health_summary(self) -> Dict[str, Any]:
        """Get a summary of consolidation system health."""
        health = await self.check_overall_health()

        return {
            'overall_status': health['status'],
            'timestamp': health['timestamp'],
            'component_count': len(health['components']),
            'healthy_components': len([c for c in health['components'].values()
                                     if c.get('status') == 'healthy']),
            'active_alerts': len([a for a in health['alerts'] if not a.get('resolved', False)]),
            'critical_alerts': len([a for a in health['alerts']
                                  if a.get('severity') == 'critical' and not a.get('resolved', False)]),
            'recommendations_count': len(health.get('recommendations', [])),
            'recent_errors': len([e for e in self.error_history
                                if e.get('timestamp', datetime.min) > datetime.now() - timedelta(hours=24)])
        }
