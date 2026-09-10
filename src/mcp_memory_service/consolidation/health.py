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
import tempfile
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


class ConsolidationHealthMonitor:
    """Monitors health of the consolidation system."""
    
    def __init__(
        self, config=None, consolidator=None, scheduler=None, schedule_config=None
    ):
        self.config = config
        self.consolidator = consolidator
        self.scheduler = scheduler
        self.schedule_config = schedule_config or {}
        self.logger = logging.getLogger(__name__)
        
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
        periods = getattr(self.config, 'retention_periods', None)
        calculator = (
            getattr(self.consolidator, 'decay_calculator', None)
            if self.consolidator else None
        )
        return {
            'status': (
                HealthStatus.HEALTHY.value
                if periods
                and all(value > 0 for value in periods.values())
                and calculator
                else HealthStatus.UNHEALTHY.value
            ),
            'checks': {
                'retention_periods': periods if periods else 'invalid',
                'decay_algorithm': 'present' if calculator else 'unverifiable'
            },
            'metrics': {
                'recent_calculations': len([h for h in self.performance_history 
                                          if h.get('component') == 'decay_calculator'
                                          and h.get('timestamp', datetime.min) > datetime.now() - timedelta(hours=1)])
            }
        }
    
    async def _check_association_engine_health(self) -> Dict[str, Any]:
        """Check association engine health."""
        recent_associations = len([h for h in self.performance_history 
                                 if h.get('component') == 'association_engine'
                                 and h.get('timestamp', datetime.min) > datetime.now() - timedelta(hours=1)])
        
        minimum = getattr(self.config, 'min_similarity', None)
        maximum = getattr(self.config, 'max_similarity', None)
        engine = (
            getattr(self.consolidator, 'association_engine', None)
            if self.consolidator else None
        )
        valid = (
            isinstance(minimum, (int, float))
            and isinstance(maximum, (int, float))
            and 0 <= minimum < maximum <= 1
        )
        return {
            'status': (
                HealthStatus.HEALTHY.value
                if valid and engine else HealthStatus.UNHEALTHY.value
            ),
            'checks': {
                'similarity_bounds': (minimum, maximum) if valid else 'invalid',
                'association_engine': 'present' if engine else 'unverifiable'
            },
            'metrics': {
                'recent_associations_discovered': recent_associations,
                'min_similarity': minimum,
                'max_similarity': maximum,
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
        engine = (
            getattr(self.consolidator, 'clustering_engine', None)
            if self.consolidator else None
        )
        try:
            algorithm = resolve_clustering_algorithm(configured)
        except ConsolidationError:
            algorithm = f"unsatisfiable: '{configured}' needs scikit-learn"

        healthy = not algorithm.startswith("unsatisfiable:") and engine is not None
        return {
            'status': (
                HealthStatus.HEALTHY.value
                if healthy else HealthStatus.UNHEALTHY.value
            ),
            'checks': {
                'clustering_algorithm': algorithm,
                'clustering_engine': 'present' if engine else 'unverifiable',
                'sklearn': 'available' if SKLEARN_AVAILABLE else 'unavailable',
                'minimum_cluster_size': getattr(
                    self.config, 'min_cluster_size', 'unverifiable'
                ),
                'embedding_processing': 'unverifiable'
            },
            'metrics': {
                'recent_clusters_created': len([h for h in self.performance_history 
                                              if h.get('component') == 'clustering_engine'
                                              and h.get('timestamp', datetime.min) > datetime.now() - timedelta(hours=1)])
            }
        }
    
    async def _check_compression_engine_health(self) -> Dict[str, Any]:
        """Check compression engine health.

        Compression is local: statistical summarisation over clusters, with no
        external service behind it. An attached engine and a usable summary
        bound are therefore the whole of what can be observed, and a
        deployment where both hold has nothing wrong with it.
        """
        enabled = bool(getattr(self.config, 'compression_enabled', False))
        engine = (
            getattr(self.consolidator, 'compression_engine', None)
            if self.consolidator else None
        )
        summary_length = getattr(self.config, 'max_summary_length', None)
        valid = (
            isinstance(summary_length, int)
            and not isinstance(summary_length, bool)
            and summary_length > 0
        )
        return {
            'status': (
                HealthStatus.HEALTHY.value if not enabled else (
                    HealthStatus.HEALTHY.value if engine and valid
                    else HealthStatus.UNHEALTHY.value
                )
            ),
            'checks': {
                'compression': (
                    'disabled by configuration' if not enabled else (
                        'engine present' if engine else 'engine not attached'
                    )
                ),
                'max_summary_length': summary_length if valid else 'invalid'
            },
            'metrics': {
                'recent_compressions': len([h for h in self.performance_history 
                                          if h.get('component') == 'compression_engine'
                                          and h.get('timestamp', datetime.min) > datetime.now() - timedelta(hours=1)])
            }
        }
    
    async def _check_forgetting_engine_health(self) -> Dict[str, Any]:
        """Check forgetting engine health."""
        enabled = bool(getattr(self.config, 'forgetting_enabled', False))
        archive = getattr(self.config, 'archive_location', None)
        relevance = getattr(self.config, 'relevance_threshold', None)
        access_days = getattr(self.config, 'access_threshold_days', None)
        engine = (
            getattr(self.consolidator, 'forgetting_engine', None)
            if self.consolidator else None
        )
        # ControlledForgettingEngine expands '~' before it writes here
        # (forgetting.py), so a probe that does not expand it reports a
        # working archive directory as unwritable.
        archive_path = os.path.expanduser(archive) if archive else None
        writable = False
        if archive_path and os.path.isdir(archive_path):
            path = None
            try:
                fd, path = tempfile.mkstemp(prefix='.health-', dir=archive_path)
                os.close(fd)
                writable = True
            except OSError:
                pass
            finally:
                if path:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass
        valid = (
            isinstance(relevance, (int, float))
            and 0 <= relevance <= 1
            and isinstance(access_days, (int, float))
            and access_days > 0
        )
        return {
            'status': (
                HealthStatus.HEALTHY.value if not enabled else (
                    HealthStatus.HEALTHY.value if writable and valid and engine
                    else HealthStatus.UNHEALTHY.value
                )
            ),
            'checks': {
                'archive_storage': (
                    'disabled by configuration' if not enabled else (
                        'writable' if writable else 'not writable'
                    )
                ),
                'thresholds': (
                    {'relevance': relevance, 'access_days': access_days}
                    if valid else 'invalid'
                ),
                'controlled_forgetting': 'engine present' if engine else 'unverifiable'
            },
            'metrics': {
                'recent_archival_operations': len([h for h in self.performance_history 
                                                 if h.get('component') == 'forgetting_engine'
                                                 and h.get('timestamp', datetime.min) > datetime.now() - timedelta(hours=1)])
            }
        }
    
    async def _check_scheduler_health(self) -> Dict[str, Any]:
        """Check scheduler health."""
        live_config = getattr(
            self.scheduler, 'schedule_config', self.schedule_config
        ) if self.scheduler else self.schedule_config
        all_disabled = bool(live_config) and all(
            value == 'disabled' for value in live_config.values()
        )
        if self.scheduler is None:
            return {
                'status': (
                    HealthStatus.HEALTHY.value if all_disabled
                    else HealthStatus.UNHEALTHY.value
                ),
                'checks': {
                    'scheduler': (
                        'disabled by configuration' if all_disabled
                        else 'scheduler not attached'
                    )
                },
                'metrics': {'schedule_config': live_config},
            }
        status = await self.scheduler.get_scheduler_status()
        if not status.get('enabled'):
            return {
                'status': (
                    HealthStatus.HEALTHY.value if all_disabled
                    else HealthStatus.UNHEALTHY.value
                ),
                'checks': {
                    'scheduler': (
                        'disabled by configuration' if all_disabled
                        else status.get('reason', 'scheduler unavailable')
                    )
                },
                'metrics': status,
            }
        if not status.get('running'):
            return {
                'status': HealthStatus.UNHEALTHY.value,
                'checks': {'scheduler': 'enabled but not running'},
                'metrics': status,
            }
        if not live_config:
            return {
                'status': HealthStatus.UNHEALTHY.value,
                'checks': {
                    'scheduler': (
                        'running with no observable schedule configuration'
                    )
                },
                'metrics': status,
            }
        expected = {
            f"consolidation_{horizon}"
            for horizon, value in live_config.items()
            if value != 'disabled'
        }
        actual = {
            job.get('id')
            for job in status.get('jobs', [])
            if job.get('id', '').startswith('consolidation_')
        }
        missing, unexpected = expected - actual, actual - expected
        if missing or unexpected:
            return {
                'status': HealthStatus.UNHEALTHY.value,
                'checks': {
                    'scheduler': 'configured jobs do not match running jobs',
                    'missing_jobs': sorted(missing),
                    'unexpected_jobs': sorted(unexpected),
                },
                'metrics': {
                    'schedule_config': live_config,
                    'scheduled_jobs': len(actual),
                },
            }
        return {
            'status': HealthStatus.HEALTHY.value,
            'checks': {
                'scheduler_running': 'running',
                'configured_jobs': sorted(expected),
                'scheduled_jobs': sorted(actual),
            },
            'metrics': {
                'scheduled_jobs': len(actual),
                'last_execution': self.scheduler.last_execution_times,
                'execution_stats': self.scheduler.execution_stats,
            },
        }
    
    async def _check_storage_backend_health(self) -> Dict[str, Any]:
        """Check storage backend health."""
        storage = (
            getattr(self.consolidator, 'storage', None)
            if self.consolidator else None
        )
        probe = getattr(storage, 'health_probe', None) if storage else None
        if not callable(probe):
            return {
                'status': HealthStatus.DEGRADED.value,
                'checks': {'storage': 'unverifiable: no health probe'},
                'metrics': {},
            }
        try:
            if await probe():
                return {
                    'status': HealthStatus.HEALTHY.value,
                    'checks': {'storage': 'health probe reachable'},
                    'metrics': {},
                }
            return {
                'status': HealthStatus.UNHEALTHY.value,
                'checks': {'storage': 'health probe failed'},
                'metrics': {},
            }
        except Exception as exc:
            return {
                'status': HealthStatus.UNHEALTHY.value,
                'checks': {'storage': 'unreachable'},
                'metrics': {},
                'error': str(exc),
            }
    
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