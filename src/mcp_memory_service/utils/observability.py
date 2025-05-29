"""
Observability utilities for EchoVault Memory Service.
Provides OpenTelemetry and Prometheus integration.
"""

import os
import time
import logging
import functools
from typing import Callable, Any, Dict, Optional

# Import OpenTelemetry modules conditionally to avoid hard dependency
try:
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

# Import Prometheus modules conditionally
try:
    import prometheus_client
    from prometheus_client import Counter, Histogram, Gauge
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)

# Global variables for metrics
_metrics_initialized = False
_counters = {}
_histograms = {}
_gauges = {}

def setup_observability(service_name: str = "echovault-memory-service"):
    """
    Set up OpenTelemetry and Prometheus for observability.
    
    Args:
        service_name: Name of the service for metrics and traces
    """
    global _metrics_initialized
    
    if _metrics_initialized:
        return
    
    # Set up OpenTelemetry if available
    if OTEL_AVAILABLE:
        try:
            # Get OTLP endpoint from environment
            otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
            
            if otlp_endpoint:
                # Create resource
                resource = Resource.create({"service.name": service_name})
                
                # Set up tracing
                tracer_provider = TracerProvider(resource=resource)
                span_processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint))
                tracer_provider.add_span_processor(span_processor)
                trace.set_tracer_provider(tracer_provider)
                
                # Set up metrics
                reader = PeriodicExportingMetricReader(
                    OTLPMetricExporter(endpoint=otlp_endpoint),
                    export_interval_millis=15000
                )
                meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
                metrics.set_meter_provider(meter_provider)
                
                logger.info(f"OpenTelemetry initialized with endpoint {otlp_endpoint}")
            else:
                logger.info("OpenTelemetry disabled (OTEL_EXPORTER_OTLP_ENDPOINT not set)")
        except Exception as e:
            logger.warning(f"Failed to initialize OpenTelemetry: {e}")
    
    # Set up Prometheus if available and enabled
    if PROMETHEUS_AVAILABLE and os.environ.get("PROMETHEUS_METRICS", "").lower() in ("true", "1", "yes"):
        try:
            # Start Prometheus HTTP server on port 8000
            prometheus_client.start_http_server(8000)
            logger.info("Prometheus metrics server started on port 8000")
            
            # Initialize standard metrics
            _init_standard_metrics()
        except Exception as e:
            logger.warning(f"Failed to initialize Prometheus: {e}")
    
    _metrics_initialized = True

def _init_standard_metrics():
    """Initialize standard metrics for the memory service."""
    global _counters, _histograms, _gauges
    
    # Counters
    _counters["memory_store_total"] = Counter(
        "memory_store_total", 
        "Total number of memory store operations"
    )
    _counters["memory_store_success"] = Counter(
        "memory_store_success", 
        "Number of successful memory store operations"
    )
    _counters["memory_store_error"] = Counter(
        "memory_store_error", 
        "Number of failed memory store operations"
    )
    _counters["memory_retrieve_total"] = Counter(
        "memory_retrieve_total", 
        "Total number of memory retrieve operations"
    )
    _counters["memory_delete_total"] = Counter(
        "memory_delete_total", 
        "Total number of memory delete operations"
    )
    _counters["blob_store_total"] = Counter(
        "blob_store_total", 
        "Total number of blob store operations"
    )
    _counters["blob_retrieve_total"] = Counter(
        "blob_retrieve_total", 
        "Total number of blob retrieve operations"
    )
    
    # Histograms
    _histograms["memory_store_duration"] = Histogram(
        "memory_store_duration_seconds", 
        "Duration of memory store operations in seconds",
        buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0)
    )
    _histograms["memory_retrieve_duration"] = Histogram(
        "memory_retrieve_duration_seconds", 
        "Duration of memory retrieve operations in seconds",
        buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0)
    )
    _histograms["embedding_generation_duration"] = Histogram(
        "embedding_generation_duration_seconds", 
        "Duration of embedding generation in seconds",
        buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0)
    )
    _histograms["memory_content_size"] = Histogram(
        "memory_content_size_bytes", 
        "Size of memory content in bytes",
        buckets=(1024, 4096, 16384, 65536, 262144, 1048576)
    )
    
    # Gauges
    _gauges["memory_count"] = Gauge(
        "memory_count", 
        "Total number of memories stored"
    )
    _gauges["blob_count"] = Gauge(
        "blob_count", 
        "Total number of blobs stored"
    )
    _gauges["connection_pool_size"] = Gauge(
        "connection_pool_size", 
        "Current size of the database connection pool"
    )
    _gauges["connection_pool_available"] = Gauge(
        "connection_pool_available", 
        "Number of available connections in the pool"
    )

def trace_function(name: Optional[str] = None):
    """
    Decorator to trace function execution with OpenTelemetry.
    
    Args:
        name: Optional name for the span. If not provided, the function name is used.
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not OTEL_AVAILABLE:
                return await func(*args, **kwargs)
            
            span_name = name or func.__name__
            tracer = trace.get_tracer(__name__)
            
            with tracer.start_as_current_span(span_name) as span:
                # Add function arguments as span attributes
                # Be careful not to include sensitive information
                span.set_attribute("function.name", func.__name__)
                
                try:
                    start_time = time.time()
                    result = await func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    # Record duration in span
                    span.set_attribute("duration_seconds", duration)
                    
                    # Record success
                    span.set_attribute("success", True)
                    
                    # Record metrics if available
                    if PROMETHEUS_AVAILABLE and _metrics_initialized:
                        if func.__name__ == "store" and "memory_store_duration" in _histograms:
                            _histograms["memory_store_duration"].observe(duration)
                            _counters["memory_store_total"].inc()
                            _counters["memory_store_success"].inc()
                        elif func.__name__ == "retrieve" and "memory_retrieve_duration" in _histograms:
                            _histograms["memory_retrieve_duration"].observe(duration)
                            _counters["memory_retrieve_total"].inc()
                    
                    return result
                except Exception as e:
                    # Record error in span
                    span.set_attribute("success", False)
                    span.set_attribute("error", True)
                    span.set_attribute("error.type", e.__class__.__name__)
                    span.set_attribute("error.message", str(e))
                    
                    # Record metrics if available
                    if PROMETHEUS_AVAILABLE and _metrics_initialized:
                        if func.__name__ == "store" and "memory_store_error" in _counters:
                            _counters["memory_store_error"].inc()
                    
                    raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not OTEL_AVAILABLE:
                return func(*args, **kwargs)
            
            span_name = name or func.__name__
            tracer = trace.get_tracer(__name__)
            
            with tracer.start_as_current_span(span_name) as span:
                # Add function arguments as span attributes
                span.set_attribute("function.name", func.__name__)
                
                try:
                    start_time = time.time()
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    # Record duration in span
                    span.set_attribute("duration_seconds", duration)
                    
                    # Record success
                    span.set_attribute("success", True)
                    
                    # Record metrics if available
                    if PROMETHEUS_AVAILABLE and _metrics_initialized:
                        if func.__name__ == "store" and "memory_store_duration" in _histograms:
                            _histograms["memory_store_duration"].observe(duration)
                            _counters["memory_store_total"].inc()
                            _counters["memory_store_success"].inc()
                        elif func.__name__ == "retrieve" and "memory_retrieve_duration" in _histograms:
                            _histograms["memory_retrieve_duration"].observe(duration)
                            _counters["memory_retrieve_total"].inc()
                    
                    return result
                except Exception as e:
                    # Record error in span
                    span.set_attribute("success", False)
                    span.set_attribute("error", True)
                    span.set_attribute("error.type", e.__class__.__name__)
                    span.set_attribute("error.message", str(e))
                    
                    # Record metrics if available
                    if PROMETHEUS_AVAILABLE and _metrics_initialized:
                        if func.__name__ == "store" and "memory_store_error" in _counters:
                            _counters["memory_store_error"].inc()
                    
                    raise
        
        # Return appropriate wrapper based on whether the function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

def update_gauge(name: str, value: float):
    """
    Update a Prometheus gauge metric.
    
    Args:
        name: Name of the gauge
        value: New value
    """
    if PROMETHEUS_AVAILABLE and _metrics_initialized and name in _gauges:
        _gauges[name].set(value)

def increment_counter(name: str, value: float = 1.0):
    """
    Increment a Prometheus counter metric.
    
    Args:
        name: Name of the counter
        value: Value to increment by (default: 1.0)
    """
    if PROMETHEUS_AVAILABLE and _metrics_initialized and name in _counters:
        _counters[name].inc(value)

def observe_histogram(name: str, value: float):
    """
    Observe a value for a Prometheus histogram metric.
    
    Args:
        name: Name of the histogram
        value: Value to observe
    """
    if PROMETHEUS_AVAILABLE and _metrics_initialized and name in _histograms:
        _histograms[name].observe(value)

def create_span(name: str, attributes: Optional[Dict[str, Any]] = None):
    """
    Create a new OpenTelemetry span.
    
    Args:
        name: Name of the span
        attributes: Optional attributes to add to the span
        
    Returns:
        A span context manager
    """
    if not OTEL_AVAILABLE:
        # Return a dummy context manager if OpenTelemetry is not available
        class DummyContextManager:
            def __enter__(self):
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
        
        return DummyContextManager()
    
    tracer = trace.get_tracer(__name__)
    span = tracer.start_span(name)
    
    if attributes:
        for key, value in attributes.items():
            span.set_attribute(key, value)
    
    return span