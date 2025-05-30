"""
OpenTelemetry and Prometheus Instrumentation for EchoVault Memory Service
Copyright (c) 2025 EchoVault
Licensed under the MIT License.

This module provides OpenTelemetry tracing and Prometheus metrics for the EchoVault Memory Service.
"""

import os
import time
import logging
import functools
from typing import Dict, Any, Optional, Callable, Awaitable, Union
import asyncio

logger = logging.getLogger(__name__)

# Import OpenTelemetry conditionally to avoid hard dependencies
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    
    # For metrics
    from opentelemetry import metrics
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    logger.warning("OpenTelemetry packages not available. Tracing will be disabled.")

# Import Prometheus conditionally
try:
    import prometheus_client
    from prometheus_client import Counter, Histogram, Gauge, Summary, Info
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("Prometheus client not available. Metrics will be disabled.")

# Global variables
_tracer_provider = None
_meter_provider = None
_tracer = None
_meter = None
_initialized = False

# Prometheus metrics
_metrics = {}

def initialize(service_name: str = "echovault-memory-service") -> bool:
    """
    Initialize OpenTelemetry and Prometheus instrumentation.
    
    Args:
        service_name: Name of the service
        
    Returns:
        True if initialization was successful
    """
    global _tracer_provider, _meter_provider, _tracer, _meter, _initialized, _metrics
    
    if _initialized:
        return True
    
    success = False
    
    # Initialize OpenTelemetry if available
    if OTEL_AVAILABLE:
        try:
            # Get OTLP endpoint from environment
            otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
            
            if otlp_endpoint:
                # Create resource with service name
                resource = Resource.create({"service.name": service_name})
                
                # Set up tracing
                _tracer_provider = TracerProvider(resource=resource)
                otlp_span_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
                span_processor = BatchSpanProcessor(otlp_span_exporter)
                _tracer_provider.add_span_processor(span_processor)
                
                # Register tracer provider
                trace.set_tracer_provider(_tracer_provider)
                _tracer = trace.get_tracer(service_name)
                
                # Set up metrics
                reader = PeriodicExportingMetricReader(
                    OTLPMetricExporter(endpoint=otlp_endpoint),
                    export_interval_millis=15000  # Export every 15 seconds
                )
                _meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
                metrics.set_meter_provider(_meter_provider)
                _meter = metrics.get_meter(service_name)
                
                logger.info(f"OpenTelemetry initialized with endpoint {otlp_endpoint}")
                success = True
            else:
                logger.info("OpenTelemetry endpoint not configured, tracing disabled")
        except Exception as e:
            logger.error(f"Failed to initialize OpenTelemetry: {str(e)}")
    
    # Initialize Prometheus if available
    if PROMETHEUS_AVAILABLE and os.environ.get("PROMETHEUS_METRICS", "").lower() in ("true", "1", "yes"):
        try:
            # Start HTTP server on port 8000
            prometheus_client.start_http_server(8000)
            
            # Define metrics
            _metrics["memory_store_count"] = Counter(
                "echovault_memory_store_total",
                "Total number of memory store operations"
            )
            
            _metrics["memory_retrieve_count"] = Counter(
                "echovault_memory_retrieve_total",
                "Total number of memory retrieve operations"
            )
            
            _metrics["memory_delete_count"] = Counter(
                "echovault_memory_delete_total",
                "Total number of memory delete operations"
            )
            
            _metrics["memory_tag_search_count"] = Counter(
                "echovault_memory_tag_search_total",
                "Total number of tag search operations"
            )
            
            _metrics["memory_store_duration"] = Histogram(
                "echovault_memory_store_duration_seconds",
                "Duration of memory store operations in seconds",
                buckets=(0.01, 0.05, 0.1, 0.5, 1, 2.5, 5, 10)
            )
            
            _metrics["memory_retrieve_duration"] = Histogram(
                "echovault_memory_retrieve_duration_seconds",
                "Duration of memory retrieve operations in seconds",
                buckets=(0.01, 0.05, 0.1, 0.5, 1, 2.5, 5, 10)
            )
            
            _metrics["memory_delete_duration"] = Histogram(
                "echovault_memory_delete_duration_seconds",
                "Duration of memory delete operations in seconds",
                buckets=(0.01, 0.05, 0.1, 0.5, 1, 2.5, 5, 10)
            )
            
            _metrics["memory_count"] = Gauge(
                "echovault_memory_count",
                "Total number of memories stored"
            )
            
            _metrics["blob_count"] = Gauge(
                "echovault_blob_count",
                "Total number of blobs stored"
            )
            
            _metrics["connection_pool_size"] = Gauge(
                "echovault_connection_pool_size",
                "Size of the connection pool"
            )
            
            _metrics["content_size"] = Histogram(
                "echovault_content_size_bytes",
                "Size of memory content in bytes",
                buckets=(1024, 4096, 16384, 32768, 65536, 131072, 262144, 524288, 1048576)
            )
            
            _metrics["vector_search_latency"] = Histogram(
                "echovault_vector_search_latency_seconds",
                "Latency of vector search operations in seconds",
                buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2.5, 5)
            )
            
            _metrics["service_info"] = Info(
                "echovault_service_info",
                "Information about the EchoVault service"
            )
            _metrics["service_info"].info({
                "version": "0.2.0",
                "service_name": service_name,
                "otel_enabled": str(OTEL_AVAILABLE).lower(),
                "prometheus_enabled": "true"
            })
            
            logger.info("Prometheus metrics initialized on port 8000")
            success = True
        except Exception as e:
            logger.error(f"Failed to initialize Prometheus metrics: {str(e)}")
    
    _initialized = success
    return success

def trace_write(content_length: int, has_payload_url: bool, tags_count: int) -> None:
    """
    Record metrics for a memory write operation.
    
    Args:
        content_length: Length of the content in bytes
        has_payload_url: Whether the content has a payload URL (stored in R2)
        tags_count: Number of tags
    """
    if not _initialized:
        return
    
    if PROMETHEUS_AVAILABLE and _metrics:
        try:
            _metrics["memory_store_count"].inc()
            _metrics["content_size"].observe(content_length)
            
            # Update memory count
            _metrics["memory_count"]._value.inc()
            
            # Update blob count if applicable
            if has_payload_url:
                _metrics["blob_count"]._value.inc()
        except Exception as e:
            logger.error(f"Failed to record write metrics: {str(e)}")

def trace_read(latency_ms: float, results_count: int) -> None:
    """
    Record metrics for a memory read operation.
    
    Args:
        latency_ms: Latency of the operation in milliseconds
        results_count: Number of results returned
    """
    if not _initialized:
        return
    
    if PROMETHEUS_AVAILABLE and _metrics:
        try:
            _metrics["memory_retrieve_count"].inc()
            _metrics["memory_retrieve_duration"].observe(latency_ms / 1000.0)  # Convert to seconds
        except Exception as e:
            logger.error(f"Failed to record read metrics: {str(e)}")

def trace_async(name: str = None):
    """
    Decorator to trace asynchronous functions with OpenTelemetry.
    
    Args:
        name: Optional name for the span. Defaults to function name.
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            if not _initialized or not OTEL_AVAILABLE:
                return await func(*args, **kwargs)
            
            span_name = name or func.__name__
            
            with _tracer.start_as_current_span(span_name) as span:
                # Add basic span attributes
                span.set_attribute("function.name", func.__name__)
                
                # Record start time for Prometheus metrics
                start_time = time.time()
                
                try:
                    # Execute the function
                    result = await func(*args, **kwargs)
                    
                    # Calculate duration for Prometheus
                    duration = time.time() - start_time
                    
                    # Record metrics based on function name
                    if PROMETHEUS_AVAILABLE and _metrics:
                        if "store" in func.__name__ and "memory_store_duration" in _metrics:
                            _metrics["memory_store_duration"].observe(duration)
                        elif "retrieve" in func.__name__ and "memory_retrieve_duration" in _metrics:
                            _metrics["memory_retrieve_duration"].observe(duration)
                        elif "delete" in func.__name__ and "memory_delete_duration" in _metrics:
                            _metrics["memory_delete_duration"].observe(duration)
                        elif "search" in func.__name__ and "vector_search_latency" in _metrics:
                            _metrics["vector_search_latency"].observe(duration)
                    
                    return result
                except Exception as e:
                    # Record error in span
                    span.set_attribute("error", True)
                    span.set_attribute("error.type", e.__class__.__name__)
                    span.set_attribute("error.message", str(e))
                    raise
        
        return wrapper
    
    return decorator

def trace_sync(name: str = None):
    """
    Decorator to trace synchronous functions with OpenTelemetry.
    
    Args:
        name: Optional name for the span. Defaults to function name.
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not _initialized or not OTEL_AVAILABLE:
                return func(*args, **kwargs)
            
            span_name = name or func.__name__
            
            with _tracer.start_as_current_span(span_name) as span:
                # Add basic span attributes
                span.set_attribute("function.name", func.__name__)
                
                # Record start time for Prometheus metrics
                start_time = time.time()
                
                try:
                    # Execute the function
                    result = func(*args, **kwargs)
                    
                    # Calculate duration for Prometheus
                    duration = time.time() - start_time
                    
                    # Record metrics based on function name
                    if PROMETHEUS_AVAILABLE and _metrics:
                        if "store" in func.__name__ and "memory_store_duration" in _metrics:
                            _metrics["memory_store_duration"].observe(duration)
                        elif "retrieve" in func.__name__ and "memory_retrieve_duration" in _metrics:
                            _metrics["memory_retrieve_duration"].observe(duration)
                        elif "delete" in func.__name__ and "memory_delete_duration" in _metrics:
                            _metrics["memory_delete_duration"].observe(duration)
                    
                    return result
                except Exception as e:
                    # Record error in span
                    span.set_attribute("error", True)
                    span.set_attribute("error.type", e.__class__.__name__)
                    span.set_attribute("error.message", str(e))
                    raise
        
        return wrapper
    
    return decorator

def update_stats(memory_count: int, blob_count: int, connection_pool_size: int, connection_pool_available: int) -> None:
    """
    Update system statistics metrics.
    
    Args:
        memory_count: Total number of memories
        blob_count: Total number of blobs
        connection_pool_size: Size of the connection pool
        connection_pool_available: Number of available connections
    """
    if not _initialized:
        return
    
    if PROMETHEUS_AVAILABLE and _metrics:
        try:
            _metrics["memory_count"].set(memory_count)
            _metrics["blob_count"].set(blob_count)
            _metrics["connection_pool_size"].set(connection_pool_size)
            
            # Calculate available connections as a gauge
            if "connection_pool_available" in _metrics:
                _metrics["connection_pool_available"].set(connection_pool_available)
        except Exception as e:
            logger.error(f"Failed to update statistics: {str(e)}")