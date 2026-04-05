"""
Configuration OpenTelemetry — tracing distribué.

Fournit un TracerProvider configuré avec export OTLP gRPC
et l'instrumentation automatique FastAPI + httpx.

Usage :
    from core.observability.otel import setup_tracing
    setup_tracing()   # appeler une fois au démarrage de l'app
"""

import logging
import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

logger = logging.getLogger(__name__)

_TRACER: trace.Tracer | None = None


def setup_tracing(app=None, sample_rate: float = 1.0) -> trace.Tracer:
    """
    Initialise le TracerProvider global et instrumente FastAPI + httpx.

    Args:
        app:         Instance FastAPI à instrumenter (optionnel).
        sample_rate: Taux d'échantillonnage [0.0, 1.0]. Défaut : 1.0 (tout).

    Returns:
        Tracer prêt à l'emploi.
    """
    global _TRACER

    endpoint    = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4317")
    service     = os.environ.get("OTEL_SERVICE_NAME", "fraud-serving")
    environment = os.environ.get("ENVIRONMENT", "production")

    resource = Resource.create({
        "service.name":        service,
        "service.version":     os.environ.get("APP_VERSION", "0.1.0"),
        "deployment.environment": environment,
    })

    provider = TracerProvider(
        resource=resource,
        sampler=TraceIdRatioBased(sample_rate),
    )
    provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(endpoint=endpoint),
            max_export_batch_size=512,
            export_timeout_millis=5_000,
        )
    )
    trace.set_tracer_provider(provider)

    # Instrumentation automatique
    HTTPXClientInstrumentor().instrument()
    if app is not None:
        FastAPIInstrumentor.instrument_app(
            app,
            excluded_urls="/healthz,/readyz,/metrics",
        )

    _TRACER = trace.get_tracer(service)
    logger.info("OTel tracing initialisé — service=%s endpoint=%s rate=%.2f", service, endpoint, sample_rate)
    return _TRACER


def get_tracer() -> trace.Tracer:
    """Retourne le tracer global. Appeler setup_tracing() d'abord."""
    if _TRACER is None:
        return trace.get_tracer(__name__)
    return _TRACER
