"""Shared utilities for sydney_demos."""

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.trace import set_tracer_provider
from pydantic_ai import Agent


def init_langsmith_tracing(console_export: bool = False):
    """Initialize OpenTelemetry to export traces to LangSmith.

    Requires these env vars (see .env.example):
        OTEL_EXPORTER_OTLP_ENDPOINT=https://api.smith.langchain.com/otel
        OTEL_EXPORTER_OTLP_HEADERS=x-api-key=<key>,Langsmith-Project=<project>
    """
    provider = TracerProvider()

    # Always send to LangSmith
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))

    # Optionally dump raw OTEL spans to console for debugging
    if console_export:
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        print("[tracing] Console span exporter enabled — raw OTEL spans will print to stdout")

    set_tracer_provider(provider)

    Agent.instrument_all()

    print("[tracing] LangSmith OTLP tracing initialized")
