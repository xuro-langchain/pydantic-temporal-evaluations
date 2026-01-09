"""Tests for the pydantic-ai agent."""

import os
from dotenv import load_dotenv

# Load environment variables before importing agent
load_dotenv()

import pytest
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import set_tracer_provider
from pydantic_ai import Agent

from agent import agent, get_weather, process_task


def init_otel():
    """Initialize OpenTelemetry to send traces to LangSmith."""
    print("Initializing OpenTelemetry tracing...")

    # Configure OTLP exporter to send to LangSmith
    exporter = OTLPSpanExporter()
    span_processor = BatchSpanProcessor(exporter)
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(span_processor)
    set_tracer_provider(tracer_provider)

    print(f"OTLP Exporter configured")
    print(f"  Endpoint: {os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT', 'not set')}")
    print(f"  Service Name: {os.getenv('OTEL_SERVICE_NAME', 'not set')}")

    # Instrument all pydantic-ai agents
    Agent.instrument_all()
    print("Pydantic-AI agents instrumented for OTEL tracing\n")

    # Return the provider so we can flush it later
    return tracer_provider


@pytest.mark.asyncio
async def test_process_task_basic():
    """Test that the agent can process a simple question."""
    question = "What is 2 + 2?"
    result = await process_task(question)

    assert result.answer is not None
    assert result.reasoning is not None
    assert isinstance(result.answer, str)
    assert isinstance(result.reasoning, str)
    assert len(result.answer) > 0
    print(f"\nQuestion: {question}")
    print(f"Answer: {result.answer}")
    print(f"Reasoning: {result.reasoning}")


@pytest.mark.asyncio
async def test_weather_tool_direct():
    """Test the weather tool directly with San Francisco coordinates."""
    # San Francisco coordinates
    latitude = 37.7749
    longitude = -122.4194

    # Call the tool directly (passing None for ctx as it's not used)
    weather_data = await get_weather(None, latitude, longitude)

    assert weather_data is not None
    assert "error" not in weather_data
    assert "temperature_fahrenheit" in weather_data
    assert "humidity_percent" in weather_data
    assert "location" in weather_data
    assert weather_data["location"]["latitude"] == latitude
    assert weather_data["location"]["longitude"] == longitude

    print(f"\nWeather data for San Francisco:")
    print(f"Temperature: {weather_data['temperature_fahrenheit']}°F")
    print(f"Humidity: {weather_data['humidity_percent']}%")
    print(f"Wind Speed: {weather_data['wind_speed_mph']} mph")


@pytest.mark.asyncio
async def test_agent_with_weather_question():
    """Test that the agent can answer weather-related questions using the tool."""
    question = "What's the weather like at latitude 40.7128, longitude -74.0060 (New York City)?"
    result = await process_task(question)

    assert result.answer is not None
    assert result.reasoning is not None
    assert len(result.answer) > 0

    # The answer should contain some weather-related information
    answer_lower = result.answer.lower()
    assert any(word in answer_lower for word in ["temperature", "weather", "degrees", "°"])

    print(f"\nQuestion: {question}")
    print(f"Answer: {result.answer}")
    print(f"Reasoning: {result.reasoning}")


if __name__ == "__main__":
    import asyncio
    import time

    # Initialize OpenTelemetry tracing
    tracer_provider = init_otel()

    print("Running agent tests with OTEL tracing enabled...\n")
    print("=" * 80 + "\n")

    asyncio.run(test_process_task_basic())
    print("\n" + "=" * 80 + "\n")
    asyncio.run(test_weather_tool_direct())
    print("\n" + "=" * 80 + "\n")
    asyncio.run(test_agent_with_weather_question())

    print("\n" + "=" * 80)
    print("All tests completed!")