"""Test Temporal workflow with OTEL tracing."""

import asyncio
import os
import time
import uuid
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

import pytest
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import set_tracer_provider
from pydantic_ai import Agent
from temporalio.client import Client
from temporalio.worker import Worker

from workflow import AgentWorkflow, run_agent_activity


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

    # Instrument all pydantic-ai agents
    Agent.instrument_all()
    print("Pydantic-AI agents instrumented\n")

    return tracer_provider


async def check_temporal_server():
    """Check if Temporal server is running."""
    try:
        client = await Client.connect("localhost:7233")
        print("✓ Temporal server is running at localhost:7233\n")
        return client
    except Exception as e:
        print(f"✗ Could not connect to Temporal server: {e}")
        print("\nPlease start Temporal server first:")
        print("  temporal server start-dev")
        return None


@pytest.mark.asyncio
async def test_workflow_simple_question():
    """Test workflow with a simple question."""
    # Initialize OTEL
    init_otel()

    # Check Temporal server
    client = await check_temporal_server()
    if not client:
        pytest.skip("Temporal server not running")

    # Start worker
    worker = Worker(
        client,
        task_queue="test-task-queue",
        workflows=[AgentWorkflow],
        activities=[run_agent_activity],
    )

    async with worker:
        # Execute workflow
        workflow_id = f"test-simple-{uuid.uuid4()}"
        question = "What is 5 + 7?"

        handle = await client.start_workflow(
            AgentWorkflow.run,
            question,
            id=workflow_id,
            task_queue="test-task-queue",
        )

        result = await handle.result()

        # Assertions
        assert result is not None
        assert "answer" in result
        assert "reasoning" in result
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0

        # Check if answer contains expected information
        answer_lower = result["answer"].lower()
        assert any(num in answer_lower for num in ["12", "twelve"])

        print(f"\n✓ Simple question test passed")
        print(f"  Question: {question}")
        print(f"  Answer: {result['answer']}")


@pytest.mark.asyncio
async def test_workflow_weather_tool():
    """Test workflow with weather tool usage."""
    # Initialize OTEL
    tracer_provider = init_otel()

    # Check Temporal server
    client = await check_temporal_server()
    if not client:
        pytest.skip("Temporal server not running")

    # Start worker
    worker = Worker(
        client,
        task_queue="test-task-queue",
        workflows=[AgentWorkflow],
        activities=[run_agent_activity],
    )

    async with worker:
        # Execute workflow
        workflow_id = f"test-weather-{uuid.uuid4()}"
        question = "What's the weather in Paris? Use latitude 48.8566 and longitude 2.3522"

        handle = await client.start_workflow(
            AgentWorkflow.run,
            question,
            id=workflow_id,
            task_queue="test-task-queue",
        )

        result = await handle.result()

        # Assertions
        assert result is not None
        assert "answer" in result
        assert "reasoning" in result
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0

        # Check if answer contains weather-related information
        answer_lower = result["answer"].lower()
        assert any(word in answer_lower for word in ["temperature", "weather", "°", "degrees"])

        print(f"\n✓ Weather tool test passed")
        print(f"  Question: {question}")
        print(f"  Answer: {result['answer'][:100]}...")

        # Flush traces
        tracer_provider.force_flush()
        time.sleep(1)


if __name__ == "__main__":
    async def main():
        print("=" * 80)
        print("TEMPORAL WORKFLOW TESTS")
        print("=" * 80 + "\n")

        try:
            await test_workflow_simple_question()
            print("\n" + "=" * 80 + "\n")
            await test_workflow_weather_tool()
        except Exception as e:
            print(f"\n✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return 1

        print("\n" + "=" * 80)
        print("All workflow tests passed!")
        print("Check LangSmith: https://smith.langchain.com/")
        print("=" * 80)
        return 0

    exit(asyncio.run(main()))
