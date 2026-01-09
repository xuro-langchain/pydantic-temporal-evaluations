"""Runner script for the Temporal workflow with LangSmith tracing."""

import asyncio
import os
import uuid

from dotenv import load_dotenv
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import set_tracer_provider
from pydantic_ai import Agent
from temporalio.client import Client
from temporalio.worker import Worker

from workflow import AgentWorkflow, run_agent_activity

# Load environment variables
load_dotenv()


def init_otel():
    """Initialize OpenTelemetry to send traces to LangSmith."""
    # Configure OTLP exporter to send to LangSmith
    exporter = OTLPSpanExporter()
    span_processor = BatchSpanProcessor(exporter)
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(span_processor)
    set_tracer_provider(tracer_provider)

    # Instrument all pydantic-ai agents
    Agent.instrument_all()


async def run_worker():
    """Start the Temporal worker."""
    # Initialize OpenTelemetry tracing
    init_otel()

    client = await Client.connect("localhost:7233")

    worker = Worker(
        client,
        task_queue="agent-task-queue",
        workflows=[AgentWorkflow],
        activities=[run_agent_activity],
    )

    print("Starting worker on task queue: agent-task-queue")
    await worker.run()


async def run_workflow(question: str):
    """Execute a workflow.

    Args:
        question: The question to process
    """
    # Initialize OpenTelemetry tracing
    init_otel()

    # Connect to Temporal
    client = await Client.connect("localhost:7233")

    # Generate a unique workflow ID
    workflow_id = f"agent-workflow-{uuid.uuid4()}"

    handle = await client.start_workflow(
        AgentWorkflow.run,
        question,
        id=workflow_id,
        task_queue="agent-task-queue",
    )
    result = await handle.result()

    print(f"\nWorkflow ID: {workflow_id}")
    print(f"Question: {question}")
    print(f"Result: {result}")
    return result


async def main():
    """Main entry point."""
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python run.py worker              - Start the worker")
        print("  python run.py execute <question>  - Execute a workflow")
        sys.exit(1)

    command = sys.argv[1]

    if command == "worker":
        await run_worker()
    elif command == "execute":
        if len(sys.argv) < 3:
            print("Error: Please provide a question")
            print("Example: python run.py execute 'What is the capital of France?'")
            sys.exit(1)
        question = " ".join(sys.argv[2:])
        await run_workflow(question)
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
