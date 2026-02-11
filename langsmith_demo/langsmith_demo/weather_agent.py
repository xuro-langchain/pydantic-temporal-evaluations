"""Demo: Minimal TemporalAgent with a weather tool, traced to LangSmith.

Run with:
    python -m langsmith_demo.weather_agent
"""

import asyncio
import uuid

import httpx
from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext
from pydantic_ai.durable_exec.temporal import (
    PydanticAIPlugin,
    PydanticAIWorkflow,
    TemporalAgent,
)
from temporalio import workflow
from temporalio.client import Client
from temporalio.contrib.opentelemetry import TracingInterceptor
from temporalio.worker import Worker
from temporalio.worker._workflow_instance import UnsandboxedWorkflowRunner

from langsmith_demo.utils import init_langsmith_tracing

load_dotenv()

TASK_QUEUE = "weather-agent-queue"

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

agent = Agent(
    "openai:gpt-4o-mini",
    name="weather-agent",
    system_prompt="You answer weather questions. Use the get_weather tool to look up current conditions.",
)


@agent.tool
async def get_weather(ctx: RunContext, latitude: float, longitude: float) -> dict:
    """Get current weather for a location using the Open-Meteo API."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
                "temperature_unit": "fahrenheit",
            },
            timeout=10.0,
        )
        resp.raise_for_status()
        current = resp.json().get("current", {})
        return {
            "temp_f": current.get("temperature_2m"),
            "humidity": current.get("relative_humidity_2m"),
            "precip_mm": current.get("precipitation"),
            "wind_mph": current.get("wind_speed_10m"),
        }


# ---------------------------------------------------------------------------
# TemporalAgent + Workflow
# ---------------------------------------------------------------------------

temporal_agent = TemporalAgent(agent, name="weather-agent")


@workflow.defn
class WeatherWorkflow(PydanticAIWorkflow):
    __pydantic_ai_agents__ = [temporal_agent]

    @workflow.run
    async def run(self, question: str) -> str:
        result = await temporal_agent.run(question)
        return result.output


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


async def main():
    init_langsmith_tracing(console_export=True)

    client = await Client.connect(
        "localhost:7233",
        interceptors=[TracingInterceptor()],
        plugins=[PydanticAIPlugin()],
    )

    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[WeatherWorkflow],
        workflow_runner=UnsandboxedWorkflowRunner(),
    ):
        result = await client.execute_workflow(
            WeatherWorkflow.run,
            args=["What's the weather in Tokyo?"],
            id=f"weather-{uuid.uuid4()}",
            task_queue=TASK_QUEUE,
        )
        print(result)

    print("\nCheck traces at: https://smith.langchain.com")
    await asyncio.sleep(2)


if __name__ == "__main__":
    asyncio.run(main())
