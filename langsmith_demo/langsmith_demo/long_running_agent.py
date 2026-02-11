"""Demo: Long-running TemporalAgent with durable execution and LangSmith tracing.

Shows a research agent that makes multiple slow tool calls (web lookups,
data analysis) orchestrated durably via TemporalAgent. If the worker
crashes mid-run, Temporal replays and resumes from where it left off.

Run with:
    python -m langsmith_demo.long_running_agent
"""

import asyncio
import random
import uuid

from dotenv import load_dotenv
from pydantic import BaseModel
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

TASK_QUEUE = "long-running-agent-queue"


# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------


class ResearchReport(BaseModel):
    title: str
    summary: str
    key_findings: list[str]
    sources_consulted: int


# ---------------------------------------------------------------------------
# Agent with slow tools that simulate long-running work
# ---------------------------------------------------------------------------

agent = Agent(
    "openai:gpt-4o-mini",
    output_type=ResearchReport,
    name="long-running-researcher",
    system_prompt=(
        "You are a thorough research assistant. When given a topic, use the "
        "available tools to gather information before writing your report. "
        "Call search_web to find sources, then call analyze_sources to "
        "synthesize the results. Always use both tools at least once."
    ),
)


@agent.tool
async def search_web(ctx: RunContext, query: str) -> dict:
    """Search the web for information on a topic.

    This simulates a slow web search that takes a few seconds.
    """
    print(f"    [tool] search_web: '{query}' — working...")
    await asyncio.sleep(random.uniform(2, 4))

    results = [
        {
            "title": f"Source {i+1}: {query[:50]}",
            "snippet": (
                f"Key finding #{i+1} related to {query[:30]}. "
                f"Research indicates significant developments in this area "
                f"with implications for policy and practice."
            ),
            "url": f"https://example.com/research/{i+1}",
        }
        for i in range(random.randint(3, 6))
    ]
    print(f"    [tool] search_web: found {len(results)} results")
    return {"query": query, "results": results}


@agent.tool
async def analyze_sources(ctx: RunContext, topic: str, num_sources: int) -> dict:
    """Perform deeper analysis on the gathered sources.

    This simulates a computationally expensive analysis step.
    """
    print(f"    [tool] analyze_sources: analyzing {num_sources} sources on '{topic}' — working...")
    await asyncio.sleep(random.uniform(3, 6))

    themes = [
        f"Theme: {topic[:25]} — aspect {i+1}" for i in range(min(num_sources, 4))
    ]
    print(f"    [tool] analyze_sources: identified {len(themes)} themes")
    return {
        "topic": topic,
        "themes_identified": themes,
        "confidence": round(random.uniform(0.75, 0.95), 2),
        "sources_analyzed": num_sources,
    }


# ---------------------------------------------------------------------------
# TemporalAgent + Workflow
# ---------------------------------------------------------------------------

temporal_agent = TemporalAgent(agent, name="long-running-researcher")


@workflow.defn
class LongRunningResearchWorkflow(PydanticAIWorkflow):
    """Durable workflow that runs the research agent.

    If the worker crashes between tool calls, Temporal replays the
    workflow history and resumes from the last completed activity.
    """

    __pydantic_ai_agents__ = [temporal_agent]

    @workflow.run
    async def run(self, question: str) -> dict:
        result = await temporal_agent.run(question)
        return result.output.model_dump()


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


async def main():
    init_langsmith_tracing()

    client = await Client.connect(
        "localhost:7233",
        interceptors=[TracingInterceptor()],
        plugins=[PydanticAIPlugin()],
    )

    question = "Research the current state of quantum computing"

    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[LongRunningResearchWorkflow],
        workflow_runner=UnsandboxedWorkflowRunner(),
    ):
        result = await client.execute_workflow(
            LongRunningResearchWorkflow.run,
            args=[question],
            id=f"long-running-{uuid.uuid4()}",
            task_queue=TASK_QUEUE,
        )

    print()
    print("=" * 60)
    print(f"Title:   {result['title']}")
    print(f"Summary: {result['summary']}")
    print(f"Sources: {result['sources_consulted']}")
    print("Key findings:")
    for finding in result["key_findings"]:
        print(f"  - {finding}")
    print("=" * 60)
    print("\nCheck traces at: https://smith.langchain.com")
    await asyncio.sleep(2)


if __name__ == "__main__":
    asyncio.run(main())
