"""Demo: TemporalAgent hits payload limits with context-heavy tool results.

TemporalAgent serializes the full message history as activity input/output
on every model call. When a tool returns large results (RAG retrieval,
search, database queries), the accumulated messages quickly exceed
Temporal's 2 MB activity payload limit.

Run with:
    python -m langsmith_demo.context_heavy_agent
"""

import asyncio
import json
import sys
import uuid

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.durable_exec.temporal import PydanticAIPlugin, PydanticAIWorkflow, TemporalAgent
from temporalio import workflow
from temporalio.client import Client
from temporalio.worker import Worker
from temporalio.worker._workflow_instance import UnsandboxedWorkflowRunner

from langsmith_demo.utils import init_langsmith_tracing

load_dotenv()


# ---------------------------------------------------------------------------
# Agent with a context-heavy retrieval tool
# ---------------------------------------------------------------------------


class AnalysisResult(BaseModel):
    summary: str
    num_sources_used: int


agent = Agent(
    "openai:gpt-4o-mini",
    output_type=AnalysisResult,
    name="research-agent",
    system_prompt=(
        "You are a research analyst. Use the search_documents tool to find "
        "relevant information, then synthesize a concise summary. Always "
        "cite how many source documents you used."
    ),
)


def _generate_documents(query: str, num_docs: int, doc_size_bytes: int) -> list[dict]:
    """Generate fake retrieved documents of a target size."""
    padding = "x" * max(0, doc_size_bytes - 200)
    return [
        {
            "id": f"doc-{i:04d}",
            "title": f"Research paper #{i}: {query[:40]}",
            "content": (
                f"This document discusses {query} in detail. "
                f"Finding {i}: significant results observed in the dataset. "
                f"{padding}"
            ),
            "relevance_score": round(1.0 - (i * 0.02), 3),
        }
        for i in range(num_docs)
    ]


# Control how large tool results are per scenario
_retrieval_config: dict = {"num_docs": 5, "doc_size": 200}


@agent.tool
async def search_documents(ctx: RunContext, query: str) -> list[dict]:
    """Search for relevant research documents."""
    docs = _generate_documents(
        query,
        num_docs=_retrieval_config["num_docs"],
        doc_size_bytes=_retrieval_config["doc_size"],
    )
    size = len(json.dumps(docs).encode())
    print(f"    [tool] search_documents returned {len(docs)} docs ({_fmt_size(size)})")
    return docs


# ---------------------------------------------------------------------------
# TemporalAgent + Workflow
# ---------------------------------------------------------------------------

temporal_agent = TemporalAgent(agent, name="research-agent")


@workflow.defn
class ResearchWorkflow(PydanticAIWorkflow):
    __pydantic_ai_agents__ = [temporal_agent]

    @workflow.run
    async def run(self, question: str) -> dict:
        result = await temporal_agent.run(question)
        return result.output.model_dump()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PAYLOAD_LIMIT = 2 * 1024 * 1024  # 2 MB


def _fmt_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    elif n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    else:
        return f"{n / (1024 * 1024):.2f} MB"


def _size_bar(current: int, width: int = 40) -> str:
    ratio = min(current / PAYLOAD_LIMIT, 1.5)
    filled = int(width * min(ratio, 1.0))
    pct = ratio * 100
    marker = " OVER LIMIT!" if current > PAYLOAD_LIMIT else ""
    return f"[{'#' * filled}{'-' * (width - filled)}] {pct:.0f}% of 2 MB{marker}"


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

SCENARIOS = [
    {
        "label": "Small context (5 docs, ~200 B each)",
        "question": "What are the key findings on climate change mitigation?",
        "num_docs": 5,
        "doc_size": 200,
    },
    {
        "label": "Medium context (20 docs, ~10 KB each)",
        "question": "Analyze the economic impacts of renewable energy policy across OECD nations.",
        "num_docs": 20,
        "doc_size": 10_000,
    },
    {
        "label": "Large context (50 docs, ~20 KB each)",
        "question": "Comprehensive literature review of machine learning applications in drug discovery.",
        "num_docs": 50,
        "doc_size": 20_000,
    },
    {
        "label": "Huge context (100 docs, ~25 KB each)",
        "question": "Full systematic review of global monetary policy effectiveness 2000-2024.",
        "num_docs": 100,
        "doc_size": 25_000,
    },
]


async def run_scenario(client: Client, scenario: dict, index: int) -> bool:
    """Run a single scenario, return True if it succeeded."""
    _retrieval_config["num_docs"] = scenario["num_docs"]
    _retrieval_config["doc_size"] = scenario["doc_size"]

    estimated = len(
        json.dumps(
            _generate_documents("x", scenario["num_docs"], scenario["doc_size"])
        ).encode()
    )

    print(f"Scenario {index}: {scenario['label']}")
    print(f"  Question: \"{scenario['question'][:70]}...\"")
    print(f"  Estimated tool result size: {_fmt_size(estimated)}")
    print(f"  {_size_bar(estimated)}")

    async with Worker(
        client,
        task_queue=f"context-heavy-queue-{index}",
        workflows=[ResearchWorkflow],
        workflow_runner=UnsandboxedWorkflowRunner(),
    ):
        workflow_id = f"context-heavy-{index}-{uuid.uuid4()}"
        try:
            result = await client.execute_workflow(
                ResearchWorkflow.run,
                scenario["question"],
                id=workflow_id,
                task_queue=f"context-heavy-queue-{index}",
            )
            print(f"  Result: \"{result['summary'][:80]}...\"")
            print(f"  Sources used: {result['num_sources_used']}")
            print()
            return True
        except Exception as e:
            err = str(e)
            print(f"  FAILED: {err[:200]}")
            print()
            print("  --> The tool result was too large for Temporal's activity payload.")
            print("      The model could handle this context fine — the limit is purely")
            print("      Temporal's 2 MB serialization boundary.")
            print()
            return False


async def main():
    init_langsmith_tracing()

    try:
        client = await Client.connect(
            "localhost:7233",
            plugins=[PydanticAIPlugin()],
        )
    except Exception:
        print("Temporal server not running. Start it with: temporal server start-dev")
        sys.exit(1)

    print()
    print("=" * 70)
    print("TemporalAgent Context-Heavy Demo")
    print("=" * 70)
    print()
    print("Each scenario uses the same agent with a search_documents tool.")
    print("The tool returns progressively larger results. TemporalAgent")
    print("serializes the full message history (including tool results)")
    print("as activity payloads — hitting Temporal's 2 MB limit.")
    print()

    for i, scenario in enumerate(SCENARIOS, 1):
        ok = await run_scenario(client, scenario, i)
        if not ok:
            remaining = SCENARIOS[i:]
            if remaining:
                print("  Remaining scenarios would also fail:")
                for j, s in enumerate(remaining, i + 1):
                    est = len(
                        json.dumps(
                            _generate_documents("x", s["num_docs"], s["doc_size"])
                        ).encode()
                    )
                    print(f"    Scenario {j}: {s['label']} — {_fmt_size(est)}")
            break

    print()
    print("=" * 70)
    print("Takeaway: TemporalAgent carries the full message history in every")
    print("activity payload. One large tool result (RAG retrieval, search,")
    print("database query) can push subsequent activities over 2 MB.")
    print()
    print("Workarounds:")
    print("  - Store large data externally (S3, Redis) and return references")
    print("  - Summarize/truncate tool results before returning")
    print("  - Use manual activities instead of TemporalAgent for data-heavy steps")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
