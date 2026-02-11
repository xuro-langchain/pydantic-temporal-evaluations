"""Demo: Multi-agent pipeline with sequential TemporalAgents.

Three TemporalAgents are called in sequence within a single Temporal workflow:

  1. Data Collector   — tools: lookup_customer, fetch_transactions
  2. Risk Analyzer    — tools: compute_risk_score, check_sanctions_list
  3. Report Generator — tools: generate_pdf_report, send_to_compliance (FAILS)

The last agent's send_to_compliance tool simulates a downstream service outage,
causing the pipeline to fail at the final stage. All three stages (including the
failure) are traced to LangSmith via OpenTelemetry.

After the workflow completes (or fails), an **insights agent** fetches the
LangSmith trace runs and analyzes what happened.

Run with:
    python -m langsmith_demo.multi_agent_pipeline
"""

import asyncio
import os
import random
import uuid
from datetime import datetime, timedelta

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

TASK_QUEUE = "multi-agent-pipeline-queue"


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------


class CustomerProfile(BaseModel):
    customer_id: str
    name: str
    email: str
    account_tier: str
    transactions: list[dict]
    total_spend: float


class RiskAssessment(BaseModel):
    risk_level: str
    risk_score: float
    flags: list[str]
    recommendation: str


class ComplianceReport(BaseModel):
    report_id: str
    summary: str
    risk_level: str
    actions_taken: list[str]
    notification_sent: bool


# ---------------------------------------------------------------------------
# Agent 1: Data Collector
# ---------------------------------------------------------------------------

collector_agent = Agent(
    "openai:gpt-4o-mini",
    output_type=CustomerProfile,
    name="data-collector",
    retries=3,
    system_prompt=(
        "You are a data collection agent. Use the available tools to gather "
        "customer information. Always call lookup_customer first, then "
        "fetch_transactions. Combine ALL results into a complete profile — "
        "you MUST include the transactions list in your output."
    ),
)


@collector_agent.tool
async def lookup_customer(ctx: RunContext, customer_id: str) -> dict:
    """Look up customer details by ID."""
    print(f"    [collector] lookup_customer: {customer_id}")
    await asyncio.sleep(random.uniform(0.5, 1.5))
    return {
        "customer_id": customer_id,
        "name": "Acme Corp",
        "email": "billing@acme.corp",
        "account_tier": "enterprise",
        "signup_date": "2023-01-15",
        "region": "US-WEST",
    }


@collector_agent.tool
async def fetch_transactions(
    ctx: RunContext, customer_id: str, days: int
) -> list[dict]:
    """Fetch recent transactions for a customer."""
    print(
        f"    [collector] fetch_transactions: {customer_id}, last {days} days"
    )
    await asyncio.sleep(random.uniform(1, 2))
    return [
        {
            "id": f"txn-{i:04d}",
            "amount": round(random.uniform(500, 45000), 2),
            "type": random.choice(["wire", "ach", "card"]),
            "destination": random.choice(["domestic", "international"]),
            "timestamp": (
                datetime.now() - timedelta(days=random.randint(0, days))
            ).isoformat(),
        }
        for i in range(random.randint(5, 12))
    ]


# ---------------------------------------------------------------------------
# Agent 2: Risk Analyzer
# ---------------------------------------------------------------------------

analyzer_agent = Agent(
    "openai:gpt-4o-mini",
    output_type=RiskAssessment,
    name="risk-analyzer",
    system_prompt=(
        "You are a financial risk analysis agent. Given customer data, use "
        "your tools to compute a risk score and check sanctions lists. "
        "Always call compute_risk_score and check_sanctions_list."
    ),
)


@analyzer_agent.tool
async def compute_risk_score(
    ctx: RunContext,
    total_spend: float,
    num_transactions: int,
    has_international: bool,
) -> dict:
    """Compute a risk score based on transaction patterns."""
    print(
        f"    [analyzer] compute_risk_score: "
        f"spend=${total_spend:,.2f}, txns={num_transactions}, intl={has_international}"
    )
    await asyncio.sleep(random.uniform(1, 2))

    base_score = 0.2
    if total_spend > 100_000:
        base_score += 0.3
    if has_international:
        base_score += 0.15
    if num_transactions > 10:
        base_score += 0.1
    score = min(base_score + random.uniform(-0.05, 0.1), 1.0)

    return {
        "score": round(score, 3),
        "factors": [
            f"transaction_volume: {'high' if num_transactions > 10 else 'normal'}",
            f"spend_level: {'elevated' if total_spend > 100_000 else 'normal'}",
            f"international_activity: {'yes' if has_international else 'no'}",
        ],
    }


@analyzer_agent.tool
async def check_sanctions_list(ctx: RunContext, entity_name: str) -> dict:
    """Check if an entity appears on sanctions lists."""
    print(f"    [analyzer] check_sanctions_list: {entity_name}")
    await asyncio.sleep(random.uniform(0.5, 1))
    return {
        "entity": entity_name,
        "match_found": False,
        "lists_checked": ["OFAC-SDN", "EU-Sanctions", "UN-Consolidated"],
        "checked_at": datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Agent 3: Report Generator (this one will fail)
# ---------------------------------------------------------------------------

reporter_agent = Agent(
    "openai:gpt-4o-mini",
    output_type=ComplianceReport,
    name="report-generator",
    system_prompt=(
        "You are a compliance report generator. Given a risk assessment, "
        "generate a formal report. First call generate_pdf_report to create "
        "the report, then call send_to_compliance to deliver it. "
        "You MUST call both tools."
    ),
)


@reporter_agent.tool
async def generate_pdf_report(
    ctx: RunContext,
    customer_name: str,
    risk_level: str,
    risk_score: float,
    flags: list[str],
) -> dict:
    """Generate a PDF compliance report."""
    print(
        f"    [reporter] generate_pdf_report: {customer_name}, risk={risk_level}"
    )
    await asyncio.sleep(random.uniform(1, 2))
    report_id = f"RPT-{uuid.uuid4().hex[:8].upper()}"
    return {
        "report_id": report_id,
        "status": "generated",
        "pages": 3,
        "storage_url": f"s3://compliance-reports/{report_id}.pdf",
    }


@reporter_agent.tool
async def send_to_compliance(
    ctx: RunContext, report_id: str, priority: str
) -> dict:
    """Send the report to the compliance team via the notification service."""
    print(f"    [reporter] send_to_compliance: {report_id}, priority={priority}")
    await asyncio.sleep(0.5)

    # --- Simulated downstream service outage ---
    raise ConnectionError(
        f"Failed to connect to compliance notification service at "
        f"https://compliance-api.internal:8443/notify — connection refused. "
        f"The service appears to be down "
        f"(status page: https://status.internal/incidents/INC-4892). "
        f"Report {report_id} was generated but could NOT be delivered."
    )


# ---------------------------------------------------------------------------
# TemporalAgents + Workflow
# ---------------------------------------------------------------------------

temporal_collector = TemporalAgent(collector_agent, name="data-collector")
temporal_analyzer = TemporalAgent(analyzer_agent, name="risk-analyzer")
temporal_reporter = TemporalAgent(reporter_agent, name="report-generator")


@workflow.defn
class MultiAgentPipelineWorkflow(PydanticAIWorkflow):
    """Three-stage compliance pipeline: collect -> analyze -> report."""

    __pydantic_ai_agents__ = [
        temporal_collector,
        temporal_analyzer,
        temporal_reporter,
    ]

    @workflow.run
    async def run(self, customer_id: str) -> dict:
        # Stage 1: Collect customer data
        collector_result = await temporal_collector.run(
            f"Look up customer {customer_id} and fetch their transactions "
            f"from the last 90 days. Return their full profile."
        )
        profile = collector_result.output

        # Stage 2: Analyze risk (feed in collector output)
        analyzer_result = await temporal_analyzer.run(
            f"Analyze risk for customer '{profile.name}' "
            f"(ID: {profile.customer_id}). "
            f"They have {len(profile.transactions)} recent transactions "
            f"totaling ${profile.total_spend:,.2f}. "
            f"Check if any transactions are international. "
            f"Also check '{profile.name}' against sanctions lists."
        )
        assessment = analyzer_result.output

        # Stage 3: Generate and send report (will fail on send_to_compliance)
        reporter_result = await temporal_reporter.run(
            f"Generate a compliance report for customer '{profile.name}'. "
            f"Risk level: {assessment.risk_level}, "
            f"score: {assessment.risk_score}. "
            f"Flags: {', '.join(assessment.flags) if assessment.flags else 'none'}. "
            f"Recommendation: {assessment.recommendation}. "
            f"Then send it to the compliance team with appropriate priority."
        )
        report = reporter_result.output

        return {
            "profile": profile.model_dump(),
            "assessment": assessment.model_dump(),
            "report": report.model_dump(),
        }


# ---------------------------------------------------------------------------
# Insights agent — fetches LangSmith traces and analyzes the pipeline run
# ---------------------------------------------------------------------------


class TraceInsight(BaseModel):
    summary: str
    stages_completed: list[str]
    failure_point: str
    root_cause: str
    recommendation: str
    total_tool_calls: int


insights_agent = Agent(
    "openai:gpt-4o-mini",
    output_type=TraceInsight,
    name="trace-insights",
    system_prompt=(
        "You are a trace analysis agent. Given LangSmith trace data from a "
        "multi-agent pipeline, analyze what happened. Identify which stages "
        "completed successfully, where the failure occurred, its root cause, "
        "and recommend a fix."
    ),
)


async def run_insights(project_name: str):
    """Fetch recent traces from LangSmith and have the insights agent analyze them."""
    from langsmith import Client as LangSmithClient

    ls_client = LangSmithClient()

    print(f"\n{'=' * 60}")
    print("INSIGHTS AGENT: Fetching LangSmith traces...")
    print(f"{'=' * 60}\n")

    runs = list(ls_client.list_runs(
        project_name=project_name,
        limit=50,
    ))

    if not runs:
        print("  No trace runs found in LangSmith yet.")
        print("  (Traces are batched — they may take a few seconds to appear.)")
        return None

    trace_lines = []
    for run in runs:
        error_msg = getattr(run, "error", None)
        duration = ""
        if run.start_time and run.end_time:
            dt = (run.end_time - run.start_time).total_seconds()
            duration = f"{dt:.1f}s"

        line = (
            f"- name={run.name!r} | type={run.run_type} | "
            f"status={run.status} | duration={duration}"
        )
        if error_msg:
            line += f" | ERROR: {error_msg[:200]}"
        trace_lines.append(line)

    trace_text = "\n".join(trace_lines)
    print(f"  Found {len(runs)} trace runs. Sending to insights agent...\n")

    prompt = (
        f"Analyze this trace from a multi-agent compliance pipeline.\n\n"
        f"The pipeline has 3 sequential stages:\n"
        f"  1. data-collector — tools: lookup_customer, fetch_transactions\n"
        f"  2. risk-analyzer  — tools: compute_risk_score, check_sanctions_list\n"
        f"  3. report-generator — tools: generate_pdf_report, send_to_compliance\n\n"
        f"LangSmith trace runs ({len(runs)} total):\n{trace_text}"
    )

    result = await insights_agent.run(prompt)
    insight = result.output

    print("--- Trace Insights ---")
    print(f"  Summary:          {insight.summary}")
    print(f"  Stages completed: {', '.join(insight.stages_completed)}")
    print(f"  Failure point:    {insight.failure_point}")
    print(f"  Root cause:       {insight.root_cause}")
    print(f"  Recommendation:   {insight.recommendation}")
    print(f"  Tool calls seen:  {insight.total_tool_calls}")
    print()

    return insight


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

    customer_id = "CUST-7291"
    workflow_id = f"multi-agent-{uuid.uuid4()}"

    print(f"Starting multi-agent pipeline: {workflow_id}")
    print(f"Customer ID: {customer_id}\n")

    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[MultiAgentPipelineWorkflow],
        workflow_runner=UnsandboxedWorkflowRunner(),
    ):
        try:
            result = await client.execute_workflow(
                MultiAgentPipelineWorkflow.run,
                args=[customer_id],
                id=workflow_id,
                task_queue=TASK_QUEUE,
            )
            print(f"\n{'=' * 60}")
            print("Pipeline completed successfully!")
            print(f"Report: {result['report']['report_id']}")
        except Exception as e:
            print(f"\n{'=' * 60}")
            print("Pipeline FAILED (expected — the last agent hits a service outage)")
            print(f"Error: {e}")

    print(f"\nWorkflow ID: {workflow_id}")
    print("Check traces at: https://smith.langchain.com")

    # Wait for the OTLP batch processor to flush spans to LangSmith
    print("\nWaiting for traces to flush to LangSmith...")
    await asyncio.sleep(5)

    # Run the insights agent on the trace
    project = os.environ.get("LANGSMITH_PROJECT", "default")
    await run_insights(project)

    await asyncio.sleep(2)


if __name__ == "__main__":
    asyncio.run(main())
