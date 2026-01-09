"""Temporal workflow for running agent tasks."""

from datetime import timedelta

from temporalio import activity, workflow


@activity.defn
async def run_agent_activity(question: str) -> dict:
    """Activity that runs the pydantic-ai agent.

    Args:
        question: The question to process

    Returns:
        Dictionary representation of TaskResult
    """
    # Import inside activity to avoid workflow sandbox restrictions
    from agent import process_task

    result = await process_task(question)
    return result.model_dump()


@workflow.defn
class AgentWorkflow:
    """Workflow that orchestrates agent execution."""

    @workflow.run
    async def run(self, question: str) -> dict:
        """Execute the workflow.

        Args:
            question: The question to process

        Returns:
            Dictionary with the agent's response
        """
        # Execute the agent activity with a timeout
        result = await workflow.execute_activity(
            run_agent_activity,
            question,
            start_to_close_timeout=timedelta(seconds=30),
        )

        return result
