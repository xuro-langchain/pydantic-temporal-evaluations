"""LangSmith evaluation for pydantic-ai agent and Temporal workflow."""

import asyncio
import os
import uuid
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langsmith import Client, aevaluate
from temporalio.client import Client as TemporalClient
from temporalio.worker import Worker
from typing import TypedDict, Annotated

load_dotenv()

from agent import process_task
from workflow import AgentWorkflow, run_agent_activity


# Initialize clients
langsmith_client = Client()

# Dataset name
DATASET_NAME = "pydantic-agent-eval"


# ============================================================================
# Dataset Creation
# ============================================================================

# Define test cases
TEST_INPUTS = [
    "What is 10 + 15?",
    "What is 8 * 7?",
    "What is the capital of France?",
    "What's the weather in Tokyo? Use latitude 35.6762 and longitude 139.6503",
    "Tell me the current weather in London at latitude 51.5074, longitude -0.1278",
    "What's the weather like in San Francisco? Coordinates: 37.7749, -122.4194",
    "How many days are in a leap year?",
    "What is 100 - 37?",
]

TEST_OUTPUTS = [
    {"expected_answer": "25", "question": TEST_INPUTS[0]},
    {"expected_answer": "56", "question": TEST_INPUTS[1]},
    {"expected_answer": "Paris", "question": TEST_INPUTS[2]},
    {"expected_answer": "A description of the current weather in Tokyo including temperature and conditions", "question": TEST_INPUTS[3]},
    {"expected_answer": "A description of the current weather in London including temperature and conditions", "question": TEST_INPUTS[4]},
    {"expected_answer": "A description of the current weather in San Francisco including temperature and conditions", "question": TEST_INPUTS[5]},
    {"expected_answer": "366 days", "question": TEST_INPUTS[6]},
    {"expected_answer": "63", "question": TEST_INPUTS[7]},
]


def ensure_dataset_exists():
    """Ensure the evaluation dataset exists in LangSmith."""
    if not langsmith_client.has_dataset(dataset_name=DATASET_NAME):
        print(f"Creating dataset: {DATASET_NAME}")
        dataset = langsmith_client.create_dataset(
            dataset_name=DATASET_NAME,
            description="Evaluation dataset for pydantic-ai agent with weather tool",
        )

        # Create examples in batch
        langsmith_client.create_examples(
            inputs=[{"question": q} for q in TEST_INPUTS],
            outputs=TEST_OUTPUTS,
            dataset_id=dataset.id,
        )

        print(f"✓ Created dataset with {len(TEST_INPUTS)} examples")
    else:
        print(f"✓ Dataset '{DATASET_NAME}' already exists")


# ============================================================================
# Run Functions
# ============================================================================

async def run_agent_direct(inputs: dict) -> dict:
    """Run the pydantic-ai agent directly."""
    question = inputs["question"]
    result = await process_task(question)

    return {
        "answer": result.answer,
        "reasoning": result.reasoning,
    }


async def run_workflow(inputs: dict) -> dict:
    """Run the Temporal workflow."""
    question = inputs["question"]

    # Connect to Temporal
    client = await TemporalClient.connect("localhost:7233")

    # Start worker in background
    worker = Worker(
        client,
        task_queue="eval-task-queue",
        workflows=[AgentWorkflow],
        activities=[run_agent_activity],
    )

    async with worker:
        # Execute workflow
        workflow_id = f"eval-workflow-{uuid.uuid4()}"
        handle = await client.start_workflow(
            AgentWorkflow.run,
            question,
            id=workflow_id,
            task_queue="eval-task-queue",
        )

        result = await handle.result()

    return {
        "answer": result["answer"],
        "reasoning": result["reasoning"],
    }


# ============================================================================
# Evaluators
# ============================================================================

# LLM-as-judge output schema for correctness
class Correctness(TypedDict):
    """Evaluate the correctness of an agent response."""
    reasoning: Annotated[str, ..., "Explain your step-by-step reasoning for the correctness assessment."]
    is_correct: Annotated[bool, ..., "True if the agent response is factually correct, otherwise False."]


# Judge LLM for correctness
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
correctness_grader_llm = model.with_structured_output(Correctness, method="json_schema", strict=True)


async def correctness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict = None) -> dict:
    """
    Evaluate if the agent's answer is correct using LLM-as-a-Judge.

    Args:
        inputs: The input question
        outputs: The agent's output (answer and reasoning)
        reference_outputs: Expected answer and metadata

    Returns:
        Evaluation result with score and reasoning
    """
    instructions = """
You are an expert evaluator grading outputs generated by an AI assistant. You are to judge whether the agent
generated an accurate and correct response for the given question. You are provided with the expected answer
as ground truth for your grading.

When grading, correct answers will have the following properties:
- The answer is factually accurate
- The answer directly addresses the question asked
- For numeric answers, the value matches the expected answer
- For descriptive answers (like weather), the answer contains relevant and accurate information
- The answer doesn't need to be word-for-word identical to the expected answer, but must be factually equivalent
"""

    user_context = f"""Please grade the following example according to the above instructions:

<example>
<question>
{inputs.get('question', '')}
</question>

<agent_answer>
{outputs.get('answer', '')}
</agent_answer>

<expected_answer>
{reference_outputs.get('expected_answer', '')}
</expected_answer>

<agent_reasoning>
{outputs.get('reasoning', '')}
</agent_reasoning>
</example>
"""

    grade = await correctness_grader_llm.ainvoke([
        {"role": "system", "content": instructions},
        {"role": "user", "content": user_context}
    ])

    return {
        "key": "correctness",
        "score": grade["is_correct"],
        "comment": grade["reasoning"]
    }


# ============================================================================
# Evaluation Runner
# ============================================================================

async def run_evaluation(
    dataset_name: str = "pydantic-agent-eval",
    use_workflow: bool = False,
    experiment_prefix: str = "pydantic-agent",
):
    """
    Run evaluation on the agent or workflow.

    Args:
        dataset_name: Name of the LangSmith dataset to use
        use_workflow: If True, evaluate the Temporal workflow; if False, evaluate agent directly
        experiment_prefix: Prefix for the experiment name in LangSmith
    """
    # Ensure dataset exists
    ensure_dataset_exists()
    print()

    # Select run function based on mode
    run_function = run_workflow if use_workflow else run_agent_direct
    mode = "workflow" if use_workflow else "agent"

    print(f"=" * 80)
    print(f"Running evaluation on: {mode}")
    print(f"Dataset: {dataset_name}")
    print(f"Experiment prefix: {experiment_prefix}-{mode}")
    print(f"=" * 80 + "\n")

    # Check if Temporal is needed and running
    if use_workflow:
        try:
            await TemporalClient.connect("localhost:7233")
            print("✓ Temporal server is running\n")
        except Exception as e:
            print(f"✗ Temporal server not running: {e}")
            print("Please start Temporal: temporal server start-dev")
            return None

    # Run evaluation
    results = await langsmith_client.aevaluate(
        run_function,
        data=dataset_name,
        evaluators=[correctness_evaluator],
        experiment_prefix=f"{experiment_prefix}-{mode}",
        num_repetitions=1,
        max_concurrency=4,
    )

    print(f"\n" + "=" * 80)
    print("Evaluation complete!")
    print(f"=" * 80)

    return results


if __name__ == "__main__":
    import sys

    print("\n" + "=" * 80)
    print("PYDANTIC-AI AGENT EVALUATION WITH LANGSMITH")
    print("=" * 80 + "\n")

    # Parse command line arguments
    use_workflow = "--workflow" in sys.argv

    # Show usage
    if "--help" in sys.argv:
        print(f"Usage: python evaluate.py [OPTIONS]")
        print(f"\nOptions:")
        print(f"  --workflow           Evaluate Temporal workflow (default: direct agent)")
        print(f"  --help               Show this help message")
        print(f"\nExamples:")
        print(f"  python evaluate.py                 # Evaluate direct agent")
        print(f"  python evaluate.py --workflow       # Evaluate Temporal workflow")
        print("=" * 80 + "\n")
        sys.exit(0)

    # Run evaluation (dataset will be created automatically if needed)
    asyncio.run(run_evaluation(
        dataset_name=DATASET_NAME,
        use_workflow=use_workflow,
        experiment_prefix="pydantic-agent",
    ))
