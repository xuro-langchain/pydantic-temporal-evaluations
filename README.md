# Pydantic AI + Temporal + LangSmith

Pydantic-AI agent with Temporal workflow orchestration and LangSmith tracing via OpenTelemetry.

## Project Structure

```
├── agent.py           # Pydantic-AI agent with weather tool
├── workflow.py        # Temporal workflow and activities
├── run.py             # Main runner with OTEL setup
├── evaluate.py        # LangSmith evaluation with LLM-as-Judge
├── tests/             # Agent and workflow tests
└── pyproject.toml     # Project dependencies
```

## Setup

1. Install dependencies:
```bash
uv sync
```

2. Configure environment:
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. Start Temporal server:
```bash
brew install temporal
temporal server start-dev  # Keep running
```

## Usage

Start worker in one terminal:
```bash
uv run python run.py worker
```

Execute workflow in another:
```bash
uv run python run.py execute "What is the capital of France?"
```

All agent executions are traced to LangSmith via OpenTelemetry.

## How It Works

- **agent.py**: Pydantic-AI agent with GPT-4o-mini, weather tool, and structured output
- **workflow.py**: Temporal workflow orchestrating agent execution
- **run.py**: OTEL configuration and workflow runner
- **evaluate.py**: LangSmith evaluation with correctness evaluator

### OpenTelemetry + LangSmith

Traces are sent to LangSmith via OTLP exporter:
- `init_otel()` configures the OTLP exporter
- `Agent.instrument_all()` enables automatic tracing
- View traces at https://smith.langchain.com/

## Testing

Run agent tests:
```bash
uv run python tests/test_agent.py
```

Run workflow tests (requires Temporal server):
```bash
uv run python tests/test_workflow.py
```

## Evaluation

Evaluate direct agent:
```bash
uv run python evaluate.py
```

Evaluate Temporal workflow (requires Temporal server):
```bash
uv run python evaluate.py --workflow
```

Dataset: 8 test cases covering math, general knowledge, and weather queries.

Evaluator: LLM-as-Judge using GPT-4o-mini with structured output to assess correctness.

View results in LangSmith dashboard (link provided in output).
