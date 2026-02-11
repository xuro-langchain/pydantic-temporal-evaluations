# LangSmith Demo

Demos for pydantic-ai's TemporalAgent integration with LangSmith tracing.

## Setup

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Copy .env.example to .env and fill in your API keys
cp .env.example .env

# Start the Temporal dev server
temporal server start-dev
```

## Demos

| File | What it shows |
|------|---------------|
| `weather_agent.py` | Minimal TemporalAgent with a live weather API tool, traced to LangSmith. |
| `long_running_agent.py` | Durable execution — agent with slow tools that survives worker crashes via Temporal replay. |
| `context_heavy_agent.py` | Progressively larger tool results hitting Temporal's 2 MB activity payload limit. |
| `multi_agent_pipeline.py` | Three sequential TemporalAgents (collect, analyze, report) with a simulated failure + LangSmith trace analysis. |

## Running

Each demo is a single script — no separate worker/client terminals needed:

```bash
uv run python -m langsmith_demo.weather_agent
uv run python -m langsmith_demo.long_running_agent
uv run python -m langsmith_demo.context_heavy_agent
uv run python -m langsmith_demo.multi_agent_pipeline
```

## Known Issues

- TemporalAgent blocks `run_stream()`, `iter()`, and `run_stream_events()` — streaming to users requires external infra (Redis, WebSocket, etc.).
- Activities that bundle agent output with supporting data (search results, datasets) exceed Temporal's 2 MB serialization limit.
