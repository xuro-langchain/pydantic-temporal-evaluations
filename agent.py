"""Simple pydantic-ai agent for text processing."""

import httpx
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext


class TaskInput(BaseModel):
    """Input for the agent task."""
    question: str


class TaskResult(BaseModel):
    """Result from the agent."""
    answer: str
    reasoning: str


# Create a simple agent that answers questions
agent = Agent(
    "openai:gpt-4o-mini",
    output_type=TaskResult,
    system_prompt=(
        "You are a helpful assistant that answers questions concisely. "
        "Provide both an answer and brief reasoning for your response. "
        "You have access to a tool to check current weather conditions for any location."
    ),
)


@agent.tool
async def get_weather(ctx: RunContext, latitude: float, longitude: float) -> dict:
    """Get current weather conditions for a location.

    Uses the free Open-Meteo API to fetch weather data.

    Args:
        ctx: The run context
        latitude: Latitude of the location
        longitude: Longitude of the location

    Returns:
        Dictionary with current weather information including temperature, conditions, etc.
    """
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": latitude,
                    "longitude": longitude,
                    "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m",
                    "temperature_unit": "fahrenheit",
                },
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()

            current = data.get("current", {})
            return {
                "temperature_fahrenheit": current.get("temperature_2m"),
                "humidity_percent": current.get("relative_humidity_2m"),
                "precipitation_mm": current.get("precipitation"),
                "wind_speed_mph": current.get("wind_speed_10m"),
                "weather_code": current.get("weather_code"),
                "location": {"latitude": latitude, "longitude": longitude},
            }
        except Exception as e:
            return {"error": f"Failed to fetch weather data: {str(e)}"}


async def process_task(question: str) -> TaskResult:
    """Process a question using the agent.

    Args:
        question: The question to answer

    Returns:
        TaskResult with answer and reasoning
    """
    result = await agent.run(question)
    return result.output
