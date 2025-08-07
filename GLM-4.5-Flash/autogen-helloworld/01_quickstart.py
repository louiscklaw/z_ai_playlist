#!/usr/bin/env python

import os, sys
import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.base import TaskResult

from dotenv import load_dotenv


def configure():
    load_dotenv()
    return os.getenv("OPENROUTER_API_KEY")


# Define a model client. You can use other model client that implements
# the `ChatCompletionClient` interface.
model_client = OpenAIChatCompletionClient(
    model="GLM-4.5-Flash",
    api_key=configure(),
    # base_url="...",
    base_url="https://api.z.ai/api/paas/v4/",
    temperature=0.8,
    # max_tokens=20000,
    model_info={
        "vision": False,
        "function_calling": True,
        "json_output": True,
        "family": "unknown",
        "structured_output": True,
    },
)


# Define a simple function tool that the agent can use.
# For this example, we use a fake weather tool for demonstration purposes.
async def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    return f"The weather in {city} is 73 degrees and Sunny."


# Define an AssistantAgent with the model, tool, system message, and reflection enabled.
# The system message instructs the agent via natural language.
agent = AssistantAgent(
    name="weather_agent",
    model_client=model_client,
    tools=[get_weather],
    system_message="You are a helpful assistant. you will answer in chinese",
    # reflect_on_tool_use=True,
    # Enable streaming tokens from the model client.
    model_client_stream=True,
)


async def runStreamAndReturn(input_agent, task="香港現在的天氣怎麽樣? 用中文回答"):
    async for message in input_agent.run_stream(task=task):
        if isinstance(message, TaskResult):
            return message
        else:
            print("-" * 120)
            print(f"Received message: ")
            print(message)


# Run the agent and stream the messages to the console.
async def main():
    result = await runStreamAndReturn(agent, "香港現在的天氣怎麽樣? 用中文回答")
    print("-" * 120)
    print("Final message")
    print(result.messages[-1].content)


if __name__ == "__main__":
    # NOTE: if running this inside a Python script you'll need to use asyncio.run(main()).
    asyncio.run(main())
