#!/usr/bin/env python

import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import ExternalTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient


import os, sys

from dotenv import load_dotenv
from autogen_core.models import ModelFamily


def configure():
    load_dotenv()
    return os.getenv("OPENROUTER_API_KEY")


# Define a model client. You can use other model client that implements
# the `ChatCompletionClient` interface.
model_client = OpenAIChatCompletionClient(
    model="GLM-4.5-Flash",
    api_key=configure(),
    base_url="https://api.z.ai/api/paas/v4/",
    temperature=1,
    # max_tokens=20000,
    model_info={
        "vision": False,
        "function_calling": True,
        "json_output": True,
        "family": "unknown",
        "structured_output": True,
    },
)


# Create the primary agent.
primary_agent = AssistantAgent(
    "primary",
    model_client=model_client,
    system_message="You are a helpful AI assistant.",
)

# Create the critic agent.
critic_agent = AssistantAgent(
    "critic",
    model_client=model_client,
    system_message=""
    "Provide constructive feedback."
    "Respond with 'APPROVE' to when your feedbacks are addressed.",
)

# Define a termination condition that stops the task if the critic approves.
text_termination = TextMentionTermination("APPROVE")

# Create a team with the primary and critic agents.
team = RoundRobinGroupChat(
    [primary_agent, critic_agent], termination_condition=text_termination
)


async def main() -> None:
    # Running a Team
    # Use `asyncio.run(...)` when running in a script.
    # result = await team.run(task="Write a short poem about the fall season.")
    # print(result)

    # Observing a Team
    # When running inside a script, use a async main function and call it from `asyncio.run(...)`.
    await team.reset()  # Reset the team for a new task.
    async for message in team.run_stream(task="Write a short poem about the fall season."):  # type: ignore
        if isinstance(message, TaskResult):
            print("Stop Reason:", message.stop_reason)
        else:
            print(message.content)
            print("-" * 80)

    pass


if __name__ == "__main__":
    import asyncio

    # # NOTE: if running this inside a Python script you'll need to use asyncio.run(main()).
    # await main()
    asyncio.run(main())

    print("helloworld")
