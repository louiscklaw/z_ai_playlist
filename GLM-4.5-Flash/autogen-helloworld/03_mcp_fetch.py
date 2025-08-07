#!/usr/bin/env python

import os, sys
import asyncio

from pathlib import Path
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import StdioServerParams, McpWorkbench
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console

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


async def main() -> None:
    # Setup server params for local filesystem access
    params = StdioServerParams(
        command="uvx",
        args=["mcp-server-fetch"],
        read_timeout_seconds=60,
    )

    # You can also use `start()` and `stop()` to manage the session.
    async with McpWorkbench(server_params=params) as workbench:
        # model_client = OpenAIChatCompletionClient(model="gpt-4.1-nano")
        assistant = AssistantAgent(
            name="Assistant",
            model_client=model_client,
            workbench=workbench,
            reflect_on_tool_use=True,
        )
        await Console(
            assistant.run_stream(
                task="Go to https://github.com/microsoft/autogen and tell me what you see."
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
