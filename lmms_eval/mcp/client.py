import asyncio
import sys
from contextlib import AsyncExitStack
from datetime import timedelta
from typing import List, Optional, Union

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import AudioContent, ImageContent, TextContent


def _tool_input_schema(tool):
    """Read tool schemas from MCP v1 (camelCase) or v2 (snake_case)."""
    input_schema = getattr(tool, "inputSchema", None)
    if input_schema is None:
        input_schema = tool.input_schema
    return input_schema


class MCPClient:
    def __init__(self, server_path: Optional[str] = None, timeout: float | timedelta = 600.0, *, server_url: Optional[str] = None):
        """
        Connect to a Python server over stdio or an existing Streamable HTTP server.
        """
        self.server_path = server_path
        self.server_url = server_url
        if bool(server_path) == bool(server_url):
            raise ValueError("Provide exactly one of server_path or server_url")
        self.timeout = timeout.total_seconds() if isinstance(timeout, timedelta) else float(timeout)
        self._session: Optional[ClientSession] = None
        self._exit_stack: Optional[AsyncExitStack] = None
        self._connection_lock = asyncio.Lock()

    def _server_params(self) -> StdioServerParameters:
        # A bare ``python`` can resolve outside the evaluator's virtual
        # environment when the launcher invokes the venv interpreter by its
        # absolute path without activating it first.
        return StdioServerParameters(command=sys.executable, args=[self.server_path])

    async def connect(self) -> ClientSession:
        """Open one MCP transport and retain its session until ``close``."""
        if self._session is not None:
            return self._session

        async with self._connection_lock:
            if self._session is not None:
                return self._session

            stack = AsyncExitStack()
            try:
                if self.server_url:
                    from mcp.client.streamable_http import streamable_http_client

                    streams = await stack.enter_async_context(streamable_http_client(self.server_url))
                    read_stream, write_stream = streams[:2]
                else:
                    read_stream, write_stream = await stack.enter_async_context(stdio_client(server=self._server_params()))
                session = await stack.enter_async_context(ClientSession(read_stream, write_stream, read_timeout_seconds=self.timeout))
                await session.initialize()
            except BaseException:
                await stack.aclose()
                raise

            self._exit_stack = stack
            self._session = session
            return session

    async def close(self) -> None:
        """Close the persistent MCP session and terminate its subprocess."""
        async with self._connection_lock:
            stack = self._exit_stack
            self._session = None
            self._exit_stack = None
            if stack is not None:
                await stack.aclose()

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        await self.close()

    async def get_function_list(self):
        """
        Connect to the MCP server and retrieve the list of available functions.
        """
        session = await self.connect()
        tools = (await session.list_tools()).tools

        functions = []
        for tool in tools:
            functions.append({"type": "function", "function": {"name": tool.name, "description": tool.description or "", "parameters": _tool_input_schema(tool)}})
        return functions

    async def run_tool(self, tool_name: str, tool_args: dict):
        """
        Run a specific tool with the given arguments.
        :param tool_name: Name of the tool to run.
        :param tool_args: Arguments for the tool.
        :return: Result of the tool execution.
        """
        session = await self.connect()
        return await session.call_tool(tool_name, tool_args)

    def convert_result_to_openai_format(self, result: Union[ImageContent, TextContent, AudioContent, List[Union[ImageContent, TextContent, AudioContent]]]) -> dict:
        """
        Convert the result from the MCP tool to OpenAI compatible format.
        :param result: Result from the MCP tool.
        :return: Converted result.
        """
        if isinstance(result, list):
            results = []
            for item in result:
                results.append(self.convert_result_to_openai_format(item))
            return results
        if isinstance(result, ImageContent):
            return [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{result.data}"}}]
        elif isinstance(result, TextContent):
            return [{"type": "text", "text": result.text}]
        elif isinstance(result, AudioContent):
            return [{"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{result.data}"}}]
        else:
            raise ValueError(f"Unsupported result type : {type(result)}")

    def get_function_list_sync(self):
        """
        Synchronous wrapper for get_function_list.
        Connect to the MCP server and retrieve the list of available functions.
        :return: List of available functions in OpenAI-compatible format.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def run():
            async with self:
                return await self.get_function_list()

        try:
            return loop.run_until_complete(run())
        finally:
            loop.close()

    def run_tool_sync(self, tool_name: str, tool_args: dict):
        """
        Synchronous wrapper for run_tool.
        Run a specific tool with the given arguments.
        :param tool_name: Name of the tool to run.
        :param tool_args: Arguments for the tool.
        :return: Result of the tool execution.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def run():
            async with self:
                return await self.run_tool(tool_name, tool_args)

        try:
            return loop.run_until_complete(run())
        finally:
            loop.close()
