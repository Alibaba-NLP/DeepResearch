from typing import Optional
import uuid
import logging
import traceback
import asyncio

from mcp import ClientSession
from mcp.client.sse import sse_client
from contextlib import asynccontextmanager
import anyio


@asynccontextmanager
async def mcp_client(
        server_url: str,
    ):
    async with sse_client(url=server_url, headers={
            "ROUTE-KEY": str(uuid.uuid4())
        }) as streams:
            async with ClientSession(*streams) as session:
                lock = asyncio.Lock()
                initialize = await session.initialize()
                print("initialize")
                print(initialize)

                print("Initialized SSE client...")
                print("Listing tools...")
                response = await session.list_tools()
                tools = response.tools
                print("\nConnected to server with tools:", [tool.name for tool in tools])

                async def _ping_loop(ping_interval_seconds: int):
                    try:
                        while True:
                            await anyio.sleep(ping_interval_seconds)
                            try:
                                async with lock:
                                    await session.list_tools()
                            except Exception as e:
                                print(f"MCPClient: Ping loop error in list_tools: {repr(e)}")
                                break
                    except anyio.get_cancelled_exc_class():
                        print("MCPClient: Ping task was cancelled.")

                session._task_group.start_soon(_ping_loop, 20)
                yield session, lock
                print(f"Exiting MCPClient context. Cleaning up...")
                print("ClientSession cleaned up.")
                print("SSE client streams cleaned up.")