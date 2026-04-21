"""
In-Process FastMCP Client
==========================
Provides a *synchronous* interface for calling FastMCP tools defined in
processors/server.py — no network server required.

LangGraph nodes are regular sync functions, so this module bridges the
async FastMCP Client to sync call sites using asyncio.
"""

import asyncio
import json
from typing import Any

from fastmcp import Client

# Lazy-loaded to allow .env to be populated before the server module
# (and its LangChain imports) are initialised.
_mcp_server = None


def _get_server():
    global _mcp_server
    if _mcp_server is None:
        from processors.server import mcp as _server
        _mcp_server = _server
    return _mcp_server


# ─────────────────────────────────────────────────────────────────────────────
# Async / sync bridge
# ─────────────────────────────────────────────────────────────────────────────

def _run_sync(coro) -> Any:
    """
    Run an async coroutine synchronously regardless of whether an event loop
    is already present (handles regular scripts, Jupyter, and nested loops).
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Already inside a running loop (e.g. Jupyter / pytest-asyncio).
        # Delegate to a fresh thread with its own event loop.
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)


# ─────────────────────────────────────────────────────────────────────────────
# Core call helper
# ─────────────────────────────────────────────────────────────────────────────

async def _call_tool_async(tool_name: str, arguments: dict) -> dict:
    """
    Connect to the FastMCP server in-process, call the requested tool,
    and return the parsed result dict.
    """
    async with Client(_get_server()) as client:
        result = await client.call_tool(tool_name, arguments)

    # FastMCP 3.x returns a CallToolResult object with a .content list.
    # Older versions returned a plain list — handle both.
    content_list = getattr(result, "content", None)
    if content_list is None and hasattr(result, "__iter__"):
        content_list = list(result)

    raw_text: str | None = None
    if content_list:
        first = content_list[0]
        raw_text = getattr(first, "text", None) or str(first)
    elif hasattr(result, "text"):
        raw_text = result.text
    elif result:
        raw_text = str(result)

    if raw_text:
        try:
            return json.loads(raw_text)
        except (json.JSONDecodeError, TypeError):
            return {"raw": raw_text}

    return {}


def call_mcp_tool(tool_name: str, arguments: dict) -> dict:
    """
    Synchronous entry point for LangGraph nodes.

    Parameters
    ----------
    tool_name  : Exact name of the FastMCP tool to call.
    arguments  : Dict matching the tool's parameter names.

    Returns
    -------
    dict — The tool's JSON-decoded result.
    """
    return _run_sync(_call_tool_async(tool_name, arguments))


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: list available tools (for debugging / introspection)
# ─────────────────────────────────────────────────────────────────────────────

async def _list_tools_async() -> list[str]:
    async with Client(_get_server()) as client:
        tools = await client.list_tools()
    return [t.name for t in tools]


def list_mcp_tools() -> list[str]:
    """Return the names of all tools registered on the FastMCP server."""
    return _run_sync(_list_tools_async())
