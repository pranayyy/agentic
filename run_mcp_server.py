"""
run_mcp_server.py
==================
Start the Employee Q&A FastMCP server as a standalone process.

This is useful for connecting external MCP clients such as:
  • Claude Desktop
  • Cursor IDE
  • Any MCP-compatible tool via stdio or SSE transport

Usage
-----
Default (stdio — for Claude Desktop / Cursor):
    python run_mcp_server.py

HTTP/SSE transport (for web clients, port 8000):
    python run_mcp_server.py --transport sse --port 8000

Inspect available tools without starting the server:
    python run_mcp_server.py --list-tools
"""

import sys
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Employee Q&A FastMCP server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport layer (default: stdio)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for SSE transport (default: 8000)",
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="Print registered tools and exit",
    )
    args = parser.parse_args()

    # Load environment before importing server (LangChain reads OPENAI_API_KEY at import)
    from dotenv import load_dotenv
    load_dotenv()

    from processors.server import mcp

    if args.list_tools:
        import asyncio
        from fastmcp import Client

        async def _list():
            async with Client(mcp) as client:
                tools = await client.list_tools()
            for t in tools:
                print(f"  • {t.name}: {t.description}")

        print(f"\nRegistered FastMCP tools on [{mcp.name}]:\n")
        asyncio.run(_list())
        print()
        return

    if args.transport == "sse":
        print(f"Starting Employee Q&A FastMCP server (SSE) on port {args.port} …")
        mcp.run(transport="sse", port=args.port)
    else:
        # stdio — standard MCP protocol for Claude Desktop / Cursor
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
