"""
FastMCP Server — Employee Q&A Processing Tools
================================================
Exposes the three MCP processing pipeline components as proper FastMCP tools.

Run standalone (stdio transport — readable by any MCP client):
    python -m processors.server

Or import in-process:
    from processors.server import mcp
"""

import json
from fastmcp import FastMCP

from processors.summarizer import summarize_and_rank as _summarize
from processors.comparator import highlight_differences as _compare
from processors.validator import flag_issues as _flag

mcp = FastMCP(
    name="Employee Q&A Tools",
    instructions=(
        "You have access to three specialised tools for processing employee questions. "
        "Always call them in order: summarize_and_rank → highlight_differences → flag_issues."
    ),
)


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1 — Summarise & Rank
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool(
    description=(
        "Summarise internal documentation and web search results, then rank all "
        "sources by relevance score. Returns JSON with keys: "
        "internal_summary (str), web_summary (str), ranked_sources (list)."
    )
)
def summarize_and_rank(
    question: str,
    internal_context: str,
    web_context: str,
    internal_docs_json: str,
    web_results_json: str,
    web_searched: bool,
) -> str:
    """
    Parameters
    ----------
    question           – The employee's original question.
    internal_context   – Concatenated internal document chunks (plain text).
    web_context        – Concatenated web search snippets (plain text).
    internal_docs_json – JSON-encoded list[dict] of internal result objects.
    web_results_json   – JSON-encoded list[dict] of web result objects.
    web_searched       – Whether a Tavily web search was performed.
    """
    internal_docs: list[dict] = json.loads(internal_docs_json or "[]")
    web_results: list[dict] = json.loads(web_results_json or "[]")

    result = _summarize(
        question=question,
        internal_context=internal_context,
        web_context=web_context,
        internal_docs=internal_docs,
        web_results=web_results,
        web_searched=web_searched,
    )
    return json.dumps(result)


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2 — Highlight Differences
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool(
    description=(
        "Compare the internal documentation summary with the external web summary. "
        "Surfaces where they agree and where they conflict. "
        "Returns JSON with keys: differences (list[str]), agreements (list[str])."
    )
)
def highlight_differences(
    question: str,
    internal_summary: str,
    web_summary: str,
    web_searched: bool,
) -> str:
    """
    Parameters
    ----------
    question          – The employee's original question.
    internal_summary  – Bullet-point summary from the internal knowledge base.
    web_summary       – Bullet-point summary from external web sources.
    web_searched      – Whether a web search was performed.
    """
    result = _compare(
        question=question,
        internal_summary=internal_summary,
        web_summary=web_summary,
        web_searched=web_searched,
    )
    return json.dumps(result)


# ─────────────────────────────────────────────────────────────────────────────
# Tool 3 — Validate & Flag
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool(
    description=(
        "Validate the collected information and flag potential quality issues. "
        "Detects outdated docs, source conflicts, and scores overall confidence. "
        "Returns JSON with keys: flagged_issues (list[str]), needs_human_review (bool), "
        "review_reasons (list[str]), confidence_score (float 0-1)."
    )
)
def flag_issues(
    question: str,
    internal_docs_json: str,
    internal_summary: str,
    web_summary: str,
    differences_json: str,
) -> str:
    """
    Parameters
    ----------
    question           – The employee's original question.
    internal_docs_json – JSON-encoded list[dict] of internal result objects.
    internal_summary   – Summary from internal documentation.
    web_summary        – Summary from external web sources.
    differences_json   – JSON-encoded list[str] of known conflict descriptions.
    """
    internal_docs: list[dict] = json.loads(internal_docs_json or "[]")
    differences: list[str] = json.loads(differences_json or "[]")

    result = _flag(
        question=question,
        internal_docs=internal_docs,
        internal_summary=internal_summary,
        web_summary=web_summary,
        differences=differences,
    )
    return json.dumps(result)


# ─────────────────────────────────────────────────────────────────────────────
# Standalone entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Runs the server over stdio so any MCP client (Claude Desktop, Cursor,
    # custom clients) can connect to it.
    mcp.run()
