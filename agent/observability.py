"""
LangFuse Observability Helpers
===============================
Provides a thin wrapper around LangFuse for tracing the Employee Q&A Agent.

Usage
-----
Set these environment variables to enable tracing:
    LANGFUSE_PUBLIC_KEY   — your LangFuse project public key
    LANGFUSE_SECRET_KEY   — your LangFuse project secret key
    LANGFUSE_HOST         — (optional) defaults to https://cloud.langfuse.com

When the keys are absent every function is a no-op so the app runs normally.
"""

import os
from functools import lru_cache


def _to_trace_id(uid: str) -> str:
    """Convert a UUID (with dashes) to the 32 lowercase hex chars Langfuse v4 expects."""
    return uid.replace("-", "").lower()


# ─────────────────────────────────────────────────────────────────────────────
# Availability check
# ─────────────────────────────────────────────────────────────────────────────

def is_enabled() -> bool:
    """Return True only when both LangFuse API keys are present."""
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


@lru_cache(maxsize=1)
def _client():
    """Return a cached Langfuse SDK client, or None when disabled."""
    if not is_enabled():
        return None
    from langfuse import Langfuse  # noqa: PLC0415
    return Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# LangChain / LangGraph callback handler (auto-traces nodes + LLM calls)
# ─────────────────────────────────────────────────────────────────────────────

def get_callback_handler(session_id: str, question: str = ""):
    """
    Return a LangFuse CallbackHandler to pass in LangGraph config['callbacks'].
    Automatically traces every node execution and LLM call including token counts.
    Returns None when LangFuse is not configured.
    """
    if not is_enabled():
        return None
    from langfuse.langchain import CallbackHandler  # noqa: PLC0415
    return CallbackHandler(trace_context={"trace_id": _to_trace_id(session_id)})


# ─────────────────────────────────────────────────────────────────────────────
# Custom spans / events logged after graph.invoke() completes
# ─────────────────────────────────────────────────────────────────────────────

def log_rag_span(trace_id: str, question: str, docs: list) -> None:
    """
    Log a RAG retrieval span with chunk metadata (query, count, scores).
    Called after invoke with result['internal_docs'].
    """
    lf = _client()
    if lf is None or not trace_id:
        return
    lf.start_observation(
        trace_context={"trace_id": _to_trace_id(trace_id)},
        name="rag_internal_search",
        as_type="retriever",
        input={"question": question},
        output={
            "chunks_retrieved": len(docs),
            "docs": [
                {
                    "title": d.get("title"),
                    "source": d.get("source"),
                    "relevance_score": d.get("relevance_score"),
                }
                for d in docs[:5]
            ],
        },
        level="DEFAULT",
    )


def log_mcp_event(
    trace_id: str,
    tools_used: list,
    flagged_issues: list,
    differences: list,
) -> None:
    """
    Log a summary event for all MCP tool calls (summarize, diff, flag).
    Called after invoke with result state.
    """
    lf = _client()
    if lf is None or not trace_id:
        return
    lf.create_event(
        trace_context={"trace_id": _to_trace_id(trace_id)},
        name="mcp_tools_summary",
        input={"tools_called": tools_used},
        output={
            "flagged_issues_count": len(flagged_issues),
            "differences_count": len(differences),
        },
        metadata={
            "flagged_issues": flagged_issues[:10],
            "differences": differences[:10],
        },
    )


def log_human_review_event(
    trace_id: str,
    review_reasons: list,
    confidence_score: float,
) -> None:
    """
    Log an event when the graph pauses for human review.
    Recorded at WARNING level so it's easy to filter in LangFuse.
    """
    lf = _client()
    if lf is None or not trace_id:
        return
    lf.create_event(
        trace_context={"trace_id": _to_trace_id(trace_id)},
        name="human_review_triggered",
        input={"review_reasons": review_reasons},
        output={"confidence_score": confidence_score},
        level="WARNING",
    )


def score_human_feedback(trace_id: str, feedback: str) -> None:
    """
    Score a trace after the human reviewer submits feedback.
    1.0 = approved as-is, 0.5 = corrections provided.
    """
    lf = _client()
    if lf is None or not trace_id:
        return
    approved = feedback.strip().lower() in ("", "approved. no changes required.")
    lf.create_score(
        trace_id=_to_trace_id(trace_id),
        name="human_review",
        value=1.0 if approved else 0.5,
        comment=feedback[:500],
    )
    lf.flush()


def get_trace_url(trace_id: str) -> str | None:
    """Return the LangFuse dashboard URL for the given trace, or None."""
    lf = _client()
    if lf is None or not trace_id:
        return None
    return lf.get_trace_url(trace_id=_to_trace_id(trace_id))


def flush() -> None:
    """Flush all pending LangFuse events to the server."""
    lf = _client()
    if lf:
        lf.flush()
