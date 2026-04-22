"""
All LangGraph node functions.

Nodes are thin orchestration wrappers that read from state,
call the appropriate RAG / MCP / LLM logic, and return state patches.
"""

import os
import json
import logging
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma
from langgraph.types import interrupt

from agent.state import AgentState
from agent.utils import parse_llm_json
from agent.mcp_client import call_mcp_tool

log = logging.getLogger("agent.nodes")

_MODEL = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")

_llm = ChatGroq(model=_MODEL, temperature=0)
_llm_json = ChatGroq(
    model=_MODEL,
    temperature=0,
    model_kwargs={"response_format": {"type": "json_object"}},
)

# ─────────────────────────────────────────────────────────────────────────────
# Node 1 – Query analyser
# ─────────────────────────────────────────────────────────────────────────────

def analyze_query_node(state: AgentState) -> dict:
    """Extract intent and search keywords from the employee's question."""
    question = state.get("question", "")
    log.info("[analyze_query] question=%r", question)

    resp = _llm_json.invoke(
        [
            SystemMessage(
                content=(
                    "You are a query analyser. "
                    "Given an employee question return JSON with:\n"
                    '{"intent": "one sentence describing what the employee wants", '
                    '"keywords": ["keyword1", "keyword2", ...]}\n'
                    "Include 3-6 keywords most useful for document retrieval."
                )
            ),
            HumanMessage(content=f"Question: {question}"),
        ]
    )

    parsed = parse_llm_json(
        resp.content,
        fallback={"intent": question, "keywords": question.split()[:5]},
    )
    log.info("[analyze_query] intent=%r  keywords=%s", parsed.get("intent"), parsed.get("keywords"))
    return {
        "query_intent": parsed.get("intent", question),
        "query_keywords": parsed.get("keywords", []),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 2 – Internal RAG search  (factory – receives vectorstore at build time)
# ─────────────────────────────────────────────────────────────────────────────

def make_search_internal_node(vectorstore: Chroma):
    """Return a node function that searches the internal ChromaDB instance."""

    def search_internal_node(state: AgentState) -> dict:
        question = state.get("question", "")
        log.info("[search_internal] searching ChromaDB for: %r", question)
        results = vectorstore.similarity_search_with_score(question, k=5)

        internal_docs: list[dict] = []
        for doc, score in results:
            internal_docs.append(
                {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "Internal Wiki"),
                    "title": doc.metadata.get("title", "Internal Document"),
                    "relevance_score": float(score),   # lower = more similar (L2)
                    "date": doc.metadata.get("date", "Unknown"),
                }
            )

        context = "\n\n---\n\n".join(
            f"[{d['title']}]\n{d['content']}" for d in internal_docs[:4]
        )
        log.info("[search_internal] found %d docs, top score=%.4f",
                 len(internal_docs), internal_docs[0]["relevance_score"] if internal_docs else 0)
        return {
            "internal_docs": internal_docs,
            "internal_context": context,
        }

    return search_internal_node


# ─────────────────────────────────────────────────────────────────────────────
# Node 3 – Evaluate sufficiency
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_sufficiency_node(state: AgentState) -> dict:
    """Decide whether internal docs adequately answer the question."""
    question = state.get("question", "")
    internal_context = state.get("internal_context", "")
    internal_docs = state.get("internal_docs", [])
    log.info("[evaluate_sufficiency] evaluating %d internal docs", len(internal_docs))

    if not internal_docs or not internal_context.strip():
        return {"internal_sufficient": False}

    resp = _llm_json.invoke(
        [
            SystemMessage(
                content=(
                    "Evaluate whether the provided internal documentation "
                    "sufficiently answers the employee's question.\n"
                    'Return JSON: {"sufficient": true/false, "reason": "..."}\n'
                    "Mark sufficient=true only if the docs give a clear, actionable answer."
                )
            ),
            HumanMessage(
                content=(
                    f"Question: {question}\n\n"
                    f"Documentation excerpt:\n{internal_context[:2500]}"
                )
            ),
        ]
    )

    parsed = parse_llm_json(resp.content, fallback={"sufficient": False})
    sufficient = bool(parsed.get("sufficient", False))
    log.info("[evaluate_sufficiency] sufficient=%s  reason=%r", sufficient, parsed.get("reason"))
    return {"internal_sufficient": sufficient}


# ─────────────────────────────────────────────────────────────────────────────
# Node 4 – Web search
# ─────────────────────────────────────────────────────────────────────────────

def search_web_node(state: AgentState) -> dict:
    """Search the internet using Tavily for supplementary information."""
    question = state.get("question", "")
    query_intent = state.get("query_intent", question)
    log.info("[search_web] Tavily query: %r", query_intent)

    try:
        from langchain_tavily import TavilySearch

        tool = TavilySearch(max_results=5)
        raw = tool.invoke({"query": f"workplace policy: {query_intent}"})

        # TavilySearch returns a list of dicts with url/content/title keys
        results_list = raw if isinstance(raw, list) else raw.get("results", [])
        web_results: list[dict] = [
            {
                "content": r.get("content", ""),
                "url": r.get("url", ""),
                "title": r.get("title", "Web Source"),
                "source": "Internet",
            }
            for r in (results_list or [])
        ]

        web_context = "\n\n---\n\n".join(
            f"[{r['title']}]\n{r['content'][:600]}" for r in web_results[:4]
        )

    except Exception as exc:
        web_results = []
        web_context = f"(Web search unavailable: {exc})"
        log.warning("[search_web] failed: %s", exc)

    log.info("[search_web] got %d web results", len(web_results))
    return {
        "web_results": web_results,
        "web_context": web_context,
        "web_searched": True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 5 – MCP: Summarise & rank  →  FastMCP tool call
# ─────────────────────────────────────────────────────────────────────────────

def summarize_rank_node(state: AgentState) -> dict:
    """MCP Tool 1 — call `summarize_and_rank` via the FastMCP server in-process."""
    log.info("[summarize_rank] calling MCP tool summarize_and_rank")
    result = call_mcp_tool(
        "summarize_and_rank",
        {
            "question": state.get("question", ""),
            "internal_context": state.get("internal_context", ""),
            "web_context": state.get("web_context", ""),
            "internal_docs_json": json.dumps(state.get("internal_docs", [])),
            "web_results_json": json.dumps(state.get("web_results", [])),
            "web_searched": state.get("web_searched", False),
        },
    )
    log.info("[summarize_rank] done, confidence=%.2f", result.get("confidence_score", 0))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Node 6 – MCP: Highlight differences  →  FastMCP tool call
# ─────────────────────────────────────────────────────────────────────────────

def highlight_differences_node(state: AgentState) -> dict:
    """MCP Tool 2 — call `highlight_differences` via the FastMCP server in-process."""
    log.info("[highlight_differences] calling MCP tool")
    result = call_mcp_tool(
        "highlight_differences",
        {
            "question": state.get("question", ""),
            "internal_summary": state.get("internal_summary", ""),
            "web_summary": state.get("web_summary", ""),
            "web_searched": state.get("web_searched", False),
        },
    )
    log.info("[highlight_differences] found %d differences", len(result.get("differences", [])))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Node 7 – MCP: Validate & flag  →  FastMCP tool call
# ─────────────────────────────────────────────────────────────────────────────

def flag_issues_node(state: AgentState) -> dict:
    """MCP Tool 3 — call `flag_issues` via the FastMCP server in-process."""
    log.info("[flag_issues] calling MCP tool")
    result = call_mcp_tool(
        "flag_issues",
        {
            "question": state.get("question", ""),
            "internal_docs_json": json.dumps(state.get("internal_docs", [])),
            "internal_summary": state.get("internal_summary", ""),
            "web_summary": state.get("web_summary", ""),
            "differences_json": json.dumps(state.get("differences", [])),
        },
    )
    log.info("[flag_issues] flagged %d issues, needs_review=%s",
             len(result.get("flagged_issues", [])), result.get("needs_human_review"))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Node 8 – Human review  (LangGraph interrupt)
# ─────────────────────────────────────────────────────────────────────────────

def human_review_node(state: AgentState) -> dict:
    """
    Pause execution for a human reviewer.
    The `interrupt()` call serialises the review payload and suspends the graph.
    Execution resumes when the caller invokes Command(resume=<feedback>).
    """
    review_payload = {
        "question": state.get("question", ""),
        "review_reasons": state.get("review_reasons", []),
        "flagged_issues": state.get("flagged_issues", []),
        "differences": state.get("differences", []),
        "internal_summary": state.get("internal_summary", "")[:600],
        "web_summary": state.get("web_summary", "")[:600],
        "confidence_score": state.get("confidence_score", 0.0),
    }

    # Blocks here until Command(resume=...) is supplied
    log.info("[human_review] PAUSED — waiting for human feedback")
    feedback = interrupt(review_payload)
    log.info("[human_review] RESUMED with feedback: %r", str(feedback)[:120])

    return {"human_feedback": feedback or "No additional reviewer feedback."}


# ─────────────────────────────────────────────────────────────────────────────
# Node 9 – Synthesise final answer
# ─────────────────────────────────────────────────────────────────────────────

def synthesize_answer_node(state: AgentState) -> dict:
    """Compose the final employee-facing answer with citations."""
    question = state.get("question", "")
    log.info("[synthesize_answer] composing final answer for: %r", question)
    internal_summary = state.get("internal_summary", "")
    web_summary = state.get("web_summary", "")
    differences = state.get("differences", [])
    flagged_issues = state.get("flagged_issues", [])
    human_feedback = state.get("human_feedback", "") or ""
    ranked_sources = state.get("ranked_sources", [])
    confidence = state.get("confidence_score", 0.75)
    web_searched = state.get("web_searched", False)

    # Build synthesis context
    ctx_parts: list[str] = []
    if internal_summary:
        ctx_parts.append(f"Internal knowledge base:\n{internal_summary}")
    if web_searched and web_summary and not web_summary.startswith("Web search was not"):
        ctx_parts.append(f"External sources:\n{web_summary}")
    if differences:
        ctx_parts.append(
            "Notable differences between sources:\n"
            + "\n".join(f"  • {d}" for d in differences)
        )
    if human_feedback and human_feedback != "No additional reviewer feedback.":
        ctx_parts.append(f"Human reviewer feedback:\n{human_feedback}")

    synthesis_context = "\n\n".join(ctx_parts) or "No information available."

    resp = _llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a helpful employee-support assistant. "
                    "Write a concise, actionable answer for the employee below.\n"
                    "Guidelines:\n"
                    "  • Lead with the most important action or fact.\n"
                    "  • Use numbered steps where a process is involved.\n"
                    "  • Mention any important caveats from the 'differences' section.\n"
                    "  • Keep the answer under 350 words.\n"
                    "  • Do NOT fabricate information not present in the context."
                )
            ),
            HumanMessage(
                content=(
                    f"Employee question: {question}\n\n"
                    f"Context:\n{synthesis_context}"
                )
            ),
        ]
    )

    final_answer: str = resp.content

    # ── Citations ──────────────────────────────────────────────────────
    citations: list[dict] = []
    seen: set[str] = set()
    for src in ranked_sources:
        key = src.get("source", "")
        if key and key not in seen:
            seen.add(key)
            citations.append(
                {
                    "title": src.get("title", "Unknown"),
                    "source": key,
                    "type": src.get("type", "unknown"),
                    "date": src.get("date", "Unknown"),
                }
            )

    # ── Metadata ───────────────────────────────────────────────────────
    sources_label = "Internal + Web" if web_searched else "Internal Only"
    answer_metadata = {
        "confidence": f"{confidence:.0%}",
        "sources_used": sources_label,
        "internal_docs_count": len(state.get("internal_docs", [])),
        "web_results_count": len(state.get("web_results", [])),
        "had_human_review": bool(
            human_feedback and human_feedback != "No additional reviewer feedback."
        ),
        "flagged_issues_count": len(flagged_issues),
    }

    return {
        "final_answer": final_answer,
        "citations": citations[:6],
        "answer_metadata": answer_metadata,
    }
