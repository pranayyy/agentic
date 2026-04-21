"""
LangGraph orchestration graph for the Employee Q&A Agent.

Workflow
────────
                    ┌─────────────────────┐
                    │    analyze_query     │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   search_internal   │  ← RAG / ChromaDB
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │ evaluate_sufficiency│
                    └──────────┬──────────┘
              sufficient?      │       not sufficient?
         ┌────────────────────►│◄────────────────────┐
         │                     │                     │
         │         ┌───────────┘             ┌───────▼──────┐
         │         │                         │  search_web  │ ← Tavily
         │         │                         └───────┬──────┘
         │         │◄────────────────────────────────┘
         │         │
         │  ┌──────▼──────────┐
         │  │ summarize_rank  │  ← MCP Component 1
         │  └──────┬──────────┘
         │         │
         │  ┌──────▼──────────────┐
         │  │highlight_differences│  ← MCP Component 2
         │  └──────┬──────────────┘
         │         │
         │  ┌──────▼──────┐
         │  │ flag_issues │  ← MCP Component 3
         │  └──────┬──────┘
         │         │
         │  needs review?         no review needed?
         │  ┌──────┘                    │
         │  │                           │
    ┌────▼──▼──────┐           ┌────────▼───────┐
    │ human_review │ (interrupt)│synthesize_answer│
    └──────┬───────┘           └────────┬───────┘
           │                            │
           └────────────────────────────┘
                                        │
                                       END
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_chroma import Chroma

from agent.state import AgentState
from agent.nodes import (
    analyze_query_node,
    make_search_internal_node,
    evaluate_sufficiency_node,
    search_web_node,
    summarize_rank_node,
    highlight_differences_node,
    flag_issues_node,
    human_review_node,
    synthesize_answer_node,
)


# ─── Routing helpers ──────────────────────────────────────────────────────────

def _route_sufficiency(state: AgentState) -> str:
    return "sufficient" if state.get("internal_sufficient", False) else "insufficient"


def _route_review(state: AgentState) -> str:
    return "needs_review" if state.get("needs_human_review", False) else "no_review"


# ─── Graph builder ────────────────────────────────────────────────────────────

def build_graph(vectorstore: Chroma) -> "CompiledGraph":
    """
    Compile and return the LangGraph StateGraph.
    A MemorySaver checkpointer is attached to enable human-in-the-loop interrupts.
    """
    builder = StateGraph(AgentState)

    # Nodes
    builder.add_node("analyze_query", analyze_query_node)
    builder.add_node("search_internal", make_search_internal_node(vectorstore))
    builder.add_node("evaluate_sufficiency", evaluate_sufficiency_node)
    builder.add_node("search_web", search_web_node)
    builder.add_node("summarize_rank", summarize_rank_node)
    builder.add_node("highlight_differences", highlight_differences_node)
    builder.add_node("flag_issues", flag_issues_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("synthesize_answer", synthesize_answer_node)

    # Edges
    builder.add_edge(START, "analyze_query")
    builder.add_edge("analyze_query", "search_internal")
    builder.add_edge("search_internal", "evaluate_sufficiency")

    builder.add_conditional_edges(
        "evaluate_sufficiency",
        _route_sufficiency,
        {"sufficient": "summarize_rank", "insufficient": "search_web"},
    )

    builder.add_edge("search_web", "summarize_rank")
    builder.add_edge("summarize_rank", "highlight_differences")
    builder.add_edge("highlight_differences", "flag_issues")

    builder.add_conditional_edges(
        "flag_issues",
        _route_review,
        {"needs_review": "human_review", "no_review": "synthesize_answer"},
    )

    builder.add_edge("human_review", "synthesize_answer")
    builder.add_edge("synthesize_answer", END)

    # Compile with in-memory checkpointer (required for interrupt/resume)
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)
