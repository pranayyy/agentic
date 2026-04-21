from typing import Optional
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    # ── Input ────────────────────────────────────────────────────────
    question: str

    # ── Query analysis ───────────────────────────────────────────────
    query_intent: str
    query_keywords: list[str]

    # ── RAG – internal knowledge base ────────────────────────────────
    internal_docs: list[dict]          # ranked chunks from ChromaDB
    internal_context: str              # concatenated raw text
    internal_sufficient: bool          # True if RAG alone is enough

    # ── Web search ───────────────────────────────────────────────────
    web_results: list[dict]
    web_context: str
    web_searched: bool

    # ── MCP: Summarise & rank ─────────────────────────────────────────
    ranked_sources: list[dict]
    internal_summary: str
    web_summary: str

    # ── MCP: Compare ─────────────────────────────────────────────────
    differences: list[str]
    agreements: list[str]

    # ── MCP: Validate / flag ─────────────────────────────────────────
    flagged_issues: list[str]
    confidence_score: float
    needs_human_review: bool
    review_reasons: list[str]

    # ── Human review ─────────────────────────────────────────────────
    human_feedback: Optional[str]

    # ── Final output ─────────────────────────────────────────────────
    final_answer: str
    citations: list[dict]
    answer_metadata: dict
