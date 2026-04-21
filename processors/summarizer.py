"""
MCP Component 1 – Summarise & Rank
===================================
Takes raw internal chunks and web search results then:
  • Produces a concise summary for each source type.
  • Ranks every source by relevance so the highest-quality
    evidence appears first in the final answer.
"""

import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from agent.utils import parse_llm_json

_MODEL = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
_llm = ChatGroq(model=_MODEL, temperature=0)
_llm_json = ChatGroq(
    model=_MODEL,
    temperature=0,
    model_kwargs={"response_format": {"type": "json_object"}},
)


def summarize_and_rank(
    question: str,
    internal_context: str,
    web_context: str,
    internal_docs: list[dict],
    web_results: list[dict],
    web_searched: bool,
) -> dict:
    """
    Returns:
        internal_summary (str)
        web_summary      (str)
        ranked_sources   (list[dict])
    """

    # ── Internal summary ──────────────────────────────────────────────
    if internal_context.strip():
        resp = _llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a concise technical writer. "
                        "Summarise only the key facts from the internal documentation "
                        "that directly answer the employee's question. "
                        "Use bullet points. Maximum 200 words."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Question: {question}\n\n"
                        f"Internal documentation:\n{internal_context[:3000]}"
                    )
                ),
            ]
        )
        internal_summary: str = resp.content
    else:
        internal_summary = "No relevant internal documentation found."

    # ── Web summary ────────────────────────────────────────────────────
    if web_searched and web_context.strip():
        resp = _llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a concise researcher. "
                        "Summarise only the key facts from the web sources "
                        "that directly answer the employee's question. "
                        "Use bullet points. Maximum 200 words."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Question: {question}\n\n"
                        f"Web sources:\n{web_context[:3000]}"
                    )
                ),
            ]
        )
        web_summary: str = resp.content
    else:
        web_summary = "Web search was not performed." if not web_searched else "No relevant web results found."

    # ── Build ranked source list ───────────────────────────────────────
    all_sources: list[dict] = []

    for doc in internal_docs:
        # ChromaDB distances: lower = more similar.  Convert to 0-1 similarity.
        similarity = max(0.0, 1.0 - float(doc.get("relevance_score", 0.5)))
        all_sources.append(
            {
                "title": doc.get("title", "Internal Doc"),
                "source": doc.get("source", "Internal Wiki"),
                "type": "internal",
                "relevance": round(similarity, 4),
                "content_preview": doc.get("content", "")[:200],
                "date": doc.get("date", "Unknown"),
            }
        )

    for result in web_results:
        all_sources.append(
            {
                "title": result.get("title", "Web Source"),
                "source": result.get("url", ""),
                "type": "web",
                "relevance": 0.70,   # web results treated as supplementary
                "content_preview": result.get("content", "")[:200],
                "date": "Current",
            }
        )

    ranked_sources = sorted(all_sources, key=lambda x: x["relevance"], reverse=True)

    return {
        "internal_summary": internal_summary,
        "web_summary": web_summary,
        "ranked_sources": ranked_sources[:8],
    }
