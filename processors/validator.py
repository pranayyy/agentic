"""
MCP Component 3 – Validate & Flag
===================================
Analyses the aggregated information to:
  • Detect signals that documentation may be outdated.
  • Identify internal contradictions or conflicts with web sources.
  • Determine whether human expert review is warranted.
  • Produce an overall confidence score (0.0 – 1.0).
"""

import json
import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from agent.utils import parse_llm_json

_MODEL = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
_llm_json = ChatGroq(
    model=_MODEL,
    temperature=0,
    model_kwargs={"response_format": {"type": "json_object"}},
)

_SYSTEM_PROMPT = """You are a quality-assurance specialist for an enterprise knowledge base.

Analyse the information provided and return ONLY valid JSON in the format below:
{
  "outdated_signals":   ["signal 1"],
  "conflict_signals":   ["conflict 1"],
  "needs_human_review": true,
  "review_reason":      "brief reason",
  "confidence_score":   0.85
}

Definitions:
- outdated_signals: phrases/facts suggesting information may be stale (e.g., old version numbers, discontinued tools).
- conflict_signals: internal contradictions OR contradictions with external sources.
- needs_human_review: true if there are outdated signals, significant conflicts, or low confidence.
- confidence_score: float 0.0–1.0 reflecting how reliably the answer can be given (1.0 = very confident).
"""


def flag_issues(
    question: str,
    internal_docs: list[dict],
    internal_summary: str,
    web_summary: str,
    differences: list[str],
) -> dict:
    """
    Returns:
        flagged_issues      (list[str])
        needs_human_review  (bool)
        review_reasons      (list[str])
        confidence_score    (float)
    """
    flagged: list[str] = []
    needs_review = False
    review_reasons: list[str] = []

    # ── Heuristic checks (fast, no LLM call) ──────────────────────────
    unknown_date_count = sum(
        1 for d in internal_docs if d.get("date", "Unknown") == "Unknown"
    )
    if unknown_date_count > 0:
        flagged.append(
            f"{unknown_date_count} internal document(s) have no publication date — "
            "content may be outdated."
        )
        needs_review = True
        review_reasons.append("Document(s) with unknown publication date(s)")

    if len(differences) >= 3:
        flagged.append(
            f"{len(differences)} conflicts detected between internal and external sources."
        )
        needs_review = True
        review_reasons.append("Multiple internal-vs-external conflicts")

    if not internal_summary or internal_summary.startswith("No relevant"):
        flagged.append("No matching internal documentation found for this question.")
        needs_review = True
        review_reasons.append("Question not covered by internal knowledge base")

    # ── LLM deep-check ────────────────────────────────────────────────
    try:
        resp = _llm_json.invoke(
            [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"Question: {question}\n\n"
                        f"Internal summary:\n{internal_summary[:1200]}\n\n"
                        f"Web summary:\n{web_summary[:1200]}\n\n"
                        f"Known conflicts:\n{json.dumps(differences)}"
                    )
                ),
            ]
        )
        parsed = parse_llm_json(
            resp.content,
            fallback={
                "outdated_signals": [],
                "conflict_signals": [],
                "needs_human_review": False,
                "review_reason": "",
                "confidence_score": 0.75,
            },
        )

        flagged += parsed.get("outdated_signals", [])
        flagged += parsed.get("conflict_signals", [])

        if parsed.get("needs_human_review", False):
            needs_review = True
            reason = parsed.get("review_reason", "").strip()
            if reason:
                review_reasons.append(reason)

        confidence: float = float(parsed.get("confidence_score", 0.75))

    except Exception as exc:
        flagged.append(f"[Validator error] {exc}")
        confidence = 0.60

    # Deduplicate flags
    flagged = list(dict.fromkeys(f for f in flagged if f))

    return {
        "flagged_issues": flagged,
        "needs_human_review": needs_review,
        "review_reasons": list(dict.fromkeys(review_reasons)),
        "confidence_score": min(max(confidence, 0.0), 1.0),
    }
