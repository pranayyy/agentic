"""
MCP Component 2 – Highlight Differences
=========================================
Compares the internal documentation summary with the external web
summary and explicitly surfaces:
  • Points where they AGREE (corroborating evidence).
  • Points where they DIFFER or CONFLICT (needs attention).
"""

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

_SYSTEM_PROMPT = """You are an expert fact-checker who compares two sources of information.

Given a question, an internal documentation summary, and an external web summary,
identify similarities and conflicts between the two.

Return ONLY valid JSON in this exact format:
{
  "differences": ["difference 1", "difference 2"],
  "agreements":  ["agreement 1",  "agreement 2"]
}

Rules:
- Each item must be a single, clear, self-contained sentence.
- "differences" = things that contradict or are markedly different between sources.
- "agreements"  = things both sources confirm.
- If a source is absent, set the relevant list to [].
- Maximum 5 items per list.
"""


def highlight_differences(
    question: str,
    internal_summary: str,
    web_summary: str,
    web_searched: bool,
) -> dict:
    """
    Returns:
        differences (list[str])
        agreements  (list[str])
    """
    if not web_searched or web_summary.startswith("Web search was not performed"):
        return {
            "differences": [],
            "agreements": [
                "Comparison not applicable — only internal sources were consulted."
            ],
        }

    resp = _llm_json.invoke(
        [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Employee question: {question}\n\n"
                    f"Internal documentation summary:\n{internal_summary}\n\n"
                    f"External web sources summary:\n{web_summary}"
                )
            ),
        ]
    )

    parsed = parse_llm_json(
        resp.content,
        fallback={"differences": [], "agreements": []},
    )

    return {
        "differences": parsed.get("differences", []),
        "agreements": parsed.get("agreements", []),
    }
