"""
Employee Q&A AI Agent — Main Entry Point
==========================================
Run with:  python main.py

Requires a .env file with:
  OPENAI_API_KEY  — your OpenAI key
  TAVILY_API_KEY  — your Tavily web-search key (optional; skipped if absent)
"""

import os
import sys

# ── Load environment variables before any LangChain imports ──────────────────
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    print(
        "ERROR: GROQ_API_KEY is not set.\n"
        "Copy .env.example \u2192 .env and add your Groq API key.\n"
        "Get a free key at https://console.groq.com"
    )
    sys.exit(1)

# ── Project imports (after env is loaded) ────────────────────────────────────
from langgraph.types import Command

from agent.graph import build_graph
from rag.document_loader import load_sample_documents
from rag.vectorstore import initialize_vectorstore


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

DIVIDER = "=" * 70
THIN = "-" * 50


def _section(title: str) -> None:
    print(f"\n{THIN}\n{title}\n{THIN}")


def display_results(result: dict) -> None:
    """Pretty-print the agent's final answer and supporting metadata."""
    if not result:
        print("(No result returned)")
        return

    print(f"\n{DIVIDER}")
    print("  ANSWER")
    print(DIVIDER)
    print(result.get("final_answer", "(No answer generated)"))

    # Citations
    citations = result.get("citations", [])
    if citations:
        _section("CITATIONS")
        for i, c in enumerate(citations, 1):
            tag = "[INTERNAL]" if c.get("type") == "internal" else "[WEB]    "
            print(f"  [{i}] {tag}  {c.get('title', 'Unknown')}")
            src = c.get("source", "")
            if src:
                print(f"         {src}")
            date = c.get("date", "")
            if date and date not in ("Unknown", "Current", ""):
                print(f"         Date: {date}")

    # Internal vs external differences
    differences = result.get("differences", [])
    if differences:
        _section("INTERNAL vs EXTERNAL — KEY DIFFERENCES")
        for d in differences:
            print(f"  ⚠  {d}")

    # Flagged issues
    flagged = result.get("flagged_issues", [])
    if flagged:
        _section("FLAGGED FOR ATTENTION")
        for f in flagged:
            print(f"  !  {f}")

    # Answer metadata
    meta = result.get("answer_metadata", {})
    if meta:
        _section("METADATA")
        print(f"  Confidence      : {meta.get('confidence', 'N/A')}")
        print(f"  Sources used    : {meta.get('sources_used', 'N/A')}")
        print(f"  Internal chunks : {meta.get('internal_docs_count', 0)}")
        if meta.get("web_results_count", 0):
            print(f"  Web results     : {meta['web_results_count']}")
        if meta.get("had_human_review"):
            print("  Human review    : Yes (reviewer feedback incorporated)")
        if meta.get("flagged_issues_count", 0):
            print(f"  Flagged issues  : {meta['flagged_issues_count']}")


# ─────────────────────────────────────────────────────────────────────────────
# Main question runner (handles human-in-the-loop interrupt/resume)
# ─────────────────────────────────────────────────────────────────────────────

def run_question(graph, question: str, thread_id: str) -> dict | None:
    """
    Send a question through the LangGraph agent.
    If the agent triggers a human-review interrupt, prompt the reviewer
    and resume execution with their feedback.
    """
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "question": question,
        "human_feedback": None,
        "web_searched": False,
        "needs_human_review": False,
    }

    print(f"\n{DIVIDER}")
    print(f"  Processing: {question!r}")
    print(DIVIDER)

    # ── First pass ────────────────────────────────────────────────────
    result: dict = graph.invoke(initial_state, config=config)

    # ── Handle any human-review interrupts ────────────────────────────
    graph_state = graph.get_state(config)
    while graph_state.next:
        # Graph paused at human_review node
        print(f"\n{DIVIDER}")
        print("  HUMAN REVIEW REQUIRED")
        print(DIVIDER)

        # Retrieve interrupt payload
        interrupt_data: dict = {}
        if graph_state.tasks and graph_state.tasks[0].interrupts:
            raw = graph_state.tasks[0].interrupts[0].value
            if isinstance(raw, dict):
                interrupt_data = raw

        _section("REVIEW REASONS")
        for r in interrupt_data.get("review_reasons", ["(none specified)"]):
            print(f"  • {r}")

        flagged = interrupt_data.get("flagged_issues", [])
        if flagged:
            _section("FLAGGED ISSUES")
            for f in flagged:
                print(f"  !  {f}")

        diffs = interrupt_data.get("differences", [])
        if diffs:
            _section("SOURCE CONFLICTS")
            for d in diffs:
                print(f"  ⚠  {d}")

        conf = interrupt_data.get("confidence_score", 0)
        print(f"\n  Confidence score: {conf:.0%}")

        internal_preview = interrupt_data.get("internal_summary", "")[:400]
        if internal_preview:
            _section("INTERNAL SUMMARY PREVIEW")
            print(f"  {internal_preview}{'...' if len(interrupt_data.get('internal_summary','')) > 400 else ''}")

        print(f"\n{THIN}")
        feedback = input(
            "Reviewer feedback (corrections/additions), or press Enter to approve:\n> "
        ).strip()

        # Resume the graph with reviewer's feedback
        result = graph.invoke(
            Command(resume=feedback or "Approved. No changes required."),
            config=config,
        )
        graph_state = graph.get_state(config)

    display_results(result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print(DIVIDER)
    print("  EMPLOYEE Q&A AI AGENT")
    print("  Powered by LangGraph · RAG (ChromaDB) · Web Search (Tavily)")
    print(DIVIDER)

    if not os.getenv("TAVILY_API_KEY"):
        print(
            "\n[Note] TAVILY_API_KEY not set — web search disabled.\n"
            "       Add it to .env to enable external source retrieval.\n"
        )

    # ── Initialise knowledge base ──────────────────────────────────────
    print("\nLoading internal knowledge base …")
    try:
        docs = load_sample_documents()
        vectorstore = initialize_vectorstore(docs)
        print(f"  ✓ {vectorstore._collection.count()} document chunks ready")
    except Exception as exc:
        print(f"ERROR initialising knowledge base: {exc}")
        sys.exit(1)

    # ── Build agent graph ──────────────────────────────────────────────
    print("Building agent graph …")
    graph = build_graph(vectorstore)
    print("  ✓ Agent ready\n")

    print("Ask any work-related question.  Type 'quit' to exit.")
    print("Example questions:")
    print("  • How do I set up VPN access?")
    print("  • What is the meal expense limit when travelling?")
    print("  • How many vacation days do I get after 4 years?")
    print("  • What MFA methods are approved for corporate systems?")
    print("  • How do I submit an expense report?\n")

    session = 0
    while True:
        print(DIVIDER)
        try:
            question = input("Your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        session += 1
        try:
            run_question(graph, question, thread_id=f"session-{session}")
        except Exception as exc:
            print(f"\n[Agent error] {exc}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
