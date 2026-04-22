"""
Employee Q&A AI Agent — Streamlit Demo UI
==========================================
Run with:  streamlit run app.py
"""

import os
import uuid
import logging

import streamlit as st
from dotenv import load_dotenv
from langgraph.types import Command

# ── Configure terminal logging before anything else ───────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("opentelemetry").setLevel(logging.WARNING)

from agent.observability import (
    get_callback_handler,
    is_enabled as lf_enabled,
    log_rag_span,
    log_mcp_event,
    log_human_review_event,
    score_human_feedback,
    flush as lf_flush,
    get_trace_url as lf_get_trace_url,
)

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Employee Q&A Agent",
    page_icon="🤖",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS tweaks
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .tag-internal  { background:#1e4d8c; color:white; border-radius:4px; padding:1px 7px; font-size:0.75rem; }
    .tag-web       { background:#276749; color:white; border-radius:4px; padding:1px 7px; font-size:0.75rem; }
    .tag-flag      { background:#7b2d00; color:white; border-radius:4px; padding:1px 7px; font-size:0.75rem; }
    .review-box    { border-left:4px solid #f59e0b; padding:10px 16px; background:#1c1a13; border-radius:4px; margin-bottom:12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Cached resource — initialise once per server session
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading knowledge base and building agent…")
def load_agent():
    from agent.graph import build_graph
    from rag.document_loader import load_sample_documents
    from rag.vectorstore import initialize_vectorstore

    docs = load_sample_documents()
    vectorstore = initialize_vectorstore(docs)
    graph = build_graph(vectorstore)
    chunk_count = vectorstore._collection.count()
    return graph, chunk_count


# ─────────────────────────────────────────────────────────────────────────────
# Session state defaults
# ─────────────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []          # chat history list of dicts
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "awaiting_review" not in st.session_state:
    st.session_state.awaiting_review = False
if "interrupt_data" not in st.session_state:
    st.session_state.interrupt_data = {}
if "last_config" not in st.session_state:
    st.session_state.last_config = None
if "last_trace_id" not in st.session_state:
    st.session_state.last_trace_id = None
if "last_trace_url" not in st.session_state:
    st.session_state.last_trace_url = None
if "trace_history" not in st.session_state:
    st.session_state.trace_history = []  # list of {question, trace_id, trace_url}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _confidence_colour(score: float) -> str:
    if score >= 0.8:
        return "normal"
    if score >= 0.6:
        return "off"
    return "inverse"


def render_result(result: dict):
    """Render the agent result in a structured, readable way."""
    if not result:
        st.warning("No result returned.")
        return

    # ── Answer ────────────────────────────────────────────────────────
    st.subheader("Answer")
    st.markdown(result.get("final_answer", "_No answer generated._"))

    # ── Metadata strip ────────────────────────────────────────────────
    meta = result.get("answer_metadata", {})
    _conf_raw = meta.get("confidence", 0) or 0
    conf = float(str(_conf_raw).rstrip("%")) / (100 if str(_conf_raw).endswith("%") else 1)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Confidence", f"{conf:.0%}")
    col2.metric("Sources", meta.get("sources_used", "—"))
    col3.metric("Internal chunks", meta.get("internal_docs_count", 0))
    col4.metric("Web results", meta.get("web_results_count", 0))
    if meta.get("had_human_review"):
        st.info("Human review feedback was incorporated into this answer.", icon="👤")

    st.divider()

    # ── Citations ──────────────────────────────────────────────────────
    citations = result.get("citations", [])
    if citations:
        with st.expander(f"📖 Citations ({len(citations)})", expanded=True):
            for i, c in enumerate(citations, 1):
                is_internal = c.get("type") == "internal"
                tag_html = (
                    '<span class="tag-internal">INTERNAL</span>'
                    if is_internal
                    else '<span class="tag-web">WEB</span>'
                )
                title = c.get("title", "Unknown source")
                src   = c.get("source", "")
                date  = c.get("date", "")

                lines = [f"**{i}.** {tag_html} &nbsp; **{title}**"]
                if src:
                    lines.append(f"&nbsp;&nbsp;&nbsp;`{src}`")
                if date and date not in ("Unknown", "Current", ""):
                    lines.append(f"&nbsp;&nbsp;&nbsp;Date: {date}")

                st.markdown(" &nbsp; ".join(lines), unsafe_allow_html=True)

    # ── Differences ────────────────────────────────────────────────────
    differences = result.get("differences", [])
    if differences:
        with st.expander(f"⚠️ Internal vs Web Differences ({len(differences)})"):
            for d in differences:
                st.warning(d, icon="⚠️")

    # ── Flagged issues ─────────────────────────────────────────────────
    flagged = result.get("flagged_issues", [])
    if flagged:
        with st.expander(f"🚩 Flagged Issues ({len(flagged)})"):
            for f in flagged:
                st.error(f, icon="🚩")


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🤖 Employee Q&A Agent")
    st.caption("Powered by LangGraph · RAG · FastMCP · Groq")

    st.divider()

    try:
        graph, chunk_count = load_agent()
        st.success(f"Agent ready — {chunk_count} knowledge chunks loaded", icon="✅")
        agent_ready = True
    except Exception as exc:
        st.error(f"Failed to load agent: {exc}", icon="❌")
        agent_ready = False

    groq_key = os.getenv("GROQ_API_KEY", "")
    tavily_key = os.getenv("TAVILY_API_KEY", "")
    st.divider()
    st.subheader("Configuration")
    st.write("**LLM:**", os.getenv("MODEL_NAME", "llama-3.1-8b-instant"))
    st.write("**Groq API:**", "✅ Set" if groq_key else "❌ Missing")
    st.write("**Web Search:**", "✅ Enabled" if tavily_key else "⚠️ Disabled")

    st.divider()
    st.subheader("Observability")
    if lf_enabled():
        st.success("LangFuse tracing active", icon="📊")
        if st.session_state.get("last_trace_url"):
            st.markdown(f"[View last trace ↗]({st.session_state.last_trace_url})")
        # Show full trace history
        history = st.session_state.get("trace_history", [])
        if history:
            with st.expander(f"All traces ({len(history)})", expanded=False):
                for i, t in enumerate(reversed(history), 1):
                    tid_short = t["trace_id"].replace("-", "")[:12] + "…"
                    label = f"{i}. {t['question']}"
                    if t.get("trace_url"):
                        st.markdown(f"[{label}]({t['trace_url']})  \n`{tid_short}`")
                    else:
                        st.code(t["trace_id"], language=None)
    else:
        st.warning("LangFuse inactive", icon="📊")
        st.caption("Set LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY in .env to enable")

    st.divider()
    st.subheader("Example Questions")
    examples = [
        "How do I set up VPN access?",
        "What is the meal expense limit when travelling?",
        "How many vacation days do I get after 4 years?",
        "What MFA methods are approved for corporate systems?",
        "How do I submit an expense report?",
        "What is the onboarding process for new employees?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.prefill = ex

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.awaiting_review = False
        st.session_state.interrupt_data = {}
        st.session_state.last_config = None
        st.session_state.last_trace_id = None
        st.session_state.last_trace_url = None
        st.session_state.trace_history = []
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Main area — render chat history
# ─────────────────────────────────────────────────────────────────────────────
st.header("Employee Knowledge Assistant")

if not agent_ready:
    st.stop()

# Replay prior messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_result(msg["content"])


# ─────────────────────────────────────────────────────────────────────────────
# Human-review interrupt panel (shown when graph is paused)
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.awaiting_review:
    data = st.session_state.interrupt_data
    st.warning("**Human review required** — the agent flagged this answer for expert verification.", icon="👤")

    with st.expander("Review details", expanded=True):
        reasons = data.get("review_reasons", [])
        if reasons:
            st.markdown("**Reasons flagged:**")
            for r in reasons:
                st.markdown(f"- {r}")

        flagged = data.get("flagged_issues", [])
        if flagged:
            st.markdown("**Issues detected:**")
            for f in flagged:
                st.error(f, icon="🚩")

        diffs = data.get("differences", [])
        if diffs:
            st.markdown("**Source conflicts:**")
            for d in diffs:
                st.warning(d, icon="⚠️")

        conf = data.get("confidence_score", 0)
        st.metric("Confidence score", f"{conf:.0%}")

        internal_preview = data.get("internal_summary", "")[:500]
        if internal_preview:
            st.markdown("**Internal summary preview:**")
            st.markdown(internal_preview + ("…" if len(data.get("internal_summary", "")) > 500 else ""))

    with st.form("review_form"):
        feedback = st.text_area(
            "Your feedback (corrections, additions, or context):",
            placeholder="Leave blank to approve as-is, or type your corrections here…",
            height=120,
        )
        submitted = st.form_submit_button("Submit feedback & continue", type="primary")

    if submitted:
        config = st.session_state.last_config
        final_feedback = feedback.strip() or "Approved. No changes required."

        # Score the trace for this conversation
        if st.session_state.get("last_trace_id"):
            score_human_feedback(st.session_state.last_trace_id, final_feedback)

        with st.spinner("Resuming agent with your feedback…"):
            try:
                result = graph.invoke(Command(resume=final_feedback), config=config)
                graph_state = graph.get_state(config)

                if graph_state.next:
                    # Still paused (shouldn't normally happen)
                    raw = graph_state.tasks[0].interrupts[0].value
                    st.session_state.interrupt_data = raw if isinstance(raw, dict) else {}
                else:
                    st.session_state.awaiting_review = False
                    st.session_state.interrupt_data = {}
                    st.session_state.messages.append({"role": "assistant", "content": result})

            except Exception as exc:
                st.error(f"Error resuming agent: {exc}")

        st.rerun()

    st.stop()   # Don't show chat input while waiting for review


# ─────────────────────────────────────────────────────────────────────────────
# Chat input
# ─────────────────────────────────────────────────────────────────────────────
prefill = st.session_state.pop("prefill", None)
question = st.chat_input("Ask a work-related question…", key="chat_input") or prefill

if question:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Run the agent
    _trace_id = str(uuid.uuid4())   # unique trace per question
    _lf_handler = get_callback_handler(
        session_id=_trace_id,
        question=question,
    )
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    if _lf_handler:
        config["callbacks"] = [_lf_handler]
    st.session_state.last_config = config

    initial_state = {
        "question": question,
        "human_feedback": None,
        "web_searched": False,
        "needs_human_review": False,
    }

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result = graph.invoke(initial_state, config=config)
                graph_state = graph.get_state(config)

                # ── Post-invoke observability ──────────────────────────
                if _lf_handler:
                    _trace_url = lf_get_trace_url(_trace_id)
                    st.session_state.last_trace_id = _trace_id
                    st.session_state.last_trace_url = _trace_url
                    # Accumulate trace history
                    st.session_state.trace_history.append({
                        "question": question[:60] + ("…" if len(question) > 60 else ""),
                        "trace_id": _trace_id,
                        "trace_url": _trace_url,
                    })
                    log_rag_span(_trace_id, question, result.get("internal_docs", []))
                    log_mcp_event(
                        _trace_id,
                        ["summarize_and_rank", "highlight_differences", "flag_issues"],
                        result.get("flagged_issues", []),
                        result.get("differences", []),
                    )

                if graph_state.next:
                    # Graph paused for human review
                    raw = graph_state.tasks[0].interrupts[0].value
                    interrupt_data = raw if isinstance(raw, dict) else {}
                    if _trace_id:
                        log_human_review_event(
                            _trace_id,
                            interrupt_data.get("review_reasons", []),
                            interrupt_data.get("confidence_score", 0),
                        )
                    st.session_state.interrupt_data = interrupt_data
                    st.session_state.awaiting_review = True
                    lf_flush()
                    st.rerun()
                else:
                    lf_flush()
                    st.session_state.messages.append({"role": "assistant", "content": result})
                    render_result(result)

            except Exception as exc:
                st.error(f"Agent error: {exc}", icon="❌")
