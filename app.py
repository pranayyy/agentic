"""
Employee Q&A AI Agent — Streamlit Demo UI
==========================================
Uses the FastAPI backend (api.py) for all agent logic.

Start the API first:
    uvicorn api:app --port 8000

Then run:
    streamlit run app.py
"""

import os
import uuid
import json

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── API base URL (override with API_BASE_URL env var) ─────────────────────────
API_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

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
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Session state defaults
# ─────────────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "awaiting_review" not in st.session_state:
    st.session_state.awaiting_review = False
if "interrupt_data" not in st.session_state:
    st.session_state.interrupt_data = {}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

# ── Node progress map ────────────────────────────────────────────────────────
# Each entry: node_name → (progress_pct, display_label)
_NODE_STEPS = {
    "analyze_query":         (10, "Analyzing your question…"),
    "search_internal":       (25, "Searching internal knowledge base…"),
    "evaluate_sufficiency":  (40, "Evaluating search results…"),
    "search_web":            (55, "Searching the web for more context…"),
    "summarize_rank":        (68, "Summarizing and ranking sources…"),
    "highlight_differences": (80, "Comparing internal vs web sources…"),
    "flag_issues":           (90, "Validating and flagging issues…"),
    "human_review":          (95, "Flagged for human review…"),
    "synthesize_answer":     (99, "Composing final answer…"),
}


def _api(method: str, path: str, **kwargs):
    """Make a request to the FastAPI backend. Raises on non-2xx."""
    resp = requests.request(method, f"{API_URL}{path}", timeout=(10, 600), **kwargs)
    resp.raise_for_status()
    return resp.json()


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

    # ── API health check ──────────────────────────────────────────────
    try:
        health = _api("GET", "/health")
        agent_ready = health.get("agent_ready", False)
        chunk_count = health.get("knowledge_chunks", 0)
        if agent_ready:
            st.success(f"API ready — {chunk_count} knowledge chunks loaded", icon="✅")
        else:
            st.warning("API reachable but agent not ready yet.", icon="⚠️")
    except Exception as exc:
        st.error(f"Cannot reach API at {API_URL}\n\n{exc}", icon="❌")
        agent_ready = False

    st.divider()
    st.subheader("Configuration")
    st.write("**API:**", API_URL)
    st.write("**Groq API:**", "✅ Set" if os.getenv("GROQ_API_KEY") else "❌ Missing")
    st.write("**Web Search:**", "✅ Enabled" if os.getenv("TAVILY_API_KEY") else "⚠️ Disabled")
    st.write("**Langfuse:**", "✅ Enabled" if (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")) else "⚠️ Disabled")

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
        with st.spinner("Resuming agent with your feedback…"):
            try:
                resp = _api(
                    "POST",
                    f"/review/{st.session_state.thread_id}",
                    json={"feedback": feedback.strip()},
                )
                st.session_state.awaiting_review = False
                st.session_state.interrupt_data = {}
                if resp.get("status") == "completed":
                    st.session_state.messages.append(
                        {"role": "assistant", "content": resp["result"]}
                    )
            except Exception as exc:
                st.error(f"Error resuming agent: {exc}")

        st.rerun()

    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Chat input
# ─────────────────────────────────────────────────────────────────────────────
prefill = st.session_state.pop("prefill", None)
question = st.chat_input("Ask a work-related question…", key="chat_input") or prefill

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        progress_bar = st.progress(0, text="Starting…")
        status_text = st.empty()
        result = None
        trace_url = None
        try:
            with requests.post(
                f"{API_URL}/ask/stream",
                json={
                    "question": question,
                    "thread_id": st.session_state.thread_id,
                },
                stream=True,
                timeout=(10, 600),
            ) as r:
                r.raise_for_status()
                for raw_line in r.iter_lines():
                    if not raw_line:
                        continue
                    if raw_line.startswith(b"data: "):
                        event = json.loads(raw_line[6:])
                        etype = event.get("type")

                        if etype == "thread_id":
                            st.session_state.thread_id = event["thread_id"]

                        elif etype == "node":
                            node = event.get("node", "")
                            pct, label = _NODE_STEPS.get(node, (50, f"Running {node}…"))
                            progress_bar.progress(pct, text=f"**{label}**")

                        elif etype == "completed":
                            progress_bar.progress(100, text="**Done!**")
                            result = event.get("result", {})
                            trace_url = event.get("trace_url")

                        elif etype == "awaiting_review":
                            progress_bar.empty()
                            status_text.empty()
                            st.session_state.interrupt_data = event.get("interrupt_data", {})
                            st.session_state.awaiting_review = True
                            st.rerun()

                        elif etype == "error":
                            raise Exception(event.get("detail", "Unknown error"))

        except Exception as exc:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Agent error: {exc}", icon="❌")

        if result:
            progress_bar.empty()
            status_text.empty()
            st.session_state.messages.append({"role": "assistant", "content": result})
            render_result(result)
            if trace_url:
                st.caption(f"[🔍 View trace in Langfuse]({trace_url})")
