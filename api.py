"""
FastAPI — Employee Q&A Agent
=============================
Wraps the LangGraph end-to-end pipeline as a REST API.

Run with:
    uvicorn api:app --reload --port 8000

Endpoints
---------
GET  /health                  → liveness + chunk count
POST /ask                     → submit a question
POST /review/{thread_id}      → submit human review feedback
GET  /status/{thread_id}      → check if a thread is awaiting review
"""

import uuid
import json
import logging
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langgraph.types import Command

load_dotenv()

from agent import observability as obs

# ── Trace context ──────────────────────────────────────────────────────────────
# Set once per request; automatically included in every log line via the filter.
_trace_id_var: ContextVar[str] = ContextVar("trace_id", default="-")


class _TraceFilter(logging.Filter):
    """Inject the current trace_id into every log record automatically."""
    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = _trace_id_var.get("-")
        return True


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [%(trace_id)s]  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
# Attach filter to root logger so it covers agent.nodes, api, and all sub-loggers
for _h in logging.root.handlers:
    _h.addFilter(_TraceFilter())

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("opentelemetry").setLevel(logging.WARNING)

log = logging.getLogger("api")

# ── Global state ──────────────────────────────────────────────────────────────
_graph = None
_chunk_count = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph, _chunk_count
    log.info("Loading knowledge base and building agent graph…")
    from agent.graph import build_graph
    from rag.document_loader import load_sample_documents
    from rag.vectorstore import initialize_vectorstore

    docs = load_sample_documents()
    vectorstore = initialize_vectorstore(docs)
    _graph = build_graph(vectorstore)
    _chunk_count = vectorstore._collection.count()
    log.info("Agent ready — %d knowledge chunks loaded", _chunk_count)
    yield
    log.info("Shutting down.")


app = FastAPI(
    title="Employee Q&A Agent API",
    description="LangGraph + RAG + FastMCP pipeline exposed as REST endpoints.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request / Response models ─────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, description="The employee's question.")
    thread_id: Optional[str] = Field(
        default=None,
        description=(
            "Reuse an existing thread for follow-up questions. "
            "A new UUID is generated if omitted."
        ),
    )


class ReviewRequest(BaseModel):
    feedback: str = Field(
        default="",
        description="Reviewer corrections or extra context. Leave empty to approve as-is.",
    )


class InterruptData(BaseModel):
    question: str
    review_reasons: list[str]
    flagged_issues: list[str]
    differences: list[str]
    internal_summary: str
    web_summary: str
    confidence_score: float


class AnswerResult(BaseModel):
    final_answer: str
    citations: list[dict]
    answer_metadata: dict
    differences: list[str]
    flagged_issues: list[str]


class AskResponse(BaseModel):
    status: str                                      # "completed" | "awaiting_review"
    thread_id: str
    trace_url: Optional[str] = None                  # Langfuse trace URL when configured
    result: Optional[AnswerResult] = None            # set when status == "completed"
    interrupt_data: Optional[InterruptData] = None   # set when status == "awaiting_review"


class StatusResponse(BaseModel):
    thread_id: str
    status: str                                      # "completed" | "awaiting_review" | "not_found"


class HealthResponse(BaseModel):
    status: str
    agent_ready: bool
    knowledge_chunks: int


# ── Helpers ───────────────────────────────────────────────────────────────────

def _answer_result(state_values: dict) -> AnswerResult:
    return AnswerResult(
        final_answer=state_values.get("final_answer", ""),
        citations=state_values.get("citations", []),
        answer_metadata=state_values.get("answer_metadata", {}),
        differences=state_values.get("differences", []),
        flagged_issues=state_values.get("flagged_issues", []),
    )


def _interrupt_data(raw: dict) -> InterruptData:
    return InterruptData(
        question=raw.get("question", ""),
        review_reasons=raw.get("review_reasons", []),
        flagged_issues=raw.get("flagged_issues", []),
        differences=raw.get("differences", []),
        internal_summary=raw.get("internal_summary", ""),
        web_summary=raw.get("web_summary", ""),
        confidence_score=float(raw.get("confidence_score", 0.0)),
    )


def _get_interrupt_payload(graph_state) -> dict:
    """Safely extract the interrupt payload from a paused graph state."""
    try:
        raw = graph_state.tasks[0].interrupts[0].value
        return raw if isinstance(raw, dict) else {}
    except (IndexError, AttributeError):
        return {}


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Meta"])
def health():
    """Liveness check — confirms the agent and vectorstore are ready."""
    return HealthResponse(
        status="ok",
        agent_ready=_graph is not None,
        knowledge_chunks=_chunk_count,
    )


@app.post("/ask", response_model=AskResponse, tags=["Agent"])
def ask(body: AskRequest):
    """
    Submit a question to the agent.

    **Normal flow** — returns `status: completed` with the full answer.

    **Review flow** — returns `status: awaiting_review` with `interrupt_data`
    and a `thread_id`. Call `POST /review/{thread_id}` to submit feedback and
    receive the final answer.
    """
    if _graph is None:
        raise HTTPException(status_code=503, detail="Agent not initialised yet.")

    thread_id = body.thread_id or str(uuid.uuid4())
    cb = obs.get_callback_handler(thread_id, body.question)
    config = {"configurable": {"thread_id": thread_id}, "callbacks": [cb] if cb else []}
    _trace_id_var.set(thread_id)   # ← all log lines from here on include trace_id

    initial_state = {
        "question": body.question,
        "human_feedback": None,
        "web_searched": False,
        "needs_human_review": False,
    }

    log.info("[ask] START  thread=%s  question=%r", thread_id, body.question)

    try:
        result = _graph.invoke(initial_state, config=config)
        graph_state = _graph.get_state(config)
    except Exception as exc:
        log.exception("[ask] agent error")
        raise HTTPException(status_code=500, detail=str(exc))

    if graph_state.next:
        obs.flush()
        trace_url = obs.get_trace_url(thread_id)
        log.info("[ask] END STATE → awaiting_review  thread=%s  trace_url=%s", thread_id, trace_url)
        return AskResponse(
            status="awaiting_review",
            thread_id=thread_id,
            trace_url=trace_url,
            interrupt_data=_interrupt_data(_get_interrupt_payload(graph_state)),
        )

    obs.log_rag_span(thread_id, body.question, result.get("internal_docs", []))
    obs.log_mcp_event(
        thread_id,
        tools_used=["summarize_rank", "highlight_differences", "flag_issues"],
        flagged_issues=result.get("flagged_issues", []),
        differences=result.get("differences", []),
    )
    obs.flush()
    trace_url = obs.get_trace_url(thread_id)
    log.info("[ask] END STATE → completed  thread=%s  trace_url=%s", thread_id, trace_url)
    return AskResponse(
        status="completed",
        thread_id=thread_id,
        trace_url=trace_url,
        result=_answer_result(result),
    )


@app.post("/ask/stream", tags=["Agent"])
def ask_stream(body: AskRequest):
    """
    Stream the agent pipeline as Server-Sent Events.

    Each event is a JSON line prefixed with `data: ` followed by two newlines.
    Event types:
    - `{"type": "thread_id", "thread_id": "...", "trace_id": "..."}` — emitted first
    - `{"type": "node", "node": "<node_name>"}` — emitted as each node completes
    - `{"type": "completed", "thread_id": "...", "trace_id": "...", "result": {...}}` — final answer
    - `{"type": "awaiting_review", "thread_id": "...", "trace_id": "...", "interrupt_data": {...}}` — review needed
    - `{"type": "error", "detail": "..."}` — error occurred
    """
    if _graph is None:
        raise HTTPException(status_code=503, detail="Agent not initialised yet.")

    thread_id = body.thread_id or str(uuid.uuid4())
    trace_id = thread_id          # same ID — one conversation = one trace end-to-end
    cb = obs.get_callback_handler(thread_id, body.question)
    config = {"configurable": {"thread_id": thread_id}, "callbacks": [cb] if cb else []}

    initial_state = {
        "question": body.question,
        "human_feedback": None,
        "web_searched": False,
        "needs_human_review": False,
    }

    def _sse(payload: dict) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    def generate():
        # Set ContextVar here — generators run lazily after the handler returns,
        # so the outer set() is in a dead frame by the time nodes execute.
        _trace_id_var.set(thread_id)
        log.info("[ask_stream] START  thread=%s  question=%r", thread_id, body.question)
        yield _sse({"type": "thread_id", "thread_id": thread_id, "trace_id": trace_id})
        try:
            for chunk in _graph.stream(initial_state, config=config):
                node_name = next(iter(chunk.keys()), None)
                if node_name:
                    log.info("[ask_stream] node=%-24s thread=%s", node_name, thread_id)
                    yield _sse({"type": "node", "node": node_name})
        except Exception as exc:
            log.warning("[ask_stream] stream ended early: %s", exc)

        try:
            graph_state = _graph.get_state(config)
            if graph_state and graph_state.next:
                raw = _get_interrupt_payload(graph_state)
                log.info(
                    "[ask_stream] END STATE → awaiting_review  thread=%s  "
                    "reasons=%s  confidence=%.2f",
                    thread_id,
                    raw.get("review_reasons", []),
                    raw.get("confidence_score", 0.0),
                )
                obs.flush()
                trace_url = obs.get_trace_url(thread_id)
                yield _sse({
                    "type": "awaiting_review",
                    "thread_id": thread_id,
                    "trace_id": trace_id,
                    "trace_url": trace_url,
                    "interrupt_data": _interrupt_data(raw).model_dump(),
                })
            else:
                state_values = graph_state.values if graph_state else {}
                result = _answer_result(state_values)
                log.info(
                    "[ask_stream] END STATE → completed  thread=%s  "
                    "confidence=%s  sources=%s  internal_docs=%d  web_results=%d  "
                    "had_review=%s  flagged=%d",
                    thread_id,
                    state_values.get("answer_metadata", {}).get("confidence", "?"),
                    state_values.get("answer_metadata", {}).get("sources_used", "?"),
                    len(state_values.get("internal_docs", [])),
                    len(state_values.get("web_results", [])),
                    state_values.get("answer_metadata", {}).get("had_human_review", False),
                    len(state_values.get("flagged_issues", [])),
                )
                obs.log_rag_span(thread_id, body.question, state_values.get("internal_docs", []))
                obs.log_mcp_event(
                    thread_id,
                    tools_used=["summarize_rank", "highlight_differences", "flag_issues"],
                    flagged_issues=state_values.get("flagged_issues", []),
                    differences=state_values.get("differences", []),
                )
                obs.flush()
                trace_url = obs.get_trace_url(thread_id)
                yield _sse({
                    "type": "completed",
                    "thread_id": thread_id,
                    "trace_id": trace_id,
                    "trace_url": trace_url,
                    "result": result.model_dump(),
                })
        except Exception as exc:
            log.exception("[ask_stream] final state error thread=%s", thread_id)
            yield _sse({"type": "error", "detail": str(exc)})

    def _traced(gen):
        """
        Starlette's StreamingResponse calls next() on a sync generator from a
        thread-pool worker — a potentially DIFFERENT thread on each call.
        ContextVar is per-thread, so re-set it here before every next() so that
        all log lines inside generate() (including agent.nodes) always see trace_id.
        """
        while True:
            _trace_id_var.set(thread_id)   # set in whichever thread Starlette is using
            try:
                yield next(gen)
            except StopIteration:
                break

    return StreamingResponse(
        _traced(generate()),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post("/review/{thread_id}", response_model=AskResponse, tags=["Agent"])
def review(thread_id: str, body: ReviewRequest):
    """
    Submit human review feedback to resume a paused graph thread.

    Use the `thread_id` from a previous `POST /ask` response where
    `status == awaiting_review`.

    - Send `feedback` with corrections or extra context.
    - Send an empty string (or omit) to approve the answer as-is.
    """
    if _graph is None:
        raise HTTPException(status_code=503, detail="Agent not initialised yet.")

    cb = obs.get_callback_handler(thread_id, "review_resume")
    config = {"configurable": {"thread_id": thread_id}, "callbacks": [cb] if cb else []}
    final_feedback = body.feedback.strip() or "Approved. No changes required."
    _trace_id_var.set(thread_id)  # ← all log lines from here on include trace_id

    # trace_id == thread_id — same conversation, same trace
    log.info("[review] START  thread=%s  feedback=%r", thread_id, final_feedback[:80])

    try:
        result = _graph.invoke(Command(resume=final_feedback), config=config)
        graph_state = _graph.get_state(config)
    except Exception as exc:
        log.exception("[review] agent error for thread=%s", thread_id)
        raise HTTPException(status_code=500, detail=str(exc))

    # Still paused (edge case — e.g. multiple interrupt points)
    if graph_state.next:
        log.warning("[review] thread=%s  trace_id=%s  still paused after resume", thread_id, thread_id)
        return AskResponse(
            status="awaiting_review",
            thread_id=thread_id,
            interrupt_data=_interrupt_data(_get_interrupt_payload(graph_state)),
        )

    obs.score_human_feedback(thread_id, final_feedback)
    obs.flush()
    trace_url = obs.get_trace_url(thread_id)
    log.info("[review] END STATE → completed  thread=%s  trace_url=%s", thread_id, trace_url)
    return AskResponse(
        status="completed",
        thread_id=thread_id,
        trace_url=trace_url,
        result=_answer_result(result),
    )


@app.get("/status/{thread_id}", response_model=StatusResponse, tags=["Agent"])
def status(thread_id: str):
    """
    Check the current state of a graph thread.

    Returns one of:
    - `completed`       — answer is ready (or was never interrupted)
    - `awaiting_review` — graph is paused, waiting for `POST /review/{thread_id}`
    - `not_found`       — unknown thread_id
    """
    if _graph is None:
        raise HTTPException(status_code=503, detail="Agent not initialised yet.")

    config = {"configurable": {"thread_id": thread_id}}
    try:
        graph_state = _graph.get_state(config)
    except Exception:
        return StatusResponse(thread_id=thread_id, status="not_found")

    if graph_state is None or not graph_state.values:
        return StatusResponse(thread_id=thread_id, status="not_found")

    thread_status = "awaiting_review" if graph_state.next else "completed"
    return StatusResponse(thread_id=thread_id, status=thread_status)
