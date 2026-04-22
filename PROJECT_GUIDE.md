# Employee Q&A AI Agent — Project Reference Document

> **Purpose:** A single document to understand the entire project — what it does, how it is built, how every piece connects, and why each design decision was made.

---

## Table of Contents

1. [What Does This Project Do?](#1-what-does-this-project-do)
2. [Technology Stack](#2-technology-stack)
3. [Project Structure](#3-project-structure)
4. [Architecture Overview](#4-architecture-overview)
5. [Agentic Flow — Step by Step](#5-agentic-flow--step-by-step)
6. [RAG — How Internal Search Works](#6-rag--how-internal-search-works)
7. [FastMCP — How Tools Are Built and Called](#7-fastmcp--how-tools-are-built-and-called)
8. [LangGraph — How the Graph Orchestrates Everything](#8-langgraph--how-the-graph-orchestrates-everything)
9. [Human-in-the-Loop](#9-human-in-the-loop)
10. [State — The Shared Memory](#10-state--the-shared-memory)
11. [MCP Transport Options](#11-mcp-transport-options)
12. [Streamlit UI](#12-streamlit-ui)
13. [Observability — Langfuse + Terminal Logging](#13-observability--langfuse--terminal-logging)
14. [File-by-File Reference](#14-file-by-file-reference)
15. [Configuration Reference](#15-configuration-reference)
16. [How to Add New Documents](#16-how-to-add-new-documents)
17. [How to Add a New MCP Tool](#17-how-to-add-a-new-mcp-tool)
18. [Common Errors & Fixes](#18-common-errors--fixes)

---

## 1. What Does This Project Do?

An employee at a company types a work-related question like _"How do I set up VPN access?"_

The system:
1. **Searches internal company documents** (wikis, policy PDFs, HR guides) using vector similarity search.
2. **Decides if internal data is enough**. If not, it searches the internet via Tavily.
3. **Runs three MCP processing tools** to summarise, compare, and validate the information.
4. **Flags low-confidence or conflicting answers** and pauses for a human expert to review if needed.
5. **Produces a final answer** with bullet points, numbered steps, and citations.

---

## 2. Technology Stack

| Component | Technology | Why |
|---|---|---|
| **Orchestration** | LangGraph | Directed graph with conditional routing + human interrupt |
| **LLM** | Groq (`llama-3.1-8b-instant`) | Fast, free inference |
| **Internal Search** | ChromaDB | Local persistent vector database |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` | Local model — no API key needed |
| **Web Search** | Tavily (`langchain-tavily`) | Real-time internet retrieval |
| **MCP Tools** | FastMCP | Exposes processing logic as MCP-protocol tools |
| **UI** | Streamlit | Interactive chat interface with sidebar controls |
| **Observability** | Langfuse v4 | Per-question tracing, spans, scores on cloud dashboard |
| **Terminal Logging** | Python `logging` | Node-level debug logs printed to terminal |
| **Environment** | python-dotenv | `.env` key management |

---

## 3. Project Structure

```
agentic/
│
├── app.py                     ← Streamlit UI entry point (run with: streamlit run app.py)
├── main.py                    ← CLI entry point. Loads KB, builds graph, runs Q&A loop.
├── demo.py                    ← Non-interactive demo with 3 preset questions.
├── run_mcp_server.py          ← Standalone FastMCP server (stdio or SSE).
├── requirements.txt
├── .env                       ← Your real API keys (never commit this).
│
├── .streamlit/
│   └── config.toml            ← Streamlit config: disables file watcher, sets log level.
│
├── agent/                     ← LangGraph agent logic
│   ├── graph.py               ← Builds and compiles the StateGraph.
│   ├── nodes.py               ← Every node function (one per workflow step) + debug logs.
│   ├── state.py               ← AgentState TypedDict — shared memory for all nodes.
│   ├── mcp_client.py          ← Bridges sync LangGraph nodes → async FastMCP Client.
│   ├── observability.py       ← Langfuse v4 tracing helpers (traces, spans, scores).
│   └── utils.py               ← Robust JSON parser for LLM responses.
│
├── processors/                ← FastMCP server + 3 tool implementations
│   ├── server.py              ← FastMCP server. Registers 3 tools via @mcp.tool.
│   ├── summarizer.py          ← Tool 1 logic: summarise + rank sources.
│   ├── comparator.py          ← Tool 2 logic: find agreements and conflicts.
│   └── validator.py           ← Tool 3 logic: flag issues + confidence score.
│
├── rag/                       ← Internal knowledge base management
│   ├── document_loader.py     ← Reads .txt/.md files, chunks them, adds metadata.
│   └── vectorstore.py         ← ChromaDB init, reuse, and reset helpers.
│
└── data/
    ├── sample_docs/           ← Internal company documents (editable)
    │   ├── vpn_setup.txt
    │   ├── employee_handbook.txt
    │   ├── it_policies.txt
    │   ├── expense_policy.txt
    │   ├── leave_policy.txt
    │   └── onboarding_guide.txt
    └── chroma_db/             ← Auto-created. Persisted vector store on disk.
```

---

## 4. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         main.py (CLI)                           │
│  Employee types question → graph.invoke() → display results     │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                  LangGraph StateGraph  (agent/graph.py)         │
│                                                                 │
│  ┌─────────────┐   ┌──────────────┐   ┌─────────────────────┐  │
│  │analyze_query│──►│search_internal│──►│evaluate_sufficiency │  │
│  └─────────────┘   └──────────────┘   └──────────┬──────────┘  │
│                     ChromaDB/HF ▲     sufficient? │             │
│                     Embeddings  │    ┌────────────┴──────────┐  │
│                                 │    │      search_web        │  │
│                                 │    │   (Tavily internet)    │  │
│                                 │    └────────────┬──────────┘  │
│                                 │                 │             │
│            ┌────────────────────┴─────────────────┘            │
│            │                                                    │
│  ┌─────────▼───────┐   FastMCP in-process call                 │
│  │ summarize_rank  │──────────────────────────────────────────► │
│  └─────────┬───────┘                           processors/      │
│            │                                   server.py        │
│  ┌─────────▼──────────────┐    ◄── Tool 1: summarize_and_rank  │
│  │ highlight_differences  │──► ◄── Tool 2: highlight_differences│
│  └─────────┬──────────────┘                                     │
│            │                   ◄── Tool 3: flag_issues          │
│  ┌─────────▼────────┐                                           │
│  │   flag_issues    │                                           │
│  └─────────┬────────┘                                           │
│            │                                                    │
│   ┌────────┴──────────────┐                                     │
│   │ needs_human_review?   │                                     │
│   └──────┬────────┬───────┘                                     │
│       Yes│        │No                                           │
│  ┌───────▼──┐  ┌──▼──────────────┐                             │
│  │ human_   │  │ synthesize_     │                             │
│  │ review   │  │ answer          │                             │
│  │(interrupt)│  │(Groq LLM)      │                             │
│  └───────┬──┘  └──┬──────────────┘                             │
│          └────────┘                                            │
│                   │                                            │
│          ┌────────▼────────┐                                   │
│          │  Final Answer   │                                   │
│          │  + Citations    │                                   │
│          │  + Metadata     │                                   │
│          └─────────────────┘                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Agentic Flow — Step by Step

### Step 1 — `analyze_query` node
**File:** `agent/nodes.py`

- Sends the question to Groq LLM.
- Extracts: `intent` (one sentence goal) and `keywords` (3-6 search terms).
- Writes to state: `query_intent`, `query_keywords`.

```
"How do I set up VPN access?"
        ↓ Groq
intent: "Employee wants to install and configure VPN"
keywords: ["VPN", "GlobalConnect", "install", "setup", "remote access"]
```

---

### Step 2 — `search_internal` node
**File:** `agent/nodes.py` + `rag/vectorstore.py`

- Converts the question to a vector using HuggingFace `all-MiniLM-L6-v2`.
- Runs cosine similarity search in ChromaDB against 45 pre-chunked internal docs.
- Returns top 5 most similar chunks with their similarity scores.
- Writes to state: `internal_docs`, `internal_context`.

---

### Step 3 — `evaluate_sufficiency` node
**File:** `agent/nodes.py`

- Sends the question + internal context to Groq.
- Groq returns JSON: `{"sufficient": true/false}`.
- If `true` → skip web search.
- If `false` → go to Step 4.
- Writes to state: `internal_sufficient`.

---

### Step 4 — `search_web` node _(only if needed)_
**File:** `agent/nodes.py`

- Calls `TavilySearch` with: `"workplace policy: {query_intent}"`.
- Returns up to 5 web results with URL, title, content snippet.
- Writes to state: `web_results`, `web_context`, `web_searched`.

---

### Step 5 — `summarize_rank` node → **MCP Tool 1**
**File:** `agent/nodes.py` → `agent/mcp_client.py` → `processors/server.py` → `processors/summarizer.py`

- Calls FastMCP tool `summarize_and_rank` in-process.
- Tool sends internal + web context to Groq → gets bullet-point summaries.
- Ranks all sources by relevance score (lower ChromaDB distance = higher rank).
- Writes to state: `internal_summary`, `web_summary`, `ranked_sources`.

---

### Step 6 — `highlight_differences` node → **MCP Tool 2**
**File:** `agent/nodes.py` → `processors/comparator.py`

- Calls FastMCP tool `highlight_differences` in-process.
- Groq compares the two summaries and returns JSON:
  ```json
  {
    "differences": ["Internal says X, web says Y"],
    "agreements":  ["Both confirm Z"]
  }
  ```
- Writes to state: `differences`, `agreements`.

---

### Step 7 — `flag_issues` node → **MCP Tool 3**
**File:** `agent/nodes.py` → `processors/validator.py`

- Calls FastMCP tool `flag_issues` in-process.
- Runs heuristic checks (unknown dates, many conflicts, no internal docs found).
- Also runs Groq deep-check for outdated signals.
- Produces a confidence score (0.0 → 1.0).
- Writes to state: `flagged_issues`, `needs_human_review`, `review_reasons`, `confidence_score`.

---

### Step 8 — `human_review` node _(only if flagged)_
**File:** `agent/nodes.py`

- Calls LangGraph `interrupt()` — **suspends the graph**.
- Prints review details to terminal: reasons, conflicts, confidence score.
- Waits for the user to type feedback and press Enter.
- Resumes with `Command(resume=feedback)`.
- Writes to state: `human_feedback`.

---

### Step 9 — `synthesize_answer` node
**File:** `agent/nodes.py`

- Combines: internal summary + web summary + differences + human feedback.
- Sends all to Groq with system prompt: _"Lead with the most important action. Use numbered steps. Under 350 words."_
- Builds citation list from `ranked_sources`.
- Writes to state: `final_answer`, `citations`, `answer_metadata`.

---

## 6. RAG — How Internal Search Works

**RAG = Retrieval-Augmented Generation**

```
Load documents
      ↓
document_loader.py reads all .txt/.md files in data/sample_docs/
      ↓
RecursiveCharacterTextSplitter → 800-char chunks, 150-char overlap
      ↓
Each chunk → HuggingFace embedding (384-dim vector)
      ↓
Stored in ChromaDB (persisted to data/chroma_db/ on disk)

────────────────────────────────

At query time:
      ↓
Question → HuggingFace embedding → 384-dim vector
      ↓
ChromaDB cosine similarity search → top 5 matching chunks
      ↓
Chunks returned with distance scores (lower = more similar)
      ↓
Used as context for LLM answer generation
```

**Key files:**
- `rag/document_loader.py` — loads, chunks, adds metadata (title, date, filename)
- `rag/vectorstore.py` — creates/reuses ChromaDB, uses `HuggingFaceEmbeddings`

**Why ChromaDB is reused:** On first run, 45 chunks are embedded and stored. On subsequent runs, `vectorstore._collection.count()` detects existing data and skips re-embedding. Delete `data/chroma_db/` to force a rebuild.

---

## 7. FastMCP — How Tools Are Built and Called

### Building a Tool (`processors/server.py`)

```python
from fastmcp import FastMCP

mcp = FastMCP("Employee Q&A Tools")   # create the server

@mcp.tool(description="Summarise and rank sources...")
def summarize_and_rank(question: str, internal_context: str, ...) -> str:
    result = _summarize(...)           # call the real logic
    return json.dumps(result)          # MCP protocol requires string return
```

- `@mcp.tool` registers the function into the tool registry at import time.
- The tool name (used by the client) = the function name.
- Parameters become the tool's JSON schema automatically.
- Must return a `str` (JSON string) — the client parses it back to dict.

---

### Calling a Tool (`agent/mcp_client.py`)

```python
async def _call_tool_async(tool_name, arguments):
    async with Client(_get_server()) as client:      # connect in-process
        result = await client.call_tool(tool_name, arguments)

    # FastMCP 3.x wraps result in CallToolResult object
    content_list = getattr(result, "content", None)
    raw_text = content_list[0].text                  # extract JSON string
    return json.loads(raw_text)                      # parse back to dict
```

- `Client(_get_server())` — passes the Python object directly (in-process).
- No HTTP, no subprocess — just function calls under the hood.
- The `_get_server()` singleton is lazy-loaded once and cached.

---

### Why MCP Instead of Direct Function Calls?

| Direct call | Via FastMCP |
|---|---|
| `summarizer.summarize_and_rank(...)` | `call_mcp_tool("summarize_and_rank", {...})` |
| Coupled to Python import | Decoupled — any MCP client can call the same tool |
| No schema | Auto-generated JSON schema + description |
| Single process only | Same tools serve Claude Desktop, Cursor, HTTP clients |

---

## 8. LangGraph — How the Graph Orchestrates Everything

### Graph = State Machine

LangGraph is a **directed graph** where:
- Each **node** = one Python function that reads/writes state.
- Each **edge** = the execution order.
- **Conditional edges** = branching logic (if/else routing).

### How State Flows

```python
# graph.py builds the graph
builder = StateGraph(AgentState)

builder.add_node("analyze_query", analyze_query_node)
builder.add_node("search_internal", make_search_internal_node(vectorstore))
...

# Regular edge — always goes next
builder.add_edge("analyze_query", "search_internal")

# Conditional edge — routes based on state value
builder.add_conditional_edges(
    "evaluate_sufficiency",
    _route_sufficiency,                         # function returns a string key
    {"sufficient": "summarize_rank",            # if "sufficient" → skip web
     "insufficient": "search_web"},             # if "insufficient" → search web
)
```

### Checkpointer — Required for Human Interrupt

```python
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
```

`MemorySaver` saves the entire state to memory at each node. This is what lets `interrupt()` pause execution and `Command(resume=...)` resume it from exactly where it stopped.

---

## 9. Human-in-the-Loop

When `flag_issues_node` sets `needs_human_review = True`, the graph routes to `human_review_node`:

```python
# nodes.py
def human_review_node(state):
    review_payload = { ...flags, reasons, summaries... }
    feedback = interrupt(review_payload)      # ← GRAPH PAUSES HERE
    return {"human_feedback": feedback}
```

```python
# main.py — handles the pause
while graph_state.next:                       # graph is paused
    interrupt_data = graph_state.tasks[0].interrupts[0].value
    # print review details to terminal
    feedback = input("Reviewer feedback > ")
    result = graph.invoke(
        Command(resume=feedback),             # ← GRAPH RESUMES
        config=config
    )
```

The reviewer's feedback is added to state as `human_feedback` and included in the final answer synthesis.

---

## 10. State — The Shared Memory

`agent/state.py` defines `AgentState` as a `TypedDict`. Every node reads input from and writes output to this shared dict. LangGraph merges each node's returned dict into the state automatically.

```
Question enters → State grows as nodes add to it → Final answer exits

State at END contains everything:
  question, intent, keywords,
  internal_docs, internal_context, internal_sufficient,
  web_results, web_context, web_searched,
  internal_summary, web_summary, ranked_sources,
  differences, agreements,
  flagged_issues, needs_human_review, confidence_score, review_reasons,
  human_feedback,
  final_answer, citations, answer_metadata
```

---

## 11. MCP Transport Options

The same `processors/server.py` can run in three modes:

### Mode 1 — In-Process (used by `main.py`)
```python
# mcp_client.py passes the Python object directly
async with Client(mcp_server_object) as client: ...
```
No port. No protocol. Fastest.

### Mode 2 — stdio (for Claude Desktop / Cursor)
```powershell
python run_mcp_server.py        # default
```
Client spawns server as subprocess. Communication via stdin/stdout pipes.

Claude Desktop `config.json`:
```json
{
  "mcpServers": {
    "employee-qa": {
      "command": "python",
      "args": ["C:/path/to/agentic/run_mcp_server.py"],
      "env": { "GROQ_API_KEY": "gsk-..." }
    }
  }
}
```

### Mode 3 — SSE/HTTP (for web clients, multi-user)
```powershell
python run_mcp_server.py --transport sse --port 8000
# http://localhost:8000
```
Client connects via HTTP. Multiple simultaneous clients supported.

---

## 12. Streamlit UI

Run the app with:
```powershell
streamlit run app.py
```

### Features
- **Chat interface** — persistent message history across questions in the same session.
- **Human-in-the-loop UI** — when the agent flags an answer for review, the UI shows the flagged issues inline and prompts for reviewer feedback without leaving the browser.
- **Answer rendering** — structured display: answer text, confidence badge, ranked source citations (Internal/Web tagged), differences expander, flagged issues expander.
- **Sidebar** — live configuration display (model, API key status), example question buttons, observability status with trace links, and a Clear conversation button.

### Session State Keys

| Key | Purpose |
|---|---|
| `messages` | Full chat history list |
| `thread_id` | Stable LangGraph checkpointer ID — one per browser session |
| `awaiting_review` | `True` when graph is paused at `human_review` interrupt |
| `interrupt_data` | Payload from the interrupt (reasons, flagged issues, confidence) |
| `last_config` | LangGraph config dict — reused on `Command(resume=...)` |
| `last_trace_id` | Langfuse trace ID of the most recent question |
| `last_trace_url` | Direct Langfuse dashboard URL for the most recent trace |
| `trace_history` | List of `{question, trace_id, trace_url}` for all questions this session |

### Key Design Decisions
- **`thread_id` vs `trace_id` are separate** — `thread_id` is stable per session (LangGraph memory), `trace_id` is a new UUID per question (Langfuse tracing). This gives Langfuse one trace per question while keeping LangGraph conversation memory intact.
- **`@st.cache_resource`** wraps the graph/vectorstore build so the heavy work (embedding model load, ChromaDB init) runs once per server process, not once per page reload.

### Streamlit Config (`.streamlit/config.toml`)
```toml
[server]
fileWatcherType = "none"   # prevents scanning transformers submodules (avoids torchvision errors)

[logger]
level = "warning"          # only show warnings+ from Streamlit itself

[runner]
fastReruns = true
```

---

## 13. Observability — Langfuse + Terminal Logging

### Overview

Two complementary layers:
1. **Langfuse** (cloud dashboard) — per-question traces with spans, LLM calls, token counts, scores.
2. **Python `logging`** (terminal) — per-node debug lines printed as code runs.

### Langfuse Setup

Get keys from [cloud.langfuse.com](https://cloud.langfuse.com) → Project Settings → API Keys, then add to `.env`:
```env
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

When keys are absent every function in `observability.py` is a no-op — the app runs normally without tracing.

### What Gets Traced

| Langfuse entry | When logged | What it contains |
|---|---|---|
| **Root trace** (auto) | Each question | All LLM calls + node spans via LangChain callback |
| `rag_internal_search` span | After `search_internal` node | Chunk count, top doc titles, relevance scores |
| `mcp_tools_summary` event | After all 3 MCP tools | Tools called, flagged issue count, difference count |
| `human_review_triggered` event | When graph pauses for review | Review reasons, confidence score (WARNING level) |
| `human_review` score | After reviewer submits feedback | 1.0 = approved, 0.5 = corrections provided |

### Trace ID Strategy

```
Per browser session:  thread_id = uuid4()   (stable — LangGraph memory)
Per question asked:   trace_id  = uuid4()   (new each time — Langfuse trace)
```

Langfuse v4 requires trace IDs as 32 lowercase hex chars. The `_to_trace_id()` helper in `observability.py` strips dashes from UUIDs automatically.

### Viewing Traces

- **Sidebar "All traces" expander** — lists every question asked this session as a clickable Langfuse dashboard link.
- **Direct URL** — each link goes to `cloud.langfuse.com/project/.../traces/<trace_id>` showing the full execution tree with latency, tokens, and custom spans.

### Terminal Logging

`app.py` configures `logging.basicConfig(level=INFO)` at startup. Every node in `nodes.py` has a dedicated logger `agent.nodes` that prints:

```
13:24:01  INFO     agent.nodes — [analyze_query] question='what is the leave policy?'
13:24:02  INFO     agent.nodes — [analyze_query] intent='Employee wants...' keywords=[...]
13:24:02  INFO     agent.nodes — [search_internal] searching ChromaDB for: '...'
13:24:02  INFO     agent.nodes — [search_internal] found 5 docs, top score=0.2341
13:24:03  INFO     agent.nodes — [evaluate_sufficiency] sufficient=True  reason='...'
13:24:03  INFO     agent.nodes — [summarize_rank] calling MCP tool summarize_and_rank
13:24:04  INFO     agent.nodes — [summarize_rank] done, confidence=0.87
13:24:04  INFO     agent.nodes — [highlight_differences] found 0 differences
13:24:05  INFO     agent.nodes — [flag_issues] flagged 0 issues, needs_review=False
13:24:05  INFO     agent.nodes — [synthesize_answer] composing final answer for: '...'
```

Noisy libraries (httpx, chromadb, langchain, opentelemetry) are set to WARNING to keep the output clean.

### `agent/observability.py` API

| Function | Description |
|---|---|
| `is_enabled()` | Returns `True` when both Langfuse keys are set |
| `get_callback_handler(trace_id, question)` | LangChain callback handler — pass in `config["callbacks"]` |
| `log_rag_span(trace_id, question, docs)` | Logs retrieval span with doc metadata |
| `log_mcp_event(trace_id, tools, issues, diffs)` | Logs MCP tool summary event |
| `log_human_review_event(trace_id, reasons, score)` | Logs human review trigger at WARNING level |
| `score_human_feedback(trace_id, feedback)` | Posts 1.0/0.5 score after reviewer submits |
| `get_trace_url(trace_id)` | Returns direct Langfuse dashboard URL |
| `flush()` | Flushes all pending events to Langfuse server |

---

## 14. File-by-File Reference

| File | Role | Key Functions |
|---|---|---|
| `app.py` | Streamlit UI entry point | `load_agent()`, `render_result()`, chat loop, human review UI |
| `main.py` | CLI entry point | `main()`, `run_question()`, `display_results()` |
| `demo.py` | Runs 3 preset questions | `run_question()` called 3 times |
| `run_mcp_server.py` | Standalone MCP server | `main()` with `--transport`, `--port`, `--list-tools` |
| `agent/graph.py` | Builds LangGraph | `build_graph(vectorstore)` → compiled graph |
| `agent/nodes.py` | All 9 node functions + logs | One function per workflow step; `log = logging.getLogger("agent.nodes")` |
| `agent/state.py` | Shared state definition | `AgentState` TypedDict |
| `agent/mcp_client.py` | Sync/async bridge | `call_mcp_tool()`, `list_mcp_tools()` |
| `agent/observability.py` | Langfuse v4 tracing | `get_callback_handler()`, `log_rag_span()`, `log_mcp_event()`, `get_trace_url()` |
| `agent/utils.py` | JSON parser | `parse_llm_json()` — handles markdown blocks, nested objects |
| `processors/server.py` | FastMCP server | `mcp = FastMCP(...)`, 3 `@mcp.tool` functions |
| `processors/summarizer.py` | Tool 1 logic | `summarize_and_rank()` |
| `processors/comparator.py` | Tool 2 logic | `highlight_differences()` |
| `processors/validator.py` | Tool 3 logic | `flag_issues()` |
| `rag/document_loader.py` | Load + chunk docs | `load_sample_documents()` |
| `rag/vectorstore.py` | ChromaDB management | `initialize_vectorstore()`, `reset_vectorstore()` |
| `.streamlit/config.toml` | Streamlit server config | File watcher disabled, log level, fast reruns |

---

## 15. Configuration Reference

All settings live in `.env`:

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | **Yes** | — | Groq API key — [console.groq.com](https://console.groq.com) |
| `TAVILY_API_KEY` | No | — | Web search — [app.tavily.com](https://app.tavily.com). Skipped if absent. |
| `MODEL_NAME` | No | `llama-3.1-8b-instant` | Any Groq model with JSON mode |
| `CHROMA_PERSIST_DIR` | No | `./data/chroma_db` | Where ChromaDB stores vectors |
| `EMBEDDING_MODEL` | No | `sentence-transformers/all-MiniLM-L6-v2` | Local HuggingFace embedding model |
| `LANGFUSE_PUBLIC_KEY` | No | — | Langfuse public key — enables Langfuse tracing |
| `LANGFUSE_SECRET_KEY` | No | — | Langfuse secret key |
| `LANGFUSE_HOST` | No | `https://cloud.langfuse.com` | Langfuse server URL |
| `TRANSFORMERS_VERBOSITY` | No | — | Set to `error` to suppress transformers warnings |
| `TRANSFORMERS_NO_ADVISORY_WARNINGS` | No | — | Set to `1` to suppress `__path__` advisory messages |
| `TOKENIZERS_PARALLELISM` | No | — | Set to `false` to suppress tokenizer fork warnings |

---

## 16. How to Add New Documents

1. Drop any `.txt` or `.md` file into `data/sample_docs/`
2. Delete `data/chroma_db/` to clear the old vector store
3. Run `python main.py` — it rebuilds automatically

```powershell
copy "my_new_policy.txt" "data\sample_docs\"
Remove-Item -Recurse -Force "data\chroma_db"
python main.py
```

The loader picks up the publication date automatically if the file contains lines like:
```
Last Updated: March 2024
Version: 2.1
```

---

## 17. How to Add a New MCP Tool

**Step 1** — Write the logic in a new file `processors/my_tool.py`:
```python
def my_analysis(question: str, context: str) -> dict:
    # ... your logic
    return {"result": "..."}
```

**Step 2** — Register it in `processors/server.py`:
```python
from processors.my_tool import my_analysis as _my_analysis

@mcp.tool(description="What this tool does...")
def my_analysis(question: str, context: str) -> str:
    return json.dumps(_my_analysis(question, context))
```

**Step 3** — Add a node in `agent/nodes.py`:
```python
def my_analysis_node(state: AgentState) -> dict:
    return call_mcp_tool("my_analysis", {
        "question": state.get("question", ""),
        "context":  state.get("internal_context", ""),
    })
```

**Step 4** — Wire it into the graph in `agent/graph.py`:
```python
builder.add_node("my_analysis", my_analysis_node)
builder.add_edge("flag_issues", "my_analysis")
builder.add_edge("my_analysis", "human_review")  # or wherever it fits
```

**Step 5** — Add the new output keys to `agent/state.py`:
```python
my_analysis_result: dict
```

---

## 18. Common Errors & Fixes

| Error | Cause | Fix |
|---|---|---|
| `Invalid API Key` | Wrong/duplicate key in `.env` | Check for duplicate `GROQ_API_KEY=` lines; remove trailing `.` from Tavily key |
| `CallToolResult is not subscriptable` | FastMCP 3.x changed return type | Fixed — `mcp_client.py` reads `.content` attribute |
| `Import "mcp" could not be resolved` | Local `mcp/` folder shadows SDK | Fixed — renamed to `processors/` |
| `TavilySearchResults deprecated` | Old `langchain_community` import | Fixed — uses `langchain_tavily.TavilySearch` |
| Empty answers | ChromaDB has no docs | Delete `data/chroma_db/` and rerun |
| Slow first start | HuggingFace model downloading | One-time ~90MB download; cached after |
| `GROQ_MODEL` not recognised | Wrong env var name | Must be `MODEL_NAME` (not `GROQ_MODEL`) |
| `No module named 'langfuse.callback'` | Langfuse v3+ removed old import | Fixed — use `from langfuse.langchain import CallbackHandler` |
| `LangchainCallbackHandler has no attribute 'get_trace_id'` | Langfuse v4 removed instance methods | Fixed — trace ID managed separately, URL via `lf.get_trace_url()` |
| `invalid literal for int() with base 16` | UUID with dashes passed as trace ID | Fixed — `_to_trace_id()` strips dashes before every Langfuse call |
| Torchvision errors in terminal | Streamlit file watcher scans transformers submodules | Fixed — `.streamlit/config.toml` sets `fileWatcherType = "none"` |
| Langfuse shows "inactive" | Keys in `.env` not saved to disk | Ensure `.env` file is saved; `load_dotenv()` reads from disk not editor buffer |
| All questions share one trace ID | `thread_id` reused as trace ID | Fixed — new `uuid4()` generated per question for Langfuse, `thread_id` kept stable for LangGraph |
