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
12. [File-by-File Reference](#12-file-by-file-reference)
13. [Configuration Reference](#13-configuration-reference)
14. [How to Add New Documents](#14-how-to-add-new-documents)
15. [How to Add a New MCP Tool](#15-how-to-add-a-new-mcp-tool)
16. [Common Errors & Fixes](#16-common-errors--fixes)

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
| **Environment** | python-dotenv | `.env` key management |

---

## 3. Project Structure

```
agentic/
│
├── main.py                    ← CLI entry point. Loads KB, builds graph, runs Q&A loop.
├── demo.py                    ← Non-interactive demo with 3 preset questions.
├── run_mcp_server.py          ← Standalone FastMCP server (stdio or SSE).
├── requirements.txt
├── .env                       ← Your real API keys (never commit this).
├── .env.example               ← Template showing all required variables.
│
├── agent/                     ← LangGraph agent logic
│   ├── graph.py               ← Builds and compiles the StateGraph.
│   ├── nodes.py               ← Every node function (one per workflow step).
│   ├── state.py               ← AgentState TypedDict — shared memory for all nodes.
│   ├── mcp_client.py          ← Bridges sync LangGraph nodes → async FastMCP Client.
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

## 12. File-by-File Reference

| File | Role | Key Functions |
|---|---|---|
| `main.py` | Entry point, CLI loop | `main()`, `run_question()`, `display_results()` |
| `demo.py` | Runs 3 preset questions | `run_question()` called 3 times |
| `run_mcp_server.py` | Standalone MCP server | `main()` with `--transport`, `--port`, `--list-tools` |
| `agent/graph.py` | Builds LangGraph | `build_graph(vectorstore)` → compiled graph |
| `agent/nodes.py` | All 9 node functions | One function per workflow step |
| `agent/state.py` | Shared state definition | `AgentState` TypedDict |
| `agent/mcp_client.py` | Sync/async bridge | `call_mcp_tool()`, `list_mcp_tools()` |
| `agent/utils.py` | JSON parser | `parse_llm_json()` — handles markdown blocks, nested objects |
| `processors/server.py` | FastMCP server | `mcp = FastMCP(...)`, 3 `@mcp.tool` functions |
| `processors/summarizer.py` | Tool 1 logic | `summarize_and_rank()` |
| `processors/comparator.py` | Tool 2 logic | `highlight_differences()` |
| `processors/validator.py` | Tool 3 logic | `flag_issues()` |
| `rag/document_loader.py` | Load + chunk docs | `load_sample_documents()` |
| `rag/vectorstore.py` | ChromaDB management | `initialize_vectorstore()`, `reset_vectorstore()` |

---

## 13. Configuration Reference

All settings live in `.env`:

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | **Yes** | — | Groq API key — [console.groq.com](https://console.groq.com) |
| `TAVILY_API_KEY` | No | — | Web search — [app.tavily.com](https://app.tavily.com). Skipped if absent. |
| `MODEL_NAME` | No | `llama-3.1-8b-instant` | Any Groq model with JSON mode |
| `CHROMA_PERSIST_DIR` | No | `./data/chroma_db` | Where ChromaDB stores vectors |
| `EMBEDDING_MODEL` | No | `sentence-transformers/all-MiniLM-L6-v2` | Local HuggingFace embedding model |

---

## 14. How to Add New Documents

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

## 15. How to Add a New MCP Tool

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

## 16. Common Errors & Fixes

| Error | Cause | Fix |
|---|---|---|
| `Invalid API Key` | Wrong/duplicate key in `.env` | Check for duplicate `GROQ_API_KEY=` lines; remove trailing `.` from Tavily key |
| `CallToolResult is not subscriptable` | FastMCP 3.x changed return type | Fixed — `mcp_client.py` reads `.content` attribute |
| `Import "mcp" could not be resolved` | Local `mcp/` folder shadows SDK | Fixed — renamed to `processors/` |
| `TavilySearchResults deprecated` | Old `langchain_community` import | Fixed — uses `langchain_tavily.TavilySearch` |
| Empty answers | ChromaDB has no docs | Delete `data/chroma_db/` and rerun |
| Slow first start | HuggingFace model downloading | One-time ~90MB download; cached after |
| `GROQ_MODEL` not recognised | Wrong env var name | Must be `MODEL_NAME` (not `GROQ_MODEL`) |
