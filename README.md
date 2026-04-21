# Employee Q&A AI Agent

An AI agent that helps employees quickly find answers to work-related questions by combining internal documentation (RAG) with real-time internet search, orchestrated via **LangGraph** and processed through **FastMCP** tools.

---

## Architecture

```
Employee Question
       │
       ▼
┌──────────────────┐
│  analyze_query   │  Extract intent & search keywords
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ search_internal  │  ChromaDB vector search over internal docs
└────────┬─────────┘
         │
         ▼
┌─────────────────────────┐
│  evaluate_sufficiency   │  Is internal data enough?
└────────┬────────────────┘
         │
    ┌────┴─────┐
    │ No       │ Yes
    ▼          │
┌──────────┐   │
│search_web│   │  Tavily internet search
└────┬─────┘   │
     └────┬────┘
          │
          ▼
┌──────────────────────────────┐
│  [FastMCP] summarize_and_rank │  Summarise + rank all sources
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────────┐
│  [FastMCP] highlight_differences  │  Internal vs external conflicts
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────┐
│     [FastMCP] flag_issues    │  Outdated docs, confidence score
└──────────────┬───────────────┘
               │
      ┌────────┴────────┐
      │ Needs review?   │
      ▼ Yes             ▼ No
┌─────────────┐  ┌──────────────────┐
│human_review │  │ synthesize_answer │
│ (interrupt) │  └────────┬─────────┘
└──────┬──────┘           │
       └──────────────────┘
                │
                ▼
         Final Answer
     (with citations, flags)
```

### Key Components

| Layer | Technology | Purpose |
|---|---|---|
| Orchestration | LangGraph `StateGraph` | Directed graph with conditional routing |
| Human-in-the-loop | LangGraph `interrupt` / `Command(resume)` | Pause & resume for expert review |
| Internal search | ChromaDB + HuggingFace Embeddings | Vector similarity search over internal docs (local, no key needed) |
| Web search | Tavily | Real-time external information retrieval |
| MCP Tools | FastMCP | Three processing pipeline tools exposed as MCP |
| LLM | Groq (llama-3.3-70b-versatile default) | Query analysis, summarisation, synthesis |

---

## Project Structure

```
agentic/
├── main.py                    # Interactive CLI — ask questions in the terminal
├── run_mcp_server.py          # Run FastMCP as a standalone server (Claude Desktop, Cursor, SSE)
├── requirements.txt
├── .env.example               # Copy to .env and fill in API keys
│
├── agent/
│   ├── graph.py               # LangGraph StateGraph definition
│   ├── nodes.py               # 9 node functions (one per workflow step)
│   ├── state.py               # AgentState TypedDict
│   ├── mcp_client.py          # Sync wrapper around async FastMCP Client
│   └── utils.py               # Robust JSON parser for LLM responses
│
├── processors/                # FastMCP server + tool implementations
│   ├── server.py              # FastMCP server — 3 registered tools
│   ├── summarizer.py          # Tool 1: Summarise & rank sources
│   ├── comparator.py          # Tool 2: Highlight internal vs external differences
│   └── validator.py           # Tool 3: Validate, flag issues, score confidence
│
├── rag/
│   ├── document_loader.py     # Load & chunk .txt/.md files from data/sample_docs/
│   └── vectorstore.py         # ChromaDB initialisation / reuse
│
└── data/
    ├── sample_docs/           # Internal knowledge base (editable)
    │   ├── vpn_setup.txt
    │   ├── employee_handbook.txt
    │   ├── it_policies.txt
    │   ├── expense_policy.txt
    │   ├── leave_policy.txt
    │   └── onboarding_guide.txt
    └── chroma_db/             # Auto-created — persisted vector store
```

---

## Setup

### 1. Prerequisites

- Python 3.11+ (3.13 not recommended — dependency compatibility issues)
- A Groq API key — free at [console.groq.com](https://console.groq.com)
- _(Optional)_ A Tavily API key for web search

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure environment

```powershell
copy .env.example .env
```

Edit `.env`:

```env
GROQ_API_KEY=gsk-...           # Required
TAVILY_API_KEY=tvly-...        # Optional — enables web search
MODEL_NAME=llama-3.3-70b-versatile  # Optional — any Groq model with JSON mode
```

---

## Running the Agent

### Interactive CLI

```powershell
python main.py
```

Example session:

```
Your question: How do I set up VPN access?

══════════════════════════════════════════════════════════════════════
  ANSWER
══════════════════════════════════════════════════════════════════════
To set up VPN access:
1. Open the Employee Self-Service Portal at https://portal.acmecorp.com
2. Navigate to IT → Downloads → VPN Client and download GlobalConnect v5.2
3. Install and launch the client; enter server: vpn.acmecorp.com
4. Sign in with your corporate account — MFA approval is required on every connection
...

──────────────────────────────────────────
CITATIONS
──────────────────────────────────────────
  [1] [INTERNAL]   Vpn Setup
         Internal Wiki – Vpn Setup
         Date: Version: 3.1 | Last Updated: January 15, 2024
```

### Human Review

When the validator flags low confidence or conflicting sources, the graph **pauses automatically** and prompts a reviewer in the terminal:

```
══════════════════════════════════════════════
  HUMAN REVIEW REQUIRED
══════════════════════════════════════════════
• Document(s) with unknown publication date(s)

Reviewer feedback (corrections/additions), or press Enter to approve:
>
```

Press **Enter** to approve, or type corrections that are incorporated into the final answer.

---

## FastMCP Server

The three processing tools are exposed as a proper **FastMCP server** and can be connected to any MCP-compatible client.

### List registered tools

```powershell
python run_mcp_server.py --list-tools
```

Output:

```
Registered FastMCP tools on [Employee Q&A Tools]:

  • summarize_and_rank: Summarise internal documentation and web search results...
  • highlight_differences: Compare the internal documentation summary with the external web summary...
  • flag_issues: Validate the collected information and flag potential quality issues...
```

### Serve to Claude Desktop / Cursor (stdio)

```powershell
python run_mcp_server.py
```

Add to your Claude Desktop `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "employee-qa": {
      "command": "python",
      "args": ["C:/path/to/agentic/run_mcp_server.py"],
      "env": {
        "GROQ_API_KEY": "gsk-...",
        "TAVILY_API_KEY": "tvly-..."
      }
    }
  }
}
```

### Serve over HTTP/SSE (web clients)

```powershell
python run_mcp_server.py --transport sse --port 8000
```

---

## Adding Internal Documents

Drop any `.txt` or `.md` file into `data/sample_docs/` and **delete** the `data/chroma_db/` folder to trigger a rebuild on next run.

```powershell
# Add your document
copy "my_policy.txt" "data\sample_docs\"

# Delete old vector store so it rebuilds
Remove-Item -Recurse -Force "data\chroma_db"

# Run — knowledge base rebuilds automatically
python main.py
```

---

## MCP Tools Reference

### `summarize_and_rank`

| Parameter | Type | Description |
|---|---|---|
| `question` | `str` | Employee's original question |
| `internal_context` | `str` | Concatenated internal document chunks |
| `web_context` | `str` | Concatenated web search snippets |
| `internal_docs_json` | `str` | JSON-encoded list of internal result objects |
| `web_results_json` | `str` | JSON-encoded list of web result objects |
| `web_searched` | `bool` | Whether Tavily search was performed |

Returns: `{ "internal_summary": str, "web_summary": str, "ranked_sources": list }`

### `highlight_differences`

| Parameter | Type | Description |
|---|---|---|
| `question` | `str` | Employee's original question |
| `internal_summary` | `str` | Summary from internal knowledge base |
| `web_summary` | `str` | Summary from external web sources |
| `web_searched` | `bool` | Whether web search was performed |

Returns: `{ "differences": list[str], "agreements": list[str] }`

### `flag_issues`

| Parameter | Type | Description |
|---|---|---|
| `question` | `str` | Employee's original question |
| `internal_docs_json` | `str` | JSON-encoded list of internal result objects |
| `internal_summary` | `str` | Internal knowledge base summary |
| `web_summary` | `str` | External web sources summary |
| `differences_json` | `str` | JSON-encoded list of known conflict strings |

Returns: `{ "flagged_issues": list[str], "needs_human_review": bool, "review_reasons": list[str], "confidence_score": float }`

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | Yes | — | Groq API key — [console.groq.com](https://console.groq.com) |
| `TAVILY_API_KEY` | No | — | Tavily web search key; web search disabled if absent |
| `MODEL_NAME` | No | `llama-3.3-70b-versatile` | Any Groq model that supports JSON mode |
| `CHROMA_PERSIST_DIR` | No | `./data/chroma_db` | Path for persistent vector store |
| `EMBEDDING_MODEL` | No | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding model (local) |
