# FinGPT Orchestrator — Software Requirements Specification
**Version:** 1.1
**Date:** 2026-03-14
**Author:** Hadi
**Status:** Reviewed — Ready for Implementation

---

## Revision History
| Version | Date | Changes |
|---|---|---|
| 1.0 | 2026-03-14 | Initial draft |
| 1.1 | 2026-03-14 | Closed all open questions; expanded yfinance schema; added history persistence; removed UI document ingestion widget; added traceability, i18n, legal sections; added US IDs; fixed acceptance criteria |

---

## 1. Introduction

### 1.1 Purpose
This document specifies the requirements for the FinGPT Orchestrator application — a locally-running system that routes financial queries through a RAG layer and dispatches them to the appropriate fine-tuned LoRA agent, all sharing a single frozen Qwen3-8B base model. It is the authoritative reference for implementation and future review.

### 1.2 Scope

**Included:**
- A CLI providing interactive sessions, batch processing, and RAG index management
- A Gradio browser-based chat interface (query and answer only — no document management in UI)
- A RAG ingestion pipeline supporting local files, directories, and live market data via yfinance
- Multi-turn conversation memory persisted to disk across sessions
- Routing logic between two trained LoRA agents (sentiment, multitask) and one planned agent (forecaster)

**Excluded:**
- REST API or network-accessible server
- Authentication or multi-user access control
- Training or fine-tuning functionality
- Production deployment, containerisation, or cloud hosting
- Any data source requiring an API key or paid subscription
- Document upload or RAG index management from the Gradio UI
- Scheduled or automatic data refresh

### 1.3 Target Audience
The sole user and developer is Hadi. This is a local portfolio development tool running on a single Windows laptop.

### 1.4 Definitions and Acronyms
| Term | Definition |
|---|---|
| Orchestrator | The central controller that classifies queries, invokes RAG, and dispatches to agents |
| Agent | A LoRA adapter hot-swapped onto the frozen Qwen3-8B base model |
| RAG | Retrieval-Augmented Generation — injecting retrieved document chunks into the prompt |
| FAISS | Facebook AI Similarity Search — the local vector store used for retrieval |
| LoRA | Low-Rank Adaptation — the fine-tuning method used for all agents |
| CLI | Command-Line Interface, built with Typer |
| UI | The Gradio browser-based chat interface |
| Session | A single continuous run of the CLI or UI from start to exit |
| Chunk | A fixed-size segment of an ingested document, stored as an embedding vector |
| yfinance | The `yfinance` Python library used to fetch financial data from Yahoo Finance |
| VRAM | GPU video memory |
| EPS | Earnings Per Share |
| History | The ordered list of prior user/assistant message pairs passed to the model |

### 1.5 References
- `orchestrator.py` — current orchestrator implementation
- `rag/retriever.py` — FAISS retriever implementation
- `rag/rag_layer.py` — RAG enrichment layer
- `docs/architecture.md` — full system design
- [yfinance GitHub](https://github.com/ranaroussi/yfinance)
- [Gradio documentation](https://gradio.app/docs)
- [Typer documentation](https://typer.tiangolo.com)

---

## 2. Goals and Objectives

### 2.1 Business Goals
This project is a self-taught ML portfolio demonstrating: multi-agent LLM architecture, RAG integration, financial domain fine-tuning, and production-quality system design — all running on consumer hardware at under $300 in training cost.

### 2.2 User Goals
- Run financial queries interactively and receive reasoned, context-aware answers grounded in real market data
- Ingest financial documents and live market data into the RAG index without writing code
- Inspect which agent handled a query and whether RAG context was used
- Process batches of queries from a file and save results for analysis
- Manage the RAG index (add, inspect, clear) from the CLI
- Resume previous conversations across sessions without losing context

### 2.3 Success Metrics
| Metric | Target | Linked Requirement |
|---|---|---|
| VRAM usage during inference | ≤ 11 GB | REQ-NFR-005 |
| Agent hot-swap time | ≤ 10 seconds | REQ-NFR-001 |
| RAG retrieval time per query | ≤ 2 seconds | REQ-NFR-002 |
| yfinance snapshot ingestion time | ≤ 30 seconds per ticker | REQ-NFR-003 |
| Gradio UI startup time | ≤ 60 seconds | REQ-NFR-004 |

---

## 3. User Stories

### US-001 — Interactive CLI Session
As a developer, I want to start an interactive CLI session where I can ask financial questions one at a time and see reasoned answers, so that I can explore the model's capabilities conversationally.

### US-002 — Batch Processing
As a developer, I want to point the CLI at a file of queries and have the orchestrator process all of them and write results to an output file, so that I can run benchmark-style evaluations without babysitting each query.

### US-003 — RAG Index Management via CLI
As a developer, I want CLI commands to add a file, add a directory, fetch a ticker from yfinance, inspect index size, and clear the index, so that I can manage the knowledge base without writing Python.

### US-004 — Gradio Chat Interface
As a developer, I want a browser-based chat interface where I can type queries, see the answer, and see which agent handled the query, so that I can demo the system to others without them needing to use a terminal.

### US-005 — Multi-turn Conversation with Persistence
As a developer, I want the system to remember prior conversation turns both within and across sessions, so that I can ask follow-up questions without restating context, even after restarting the application.

### US-006 — Live Market Data in RAG
As a developer, I want to fetch a ticker's full financial data profile via yfinance and index it into the RAG store, so that model answers are grounded in real, current market data including earnings history.

### US-007 — RAG Failure Visibility
As a developer, I want to know when RAG failed and why — empty index, no match, low confidence, or fetch error — so that I can diagnose retrieval problems and tune the system without guessing.

---

## 4. Functional Requirements

Priority: **H** = High (must have for v1), **M** = Medium (important but deferrable), **L** = Low.

### 4.1 CLI

| ID | Requirement | Priority | Traces To |
|---|---|---|---|
| REQ-CLI-001 | The CLI MUST provide an `interactive` command that starts a REPL accepting natural language queries | H | US-001 |
| REQ-CLI-002 | The CLI MUST display the agent name used for each response in interactive mode | H | US-001 |
| REQ-CLI-003 | The CLI MUST display the RAG status and warning message for each response in interactive mode | H | US-007 |
| REQ-CLI-004 | The CLI MUST maintain and prepend conversation history across turns within an interactive session | H | US-005 |
| REQ-CLI-005 | The CLI MUST provide a `batch` command accepting an input file path and an output file path | H | US-002 |
| REQ-CLI-006 | The `batch` command MUST support `.txt` (one query per line) and `.jsonl` input formats | H | US-002 |
| REQ-CLI-007 | The `batch` command MUST write results as `.jsonl` with fields: `query`, `answer`, `agent_used`, `rag_status`, `rag_warning` | H | US-002 |
| REQ-CLI-008 | The CLI MUST provide an `index` subcommand group for RAG index management | H | US-003 |
| REQ-CLI-009 | `index add-file <path>` MUST ingest a single `.txt`, `.pdf`, or `.md` file into the RAG index | H | US-003 |
| REQ-CLI-010 | `index add-dir <path>` MUST recursively ingest all `.txt`, `.pdf`, and `.md` files in a directory | H | US-003 |
| REQ-CLI-011 | `index fetch <ticker>` MUST fetch the full financial data profile for the given ticker via yfinance and index it | H | US-006 |
| REQ-CLI-012 | `index status` MUST print the total number of indexed chunks and the name of the active embedding model | H | US-003 |
| REQ-CLI-013 | `index clear` MUST delete all indexed documents and the persisted index files after an explicit confirmation prompt | H | US-003 |
| REQ-CLI-014 | The CLI MUST print a startup summary showing loaded agents and current index size before accepting input | M | US-001 |
| REQ-CLI-015 | The CLI MUST support a `--verbose` flag that prints retrieved chunks and the routing decision for each query | M | US-007 |
| REQ-CLI-016 | The CLI MUST support an `--agent` option to override automatic agent routing for a session | M | US-001 |
| REQ-CLI-017 | The CLI MUST provide a `history clear` command that deletes the persisted conversation history file | M | US-005 |

### 4.2 Gradio Chat UI

| ID | Requirement | Priority | Traces To |
|---|---|---|---|
| REQ-UI-001 | The UI MUST launch a Gradio interface accessible at `http://localhost:7860` | H | US-004 |
| REQ-UI-002 | The UI MUST display the model's answer in a chat bubble format | H | US-004 |
| REQ-UI-003 | The UI MUST show the agent name used for each response as a label below the answer bubble | H | US-004 |
| REQ-UI-004 | The UI MUST display the full conversation history in the chat panel within the session | H | US-005 |
| REQ-UI-005 | The UI MUST show a loading indicator while the model is generating a response | H | US-004 |
| REQ-UI-006 | The UI MUST display a "Not financial advice" disclaimer prominently on the page | H | SEC-001 |
| REQ-UI-007 | The UI SHOULD display the current RAG index size in the sidebar | L | US-004 |
| REQ-UI-008 | The UI SHOULD display the RAG warning string when `rag_warning` is not None | M | US-007 |

### 4.3 Orchestrator Core

| ID | Requirement | Priority | Traces To |
|---|---|---|---|
| REQ-ORC-001 | The orchestrator MUST load the Qwen3-8B base model once at startup in 4-bit quantisation | H | US-001 |
| REQ-ORC-002 | The orchestrator MUST hot-swap LoRA adapters without reloading the base model between queries | H | US-001 |
| REQ-ORC-003 | The orchestrator MUST maintain an ordered conversation history list and include it in every query prompt | H | US-005 |
| REQ-ORC-004 | The orchestrator MUST truncate conversation history if the combined prompt would exceed `MAX_LEN` tokens, dropping the oldest turns first | H | US-005 |
| REQ-ORC-005 | The orchestrator MUST route to `multitask` when `forecaster` is requested but not yet present in the `AGENTS` dict | H | US-001 |
| REQ-ORC-006 | The orchestrator MUST include `rag_status` and `rag_warning` in every response dict | H | US-007 |
| REQ-ORC-007 | The orchestrator MUST strip `<think>...</think>` blocks from model output before returning the answer | H | US-001 |
| REQ-ORC-008 | The orchestrator MUST load conversation history from the persisted history file on startup if it exists | H | US-005 |
| REQ-ORC-009 | The orchestrator MUST append each completed turn to the persisted history file immediately after generation | H | US-005 |
| REQ-ORC-010 | The orchestrator MUST expose a `reset_history()` method that clears both in-memory history and the persisted file | M | US-005 |

### 4.4 RAG Ingestion Pipeline

| ID | Requirement | Priority | Traces To |
|---|---|---|---|
| REQ-RAG-001 | The ingestion pipeline MUST support `.txt` files | H | US-003 |
| REQ-RAG-002 | The ingestion pipeline MUST support `.md` files | H | US-003 |
| REQ-RAG-003 | The ingestion pipeline MUST support `.pdf` files using `pypdf` | H | US-003 |
| REQ-RAG-004 | The ingestion pipeline MUST support bulk ingestion from a directory, processing all supported file types recursively | H | US-003 |
| REQ-RAG-005 | The ingestion pipeline MUST support fetching the full financial data profile for a ticker via `yfinance`, including: current price, P/E ratio, market cap, 52-week high/low, forward EPS, trailing EPS; quarterly earnings history (revenue, net income, EPS, profit margin) for the last 8 quarters; annual income statement summary (revenue, EBITDA, free cash flow) for the last 4 years; and the 10 most recent news headlines with publisher and summary | H | US-006 |
| REQ-RAG-006 | Each ingested document MUST be stored with metadata: `source`, `ticker` (if applicable), `date` (ISO 8601) | H | US-003, US-006 |
| REQ-RAG-007 | If yfinance fetching raises any exception, the pipeline MUST raise and halt — no partial indexing | H | US-006, US-007 |
| REQ-RAG-008 | The ingestion pipeline MUST print a summary showing chunks added and total index size after each operation | H | US-003 |
| REQ-RAG-009 | PDF ingestion SHOULD extract text page by page, treating each page as an independent chunking unit | M | US-003 |

### 4.5 RAG Retrieval

| ID | Requirement | Priority | Traces To |
|---|---|---|---|
| REQ-RET-001 | The retriever MUST use `FinLang/finance-embeddings-investopedia` as the primary embedding model | H | US-006 |
| REQ-RET-002 | The retriever MUST fall back to `all-MiniLM-L6-v2` if the primary model fails to load | H | US-007 |
| REQ-RET-003 | The retriever MUST return one of five explicit statuses: `OK`, `EMPTY_INDEX`, `NO_MATCH`, `LOW_CONFIDENCE`, `ERROR` | H | US-007 |
| REQ-RET-004 | The retriever MUST persist the FAISS index to disk after every `add` operation | H | US-003 |
| REQ-RET-005 | The retriever MUST load the persisted index from disk on startup if it exists | H | US-003 |

---

## 5. Non-Functional Requirements

### 5.1 Performance
- **REQ-NFR-001:** Agent hot-swap ≤ 10 seconds on RTX 5070 Ti
- **REQ-NFR-002:** RAG retrieval ≤ 2 seconds per query
- **REQ-NFR-003:** yfinance snapshot ≤ 30 seconds per ticker; full earnings MAY take longer with progress message
- **REQ-NFR-004:** Gradio UI accessible within 60 seconds of `python ui.py`
- **REQ-NFR-005:** VRAM ≤ 11 GB during inference

### 5.2 Security
- **REQ-NFR-006:** No API keys required or stored in v1
- **REQ-NFR-007:** Gradio binds to `127.0.0.1` only — never `0.0.0.0`

### 5.3 Usability
- **REQ-NFR-008:** Every CLI subcommand MUST expose `--help`
- **REQ-NFR-009:** All error messages MUST suggest a corrective action
- **REQ-NFR-010:** CLI startup banner shows model, agents, index size, history path
- **REQ-NFR-011:** `index clear` requires typing `yes` after showing chunk count

### 5.4 Reliability
- **REQ-NFR-012:** Batch failures log error, write `"error"` field, continue remaining queries
- **REQ-NFR-013:** FAISS index saved via `.tmp` + atomic rename — crash-safe

### 5.5 Maintainability
- **REQ-NFR-014:** CLI, UI, orchestrator, ingestion, RAG in separate modules — no circular imports
- **REQ-NFR-015:** Adding an agent requires only one entry in `AGENTS` dict

### 5.6 Portability
- **REQ-NFR-016:** Windows 11, Python 3.11
- **REQ-NFR-017:** All deps installable via `pip`

### 5.7 Data Requirements
- **REQ-NFR-018:** History persisted to `history/session.jsonl`, loaded on startup
- **REQ-NFR-019:** RAG index persisted to `rag/index/`, survives restarts
- **REQ-NFR-020:** Batch output `.jsonl` — each line independently parseable
- **REQ-NFR-021:** History format: `{"role": "user"|"assistant", "content": "...", "ts": "<ISO8601>"}`
- **REQ-NFR-022:** History pruned to 500 most recent turns on load

### 5.8 Error Handling and Logging
- **REQ-NFR-023:** No bare `except:` — all exceptions caught with typed clauses
- **REQ-NFR-024:** `logging` module for internal logic; `print()` for user-facing output only

### 5.9 Internationalisation
- **REQ-NFR-025:** English-only. No multi-language support planned for v1.

### 5.10 Accessibility
- **REQ-NFR-026:** Gradio UI works in Chrome and Edge on Windows 11

### 5.11 Legal and Compliance
- **SEC-001:** Gradio UI MUST display: *"This tool is for research and portfolio demonstration purposes only. Nothing produced by this system constitutes financial advice."*
- **SEC-002:** CLI MUST print the same disclaimer on startup in interactive mode

---

## 6. Technical Requirements

### 6.1 Technology Stack
| Component | Technology |
|---|---|
| Language | Python 3.11 |
| LLM runtime | Unsloth + HuggingFace Transformers |
| LoRA | PEFT |
| Vector store | FAISS (`faiss-cpu`) |
| Embeddings | `FinLang/finance-embeddings-investopedia` |
| Chat UI | Gradio ≥ 4.0 |
| CLI framework | Typer ≥ 0.12 |
| Terminal formatting | Rich (via `typer[all]`) |
| Market data | yfinance ≥ 0.2 |
| PDF parsing | pypdf ≥ 4.0 |

### 6.2 New Dependencies
```
typer[all]>=0.12
gradio>=4.0
yfinance>=0.2
pypdf>=4.0
faiss-cpu>=1.7
sentence-transformers>=2.7
python-dotenv>=1.0
```

### 6.3 Module Structure
```
FinGPT/
├── orchestrator.py      # EXTEND — add history persistence
├── cli.py               # NEW — Typer CLI
├── ui.py                # NEW — Gradio chat interface
├── history/
│   └── session.jsonl    # AUTO-CREATED
├── rag/
│   ├── __init__.py
│   ├── retriever.py
│   ├── rag_layer.py
│   └── ingest.py        # NEW — file/dir/yfinance ingestion
└── rag/index/           # AUTO-CREATED
```

### 6.4 Entry Points
| Command | Entry Point |
|---|---|
| Interactive CLI | `python cli.py interactive` |
| Batch | `python cli.py batch <in.jsonl> <out.jsonl>` |
| Index management | `python cli.py index fetch AAPL` |
| Gradio UI | `python ui.py` |

---

## 7. Design Considerations

### 7.1 Conversation History Persistence
History is `{"role": "user"|"assistant", "content": str, "ts": str}` dicts matching Qwen3 chat template format. Loaded from `history/session.jsonl` on startup (max 500 turns). Appended after every turn (append-only). `reset_history()` deletes the file. Oldest turns are truncated from memory (not from file) when token budget is exceeded.

### 7.2 yfinance Data Schema
Four sections per ticker: current snapshot → quarterly earnings (8 quarters) → annual financials (4 years) → recent news (10 headlines). If any section fails, pipeline raises and halts — no partial indexing.

### 7.3 Batch File Formats
Input `.txt`: one query per line. Input `.jsonl`: `{"query": "..."}` per line.
Output `.jsonl`: `{"query", "answer", "agent_used", "rag_status", "rag_warning"}` per line; failed queries add `"error"` field and null model fields.

---

## 8. Acceptance Criteria

| Requirement | Test |
|---|---|
| REQ-CLI-001 | `python cli.py interactive` starts and accepts a query |
| REQ-CLI-004 | `--verbose` shows previous turn in history on second query |
| REQ-CLI-007 | Each output line has all 5 fields and passes `json.loads()` |
| REQ-CLI-011 | `index fetch AAPL` increases chunk count shown by `index status` |
| REQ-CLI-013 | Abort without `yes`; clear with `yes` leaves empty index |
| REQ-UI-001 | `python ui.py` opens `localhost:7860` within 60 seconds |
| REQ-UI-003 | Agent name label appears below every answer |
| REQ-UI-006 | Disclaimer visible without scrolling |
| REQ-ORC-008 | Restart + "what did I ask earlier?" references previous session |
| REQ-RAG-005 | `index fetch MSFT` indexes all 4 sections |
| REQ-RAG-007 | Invalid ticker raises error, leaves index unchanged |
| SEC-001 | Disclaimer visible in Gradio UI |
| SEC-002 | Disclaimer printed on CLI interactive startup |

---

## 9. Future Considerations
- REST API (FastAPI wrapper)
- Forecaster agent (Round 3)
- Scheduled yfinance refresh
- FAISS IVF index for scale
- HuggingFace Space deployment
- SEC EDGAR and news RSS data sources

---

## 10. Resolved Questions
| # | Question | Resolution |
|---|---|---|
| 1 | yfinance scope | All data: snapshot + quarterly earnings + annual financials + news (REQ-RAG-005) |
| 2 | History persistence | Yes — `history/session.jsonl`, loaded on startup (REQ-NFR-018) |
| 3 | UI document ingestion | No — UI is chat-only, all RAG management via CLI |
