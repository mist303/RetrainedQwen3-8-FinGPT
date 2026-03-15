# Architecture — Multi-Agent Financial AI System

## Overview

The system is built around a core principle from the FinGPT ecosystem:
**specialized agents outperform generalist models on financial NLP tasks.**

Each agent is a lightweight LoRA adapter (~80MB) trained on specific tasks.
All agents share the same frozen Qwen3-8B base model in memory. The
orchestrator decides which adapter to activate per query, and the RAG layer
enriches every query with retrieved financial context before it reaches any agent.

---

## Full System Architecture

```
                        User Query
                              │
                              ▼
                     ┌─────────────────┐
                     │   Orchestrator  │  classify_query() → keyword router
                     └────────┬────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │    RAG Layer    │  enrich() → FAISS vector search
                     │                │  inject context into system prompt
                     │  ┌───────────┐ │  status: OK / EMPTY / NO_MATCH /
                     │  │   FAISS   │ │          LOW_CONFIDENCE / ERROR
                     │  │  Index    │ │
                     │  └───────────┘ │
                     └────────┬────────┘
                              │ enriched prompt
                              ▼
           ┌──────────────────┼──────────────────┐
           ▼                  ▼                  ▼
   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
   │  sentiment   │  │  multitask   │  │  forecaster  │
   │  agent       │  │  agent       │  │  agent       │
   │  r=8         │  │  r=16        │  │  (planned)   │
   └──────────────┘  └──────────────┘  └──────────────┘
           │                  │
           └──────────┬───────┘
                      ▼
             Shared Base Model
             Qwen3-8B (frozen, 4bit)
             ~5.5 GB VRAM
```

---

## The Five-Layer FinGPT Framework

This project implements layers 3–5:

```
Layer 5 │ Application     │ Orchestrator + CLI + Gradio UI
Layer 4 │ Task            │ Specialized LoRA agents (this project)
Layer 3 │ LLMs            │ Qwen3-8B + Unsloth fine-tuning (this project)
Layer 2 │ Data Engineering│ FinNLP preprocessing pipelines
Layer 1 │ Data Source     │ FinGPT HuggingFace datasets + yfinance live data
```

---

## Why LoRA Adapters as Agents

Standard multi-task training mixes diverse objectives into a single weight set.
For financial NLP this causes task interference:

- Sentiment analysis expects rigid 3-class output: `{negative/neutral/positive}`
- Q&A is generative and open-ended
- Headline classification is binary Yes/No
- Forecasting requires multi-paragraph structured analysis

LoRA adapters isolate each task into its own small set of trainable parameters
(~40M out of 8.2B total, ~0.5%). The base model's general language understanding
stays intact and is shared across all agents.

### Memory efficiency

```
Separate models:  4 × 8B × ~5.5GB = ~22GB  — impossible on 12GB VRAM
Adapter approach: 5.5GB base + ~80MB adapter = ~5.6GB total  ✓
```

### Adapter hot-swapping

Swapping adapters takes under 10 seconds with no model reload or VRAM spike:

```python
state_dict = load_file("adapter_multitask.safetensors")
set_peft_model_state_dict(model, state_dict)
```

---

## Agent Roster

### `sentiment` — Sentiment Specialist
- **File:** `qwen3-8b-fingpt-lora/adapter_model.safetensors`
- **Training data:** `FinGPT/fingpt-sentiment-train` (76.8K samples)
- **Task:** 3-class sentiment classification on financial news and tweets
- **Output:** One word — `positive`, `neutral`, or `negative`
- **LoRA config:** r=8, alpha=8
- **Status:** ✅ Complete — benchmarked

### `multitask` — Multi-Task Agent
- **File:** `qwen3-8b-round2-lora/adapter_model.safetensors`
- **Training data:**
  - `FinGPT/fingpt-sentiment-train` (76.8K)
  - `FinGPT/fingpt-fiqa_qa` (17.1K)
  - `FinGPT/fingpt-headline` (82.2K)
  - Total: ~174K samples
- **Tasks:** Sentiment analysis, open-ended financial Q&A, binary headline classification
- **LoRA config:** r=16, alpha=16 (higher rank for multi-task capacity)
- **Status:** ✅ Complete — benchmarked

### `forecaster` — Forecaster Agent *(planned)*
- **Training data:** `FinGPT/fingpt-forecaster-dow30-202305-202405`
- **Task:** Multi-paragraph stock price movement prediction given news + financials
- **Output:** [Positive Developments] / [Potential Concerns] / [Prediction & Analysis]
- **Status:** 📋 Planned — Round 3 training

---

## RAG Layer

The RAG layer sits between the orchestrator and the agents. It enriches every
query with retrieved financial context before the prompt reaches any agent.

### Design

```
query → embed (FinLang/finance-embeddings-investopedia, 384-dim)
      → FAISS flat L2 nearest-neighbour search
      → L2 distance filter (threshold = 1.2)
      → inject passing chunks into system prompt
      → return explicit RAGStatus to orchestrator
```

### RAG status enum

The orchestrator always receives an explicit status — never a silent fallback:

| Status | Meaning | Orchestrator response |
|---|---|---|
| `OK` | Context injected | Proceed with enriched prompt |
| `EMPTY_INDEX` | No documents indexed | Warn once per session, proceed |
| `NO_MATCH` | Index exists but nothing found | Note in result, proceed |
| `LOW_CONFIDENCE` | Candidates found, all too dissimilar | Log best score, proceed |
| `ERROR` | FAISS exception | Log error, flag in result |

### Embedding model

`FinLang/finance-embeddings-investopedia` — MiniLM-L6-v2 fine-tuned on
Investopedia financial text. Same 384-dim as vanilla MiniLM so the index
format is compatible, but financial jargon (EBITDA, yield curve, basis points,
WACC, P/E) maps to correct embedding space. Falls back to `all-MiniLM-L6-v2`
if the finance model is unavailable.

### Data sources supported

- `.txt`, `.pdf`, `.md` files (local)
- Directories (bulk ingest, recursive)
- Yahoo Finance via `yfinance` — current price, P/E, market cap, quarterly
  earnings history (8 quarters), annual financials (4 years), recent news (10 headlines)

---

## Orchestrator Query Router

Keyword-based routing — intentionally simple, no ML needed:

```python
def classify_query(query: str) -> str:
    q = query.lower()
    if any(kw in q for kw in ["forecast","predict","next week","will rise"]):
        return "forecaster"
    if any(kw in q for kw in ["sentiment","bullish","bearish","what is the tone"]):
        return "sentiment"
    if any(kw in q for kw in ["headline","price going up","does this headline"]):
        return "multitask"
    return "multitask"   # default: handles QA and ambiguous queries
```

Upgrade path: replace with a trained classifier once more agents exist and
routing ambiguity increases.

---

## Benchmark Methodology

Benchmarks run at **orchestrator level** — the full pipeline including routing
is evaluated, not individual adapters in isolation. This matches the FinGPT
paper methodology and produces scores directly comparable to their published
reference numbers.

The `routing` field in each result confirms which agent handled each task,
allowing routing correctness to be verified alongside accuracy scores.

See `results/benchmark_results.json` for current scores.
