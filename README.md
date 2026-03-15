# FinGPT-Qwen3 — Self-Taught Financial AI Portfolio

> Fine-tuning open-source LLMs on financial data for sentiment analysis,
> Q&A, headline classification, and stock forecasting —
> built on a consumer laptop with a single GPU.

---

## What This Project Is

A personal portfolio project documenting the design and training of a
multi-agentic financial AI system where each agent is a specialized LoRA
adapter fine-tuned on [FinGPT](https://github.com/AI4Finance-Foundation/FinGPT)
ecosystem datasets. The system runs entirely locally on a consumer laptop (mine)
using [Unsloth](https://github.com/unslothai/unsloth) for 2× faster training.

---

## System Architecture (to be updated as I research more tools and architectures)

```
                        User Query
                              │
                              ▼
                     ┌─────────────────┐
                     │   Orchestrator  │  keyword router
                     └────────┬────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │    RAG Layer    │  FAISS vector search
                     │                │  finance-domain embeddings
                     │  ┌───────────┐ │  explicit status: OK / NO_MATCH /
                     │  │   FAISS   │ │  LOW_CONFIDENCE / ERROR
                     │  │  Index    │ │
                     │  └───────────┘ │
                     └────────┬────────┘
                              │ enriched prompt
                              ▼
           ┌──────────────────┼──────────────────┐
           ▼                  ▼                  ▼
   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
   │  sentiment   │  │  multitask   │  │  forecaster  │
   │  agent  r=8  │  │  agent r=16  │  │  (planned)   │
   └──────────────┘  └──────────────┘  └──────────────┘
           └──────────────────┘
                      │
                      ▼
             Shared Base Model
             Qwen3-8B (frozen, 4bit)
```

All agents share the same frozen Qwen3-8B base in memory. The orchestrator
hot-swaps LoRA adapters (~80MB each) at inference time — no model reload,
no VRAM spike between agents (~5.6GB total vs ~22GB for 4 separate models).

---

## Benchmark Results

to be updated

---

## Agent Details

| Agent | Adapter | Training Data | Tasks | Status |
|---|---|---|---|---|
| `sentiment` | `qwen3-8b-fingpt-lora` | fingpt-sentiment-train (76.8K) | Sentiment | ✅ Done |
| `multitask` | `qwen3-8b-round2-lora` | sentiment + fiqa_qa + headline (174K) | Sentiment, Q&A, Headline | ✅ Done |
| `forecaster` | `qwen3-8b-round3-lora` | fingpt-forecaster-dow30 | Price Forecasting | 📋 Planned |

### Round 2 Training Config

```
Base model:   Qwen3-8B (unsloth 4bit)
LoRA rank:    r=16, alpha=16
Datasets:     fingpt-sentiment-train (76.8K) +
              fingpt-fiqa_qa (17.1K) +
              fingpt-headline (82.2K)
              ─────────────────────────────
              Total: ~174K samples
Epochs:       1  (10,888 steps)
Batch size:   2 (device) × 8 (grad accum) = 16 effective
Optimizer:    paged_adamw_8bit
Precision:    bf16
Hardware:     RTX 5070 Ti Laptop, 12GB VRAM
Duration:     ~24 hours
```

---

## RAG Layer

The RAG layer enriches queries with retrieved financial context before they
reach any agent. It uses a local FAISS vector store with
`FinLang/finance-embeddings-investopedia` — a finance-domain fine-tune of
MiniLM-L6-v2 that correctly maps financial jargon (EBITDA, yield curve,
basis points, P/E, WACC) in embedding space.

**Data sources:**
- Local `.txt`, `.pdf`, `.md` files
- Directory bulk ingestion
- Live market data via `yfinance` — current snapshot, quarterly earnings
  history (8 quarters), annual financials (4 years), recent news headlines

**Failure handling:** every retrieval returns an explicit `RAGStatus` enum
(`OK`, `EMPTY_INDEX`, `NO_MATCH`, `LOW_CONFIDENCE`, `ERROR`) — the orchestrator
reacts to each case explicitly with no silent fallbacks.

---

## Key Design Decisions

**Why separate LoRA adapters instead of one big model?**
Mixing rigid classification (sentiment) with open-ended generation (Q&A) and
binary prediction (headline) in a single training run causes task interference.
Separate adapters stay specialized and can be individually retrained as new
data arrives without touching other agents.

**Why Qwen3-8B?**
Qwen3 includes a native thinking mode (`<think>...</think>`) — the model
reasons through financial problems before answering. For forecasting and
analysis tasks, this chain-of-thought reasoning is a meaningful advantage
over earlier models that output one word directly.

**Why benchmark at orchestrator level?**
FinGPT publishes scores for their deployed system, not isolated adapters.
Benchmarking at orchestrator level produces comparable numbers and also
validates that routing is working correctly — the `routing` field in every
result confirms which agent handled each task.

**Why local instead of API?**
The whole point is to own the weights. A locally-trained adapter can be
updated weekly with new financial data. BloombergGPT cost $2.67M to train.
FinGPT brought that down to ~$300. This project brings it down further to
a consumer GPU and one electricity bill.

---

## Repository Structure

```
FinGPT-Portfolio/
│
├── README.md
│
├── orchestrator.py            ← Production orchestrator (classify → RAG → agent)
│
├── rag/
│   ├── __init__.py
│   ├── retriever.py           ← FAISS vector store + finance embeddings
│   └── rag_layer.py           ← RAG enrichment + explicit status handling
│
├── scripts/
│   ├── round1.py              ← Round 1 sentiment specialist training
│   ├── round2.py              ← Round 2 multi-task training
│   └── benchmark.py           ← Orchestrator-level benchmark (full pipeline)
│
├── docs/
│   ├── architecture.md        ← Full system design with diagrams
│   ├── roadmap.md             ← What's done, in progress, and planned
│   └── orchestrator_requirements.md  ← Full SRS for CLI + UI + ingestion
│
└── results/
    └── benchmark_results.json ← Benchmark output with routing metadata
```

---

## Roadmap

- [x] Round 1: Sentiment specialist adapter
- [x] Round 2: Multi-task adapter (sentiment + Q&A + headline)
- [x] RAG layer with FAISS + finance-domain embeddings
- [x] Orchestrator with keyword routing + explicit RAG status
- [x] Benchmark at orchestrator level vs FinGPT reference
- [ ] CLI (Typer) — interactive, batch, index management
- [ ] Gradio chat UI
- [ ] Round 3: Forecaster adapter (fingpt-forecaster-dow30)
- [ ] HuggingFace Space for public demo

---

## Built On

- [FinGPT](https://github.com/AI4Finance-Foundation/FinGPT) — AI4Finance Foundation
- [Unsloth](https://github.com/unslothai/unsloth) — 2× faster fine-tuning
- [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) — Alibaba Cloud
- [HuggingFace PEFT](https://github.com/huggingface/peft) — LoRA implementation
- [FAISS](https://github.com/facebookresearch/faiss) — vector similarity search
- [FinLang embeddings](https://huggingface.co/FinLang/finance-embeddings-investopedia) — finance-domain retrieval

---

## Disclaimer

This project is for educational and portfolio purposes only.
Nothing here constitutes financial advice or a recommendation to trade.
Do not make trading or investment decisions based on this system's output.
