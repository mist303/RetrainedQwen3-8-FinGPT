# FinGPT-Qwen3 — Self-Taught Financial AI Portfolio

> Fine-tuning open-source LLMs on financial data for sentiment analysis,
> Q&A, headline classification, and stock forecasting —
> built on a consumer laptop with a single GPU.

---

## What This Project Is

A personal portfolio project documenting the design and training of a
multi-agent financial AI system where each agent is a specialized LoRA
adapter fine-tuned on [FinGPT](https://github.com/AI4Finance-Foundation/FinGPT)
ecosystem datasets. The system runs entirely locally on a consumer laptop
using [Unsloth](https://github.com/unslothai/unsloth) for 2× faster training.

The core idea: instead of one model that does everything poorly, route each
financial query to the specialist adapter that was trained for exactly that task.

---

## System Architecture

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
hot-swaps LoRA adapters (~80MB each) at inference time to avoid  model reload and VRAM spike between agents (~5.6GB total vs ~22GB for the 4 separate models found in FinGPT).

---

## Benchmark Results

Evaluated using the official FinGPT benchmark datasets. Every query routes
through the full orchestrator pipeline seen in the above system architecture with the keyword classifier deciding which
agent handles each task. I will be optimizing this further as my understanding deepens.

| Task | Agent Used | F1 Weighted | FinGPT v3.3 |
|---|---|---|---|
| FPB (Financial PhraseBank) | sentiment | 0.577 | 0.882 |
| FiQA-SA | sentiment | 0.935 | 0.874 |
| Headline | multitask | 0.881 | ~0.970 |
| QA (keyword score) | multitask | 0.562 | — |

The system routes sentiment classification tasks to the sentiment specialist adapter and
headline/Q&A tasks to the Round 2 multi-task adapter. This adapted will be redone as I made a mistake to train it on sentiment analysis as well. 

**FiQA-SA at 0.935** beats the FinGPT reference (0.874). Re-evaluation tbd to clean out the multitasker and see if improvements can be made. 

**Headline at 0.881** is the cleanest result with no data leakage, genuinely
held-out test split, trained for 1 epoch on a 12GB consumer GPU.

**Hardware:** RTX 5070 Ti Laptop (12GB VRAM and 32 GB RAM)· Unsloth 4-bit quantization ·
training cost is limited to electricity only · No cloud compute

---

## Agent Details

| Agent | Tasks | Training Data | Config | Status |
|---|---|---|---|---|
| `sentiment` | Sentiment classification | fingpt-sentiment-train (76.8K) | r=8, alpha=8, 1 epoch | ✅ Done |
| `multitask` | Financial Q&A, Headline classification | fingpt-fiqa_qa (17.1K) + fingpt-headline (82.2K) | r=16, alpha=32, 3 epochs | 🔄 Retraining |
| `forecaster` | Stock price forecasting | fingpt-forecaster-dow30 | r=16, alpha=32 | 📋 Planned |

### Why these boundaries?

**`sentiment`** handles all sentiment classification tasks (FPB, FiQA-SA).
It was trained exclusively on sentiment data so it stays sharp on that one
output format — one word: `positive`, `neutral`, or `negative`.

**`multitask`** handles open-ended financial Q&A and binary headline
classification. Sentiment data was deliberately excluded from its training.
Early benchmarks showed including sentiment caused 2.6pp task interference
on FiQA-SA — the agent performed worse on sentiment than the specialist
despite seeing the same data. Keeping these agents separate is not just
architectural preference — it's empirically validated by the benchmark results.

**`forecaster`** will handle multi-paragraph stock price prediction given
news and financials. Its output format (structured analysis + prediction) is
incompatible with the other agents' training data, so it gets its own adapter.

### `multitask` Training Config (current retraining run)

```
Base model:   Qwen3-8B (unsloth 4bit)
LoRA rank:    r=16, alpha=32
Datasets:     fingpt-fiqa_qa  (17.1K)
              fingpt-headline (82.2K)
              ─────────────────────────────
              Total: ~99.3K samples
              (sentiment excluded — owned by the sentiment agent)
Epochs:       3
Batch size:   2 (device) × 8 (grad accum) = 16 effective
Optimizer:    paged_adamw_8bit
Precision:    bf16
Hardware:     RTX 5070 Ti Laptop, 12GB VRAM
Checkpoints:  every ~10% (~4h intervals) — safe to stop and resume
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
Benchmarks confirmed this empirically — the sentiment specialist outperforms
the multi-task agent on sentiment tasks despite seeing the same training data.
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
│   ├── sentiment_agent.py     ← Sentiment agent training
│   ├── multitask_agent.py     ← Multitask agent training (Q&A + Headline)
│   ├── benchmark.py           ← Orchestrator-level benchmark (full pipeline)
│   └── round1.py / round2.py  ← Original training runs (archived reference)
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

- [x] `sentiment` agent trained and benchmarked
- [x] RAG layer with FAISS + finance-domain embeddings
- [x] Orchestrator with keyword routing + explicit RAG status
- [x] Benchmark at orchestrator level vs FinGPT reference
- [ ] `multitask` agent retraining (Q&A + Headline only, 3 epochs) — in progress
- [ ] CLI (Typer) — interactive, batch, index management
- [ ] Gradio chat UI
- [ ] `forecaster` agent (fingpt-forecaster-dow30)
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
