# FinGPT-Qwen3 — Self-Taught Financial AI Portfolio

> Fine-tuning open-source LLMs on financial data for sentiment analysis,
> Q&A, and stock forecasting — built on a consumer laptop with a single GPU.

---

## What This Project Is

This is a personal portfolio project documenting my journey fine-tuning
[Qwen3-8B](https://huggingface.co/unsloth/Qwen3-8B) on the
[FinGPT](https://github.com/AI4Finance-Foundation/FinGPT) ecosystem datasets.

The goal is to build a **multi-agent financial AI system** where each agent
is a specialized LoRA adapter trained for a distinct financial NLP task —
sentiment analysis, financial Q&A, headline classification, and eventually
stock price forecasting.

Everything here is trained on a **consumer laptop** (RTX 5070 Ti, 12GB VRAM,
32GB RAM) using [Unsloth](https://github.com/unslothai/unsloth) for 2x faster
training and 4-bit quantization to fit within VRAM constraints.

---

## System Architecture

```
                        User Query
                            │
                            ▼
                   ┌─────────────────┐
                   │   Orchestrator  │  ← routes query to the right agent
                   └────────┬────────┘
                            │
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                 ▼
  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
  │   Agent 1    │  │   Agent 2    │  │   Agent 3    │
  │  Sentiment   │  │  Multi-Task  │  │  Forecaster  │
  │  Specialist  │  │  (QA+Hdln)   │  │  (Planned)   │
  └──────────────┘  └──────────────┘  └──────────────┘
          │                 │
          └────────┬────────┘
                   ▼
          Shared Base Model
          Qwen3-8B (frozen, 4bit)
```

All agents share the same frozen Qwen3-8B base in memory.
At inference time, the orchestrator hot-swaps LoRA adapters
(`.safetensors` files) — no model reload required between agents.

---

## Training Progress

| Adapter | Base Model | Datasets | Status | Tasks |
|---|---|---|---|---|
| `adapter_round1` | Qwen3-8B | fingpt-sentiment-train | ✅ Done | Sentiment |
| `adapter_round2` | Qwen3-8B | sentiment + fiqa_qa + headline | 🔄 Training | Sentiment, Q&A, Headline |
| `adapter_finred` | Qwen3-8B | fingpt-finred | 📋 Planned | Relation Extraction |
| `adapter_forecaster` | Qwen3-8B | fingpt-forecaster-dow30 | 📋 Planned | Price Forecasting |

### Round 2 Training Details

```
Model:       Qwen3-8B (unsloth 4bit)
LoRA rank:   r=16, alpha=16
Datasets:    fingpt-sentiment-train (76.8K) +
             fingpt-fiqa_qa (17.1K) +
             fingpt-headline (82.2K)
             ────────────────────────────
             Total: ~174K samples
Epochs:      1
Batch size:  2 (device) × 8 (grad accum) = 16 effective
Warmup:      200 steps
Total steps: 10,888
Hardware:    RTX 5070 Ti (12GB VRAM), 32GB RAM
Optimizer:   paged_adamw_8bit
Precision:   bf16
```

---

## Key Design Decisions

**Why separate LoRA adapters instead of one big model?**
Mixing tasks like sentiment classification (rigid output) and open-ended Q&A
(generative) in a single training run causes task interference — the model
learns to be mediocre at both. Separate adapters stay specialized and can be
individually retrained when new data arrives, without touching other agents.

**Why Qwen3-8B?**
Qwen3 is the latest generation with strong multilingual support and a 32k
context window. It's a natural fit for financial text which often contains
mixed-language content and long-form analysis.

**Why Unsloth?**
2x faster training + gradient checkpointing + 4-bit quantization makes
training a 8B model feasible on 12GB VRAM. Without it, this project wouldn't
run on consumer hardware at all.

**Why build this locally instead of using the API?**
The whole point is to own the weights. A locally-trained adapter can be
updated weekly with new financial data for under $1 in electricity.
BloombergGPT cost $2.67M to train. FinGPT-style lightweight adaptation
brings that down to under $300. This project brings it down further to
a consumer GPU.

---

## Benchmark Results

Benchmarks run using the official FinGPT evaluation methodology
(sklearn F1 Weighted, matching the BloombergGPT comparison metric).

> Results will be populated after Round 2 training completes.
> See `results/benchmark_results.json` for raw output.

| Task | Dataset | Round 2 Score | FinGPT Reference | BloombergGPT |
|---|---|---|---|---|
| Sentiment | FPB | _pending_ | 0.882 | 0.511 |
| Sentiment | FiQA-SA | _pending_ | 0.903 | — |
| Headline | fingpt-headline | _pending_ | ~0.970 | — |
| Q&A | fiqa_qa (keyword) | _pending_ | — | — |

---

## Repository Structure

```
FinGPT-Portfolio/
│
├── README.md                  ← You are here
│
├── scripts/
│   ├── round1.py              ← Round 1 sentiment specialist training + merge
│   ├── round2.py              ← Round 2 multi-task training + merge
│   └── benchmark.py           ← Official FinGPT benchmark evaluation
│
├── docs/
│   ├── architecture.md        ← Full multi-agent system design
│   ├── training.md            ← Training decisions, configs, lessons learned
│   └── roadmap.md             ← What comes next
│
└── results/
    └── benchmark_results.json ← Benchmark output (populated after eval)
```

---

## Roadmap

See [`docs/roadmap.md`](docs/roadmap.md) for the full plan.

Short version:
- [x] Round 1: Sentiment specialist adapter
- [ ] Round 2: Multi-task adapter (sentiment + Q&A + headline)
- [ ] Benchmark & compare adapters against FinGPT reference scores
- [ ] Round 3: Relation extraction adapter (fingpt-finred)
- [ ] Orchestrator: Router that selects the right adapter per query
- [ ] Forecaster: Stock price movement prediction (fingpt-forecaster-dow30)
- [ ] RAG layer: Real-time news injection via FinGPT-RAG

---

## Built On

- [FinGPT](https://github.com/AI4Finance-Foundation/FinGPT) — AI4Finance Foundation
- [Unsloth](https://github.com/unslothai/unsloth) — 2x faster fine-tuning
- [Qwen3](https://huggingface.co/Qwen/Qwen3-8B) — Alibaba Cloud
- [HuggingFace PEFT](https://github.com/huggingface/peft) — LoRA implementation

---

## Disclaimer

This project is for educational and portfolio purposes only.
Nothing here constitutes financial advice or a recommendation to trade.
