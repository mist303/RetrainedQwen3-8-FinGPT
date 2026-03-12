# Architecture — Multi-Agent Financial AI System

## Overview

The system is designed around a core principle from the FinGPT ecosystem:
**specialized agents outperform generalist models on financial NLP tasks.**

Rather than training one model to do everything, each agent is a lightweight
LoRA adapter (~40MB) trained on a specific task. They all share the same
frozen Qwen3-8B base model in memory, and the orchestrator decides which
adapter to activate per query.

---

## The Five-Layer FinGPT Framework

This project implements layers 3–5 of the FinGPT architecture:

```
Layer 5 │ Application     │ Orchestrator + user-facing interface
Layer 4 │ Task            │ Specialized LoRA agents (this project)
Layer 3 │ LLMs            │ Qwen3-8B + Unsloth fine-tuning (this project)
Layer 2 │ Data Engineering│ FinNLP preprocessing pipelines
Layer 1 │ Data Source     │ FinGPT HuggingFace datasets
```

---

## Agent Design

### Why LoRA Adapters as Agents?

Standard multi-task training mixes diverse objectives into a single set of
weights. For financial NLP, this creates problems:

- Sentiment analysis expects rigid 3-class output: `{negative/neutral/positive}`
- Q&A is generative and open-ended
- Relation extraction outputs structured entity pairs
- Forecasting requires multi-paragraph analysis

Training these together causes **task interference** — the gradient updates
from one task partially undo what another task learned. The result is a model
that's mediocre at everything rather than excellent at one thing.

LoRA adapters solve this by isolating each task into its own small set of
trainable parameters (~40M out of 8.2B total, ~0.5%). The base model's
general language understanding stays intact.

### Memory Efficiency of the Multi-Agent Approach

On 12GB VRAM, loading Qwen3-8B in 4-bit takes approximately 5–6GB.
That leaves headroom for inference. Because all adapters share the same base:

```
Memory used = base_model (5.5GB) + active_adapter (~80MB) = ~5.6GB total
```

Compare this to loading 4 separate 8B models: ~22GB — impossible on a
consumer GPU. The adapter approach makes multi-agent feasible on a laptop.

### Adapter Hot-Swapping

At inference time, swapping adapters takes milliseconds:

```python
# Swap from sentiment agent to Q&A agent
state_dict = load_file("adapter_qa.safetensors")
set_peft_model_state_dict(model, state_dict)
```

No model reload, no VRAM spike, no latency overhead.

---

## Current Agent Roster

### Agent 1 — Sentiment Specialist (`adapter_round1`)
- **Training data:** `FinGPT/fingpt-sentiment-train` (76.8K samples)
- **Task:** 3-class sentiment classification on financial news and tweets
- **Output format:** One word — `positive`, `neutral`, or `negative`
- **LoRA config:** r=8, alpha=16
- **Status:** ✅ Complete

### Agent 2 — Multi-Task (`adapter_round2`)
- **Training data:**
  - `FinGPT/fingpt-sentiment-train` (76.8K)
  - `FinGPT/fingpt-fiqa_qa` (17.1K)
  - `FinGPT/fingpt-headline` (82.2K)
  - Total: ~174K samples
- **Tasks:**
  - Sentiment analysis on financial news/tweets
  - Open-ended financial Q&A
  - Binary headline price movement classification
- **LoRA config:** r=16, alpha=16 (higher rank for multi-task capacity)
- **Status:** 🔄 Training

### Agent 3 — Relation Extraction (`adapter_finred`) — Planned
- **Training data:** `FinGPT/fingpt-finred` (27.6K samples)
- **Task:** Extract entity-relation pairs from financial text
- **Output format:** `relation: entity1, entity2; relation2: entity3, entity4`
- **Why separate:** Structural output format is incompatible with
  sentiment/QA training — would cause catastrophic forgetting if mixed
- **Status:** 📋 Planned

### Agent 4 — Forecaster (`adapter_forecaster`) — Planned
- **Training data:** `FinGPT/fingpt-forecaster-dow30-202305-202405`
- **Task:** Multi-paragraph stock price movement prediction given news + financials
- **Output format:** [Positive Developments] / [Potential Concerns] / [Prediction]
- **Status:** 📋 Planned

---

## Orchestrator Design (Planned)

The orchestrator routes queries to the correct agent. The routing logic is
intentionally simple — no ML needed:

```python
def route(query: str) -> str:
    q = query.lower()
    if any(k in q for k in ["sentiment", "feeling", "tone", "positive", "negative"]):
        return "sentiment"
    elif any(k in q for k in ["predict", "forecast", "next week", "price movement"]):
        return "forecaster"
    elif any(k in q for k in ["relation", "who owns", "founded by", "subsidiary"]):
        return "finred"
    else:
        return "multitask"   # default: Q&A agent handles open-ended questions
```

For ambiguous queries, the default is the multi-task agent since it handles
the broadest range of financial topics.

---

## Planned RAG Integration

Once the base agents are working, the next layer is
[FinGPT-RAG](https://github.com/AI4Finance-Foundation/FinGPT/tree/master/fingpt/FinGPT_RAG)
for real-time news injection:

```
User Query
    │
    ├── RAG Layer ──► fetch recent news ──► inject into context
    │
    └── Agent ──► answer with live context
```

This closes the knowledge gap for time-sensitive financial queries where
training data is already stale.
