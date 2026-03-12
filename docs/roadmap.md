# Roadmap

A living document tracking what's done, what's in progress, and what comes next.

---

## Phase 1 — Foundation (In Progress)

Build the core specialist adapters and establish benchmark baselines.

- [x] Set up Unsloth + Qwen3-8B training pipeline on consumer hardware
- [x] Migrate FinGPT ecosystem to Qwen3 architecture
- [x] Round 1: Train sentiment specialist adapter
- [ ] Round 2: Train multi-task adapter (sentiment + Q&A + headline)
- [ ] Run official FinGPT benchmarks on both adapters
- [ ] Compare F1 Weighted scores against published FinGPT reference numbers
- [ ] Export adapters to GGUF for local inference via llama.cpp

---

## Phase 2 — Specialization

Add task-specific agents that Round 2 deliberately left out.

- [ ] Round 3: Train relation extraction adapter on `fingpt-finred`
- [ ] Round 4: Train forecaster adapter on `fingpt-forecaster-dow30`
  - Requires bumping max_seq_length to 2048 for long financial reports
  - Will benchmark against FinGPT-Forecaster demo on HuggingFace

---

## Phase 3 — Orchestration

Wire the agents together into a working multi-agent system.

- [ ] Build query router (rule-based, no ML needed for v1)
- [ ] Implement adapter hot-swapping in inference loop
- [ ] Build simple CLI interface: ask a financial question, get routed answer
- [ ] Test multi-turn conversations where different turns hit different agents

---

## Phase 4 — RAG Integration

Give the system access to real-time financial data.

- [ ] Integrate FinGPT-RAG for live news retrieval
- [ ] Connect to a financial news API (e.g. Alpha Vantage, Yahoo Finance)
- [ ] Inject retrieved context into agent prompts at inference time
- [ ] Benchmark RAG vs no-RAG on time-sensitive questions

---

## Phase 5 — Interface

Make the system actually usable.

- [ ] Simple web UI (Gradio or Streamlit)
- [ ] Input: ticker symbol or free-text financial question
- [ ] Output: structured analysis with source citations
- [ ] Show which agent handled the query and why

---

## Stretch Goals

Things that would be cool but are not the immediate focus.

- [ ] RLHF layer: collect preference data and train reward model
- [ ] Real-time data pipeline with scheduled weekly fine-tuning updates
- [ ] Multi-stock portfolio analysis: run forecaster across N tickers
- [ ] Confidence scoring: model outputs uncertainty estimates
- [ ] Comparison study: Qwen3-8B adapters vs GPT-4 on financial benchmarks

---

## What This Project Does NOT Plan to Do

- Trade real money (this is educational only)
- Replace professional financial advice
- Scale to production serving (out of scope for a portfolio project)
