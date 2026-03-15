# Roadmap

A living document tracking what's done, what's in progress, and what comes next.

---

## Phase 1 — Foundation ✅ Complete

- [x] Set up Unsloth + Qwen3-8B training pipeline on consumer hardware
- [x] Round 1: Train `sentiment` adapter (fingpt-sentiment-train, r=8, 1 epoch)
- [x] Round 2: Train `multitask` adapter (sentiment + fiqa_qa + headline, r=16, 1 epoch)
- [x] Implement adapter hot-swapping on shared frozen base model
- [x] Run official FinGPT benchmarks at orchestrator level
- [x] Build RAG layer — FAISS vector store with finance-domain embeddings
- [x] Build orchestrator — keyword router + explicit RAG status handling
- [x] Write orchestrator requirements specification (v1.1)

---

## Phase 2 — Orchestrator MVP (In Progress)

Build the full interactive system around the trained agents.

- [ ] Round 2b: Retrain `multitask` adapter — sentiment removed, 3 epochs, alpha=32
  - Script ready: `scripts/round2b.py`
  - ~99.3K samples (fiqa_qa + headline only)
  - Checkpoints every ~10% (~4h intervals) — safe to stop and resume
  - Estimated runtime: ~40 hours
  - Rationale: benchmarks showed 2.6pp interference from sentiment data;
    dropping it keeps the multi-agent architecture properly specialized
- [ ] `rag/ingest.py` — ingestion pipeline (file, directory, yfinance)
- [ ] `cli.py` — Typer CLI (interactive, batch, index management)
- [ ] `ui.py` — Gradio chat interface
- [ ] Conversation history persistence (`history/session.jsonl`)
- [ ] Re-run benchmark with strip_thinking fix for clean scores
- [ ] Update orchestrator AGENTS dict to point at round2b adapter after training

---

## Phase 3 — Specialization

Add task-specific agents that expand system coverage.

- [ ] Round 3: Train `forecaster` adapter on `fingpt-forecaster-dow30`
  - Requires bumping max_seq_length to 2048 for long financial reports
  - Will benchmark against FinGPT-Forecaster demo on HuggingFace
  - Training config: lora_alpha=32, 3 epochs (matching round2b standards)

---

## Phase 4 — Interface & Deployment

Make the system publicly demoable.

- [ ] Push Gradio UI to HuggingFace Space for public portfolio demonstration
- [ ] Upload adapter weights to HuggingFace Hub (`hadioma/fingpt-*`)
- [ ] Write model cards for each adapter with benchmark results

---

## Stretch Goals

- [ ] RLHF layer: collect preference data and train reward model
- [ ] Real-time yfinance refresh on a configurable schedule
- [ ] Multi-stock portfolio analysis: run forecaster across N tickers
- [ ] FAISS IVF index for faster retrieval as document store scales beyond 100K chunks
- [ ] Confidence scoring: model outputs uncertainty estimates
- [ ] Additional data sources: SEC EDGAR filings, financial news RSS

---

## What This Project Does NOT Plan to Do

- Trade real money (educational and portfolio purposes only)
- Replace professional financial advice
- Scale to production serving
