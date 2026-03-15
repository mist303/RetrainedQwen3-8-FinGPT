"""
FinGPT Orchestrator Benchmark
==============================
Evaluates the full production pipeline end-to-end:
  classify → RAG → agent dispatch → generate → score

This matches how FinGPT benchmarks their published models — the score reflects
the deployed system, not individual components in isolation. Results are directly
comparable to the FinGPT paper reference numbers.

Why orchestrator-level (not per-adapter):
  - Routing is part of the product. If a query misroutes, the user gets a
    worse answer — that's a real quality signal worth capturing.
  - FinGPT's published F1 scores reflect their full pipeline, not isolated adapters.
  - The output shows which agent handled each task, so routing can be verified
    alongside accuracy.

Tasks and expected routing:
  FPB      — "What is the sentiment...?" → sentiment agent
  FiQA-SA  — "What is the sentiment...?" → sentiment agent
  Headline — "Does this headline...?"    → multitask agent
  QA       — open-ended questions        → multitask agent (default)

Metrics match the published FinGPT paper:
  F1 Weighted — primary (used in BloombergGPT comparisons)
  Accuracy, F1 Macro, F1 Micro — secondary

Qwen3 thinking mode:
  The orchestrator already calls strip_thinking() internally.
  Benchmark receives the clean final answer.

RAG note:
  If the RAG index is empty, the orchestrator proceeds without context
  and logs a warning. Benchmark scores reflect model knowledge only in
  that case — populate the index with `index fetch <ticker>` for
  context-aware production scores.

Usage (from repo root):
  python scripts/benchmark.py

Results: results/benchmark_results.json
Previous results archived to results/benchmark_results_YYYYMMDD_HHMMSS.json.

Dependencies:
  pip install scikit-learn unsloth peft safetensors datasets tqdm
"""

import os
import json
import warnings
import torch
from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orchestrator import Orchestrator

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

RESULTS_PATH = "./results/benchmark_results.json"
BATCH_SIZE   = 4

REFERENCES = {
    "FPB":      {"FinGPT_v3.3": 0.882, "BloombergGPT": 0.511, "GPT-4": 0.833},
    "FiQA-SA":  {"FinGPT_v3.3": 0.903, "BloombergGPT": None},
    "Headline": {"FinGPT_v3.3": 0.970, "BloombergGPT": None},
    "QA":       {},
}


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def archive_existing_results(path: str):
    if not os.path.exists(path):
        return
    ts        = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y%m%d_%H%M%S")
    base, ext = os.path.splitext(path)
    os.rename(path, f"{base}_{ts}{ext}")
    print(f"  Archived previous results → {base}_{ts}{ext}")


def normalize_sentiment(text: str) -> str:
    t = text.lower()
    if "positive" in t: return "positive"
    if "negative" in t: return "negative"
    return "neutral"


def normalize_headline(text: str) -> int:
    return 1 if "yes" in text.lower() else 0


def run_queries(orc: Orchestrator, prompts: list[str], desc: str) -> list[dict]:
    results = []
    for i, prompt in enumerate(tqdm(prompts, desc=desc)):
        r = orc.query(prompt, max_new_tokens=768)
        results.append({
            "answer":      r["answer"],
            "agent_used":  r["agent_used"],
            "rag_used":    r["rag_used"],
            "rag_warning": r["rag_warning"],
        })
        if (i + 1) % BATCH_SIZE == 0:
            torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK TASKS
# ─────────────────────────────────────────────────────────────────────────────

def run_fpb(orc: Orchestrator) -> dict:
    print("\n── FPB (Financial PhraseBank Sentiment) ──")
    ds = load_dataset("FinGPT/fingpt-sentiment-train", split="train")
    ds = ds.train_test_split(test_size=0.2, seed=42)["test"]
    instr   = "What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}."
    prompts = [f"{instr}\nInput: {row['input']}" for row in ds]
    targets = [normalize_sentiment(row["output"]) for row in ds]
    outputs = run_queries(orc, prompts, "FPB")
    preds   = [normalize_sentiment(o["answer"]) for o in outputs]
    agents  = [o["agent_used"] for o in outputs]
    acc = accuracy_score(targets, preds)
    f1w = f1_score(targets, preds, average="weighted", zero_division=0)
    f1a = f1_score(targets, preds, average="macro",    zero_division=0)
    f1i = f1_score(targets, preds, average="micro",    zero_division=0)
    agent_counts = {a: agents.count(a) for a in set(agents)}
    print(f"  Acc: {acc:.4f} | F1 Weighted: {f1w:.4f} | F1 Macro: {f1a:.4f} | F1 Micro: {f1i:.4f}")
    print(f"  Routing: {agent_counts}  |  Reference — FinGPT v3.3: 0.882 | BloombergGPT: 0.511")
    return {"task": "FPB", "n_samples": len(targets), "accuracy": round(acc,4),
            "f1_weighted": round(f1w,4), "f1_macro": round(f1a,4),
            "f1_micro": round(f1i,4), "routing": agent_counts}


def run_fiqa(orc: Orchestrator) -> dict:
    print("\n── FiQA-SA (Financial QA Sentiment) ──")
    ds = load_dataset("FinGPT/fingpt-fiqa_qa", split="train")
    ds = ds.train_test_split(test_size=0.2, seed=42)["test"]
    def get_instr(row):
        return ("What is the sentiment of this tweet? Please choose an answer from {negative/neutral/positive}."
                if row.get("format","") == "post"
                else "What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}.")
    prompts = [f"{get_instr(row)}\nInput: {row['input']}" for row in ds]
    targets = [normalize_sentiment(row["output"]) for row in ds]
    outputs = run_queries(orc, prompts, "FiQA")
    preds   = [normalize_sentiment(o["answer"]) for o in outputs]
    agents  = [o["agent_used"] for o in outputs]
    acc = accuracy_score(targets, preds)
    f1w = f1_score(targets, preds, average="weighted", zero_division=0)
    f1a = f1_score(targets, preds, average="macro",    zero_division=0)
    f1i = f1_score(targets, preds, average="micro",    zero_division=0)
    agent_counts = {a: agents.count(a) for a in set(agents)}
    print(f"  Acc: {acc:.4f} | F1 Weighted: {f1w:.4f} | F1 Macro: {f1a:.4f} | F1 Micro: {f1i:.4f}")
    print(f"  Routing: {agent_counts}  |  Reference — FinGPT v3.3: 0.903")
    return {"task": "FiQA-SA", "n_samples": len(targets), "accuracy": round(acc,4),
            "f1_weighted": round(f1w,4), "f1_macro": round(f1a,4),
            "f1_micro": round(f1i,4), "routing": agent_counts}


def run_headline(orc: Orchestrator) -> dict:
    print("\n── Headline (Financial Headline Classification) ──")
    ds = load_dataset("FinGPT/fingpt-headline", split="test")
    prompts = [f"{row['instruction']}\nInput: {row['input']}" for row in ds]
    targets = [1 if "yes" in row["output"].lower() else 0 for row in ds]
    outputs = run_queries(orc, prompts, "Headline")
    preds   = [normalize_headline(o["answer"]) for o in outputs]
    agents  = [o["agent_used"] for o in outputs]
    acc = accuracy_score(targets, preds)
    f1w = f1_score(targets, preds, average="weighted", zero_division=0)
    f1b = f1_score(targets, preds, average="binary",   zero_division=0)
    agent_counts = {a: agents.count(a) for a in set(agents)}
    print(f"  Acc: {acc:.4f} | F1 Weighted: {f1w:.4f} | F1 Binary: {f1b:.4f}")
    print(f"  Routing: {agent_counts}  |  Reference — FinGPT multi-task: ~0.970")
    return {"task": "Headline", "n_samples": len(targets), "accuracy": round(acc,4),
            "f1_weighted": round(f1w,4), "f1_binary": round(f1b,4),
            "routing": agent_counts}


QA_CASES = [
    ("What is the relationship between interest rates and bond prices?",
     ["inverse","fall","rise","yield","price","duration"], "core"),
    ("What does the P/E ratio indicate about a stock?",
     ["earnings","price","valuation","multiple","overvalued","undervalued"], "core"),
    ("Explain the difference between a bull market and a bear market.",
     ["rise","fall","bull","bear","decline","growth","optimism","pessimism"], "core"),
    ("What is quantitative easing and what are its effects?",
     ["money","supply","central bank","bonds","inflation","liquidity","stimulus"], "intermediate"),
    ("What are the key risks when investing in emerging market equities?",
     ["currency","political","volatility","liquidity","regulatory","risk"], "intermediate"),
    ("How does the yield curve inversion signal a recession?",
     ["short","long","invert","recession","rates","term","predict"], "advanced"),
    ("What is the difference between systematic and unsystematic risk?",
     ["market","diversif","specific","beta","portfolio","company"], "advanced"),
    ("Explain how a discounted cash flow (DCF) model works.",
     ["cash flow","discount","present value","rate","terminal","future"], "advanced"),
]


def run_qa(orc: Orchestrator) -> dict:
    print("\n── QA (Open-Ended Financial Q&A) ──")
    outputs = run_queries(orc, [c[0] for c in QA_CASES], "QA")
    scores, per_case = [], []
    for (q, kws, lvl), out in zip(QA_CASES, outputs):
        hits  = [kw for kw in kws if kw.lower() in out["answer"].lower()]
        score = len(hits) / len(kws)
        scores.append(score)
        print(f"\n  [{lvl.upper()}] {score:.0%}  agent={out['agent_used']}  Q: {q}")
        print(f"  A: {out['answer'][:250]}{'...' if len(out['answer'])>250 else ''}")
        print(f"  Keywords hit: {hits}")
        per_case.append({"question": q, "response": out["answer"],
                         "agent_used": out["agent_used"], "score": round(score,4),
                         "keywords_hit": hits, "difficulty": lvl})
    avg = sum(scores) / len(scores)
    agents = [o["agent_used"] for o in outputs]
    agent_counts = {a: agents.count(a) for a in set(agents)}
    print(f"\n  Average QA keyword score: {avg:.1%}  |  Routing: {agent_counts}")
    return {"task": "QA", "n_samples": len(QA_CASES),
            "avg_keyword_score": round(avg,4), "routing": agent_counts, "per_case": per_case}


# ─────────────────────────────────────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def print_system_summary(task_results: list):
    print(f"\n{'='*72}")
    print("  SYSTEM RESULTS vs FinGPT Reference")
    print(f"{'='*72}")
    print(f"  {'Task':<14} {'This System':>12} {'FinGPT v3.3':>12} {'Δ':>8}  Routing")
    print(f"  {'-'*68}")
    for r in task_results:
        task  = r["task"]
        score = r.get("avg_keyword_score") if task == "QA" else r["f1_weighted"]
        ref   = REFERENCES.get(task, {}).get("FinGPT_v3.3")
        print(f"  {task:<14} {score:>12.3f} {ref if ref else '—':>12}  "
              f"{f'{score-ref:+.3f}' if ref else 'n/a':>8}  {r.get('routing',{})}")
    print(f"\n  QA score = keyword hit rate (no published reference).")
    print(f"  Routing column shows which agent handled each task.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    archive_existing_results(RESULTS_PATH)

    print("Initialising orchestrator...")
    orc = Orchestrator(load_in_4bit=True, rag_top_k=5)

    task_results = [run_fpb(orc), run_fiqa(orc), run_headline(orc), run_qa(orc)]
    print_system_summary(task_results)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "_meta": {
                "run_date":    datetime.now().strftime("%Y-%m-%d"),
                "description": "Orchestrator-level benchmark — full pipeline including routing",
                "references":  REFERENCES,
            },
            "system_results": task_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Results saved to {RESULTS_PATH}")
