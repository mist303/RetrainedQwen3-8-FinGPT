"""
FinGPT Orchestrator Benchmark
==============================
Evaluates the full production pipeline end-to-end:
  user prompt → orchestrator → classify → RAG → agent dispatch → generate → score

All four tasks and dataset sources match the OFFICIAL FinGPT benchmark exactly,
making our F1 scores directly comparable to the published FinGPT v3.3 numbers.

  Task      Dataset                                      Split
  ───────── ──────────────────────────────────────────── ──────────────────────
  FPB       financial_phrasebank / sentences_50agree     train_test_split(seed=42) default
  FiQA-SA   pauri32/fiqa-2018                            concat splits, test_size=0.226
  TFNS      zeroshot/twitter-financial-news-sentiment    validation split
  NWGI      oliverwang15/news_with_gpt_instructions      train split (no explicit test)
  Headline  FinGPT/fingpt-headline                       explicit test split

Reference scores (FinGPT v3.3, Llama2-13B LoRA single-task):
  FPB F1w:      0.882
  FiQA-SA F1w:  0.874
  TFNS F1w:     0.903
  NWGI F1w:     0.643
  Headline:     ~0.970

Routing through the orchestrator (keyword router, no bypass):
  FPB / FiQA / TFNS / NWGI  → contain "sentiment" → sentiment agent
  Headline                   → contains "headline"  → multitask agent
  QA                         → open-ended           → multitask agent (default)

History is reset between tasks to prevent prior turns contaminating results.
"""

# unsloth MUST be first import — patches torch/transformers/peft at import time
import unsloth  # noqa: F401

import os
import json
import warnings
import torch
from pathlib import Path
import datasets as hf_datasets
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import platform

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from orchestrator import Orchestrator

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CROSS-PLATFORM COMPATIBILITY
# ─────────────────────────────────────────────────────────────────────────────

OS_TYPE = platform.system()  # "Linux", "Windows", "Darwin"
print(f"Running on: {OS_TYPE} (Python {platform.python_version()})")

# Ensure CUDA libraries are accessible on Linux
if OS_TYPE == "Linux":
    os.environ["CUDA_HOME"] = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    cuda_lib = os.path.join(os.environ["CUDA_HOME"], "lib64")
    if cuda_lib not in os.environ.get("LD_LIBRARY_PATH", ""):
        os.environ["LD_LIBRARY_PATH"] = f"{cuda_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}"

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Use Path objects for cross-platform path handling
RESULTS_DIR  = Path("./results")
RESULTS_PATH = RESULTS_DIR / "benchmark_results.json"

# Per-task token budgets.
# Classification only needs 1-3 tokens (positive/neutral/negative/yes/no).
# Using a large number causes repetitive looping and wastes ~30s per sample.
MAX_TOKENS_CLASSIFY = 16
MAX_TOKENS_QA       = 256

# Official FinGPT v3.3 reference scores (Llama2-13B single-task LoRA).
# Source: https://github.com/AI4Finance-Foundation/FinGPT README benchmark table.
# FiQA-SA is 0.874 — NOT 0.903 (that's TFNS).
REFERENCES = {
    "FPB":      {"FinGPT_v3.3": 0.882, "BloombergGPT": 0.511, "GPT-4": 0.833},
    "FiQA-SA":  {"FinGPT_v3.3": 0.874, "BloombergGPT": None,  "GPT-4": 0.630},
    "TFNS":     {"FinGPT_v3.3": 0.903, "BloombergGPT": None},
    "NWGI":     {"FinGPT_v3.3": 0.643, "BloombergGPT": None},
    "Headline": {"FinGPT_v3.3": 0.970, "BloombergGPT": None},
    "QA":       {},   # no published reference — internal keyword metric only
}

# ─────────────────────────────────────────────────────────────────────────────
# LABEL NORMALIZERS  (mirror FinGPT's change_target / map_output exactly)
# ─────────────────────────────────────────────────────────────────────────────

def normalize_sentiment(text: str) -> str:
    """
    Mirror of FinGPT's change_target().
    Checks for keyword anywhere in the model output — the model may output
    'The sentiment is positive.' rather than just 'positive'.
    Priority: positive > negative > neutral (neutral is the fallback).
    """
    t = text.lower()
    if "positive" in t: return "positive"
    if "negative" in t: return "negative"
    return "neutral"


def normalize_headline(text: str) -> int:
    """Mirror of FinGPT's map_output(). 1=yes, 0=no."""
    return 1 if "yes" in text.lower() else 0


def fiqa_score_to_label(score: float) -> str:
    """
    Mirror of FinGPT's make_label().
    Thresholds: < -0.1 → negative | -0.1 to 0.1 → neutral | ≥ 0.1 → positive
    """
    if score < -0.1: return "negative"
    if score >= 0.1: return "positive"
    return "neutral"


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR INFERENCE WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

def run_queries(
    orc:            Orchestrator,
    prompts:        list[str],
    desc:           str,
    max_new_tokens: int = MAX_TOKENS_CLASSIFY,
    agent_override: str | None = None,
) -> list[dict]:
    """
    Route every prompt through the full orchestrator pipeline:
      classify_query → RAG enrich → _load_agent → _generate

    We never bypass routing — agent_override is None for all standard tasks
    so the keyword router picks the agent naturally. This is what makes our
    benchmark an orchestrator-level test rather than an isolated adapter test.

    Flushes CUDA cache every 16 queries to prevent VRAM fragmentation
    across the 15K+ sample FPB and 20K+ sample Headline runs.
    """
    results = []
    for i, prompt in enumerate(tqdm(prompts, desc=desc)):
        result = orc.query(
            prompt,
            agent_override=agent_override,
            max_new_tokens=max_new_tokens,
            record_history=False,  # prevent accumulation truncating later prompts
        )
        results.append({
            "answer":      result["answer"],
            "agent_used":  result["agent_used"],
            "rag_used":    result["rag_used"],
            "rag_warning": result["rag_warning"],
        })
        if (i + 1) % 16 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1: FPB — Financial PhraseBank Sentiment
# ─────────────────────────────────────────────────────────────────────────────
# Source:   financial_phrasebank, sentences_50agree config
# Split:    train_test_split(seed=42) — NO test_size (matches FinGPT exactly,
#           HuggingFace default is 0.25, not 0.2)
# Labels:   {0: negative, 1: neutral, 2: positive}
# Routing:  "sentiment" in prompt → sentiment agent
# ─────────────────────────────────────────────────────────────────────────────

# Local data directory — populated by running download_benchmark_data.py once
_DATA_DIR = Path(__file__).parent / "fingpt" / "FinGPT_Benchmark" / "data"


def _load_fpb() -> hf_datasets.Dataset:
    """
    Load Financial PhraseBank sentences_50agree (4,846 sentences).

    financial_phrasebank and all its mirrors use a legacy loading script
    that newer datasets versions no longer support. We use:
      - Disk cache (fingpt/FinGPT_Benchmark/data/financial_phrasebank) if available
      - atrost/financial_phrasebank as fallback — clean Parquet re-host,
        same sentence/label columns, no loading script

    atrost splits the data as train/valid/test. We concatenate all splits
    and do our own seed=42 split to match FinGPT's exact evaluation call.
    """
    local = _DATA_DIR / "financial_phrasebank"
    if local.exists():
        ds = hf_datasets.load_from_disk(str(local))
        # Concatenate all available splits into one pool
        all_splits = hf_datasets.concatenate_datasets(
            [ds[s] for s in ds.keys()]
        )
    else:
        print("  [FPB] Downloading atrost/financial_phrasebank (no loading script)...")
        raw = hf_datasets.load_dataset("atrost/financial_phrasebank")
        all_splits = hf_datasets.concatenate_datasets(
            [raw[s] for s in raw.keys()]
        )
    # No test_size → HuggingFace default 0.25, matches FinGPT's exact call
    return all_splits.train_test_split(seed=42)["test"]


def run_fpb(orc: Orchestrator) -> dict:
    print("\n── FPB (Financial PhraseBank) ──")
    ds = _load_fpb()

    # FPB label mapping: {0: negative, 1: neutral, 2: positive}
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    instr   = "What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}."
    prompts = [f"{instr}\nInput: {row['sentence']}" for row in ds]
    targets = [label_map[row["label"]] for row in ds]

    outputs = run_queries(orc, prompts, "FPB")
    preds   = [normalize_sentiment(o["answer"]) for o in outputs]
    agents  = [o["agent_used"] for o in outputs]

    acc = accuracy_score(targets, preds)
    f1w = f1_score(targets, preds, average="weighted", zero_division=0)
    f1a = f1_score(targets, preds, average="macro",    zero_division=0)
    f1i = f1_score(targets, preds, average="micro",    zero_division=0)
    agent_counts = {a: agents.count(a) for a in set(agents)}

    print(f"  n={len(targets)} | Acc={acc:.4f} | F1w={f1w:.4f} | F1a={f1a:.4f} | F1i={f1i:.4f}")
    print(f"  Routing: {agent_counts}")
    ref = REFERENCES["FPB"]
    print(f"  Reference: FinGPT v3.3={ref['FinGPT_v3.3']} | BloombergGPT={ref['BloombergGPT']}")

    return {
        "task": "FPB",
        "dataset": "financial_phrasebank/sentences_50agree (seed=42 default split)",
        "n_samples": len(targets),
        "accuracy": round(acc, 4), "f1_weighted": round(f1w, 4),
        "f1_macro": round(f1a, 4), "f1_micro":    round(f1i, 4),
        "routing": agent_counts,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TASK 2: FiQA-SA — Financial QA Sentiment Analysis
# ─────────────────────────────────────────────────────────────────────────────
# Source:   pauri32/fiqa-2018, train + validation + test concatenated
# Split:    test_size=0.226, seed=42  (matches FinGPT exactly)
# Labels:   derived from sentiment_score float via make_label() thresholds
# Routing:  "sentiment" in prompt → sentiment agent
# ─────────────────────────────────────────────────────────────────────────────

def run_fiqa(orc: Orchestrator) -> dict:
    print("\n── FiQA-SA (Financial QA Sentiment) ──")
    raw      = hf_datasets.load_dataset("pauri32/fiqa-2018")
    combined = hf_datasets.concatenate_datasets(
        [raw["train"], raw["validation"], raw["test"]]
    )
    ds = combined.train_test_split(test_size=0.226, seed=42)["test"]

    def get_instr(row) -> str:
        """Mirror of FinGPT's add_instructions() using the format field."""
        if row.get("format", "") == "post":
            return "What is the sentiment of this tweet? Please choose an answer from {negative/neutral/positive}."
        return "What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}."

    prompts = [f"{get_instr(row)}\nInput: {row['sentence']}" for row in ds]
    targets = [fiqa_score_to_label(row["sentiment_score"]) for row in ds]

    outputs = run_queries(orc, prompts, "FiQA")
    preds   = [normalize_sentiment(o["answer"]) for o in outputs]
    agents  = [o["agent_used"] for o in outputs]

    acc = accuracy_score(targets, preds)
    f1w = f1_score(targets, preds, average="weighted", zero_division=0)
    f1a = f1_score(targets, preds, average="macro",    zero_division=0)
    f1i = f1_score(targets, preds, average="micro",    zero_division=0)
    agent_counts = {a: agents.count(a) for a in set(agents)}

    print(f"  n={len(targets)} | Acc={acc:.4f} | F1w={f1w:.4f} | F1a={f1a:.4f} | F1i={f1i:.4f}")
    print(f"  Routing: {agent_counts}")
    print(f"  Reference: FinGPT v3.3={REFERENCES['FiQA-SA']['FinGPT_v3.3']}")

    return {
        "task": "FiQA-SA",
        "dataset": "pauri32/fiqa-2018 (concat train+val+test, split=0.226)",
        "n_samples": len(targets),
        "accuracy": round(acc, 4), "f1_weighted": round(f1w, 4),
        "f1_macro": round(f1a, 4), "f1_micro":    round(f1i, 4),
        "routing": agent_counts,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TASK 3: TFNS — Twitter Financial News Sentiment
# ─────────────────────────────────────────────────────────────────────────────
# Source:   zeroshot/twitter-financial-news-sentiment, validation split
# Labels:   {0: negative, 1: positive, 2: neutral}
#           ⚠ DIFFERENT ordering from FPB (1=positive here, 1=neutral in FPB)
# Routing:  "sentiment" in prompt → sentiment agent
# ─────────────────────────────────────────────────────────────────────────────

def run_tfns(orc: Orchestrator) -> dict:
    print("\n── TFNS (Twitter Financial News Sentiment) ──")
    raw = hf_datasets.load_dataset("zeroshot/twitter-financial-news-sentiment")
    ds  = raw["validation"]

    # CRITICAL: TFNS label ordering differs from FPB.
    # FPB:  {0: negative, 1: neutral,  2: positive}
    # TFNS: {0: negative, 1: positive, 2: neutral}   ← positive and neutral swapped
    label_map = {0: "negative", 1: "positive", 2: "neutral"}
    instr   = "What is the sentiment of this tweet? Please choose an answer from {negative/neutral/positive}."
    prompts = [f"{instr}\nInput: {row['text']}" for row in ds]
    targets = [label_map[row["label"]] for row in ds]

    outputs = run_queries(orc, prompts, "TFNS")
    preds   = [normalize_sentiment(o["answer"]) for o in outputs]
    agents  = [o["agent_used"] for o in outputs]

    acc = accuracy_score(targets, preds)
    f1w = f1_score(targets, preds, average="weighted", zero_division=0)
    f1a = f1_score(targets, preds, average="macro",    zero_division=0)
    f1i = f1_score(targets, preds, average="micro",    zero_division=0)
    agent_counts = {a: agents.count(a) for a in set(agents)}

    print(f"  n={len(targets)} | Acc={acc:.4f} | F1w={f1w:.4f} | F1a={f1a:.4f} | F1i={f1i:.4f}")
    print(f"  Routing: {agent_counts}")
    print(f"  Reference: FinGPT v3.3={REFERENCES['TFNS']['FinGPT_v3.3']}")

    return {
        "task": "TFNS",
        "dataset": "zeroshot/twitter-financial-news-sentiment (validation)",
        "n_samples": len(targets),
        "accuracy": round(acc, 4), "f1_weighted": round(f1w, 4),
        "f1_macro": round(f1a, 4), "f1_micro":    round(f1i, 4),
        "routing": agent_counts,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TASK 4: NWGI — News With GPT Instructions
# ─────────────────────────────────────────────────────────────────────────────
# Source:   oliverwang15/news_with_gpt_instructions, train split
# Labels:   7-class → 3-class mapping (mildly positive/negative → neutral)
# Routing:  "sentiment" in prompt → sentiment agent
# ─────────────────────────────────────────────────────────────────────────────

# Mirror of FinGPT's nwgi.py dic — mildly variants collapse to neutral
NWGI_LABEL_MAP = {
    "strong negative":    "negative",
    "moderately negative":"negative",
    "mildly negative":    "neutral",   # ← collapses to neutral, not negative
    "neutral":            "neutral",
    "mildly positive":    "neutral",   # ← collapses to neutral, not positive
    "moderately positive":"positive",
    "strong positive":    "positive",
}

def run_nwgi(orc: Orchestrator) -> dict:
    print("\n── NWGI (News With GPT Instructions) ──")
    raw = hf_datasets.load_dataset("oliverwang15/news_with_gpt_instructions")

    # FinGPT loads the full dataset from disk with no split.
    # HuggingFace version has a 'train' split — use that to match FinGPT's full-set eval.
    split_name = "train" if "train" in raw else list(raw.keys())[0]
    ds = raw[split_name]

    instr   = "What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}."
    prompts = [f"{instr}\nInput: {row['news']}" for row in ds]

    # Apply 7→3 label mapping — same as FinGPT's dic
    targets = []
    for row in ds:
        raw_label = row["label"].strip().lower()
        targets.append(NWGI_LABEL_MAP.get(raw_label, "neutral"))

    outputs = run_queries(orc, prompts, "NWGI")
    preds   = [normalize_sentiment(o["answer"]) for o in outputs]
    agents  = [o["agent_used"] for o in outputs]

    acc = accuracy_score(targets, preds)
    f1w = f1_score(targets, preds, average="weighted", zero_division=0)
    f1a = f1_score(targets, preds, average="macro",    zero_division=0)
    f1i = f1_score(targets, preds, average="micro",    zero_division=0)
    agent_counts = {a: agents.count(a) for a in set(agents)}

    print(f"  n={len(targets)} | Acc={acc:.4f} | F1w={f1w:.4f} | F1a={f1a:.4f} | F1i={f1i:.4f}")
    print(f"  Routing: {agent_counts}")
    print(f"  Reference: FinGPT v3.3={REFERENCES['NWGI']['FinGPT_v3.3']}")

    return {
        "task": "NWGI",
        "dataset": f"oliverwang15/news_with_gpt_instructions ({split_name})",
        "n_samples": len(targets),
        "accuracy": round(acc, 4), "f1_weighted": round(f1w, 4),
        "f1_macro": round(f1a, 4), "f1_micro":    round(f1i, 4),
        "routing": agent_counts,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TASK 5: Headline — Financial Headline Price Movement
# ─────────────────────────────────────────────────────────────────────────────
# Source:   FinGPT/fingpt-headline, explicit test split
# Labels:   Yes/No binary
# Routing:  "headline" in instruction → multitask agent
# ─────────────────────────────────────────────────────────────────────────────

def run_headline(orc: Orchestrator) -> dict:
    print("\n── Headline (Financial Headline Classification) ──")
    ds = hf_datasets.load_dataset("FinGPT/fingpt-headline", split="test")

    prompts = [f"{row['instruction']}\nInput: {row['input']}" for row in ds]
    targets = [1 if "yes" in row["output"].lower() else 0 for row in ds]

    outputs = run_queries(orc, prompts, "Headline")
    preds   = [normalize_headline(o["answer"]) for o in outputs]
    agents  = [o["agent_used"] for o in outputs]

    acc = accuracy_score(targets, preds)
    f1w = f1_score(targets, preds, average="weighted", zero_division=0)
    f1b = f1_score(targets, preds, average="binary",   zero_division=0)
    agent_counts = {a: agents.count(a) for a in set(agents)}

    print(f"  n={len(targets)} | Acc={acc:.4f} | F1w={f1w:.4f} | F1b={f1b:.4f}")
    print(f"  Routing: {agent_counts}")
    print(f"  Reference: FinGPT v3.3≈{REFERENCES['Headline']['FinGPT_v3.3']}")

    return {
        "task": "Headline",
        "dataset": "FinGPT/fingpt-headline (test split)",
        "n_samples": len(targets),
        "accuracy": round(acc, 4), "f1_weighted": round(f1w, 4),
        "f1_binary": round(f1b, 4),
        "routing": agent_counts,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TASK 6: QA — Open-Ended Financial Q&A (no FinGPT reference)
# ─────────────────────────────────────────────────────────────────────────────

QA_CASES = [
    ("What is the relationship between interest rates and bond prices?",
     ["inverse", "fall", "rise", "yield", "price", "duration"], "core"),
    ("What does the P/E ratio indicate about a stock?",
     ["earnings", "price", "valuation", "multiple", "overvalued", "undervalued"], "core"),
    ("Explain the difference between a bull market and a bear market.",
     ["rise", "fall", "bull", "bear", "decline", "growth", "optimism", "pessimism"], "core"),
    ("What is quantitative easing and what are its effects?",
     ["money", "supply", "central bank", "bonds", "inflation", "liquidity", "stimulus"], "intermediate"),
    ("What are the key risks when investing in emerging market equities?",
     ["currency", "political", "volatility", "liquidity", "regulatory", "risk"], "intermediate"),
    ("How does the yield curve inversion signal a recession?",
     ["short", "long", "invert", "recession", "rates", "term", "predict"], "advanced"),
    ("What is the difference between systematic and unsystematic risk?",
     ["market", "diversif", "specific", "beta", "portfolio", "company"], "advanced"),
    ("Explain how a discounted cash flow (DCF) model works.",
     ["cash flow", "discount", "present value", "rate", "terminal", "future"], "advanced"),
]


def run_qa(orc: Orchestrator) -> dict:
    print("\n── QA (Open-Ended Financial Q&A) ──")
    prompts = [c[0] for c in QA_CASES]
    outputs = run_queries(orc, prompts, "QA", max_new_tokens=MAX_TOKENS_QA)

    scores, per_case = [], []
    for (q, kws, lvl), out in zip(QA_CASES, outputs):
        r    = out["answer"].lower()
        hits = [kw for kw in kws if kw.lower() in r]
        s    = len(hits) / len(kws)
        scores.append(s)
        print(f"\n  [{lvl.upper()}] {s:.0%}  agent={out['agent_used']}")
        print(f"  Q: {q}")
        print(f"  A: {out['answer'][:200]}{'...' if len(out['answer'])>200 else ''}")
        print(f"  Keywords hit: {hits}")
        per_case.append({
            "question": q, "response": out["answer"],
            "agent_used": out["agent_used"], "score": round(s, 4),
            "keywords_hit": hits, "difficulty": lvl,
        })

    avg = sum(scores) / len(scores)
    agent_counts = {a: [o["agent_used"] for o in outputs].count(a)
                    for a in set(o["agent_used"] for o in outputs)}
    print(f"\n  Avg keyword score: {avg:.1%} | Routing: {agent_counts}")

    return {
        "task": "QA", "n_samples": len(QA_CASES),
        "avg_keyword_score": round(avg, 4),
        "routing": agent_counts, "per_case": per_case,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(task_results: list):
    print(f"\n{'='*78}")
    print("  SYSTEM vs FinGPT v3.3 Reference  (F1 Weighted)")
    print(f"{'='*78}")
    print(f"  {'Task':<12} {'Ours':>10} {'FinGPT v3.3':>12} {'Δ':>8}  Routing")
    print(f"  {'-'*74}")
    for r in task_results:
        task  = r["task"]
        score = r.get("avg_keyword_score") if task == "QA" else r.get("f1_weighted", 0)
        ref   = REFERENCES.get(task, {}).get("FinGPT_v3.3")
        s_str = f"{score:.3f}"
        r_str = f"{ref:.3f}" if ref else "      —"
        d_str = f"{score - ref:+.3f}" if ref else "    n/a"
        print(f"  {task:<12} {s_str:>10} {r_str:>12} {d_str:>8}  {r.get('routing', {})}")
    print(f"\n  QA = keyword hit rate (no published FinGPT reference).")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Create results directory with proper permissions on Linux
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Archive previous results with timestamp so no run is lost
    if RESULTS_PATH.exists():
        ts = datetime.fromtimestamp(RESULTS_PATH.stat().st_mtime).strftime("%Y%m%d_%H%M%S")
        archived_path = RESULTS_PATH.parent / f"{RESULTS_PATH.stem}_{ts}{RESULTS_PATH.suffix}"
        RESULTS_PATH.rename(archived_path)
        print(f"Archived previous results → {archived_path}")

    orc = Orchestrator(load_in_4bit=True, rag_top_k=5)

    # History reset between tasks is CRITICAL.
    # Without it, FiQA runs with 15K FPB turns in Qwen3's context window,
    # TFNS runs with 15K + 3K turns, etc. — completely corrupting results.
    task_fns = [run_fpb, run_fiqa, run_tfns, run_nwgi, run_headline, run_qa]
    task_results = []
    for fn in task_fns:
        orc.reset_history()
        task_results.append(fn(orc))

    print_summary(task_results)

    output = {
        "_meta": {
            "run_date":   datetime.now().strftime("%Y-%m-%d %H:%M"),
            "datasets": {
                "FPB":      "takala/financial_phrasebank/sentences_50agree (seed=42, default split ~25%) — same data as financial_phrasebank, Parquet re-host",
                "FiQA-SA":  "pauri32/fiqa-2018 (concat train+val+test, test_size=0.226, seed=42)",
                "TFNS":     "zeroshot/twitter-financial-news-sentiment (validation split)",
                "NWGI":     "oliverwang15/news_with_gpt_instructions (train split)",
                "Headline": "FinGPT/fingpt-headline (explicit test split)",
            },
            "references": REFERENCES,
        },
        "results": task_results,
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    # Ensure proper file permissions on Linux
    if OS_TYPE == "Linux":
        RESULTS_PATH.chmod(0o644)  # readable by owner and group
    
    print(f"\n✓ Results saved to {RESULTS_PATH}")
