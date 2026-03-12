# benchmark.py — Official FinGPT Benchmark Evaluation
# Evaluates trained LoRA adapters using the same metrics as the FinGPT paper.
# Adapts fingpt/FinGPT_Benchmark/benchmarks/ for Unsloth + Qwen3 chat template.
#
# Usage:
#   python scripts/benchmark.py
#
# Adapters are hot-swapped on the same frozen base model — no reload between runs.
# Results saved to results/benchmark_results.json
#
# Dependencies: pip install scikit-learn unsloth peft safetensors datasets tqdm

import os
import json
import warnings
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from unsloth import FastLanguageModel
from safetensors.torch import load_file
from peft import set_peft_model_state_dict

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL  = "unsloth/Qwen3-8B"
MAX_LEN     = 512
BATCH_SIZE  = 4      # safe for 12GB VRAM at inference (no gradients)
MAX_NEW_TOK = 32     # sentiment/headline only need a word; keeps eval fast

# Adapter paths relative to the FinGPT project root
# Update these paths to point to your actual adapter locations
ADAPTERS = {
    "Round1_Sentiment": "./qwen3-8b-fingpt-lora/adapter_model.safetensors",
    "Round2_MultiTask": "./qwen3-8b-round2-lora/adapter_model.safetensors",
}

# Must exactly match the system prompt used during training
SYSTEM = (
    "You are an expert financial analyst. "
    "Reason carefully, cite your logic, and provide structured, professional analysis."
)


# ── Label Normalization ───────────────────────────────────────────────────────
# Adapted from fingpt/FinGPT_Benchmark/benchmarks/fpb.py and fiqa.py

def normalize_sentiment(text: str) -> str:
    """Coerce any model output to one of three canonical sentiment labels."""
    t = text.lower()
    if "positive" in t:   return "positive"
    elif "negative" in t: return "negative"
    else:                 return "neutral"   # default when model is uncertain


def normalize_headline(text: str) -> int:
    """Coerce headline output to binary 0/1."""
    return 1 if "yes" in text.lower() else 0


# ── Model Loading ─────────────────────────────────────────────────────────────

def load_base_model():
    """
    Load Qwen3-8B in 4-bit with an empty r=16 LoRA scaffold.
    Actual trained weights are loaded per-adapter via load_adapter().
    Using r=16 to match Round 2 training; Round 1 (r=8) still loads
    correctly because set_peft_model_state_dict matches by layer name.
    """
    print(f"\nLoading base model: {BASE_MODEL}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_LEN,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16, lora_dropout=0, bias="none",
        use_gradient_checkpointing=False,  # not needed at inference
        random_state=42,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def load_adapter(model, path: str, name: str):
    """Hot-swap adapter weights without reloading the base model."""
    print(f"\nLoading adapter: {name}")
    set_peft_model_state_dict(model, load_file(path))
    print(f"  ✓ {path}")


# ── Batched Inference ─────────────────────────────────────────────────────────

def batch_generate(model, tokenizer, prompts: list[str],
                   max_new_tokens: int = MAX_NEW_TOK) -> list[str]:
    """
    Wrap prompts in Qwen3 chat template, tokenize as a left-padded batch,
    run greedy decoding, and return only the newly generated tokens.
    Greedy (do_sample=False) ensures deterministic, reproducible eval results.
    """
    formatted = [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM},
             {"role": "user",   "content": p}],
            tokenize=False, add_generation_prompt=True,
        )
        for p in prompts
    ]
    tokenizer.padding_side = "left"
    inputs = tokenizer(
        formatted, return_tensors="pt", padding=True,
        truncation=True, max_length=MAX_LEN, return_token_type_ids=False,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    input_len = inputs["input_ids"].shape[1]
    return [
        tokenizer.decode(out[input_len:], skip_special_tokens=True).strip()
        for out in outputs
    ]


# ── Task 1: FPB ───────────────────────────────────────────────────────────────
# Financial PhraseBank sentiment — same 20%/seed=42 test split as official benchmark

def run_fpb(model, tokenizer) -> dict:
    print("\n── FPB (Financial PhraseBank Sentiment) ──")
    ds = load_dataset("FinGPT/fingpt-sentiment-train", split="train")
    ds = ds.train_test_split(test_size=0.2, seed=42)["test"]

    instruction = "What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}."
    prompts  = [f"{instruction}\nInput: {row['input']}"   for row in ds]
    targets  = [normalize_sentiment(row["output"])         for row in ds]

    preds = []
    for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="FPB"):
        preds.extend([normalize_sentiment(r) for r in batch_generate(model, tokenizer, prompts[i:i+BATCH_SIZE])])
        torch.cuda.empty_cache()

    acc = accuracy_score(targets, preds)
    f1w = f1_score(targets, preds, average="weighted", zero_division=0)
    f1a = f1_score(targets, preds, average="macro",    zero_division=0)
    f1i = f1_score(targets, preds, average="micro",    zero_division=0)
    print(f"  Acc:{acc:.4f}  F1w:{f1w:.4f}  F1a:{f1a:.4f}  F1i:{f1i:.4f}")
    print(f"  (FinGPT v3.3 reference — Acc:0.882 F1w:0.882)")
    return {"task":"FPB","n_samples":len(targets),"accuracy":round(acc,4),
            "f1_weighted":round(f1w,4),"f1_macro":round(f1a,4),"f1_micro":round(f1i,4)}


# ── Task 2: FiQA-SA ───────────────────────────────────────────────────────────
# Financial tweet/news sentiment from continuous scores

def run_fiqa(model, tokenizer) -> dict:
    print("\n── FiQA-SA (Financial QA Sentiment) ──")
    ds = load_dataset("FinGPT/fingpt-fiqa_qa", split="train")
    ds = ds.train_test_split(test_size=0.2, seed=42)["test"]

    def instr(row):
        return ("What is the sentiment of this tweet? Please choose an answer from {negative/neutral/positive}."
                if row.get("format","") == "post" else
                "What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}.")

    prompts = [f"{instr(row)}\nInput: {row['input']}" for row in ds]
    targets = [normalize_sentiment(row["output"])      for row in ds]

    preds = []
    for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="FiQA"):
        preds.extend([normalize_sentiment(r) for r in batch_generate(model, tokenizer, prompts[i:i+BATCH_SIZE])])
        torch.cuda.empty_cache()

    acc = accuracy_score(targets, preds)
    f1w = f1_score(targets, preds, average="weighted", zero_division=0)
    f1a = f1_score(targets, preds, average="macro",    zero_division=0)
    f1i = f1_score(targets, preds, average="micro",    zero_division=0)
    print(f"  Acc:{acc:.4f}  F1w:{f1w:.4f}  F1a:{f1a:.4f}  F1i:{f1i:.4f}")
    print(f"  (FinGPT v3.3 reference — Acc:0.874 F1w:0.903)")
    return {"task":"FiQA-SA","n_samples":len(targets),"accuracy":round(acc,4),
            "f1_weighted":round(f1w,4),"f1_macro":round(f1a,4),"f1_micro":round(f1i,4)}


# ── Task 3: Headline ──────────────────────────────────────────────────────────
# Binary Yes/No price movement classification on financial headlines

def run_headline(model, tokenizer) -> dict:
    print("\n── Headline (Price Movement Classification) ──")
    ds      = load_dataset("FinGPT/fingpt-headline", split="test")
    prompts = [f"{row['instruction']}\nInput: {row['input']}" for row in ds]
    targets = [1 if "yes" in row["output"].lower() else 0    for row in ds]

    preds = []
    for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="Headline"):
        preds.extend([normalize_headline(r) for r in batch_generate(model, tokenizer, prompts[i:i+BATCH_SIZE])])
        torch.cuda.empty_cache()

    acc = accuracy_score(targets, preds)
    f1w = f1_score(targets, preds, average="weighted", zero_division=0)
    f1b = f1_score(targets, preds, average="binary",   zero_division=0)
    print(f"  Acc:{acc:.4f}  F1w:{f1w:.4f}  F1b:{f1b:.4f}")
    print(f"  (FinGPT multi-task reference — F1w ~0.97)")
    return {"task":"Headline","n_samples":len(targets),"accuracy":round(acc,4),
            "f1_weighted":round(f1w,4),"f1_binary":round(f1b,4)}


# ── Task 4: QA ────────────────────────────────────────────────────────────────
# Open-ended Q&A with keyword hit-rate as quality proxy (no ground-truth labels)

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

def run_qa(model, tokenizer) -> dict:
    print("\n── QA (Open-Ended Financial Q&A) ──")
    prompts  = [c[0] for c in QA_CASES]
    keywords = [c[1] for c in QA_CASES]
    levels   = [c[2] for c in QA_CASES]

    # QA needs more tokens — call generate directly with override
    responses = []
    for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="QA"):
        responses.extend(batch_generate(model, tokenizer, prompts[i:i+BATCH_SIZE], max_new_tokens=200))
        torch.cuda.empty_cache()

    scores, per_case = [], []
    for q, r, kws, lvl in zip(prompts, responses, keywords, levels):
        hits  = [kw for kw in kws if kw.lower() in r.lower()]
        score = len(hits) / len(kws)
        scores.append(score)
        print(f"\n  [{lvl.upper()}] {score:.0%} | Q: {q}")
        print(f"  A: {r[:200]}{'...' if len(r)>200 else ''}")
        print(f"  Hit: {hits}")
        per_case.append({"question":q,"response":r,"score":round(score,4),
                         "keywords_hit":hits,"difficulty":lvl})

    avg = sum(scores) / len(scores)
    print(f"\n  Avg keyword score: {avg:.1%}")
    return {"task":"QA","n_samples":len(QA_CASES),
            "avg_keyword_score":round(avg,4),"per_case":per_case}


# ── Display Helpers ───────────────────────────────────────────────────────────

def print_summary(name: str, results: list):
    print(f"\n{'='*62}\n  SUMMARY: {name}\n{'='*62}")
    print(f"  {'Task':<14} {'Metric':<20} {'Score':>8}  Bar")
    print(f"  {'-'*58}")
    for r in results:
        score  = r.get("avg_keyword_score") if r["task"]=="QA" else r.get("f1_weighted",0)
        metric = "Keyword Hit Rate" if r["task"]=="QA" else "F1 Weighted"
        bar    = "█"*int(score*24) + "░"*(24-int(score*24))
        print(f"  {r['task']:<14} {metric:<20} {score:>7.1%}  [{bar}]")


def print_comparison(r1: list, r2: list):
    print(f"\n{'='*62}\n  HEAD-TO-HEAD\n{'='*62}")
    print(f"  {'Task':<14} {'Round1':>9} {'Round2':>9} {'Δ':>7}  Winner")
    print(f"  {'-'*58}")
    get = lambda r: r.get("avg_keyword_score") if r["task"]=="QA" else r.get("f1_weighted",0)
    m1  = {r["task"]: get(r) for r in r1}
    m2  = {r["task"]: get(r) for r in r2}
    for task in ["FPB","FiQA-SA","Headline","QA"]:
        s1, s2 = m1.get(task), m2.get(task)
        if s1 is None or s2 is None: continue
        d = s2 - s1
        w = "Round2 ✓" if d > 0.005 else ("Round1 ✓" if d < -0.005 else "Tie")
        print(f"  {task:<14} {s1:>8.1%}  {s2:>8.1%}  {'▲' if d>0 else '▼'}{abs(d):>5.1%}  {w}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model, tokenizer = load_base_model()
    all_results = {}

    for name, path in ADAPTERS.items():
        if not os.path.exists(path):
            print(f"\nSKIPPING {name} — not found at {path}")
            continue
        load_adapter(model, path, name)
        results = [run_fpb(model, tokenizer), run_fiqa(model, tokenizer),
                   run_headline(model, tokenizer), run_qa(model, tokenizer)]
        print_summary(name, results)
        all_results[name] = results

    if "Round1_Sentiment" in all_results and "Round2_MultiTask" in all_results:
        print_comparison(all_results["Round1_Sentiment"], all_results["Round2_MultiTask"])

    out = "./results/benchmark_results.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Results saved to {out}")
