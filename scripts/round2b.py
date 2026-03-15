"""
round2b.py — Round 2b Multi-Task LoRA Training (Revised)
=========================================================
Trains the `multitask` agent on headline classification and financial Q&A.
Sentiment deliberately excluded — the `sentiment` agent owns that task.

Why round2 was retrained:
  Round 2 included fingpt-sentiment-train, which the sentiment specialist
  already covers. Benchmarks showed task interference: the multitask agent
  scored 2.6pp lower on FiQA-SA than the dedicated sentiment specialist.
  Removing sentiment keeps the multi-agent architecture clean.

Changes vs round2:
  - Removed: FinGPT/fingpt-sentiment-train (76.8K samples dropped)
  - Dataset: fingpt-fiqa_qa (17.1K) + fingpt-headline (82.2K) = ~99.3K
  - lora_alpha: 16 → 32  (stronger update scaling)
  - num_train_epochs: 1 → 3  (more signal per sample)
  - warmup_steps: 200 → 300
  - save_steps: 1900  (~10% of ~18.6K total steps — safe stop/resume points)
  - save_total_limit: 10  (keeps all checkpoints)

Estimated training time: ~40 hours  |  Checkpoint every ~4 hours (~10%)
Hardware: RTX 5070 Ti (12GB VRAM), 32GB RAM, Windows 11

To stop safely: Ctrl+C between checkpoints — last saved checkpoint is intact.
To resume:      trainer.train(resume_from_checkpoint=True) — auto-finds latest.
"""

import os
import multiprocessing
import gc

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

try:
    import psutil
    psutil.Process(os.getpid()).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except ImportError:
    print("Tip: pip install psutil for background priority management.")


SYSTEM = ("You are an expert financial analyst. "
          "Reason carefully, cite your logic, and provide structured, professional analysis.")

def process_and_filter(sample, tokenizer):
    instruction = str(sample.get("instruction", "") or "")
    output      = str(sample.get("output",      "") or "")
    text = tokenizer.apply_chat_template(
        [
            {"role": "system",    "content": SYSTEM},
            {"role": "user",      "content": instruction},
            {"role": "assistant", "content": output},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    token_len = len(tokenizer(text, truncation=False)["input_ids"])
    return {"text": text, "valid_length": token_len <= 512 and text.strip() != ""}

def prepare(ds, tokenizer, cores):
    ds = ds.map(process_and_filter, fn_kwargs={"tokenizer": tokenizer}, num_proc=cores)
    ds = ds.filter(lambda s: s["valid_length"], num_proc=cores)
    return ds.select_columns(["text"])

def pre_tokenize(sample, tokenizer):
    return {"input_ids": tokenizer(
        sample["text"], truncation=True, max_length=512, padding=False,
    )["input_ids"]}


if __name__ == "__main__":

    SAFE_CORES = min(4, multiprocessing.cpu_count() - 2)

    # ── Step 1: Base model ────────────────────────────────────────────────────
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-8B",
        max_seq_length=512,
        dtype=None,
        load_in_4bit=True,
    )

    # ── Step 2: LoRA scaffold ─────────────────────────────────────────────────
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # ── Step 3: Load datasets (no sentiment) ─────────────────────────────────
    print("Loading datasets...")
    from concurrent.futures import ThreadPoolExecutor

    def load_ds(info): return load_dataset(info[1], split="train")

    with ThreadPoolExecutor(max_workers=2) as ex:
        ds_fiqa, ds_headline = list(ex.map(load_ds, [
            ("fiqa",     "FinGPT/fingpt-fiqa_qa"),
            ("headline", "FinGPT/fingpt-headline"),
        ]))

    print(f"  fiqa_qa:  {len(ds_fiqa):,}")
    print(f"  headline: {len(ds_headline):,}")

    # ── Step 4: Format + concatenate ─────────────────────────────────────────
    print("Formatting and filtering...")
    dataset = concatenate_datasets([ds_fiqa, ds_headline])
    dataset = prepare(dataset, tokenizer, SAFE_CORES).shuffle(seed=42)
    print(f"Dataset ready: {len(dataset):,} rows")

    # ── Step 5: Pre-tokenize ──────────────────────────────────────────────────
    print("Pre-tokenizing...")
    dataset = dataset.map(
        pre_tokenize,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=1,
        remove_columns=["text"],
    )
    dataset.set_format(type="torch", columns=["input_ids"])

    # ── Step 6: Train ─────────────────────────────────────────────────────────
    # ~18,600 total steps. Checkpoint every 1,900 steps ≈ every 10% ≈ every 4h.
    # Ctrl+C safely between checkpoints. Resume with resume_from_checkpoint=True.
    torch.cuda.empty_cache()

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            warmup_steps=300,
            num_train_epochs=3,
            learning_rate=5e-5,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=100,
            save_strategy="steps",
            save_steps=1900,        # ~10% of total steps
            save_total_limit=10,    # keep all 10 checkpoints (~6.4GB total disk)
            output_dir="./qwen3-8b-round2b",
            optim="paged_adamw_8bit",
            report_to="none",
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
        ),
    )

    trainer.train()

    # ── Step 7: Save adapter ──────────────────────────────────────────────────
    model.save_pretrained("./qwen3-8b-round2b-lora")
    tokenizer.save_pretrained("./qwen3-8b-round2b-lora")
    print("Round 2b LoRA adapter saved to ./qwen3-8b-round2b-lora")

    # ── Step 8: Merge ─────────────────────────────────────────────────────────
    del trainer, dataset
    gc.collect()
    torch.cuda.empty_cache()

    model.save_pretrained_merged(
        "./qwen3-8b-round2b-merged",
        tokenizer,
        save_method="merged_16bit",
    )
    print("Round 2b merged model saved to ./qwen3-8b-round2b-merged")
