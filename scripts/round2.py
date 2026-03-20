import os
import multiprocessing
import gc

# Maximize memory efficiency before PyTorch initializes
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

# Optional but highly recommended for background laptop training
try:
    import psutil
    p = psutil.Process(os.getpid())
    # Lowers priority so your laptop doesn't freeze while training
    p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS) 
except ImportError:
    print("Tip: 'pip install psutil' to enable background priority management.")


# ── GLOBAL FUNCTIONS: MOVED OUTSIDE __main__ FOR WINDOWS MULTIPROCESSING ──────
SYSTEM = ("You are an expert financial analyst. "
          "Reason carefully, cite your logic, and provide structured, professional analysis.")

def process_and_filter(sample, tokenizer):
    # Handle dict edge cases safely
    instruction = str(sample.get("instruction", "") or "")
    output = str(sample.get("output", "") or "")
    
    text = tokenizer.apply_chat_template(
        [
            {"role": "system",    "content": SYSTEM},
            {"role": "user",      "content": instruction},
            {"role": "assistant", "content": output},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    
    # Tokenize once to get both the length and prepare for filtering
    token_len = len(tokenizer(text, truncation=False)["input_ids"])
    
    return {
        "text": text,
        "valid_length": token_len <= 512 and text.strip() != ""
    }

def prepare(ds, tokenizer, cores):
    # Format and filter samples
    ds = ds.map(process_and_filter, fn_kwargs={"tokenizer": tokenizer}, num_proc=cores)
    ds = ds.filter(lambda s: s["valid_length"], num_proc=cores)
    ds = ds.select_columns(["text"])
    return ds


def pre_tokenize(sample, tokenizer):
    """
    Pre-tokenize each sample into input_ids only.
    Labels are NOT set here — DataCollatorForLanguageModeling handles that
    at collation time by copying input_ids → labels and masking padding with -100.
    This is the correct pattern for CLM training with pre-tokenized data.
    """
    ids = tokenizer(
        sample["text"],
        truncation=True,
        max_length=512,
        padding=False,
    )["input_ids"]
    return {"input_ids": ids}


# ── WINDOWS GUARD REQUIRED FOR MULTIPROCESSING ────────────────────────────────
if __name__ == "__main__":
    
    # Cap cores on Windows to prevent WinError 1455 and keep the system responsive
    SAFE_CORES = min(4, multiprocessing.cpu_count() - 2) 

    # ── Step 1: Load base model ───────────────────────────────────────────────
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-8B",
        max_seq_length=512,  # must match pre_tokenize truncation length
        dtype=None,
        load_in_4bit=True,
    )

    # ── Step 2: Attach LoRA adapters ──────────────────────────────────────────
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",  # re-enabled: faster than disabled
        random_state=42,
    )

    # ── Step 3: Load datasets in parallel ────────────────────────────────────
    print("Loading datasets...")
    from concurrent.futures import ThreadPoolExecutor
    
    def load_single_dataset(name_and_path):
        return load_dataset(name_and_path[1], split="train")
    
    datasets_info = [
        ("sentiment", "FinGPT/fingpt-sentiment-train"),
        ("fiqa", "FinGPT/fingpt-fiqa_qa"),
        ("headline", "FinGPT/fingpt-headline"),
    ]
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        loaded_datasets = list(executor.map(load_single_dataset, datasets_info))
    
    ds_sentiment, ds_fiqa, ds_headline = loaded_datasets

    # ── Step 4: Format, filter, pre-tokenize ─────────────────────────────────
    print("Formatting and filtering datasets...")
    dataset = concatenate_datasets([ds_sentiment, ds_fiqa, ds_headline])
    dataset = prepare(dataset, tokenizer, SAFE_CORES).shuffle(seed=42)
    print(f"Final dataset ready: {len(dataset)} rows")

    # Pre-tokenize the entire dataset into input_ids before training.
    # Moves all CPU tokenization work out of the training loop entirely.
    # The DataLoader will only batch integers at train time — GPU never waits.
    print("Pre-tokenizing dataset (one-time cost, much faster training after)...")
    dataset = dataset.map(
        pre_tokenize,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=1,              # must be 1 — multiprocessing re-imports Unsloth on Windows
        remove_columns=["text"],
    )
    dataset.set_format(type="torch", columns=["input_ids"])
    print(f"Pre-tokenization complete. Ready to train.")

    # ── Step 7: Train ─────────────────────────────────────────────────────────
    # Clear VRAM before starting the run
    torch.cuda.empty_cache()

    # DataCollatorForLanguageModeling handles label creation at collation time:
    # it copies input_ids → labels and sets padding positions to -100 (ignored by loss).
    # This is the correct pattern for pre-tokenized CLM data and avoids the
    # SFTTrainer internal label manipulation that caused the 512 vs 524 shape crash.
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,   # CLM (causal), not masked language modelling
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        data_collator=collator,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,   # effective batch = 16
            warmup_steps=200,
            num_train_epochs=1,
            learning_rate=5e-5,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=100,
            save_strategy="steps",
            save_steps=1000,
            save_total_limit=3,
            output_dir="./qwen3-8b-round2",
            optim="paged_adamw_8bit",
            report_to="none",
            dataloader_num_workers=0,        # must be 0 on Windows — workers re-init Unsloth
            dataloader_pin_memory=False,
        ),
    )

    trainer.train()

    # ── Step 8: Save LoRA adapter ─────────────────────────────────────────────
    model.save_pretrained("./qwen3-8b-round2-lora")
    tokenizer.save_pretrained("./qwen3-8b-round2-lora")
    print("Round 2 LoRA adapter saved.")

    # ── Step 9: Merge & export ────────────────────────────────────────────────
    print("Cleaning up memory before 16-bit merge...")
    # Delete trainer and dataset from RAM to free up space for the merge
    del trainer
    del dataset
    gc.collect()
    torch.cuda.empty_cache()

    print("Merging model to 16-bit (This takes significant RAM)...")
    model.save_pretrained_merged(
        "./qwen3-8b-round2-merged",
        tokenizer,
        save_method="merged_16bit",
    )
    print("Round 2 merged model saved to ./qwen3-8b-round2-merged")