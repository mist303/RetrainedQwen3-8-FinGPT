# round2.py — Round 2 Multi-Task LoRA Training
# Trains Qwen3-8B from scratch on three FinGPT datasets:
#   - fingpt-sentiment-train (76.8K)
#   - fingpt-fiqa_qa (17.1K)
#   - fingpt-headline (82.2K)
# Produces a single independent adapter: ./qwen3-8b-round2-lora/
# Hardware target: RTX 5070 Ti (12GB VRAM), 32GB RAM, Windows 11

import os
import multiprocessing
import gc

# Set CUDA memory allocator to use expandable segments.
# This reduces fragmentation when batch sizes vary between steps.
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
from datasets import load_dataset, concatenate_datasets
from trl import SFTTrainer
from transformers import TrainingArguments

# Lower process priority so the laptop stays usable during the multi-hour run
try:
    import psutil
    p = psutil.Process(os.getpid())
    p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except ImportError:
    print("Tip: 'pip install psutil' to enable background priority management.")


# ── GLOBAL SCOPE: Required for Windows multiprocessing ────────────────────────
# On Windows, dataset.map() with num_proc > 1 spawns new processes.
# Functions and constants used inside those processes must be at module level,
# not inside __main__, or Windows will fail to pickle them.

SYSTEM = (
    "You are an expert financial analyst. "
    "Reason carefully, cite your logic, and provide structured, professional analysis."
)


def process_and_filter(sample, tokenizer):
    """
    Format a single dataset sample into Qwen3's chat template and check length.
    Returns the formatted text plus a boolean flag for filtering.

    The chat template wraps the instruction/output in:
      <|im_start|>system ... <|im_end|>
      <|im_start|>user   ... <|im_end|>
      <|im_start|>assistant ... <|im_end|>

    We filter out any sample whose total token count exceeds 512 to prevent
    truncation from corrupting gradient signals during training.
    """
    instruction = str(sample.get("instruction", "") or "")
    output      = str(sample.get("output", "")      or "")

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

    return {
        "text":         text,
        "valid_length": token_len <= 512 and text.strip() != "",
    }


def prepare(ds, tokenizer, cores):
    """
    Apply chat template formatting and filter out over-length samples.
    Uses multiple CPU cores for parallel processing (capped for Windows stability).
    """
    ds = ds.map(process_and_filter, fn_kwargs={"tokenizer": tokenizer}, num_proc=cores)
    ds = ds.filter(lambda s: s["valid_length"], num_proc=cores)
    return ds.select_columns(["text"])


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Cap CPU cores: Windows throws WinError 1455 if too many worker processes
    # are spawned simultaneously. 4 is safe on most modern laptops.
    SAFE_CORES = min(4, multiprocessing.cpu_count() - 2)

    # ── Step 1: Load base model ───────────────────────────────────────────────
    # 4-bit quantization keeps the 8B model within 6GB VRAM.
    # dtype=None lets Unsloth auto-select bf16 (optimal for RTX 5070 Ti).
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-8B",
        max_seq_length=512,
        dtype=None,
        load_in_4bit=True,
    )

    # ── Step 2: Attach fresh LoRA scaffold ───────────────────────────────────
    # r=16 (vs r=8 in Round 1) gives more adapter capacity for 3 different tasks.
    # All base model weights remain frozen — only the LoRA matrices are trained.
    # Trainable params: ~43M out of 8.2B total (~0.53%)
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",  # trades compute for VRAM
        random_state=42,
    )

    # ── Step 3: Load all three datasets ──────────────────────────────────────
    print("Loading datasets...")
    ds_sentiment = load_dataset("FinGPT/fingpt-sentiment-train", split="train")
    ds_fiqa      = load_dataset("FinGPT/fingpt-fiqa_qa",         split="train")
    ds_headline  = load_dataset("FinGPT/fingpt-headline",        split="train")

    # ── Step 4: Format and filter ─────────────────────────────────────────────
    print("Formatting and filtering datasets...")
    dataset = concatenate_datasets([
        prepare(ds_sentiment, tokenizer, SAFE_CORES),
        prepare(ds_fiqa,      tokenizer, SAFE_CORES),
        prepare(ds_headline,  tokenizer, SAFE_CORES),
    ]).shuffle(seed=42)
    print(f"Final dataset ready: {len(dataset)} rows")

    # ── Step 5: Train ─────────────────────────────────────────────────────────
    torch.cuda.empty_cache()

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=512,
        packing=True,           # packs short sequences together for GPU efficiency
        args=TrainingArguments(
            per_device_train_batch_size=2,   # fits with 4bit Qwen3-8B on 12GB
            gradient_accumulation_steps=8,   # effective batch = 2 × 8 = 16
            warmup_steps=200,                # scaled for ~174K row dataset
            num_train_epochs=1,
            learning_rate=1e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),    # RTX 5070 Ti supports bf16 natively
            logging_steps=25,
            save_strategy="epoch",
            output_dir="./qwen3-8b-round2",
            optim="paged_adamw_8bit",        # keeps optimizer state in CPU RAM
            report_to="none",
            dataloader_num_workers=2,
            average_tokens_across_devices=False,
        ),
    )

    trainer.train()

    # ── Step 6: Save LoRA adapter ─────────────────────────────────────────────
    # Saves only the adapter weights (~80MB), not the full 16GB merged model.
    model.save_pretrained("./qwen3-8b-round2-lora")
    tokenizer.save_pretrained("./qwen3-8b-round2-lora")
    print("Round 2 LoRA adapter saved.")

    # ── Step 7: Merge and export ──────────────────────────────────────────────
    # Merge LoRA weights into the base model and save as float16.
    # This is needed for GGUF conversion via llama.cpp for local inference.
    del trainer
    del dataset
    gc.collect()
    torch.cuda.empty_cache()

    model.save_pretrained_merged(
        "./qwen3-8b-round2-merged",
        tokenizer,
        save_method="merged_16bit",
    )
    print("Round 2 merged model saved to ./qwen3-8b-round2-merged")
