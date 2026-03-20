# forecaster.py — Round 3 Forecaster LoRA Training
# Trains Qwen3-8B on the FinGPT DOW30 forecaster dataset.
# Produces a structured stock prediction adapter: ./qwen3-8b-forecaster-lora/
#
# This adapter's output format is fundamentally different from all other adapters:
#   [Positive Developments]: 1. ... 2. ...
#   [Potential Concerns]:    1. ... 2. ...
#   [Prediction & Analysis]: Prediction: up/down X%. Analysis: ...
#
# That structured multi-paragraph format is why this MUST be a separate adapter —
# mixing it with classification tasks (sentiment, headline) or short Q&A would
# cause output format collapse in both directions.
#
# Hardware target: RTX 5070 Ti (12GB VRAM), 32GB RAM, Windows 11
# Estimated runtime: ~15-25 hours (dataset ~10K samples × 3 epochs)
#
# Key differences from round2:
#   - max_seq_length = 2048  (long inputs: company intro + news + financials)
#   - batch_size = 1         (2048 tokens × batch2 risks OOM on 12GB)
#   - grad_accum = 16        (keeps effective batch = 16)
#   - r=16, alpha=32         (more expressiveness for generative task)
#   - 3 epochs               (smaller dataset, needs more passes)
#   - Dataset columns: prompt/answer (not instruction/input/output)
#   - Specific forecaster system prompt

import os
import gc
import multiprocessing

# Set CUDA memory allocator before PyTorch initializes.
# expandable_segments reduces fragmentation from variable-length 2048-token batches.
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

# Lower process priority so the laptop stays usable during the multi-day run
try:
    import psutil
    p = psutil.Process(os.getpid())
    p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except ImportError:
    print("Tip: 'pip install psutil' to enable background priority management.")


# ── GLOBAL SCOPE: Required for Windows multiprocessing ────────────────────────
# Any function called inside dataset.map(num_proc>1) must live at module level.
# Forecaster pre_tokenize uses num_proc=1 anyway (Windows Unsloth re-init issue),
# but process_and_filter uses SAFE_CORES — keep it global to be safe.

# Forecaster system prompt — must match exactly at inference time.
# Source: FinGPT-Forecaster paper / HuggingFace Space.
FORECASTER_SYSTEM = (
    "You are a seasoned stock market analyst. Your task is to list the positive "
    "developments and potential concerns for companies based on relevant news and "
    "basic financials from the past weeks, then provide an analysis and prediction "
    "for the companies' stock price movement for the upcoming week. "
    "Your answer format should be as follows:\n\n"
    "[Positive Developments]:\n1. ...\n\n"
    "[Potential Concerns]:\n1. ...\n\n"
    "[Prediction & Analysis]:\n..."
)

# Max token length for this adapter.
# Forecaster prompts contain company intro + multiple weeks of news + financials.
# 512 would truncate most samples. 2048 keeps ~95%+ of the dataset intact.
MAX_SEQ_LEN = 2048


def process_and_filter(sample, tokenizer):
    """
    Format a forecaster sample into Qwen3's chat template and check length.

    Dataset columns:
      prompt — fully assembled input text (company intro + news + financials + question)
      answer — structured response ([Positive Developments] / [Concerns] / [Prediction])

    Unlike round1/round2 which had instruction + input + output columns, the
    forecaster dataset already has a fully assembled prompt string. We just
    wrap it in Qwen3's chat template with the forecaster-specific system prompt.

    Samples exceeding MAX_SEQ_LEN are flagged for filtering to prevent
    truncation from producing incomplete [Prediction & Analysis] sections,
    which would corrupt the model's understanding of the expected output format.
    """
    prompt = str(sample.get("prompt", "") or "").strip()
    answer = str(sample.get("answer", "") or "").strip()

    if not prompt or not answer:
        return {"text": "", "valid_length": False}

    text = tokenizer.apply_chat_template(
        [
            {"role": "system",    "content": FORECASTER_SYSTEM},
            {"role": "user",      "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )

    # Count tokens without truncation to get the true length
    token_len = len(tokenizer(text, truncation=False)["input_ids"])

    return {
        "text":         text,
        "valid_length": token_len <= MAX_SEQ_LEN and text.strip() != "",
    }


def prepare(ds, tokenizer, cores):
    """
    Apply chat template formatting and filter out over-length samples.
    Over-length filtering is more aggressive here than in round1/round2
    because truncated forecaster outputs produce malformed structured text.
    """
    ds = ds.map(process_and_filter, fn_kwargs={"tokenizer": tokenizer}, num_proc=cores)
    ds = ds.filter(lambda s: s["valid_length"], num_proc=cores)
    return ds.select_columns(["text"])


def pre_tokenize(sample, tokenizer):
    """
    Pre-tokenize each sample into input_ids only, before training begins.
    Labels are handled by DataCollatorForLanguageModeling at collation time.

    num_proc=1 is mandatory for this step on Windows — multiprocessing here
    spawns new processes that re-import Unsloth, causing the training banner
    to print multiple times and massively slowing down startup.

    This pre-tokenization moves all CPU tokenization work OUT of the training
    loop so the DataLoader only passes integer tensors to the GPU each step.
    """
    ids = tokenizer(
        sample["text"],
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding=False,
    )["input_ids"]
    return {"input_ids": ids}


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Cap CPU cores for Windows stability
    SAFE_CORES = min(4, multiprocessing.cpu_count() - 2)

    # ── Step 1: Load base model ───────────────────────────────────────────────
    # max_seq_length=2048 is critical — Unsloth compiles kernels for this length.
    # Using 512 (like round2) would truncate most forecaster samples and produce
    # Unsloth warnings + corrupted gradient signals.
    print("Loading base model (Qwen3-8B, 4bit, max_seq_len=2048)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-8B",
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,           # auto-detect: bf16 on RTX 5070 Ti
        load_in_4bit=True,    # ~5.5GB VRAM for 8B params
    )

    # ── Step 2: Attach fresh LoRA scaffold ───────────────────────────────────
    # r=16, alpha=32: higher alpha/rank ratio than round2 (alpha=16).
    # Forecasting is a generative long-form task — it benefits from stronger
    # gradient scaling (alpha controls the effective learning rate of LoRA).
    # alpha=32 with r=16 means LoRA updates are scaled by 32/16=2x relative
    # to the base model weights, encouraging faster specialization.
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",  # essential for 2048-token sequences
        random_state=42,
    )

    # ── Step 3: Load dataset ──────────────────────────────────────────────────
    # fingpt-forecaster-dow30-202305-202405:
    #   DOW30 stocks, May 2023 – April 2024 (~10K training samples)
    #   Columns: prompt (str), answer (str), label (str), symbol (str), period (str)
    # We use only the train split; test split is used for benchmarking only.
    print("Loading forecaster dataset...")
    ds = load_dataset("FinGPT/fingpt-forecaster-dow30-202305-202405", split="train")
    print(f"Raw dataset: {len(ds)} samples")

    # ── Step 4: Format, filter, and pre-tokenize ──────────────────────────────
    print("Formatting and filtering dataset...")
    dataset = prepare(ds, tokenizer, SAFE_CORES).shuffle(seed=42)
    print(f"After filtering: {len(dataset)} samples retained")

    # Print retention rate so we can see how many samples were too long
    retention = len(dataset) / len(ds) * 100
    print(f"Retention rate: {retention:.1f}% (samples within {MAX_SEQ_LEN} tokens)")

    # Pre-tokenize the entire dataset into input_ids before training.
    # This is the key fix from round2 debugging: SFTTrainer's internal
    # tokenization during the training loop caused 80s/it on Windows
    # due to multiprocessing re-importing Unsloth. Pre-tokenizing here
    # means the DataLoader only moves integer tensors during training.
    print("Pre-tokenizing (one-time cost, num_proc=1 for Windows compatibility)...")
    dataset = dataset.map(
        pre_tokenize,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=1,              # MUST be 1 on Windows — see docstring above
        remove_columns=["text"],
    )
    dataset.set_format(type="torch", columns=["input_ids"])
    print(f"Pre-tokenization complete. {len(dataset)} samples ready.")

    # ── Step 5: Train ─────────────────────────────────────────────────────────
    torch.cuda.empty_cache()

    # DataCollatorForLanguageModeling:
    # - Copies input_ids → labels
    # - Sets padding token positions to -100 (ignored by cross-entropy loss)
    # - mlm=False: causal LM (predict next token), not masked LM (BERT-style)
    # This is the correct collator for pre-tokenized CLM training.
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Estimate total steps for save_steps calculation
    # ~10K samples / 1 (batch) / 16 (accum) × 3 epochs ≈ 1875 steps
    # Save every ~200 steps (roughly once per 10% of training)
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        data_collator=collator,
        args=TrainingArguments(
            # Batch size: 1 device × 16 grad accum = 16 effective batch size.
            # batch_size=1 is necessary for 2048-token sequences on 12GB VRAM.
            # With gradient checkpointing, Qwen3-8B (4bit) + 2048 tokens
            # uses ~9-10GB. batch_size=2 risks OOM during the backward pass.
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,  # effective batch = 16

            warmup_steps=100,                # ~5% of total steps; dataset is small
            num_train_epochs=3,              # 3 passes needed — forecaster is ~10K samples
            learning_rate=2e-5,             # lower than round2 (5e-5) — 3 epochs risks
                                            # overfitting with higher LR on ~10K samples

            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),   # RTX 5070 Ti natively supports bf16

            logging_steps=50,               # log more frequently (fewer total steps)
            save_strategy="steps",
            save_steps=200,                 # ~every 10% of an epoch
            save_total_limit=5,             # keep 5 checkpoints (~last 1 epoch)

            output_dir="./qwen3-8b-forecaster",
            optim="paged_adamw_8bit",       # keeps optimizer state in CPU RAM
            report_to="none",
            dataloader_num_workers=0,       # MUST be 0 on Windows — see round2 debugging
            dataloader_pin_memory=False,    # only useful when num_workers > 0
        ),
    )

    print("Starting forecaster training...")
    print(f"Config: r=16 alpha=32, 3 epochs, lr=2e-5, batch=1×16=16 effective")
    print(f"Estimated steps: ~{len(dataset) // 16 * 3} total")
    trainer.train()

    # ── Step 6: Save LoRA adapter ─────────────────────────────────────────────
    # The adapter is ~80MB — only the LoRA matrices, not the 16GB base model.
    model.save_pretrained("./qwen3-8b-forecaster-lora")
    tokenizer.save_pretrained("./qwen3-8b-forecaster-lora")
    print("Forecaster LoRA adapter saved to ./qwen3-8b-forecaster-lora")

    # ── Step 7: Merge and export to float16 ──────────────────────────────────
    # Merges LoRA weights into the base model for GGUF conversion via llama.cpp.
    # This requires significant RAM (~32GB) — the 32GB laptop RAM should be fine.
    print("Freeing VRAM before merge...")
    del trainer
    del dataset
    gc.collect()
    torch.cuda.empty_cache()

    print("Merging to float16 (requires ~32GB RAM)...")
    model.save_pretrained_merged(
        "./qwen3-8b-forecaster-merged",
        tokenizer,
        save_method="merged_16bit",
    )
    print("Merged model saved to ./qwen3-8b-forecaster-merged")
    print("Next: add forecaster agent to orchestrator AGENTS dict")
