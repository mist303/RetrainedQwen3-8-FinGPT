# round1.py — Round 1 Sentiment Specialist LoRA Training
# Trains Qwen3-8B on the FinGPT sentiment dataset only.
# Produces a single specialist adapter: ./qwen3-8b-fingpt-lora/
#
# This was the first training run. The resulting adapter is intentionally
# narrow — it only knows how to classify financial sentiment.
# It serves as the baseline specialist in the multi-agent architecture,
# and as a benchmark comparison against the broader Round 2 adapter.
#
# Hardware target: RTX 5070 Ti (12GB VRAM), 32GB RAM, Windows 11

from unsloth import FastLanguageModel
import torch
import os
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

# ── Step 1: Load base model ───────────────────────────────────────────────────
# 4-bit quantization keeps the 8B model within ~6GB VRAM.
# dtype=None lets Unsloth auto-detect bf16 (optimal for RTX 5070 Ti).
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-8B",
    max_seq_length=512,
    dtype=None,
    load_in_4bit=True,
)

# ── Step 2: Attach fresh LoRA scaffold ───────────────────────────────────────
# r=8 is sufficient for a single-task classification adapter.
# Only ~0.5% of total parameters are trainable — the base model is frozen.
model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",  # trades compute for VRAM
    random_state=42,
)

# ── Step 3: Load and format dataset ──────────────────────────────────────────
# fingpt-sentiment-train contains 76.8K instruction-formatted samples.
# Each sample has an 'instruction', 'input' (news text), and 'output' (label).
SYSTEM = (
    "You are an expert financial analyst. "
    "Reason carefully, cite your logic, and provide structured, professional analysis."
)

ds = load_dataset("FinGPT/fingpt-sentiment-train", split="train")

def format_sample(sample):
    """Wrap each sample in Qwen3's chat template for supervised fine-tuning."""
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
    # Filter flag: drop samples that are too long or empty
    token_len = len(tokenizer(text, truncation=False)["input_ids"])
    return {"text": text, "valid": token_len <= 512 and text.strip() != ""}

ds = ds.map(format_sample)
ds = ds.filter(lambda s: s["valid"]).select_columns(["text"])
print(f"Dataset ready: {len(ds)} rows")

# ── Step 4: Train ─────────────────────────────────────────────────────────────
torch.cuda.empty_cache()

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds,
    dataset_text_field="text",
    max_seq_length=512,
    packing=True,           # pack short sequences together for GPU efficiency
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,  # effective batch = 16
        warmup_steps=50,
        num_train_epochs=1,
        learning_rate=1e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),    # RTX 5070 Ti supports bf16 natively
        logging_steps=25,
        save_strategy="epoch",
        output_dir="./qwen3-8b-fingpt",
        optim="paged_adamw_8bit",        # keeps optimizer state in CPU RAM
        report_to="none",
    ),
)

trainer.train()

# ── Step 5: Save LoRA adapter ─────────────────────────────────────────────────
# Saves only the adapter weights (~40MB at r=8), not the full merged model.
model.save_pretrained("./qwen3-8b-fingpt-lora")
tokenizer.save_pretrained("./qwen3-8b-fingpt-lora")
print("Round 1 LoRA adapter saved to ./qwen3-8b-fingpt-lora")

# ── Step 6: Merge and export to float16 ──────────────────────────────────────
# Merges LoRA weights into the base model and saves as float16.
# Required for GGUF conversion via llama.cpp for local inference.
print("Merging to float16 for GGUF export...")
model.save_pretrained_merged(
    "./qwen3-8b-fingpt-merged",
    tokenizer,
    save_method="merged_16bit",
)
print("Merged model saved to ./qwen3-8b-fingpt-merged")
print("Next: convert to GGUF using llama.cpp")
