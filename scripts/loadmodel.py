from unsloth import FastLanguageModel

# Step 1: Load base model with LoRA attached
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-8B",
    max_seq_length=512,
    dtype=None,
    load_in_4bit=True,
)

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
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# Step 2: Load trained weights from checkpoint into the LoRA model
from peft import set_peft_model_state_dict
import torch, os

checkpoint_path = "./qwen3-8b-fingpt/checkpoint-10131"
adapter_path = os.path.join(checkpoint_path, "adapter_model.safetensors")

if not os.path.exists(adapter_path):
    # Try bin format
    adapter_path = os.path.join(checkpoint_path, "adapter_model.bin")

print(f"Loading adapter from: {adapter_path}")

from safetensors.torch import load_file
state_dict = load_file(adapter_path)
set_peft_model_state_dict(model, state_dict)
print("LoRA adapter loaded successfully.")

# Step 3: Save merged model in float16 for GGUF conversion
print("Merging and saving to float16...")
model.save_pretrained_merged(
    "./qwen3-8b-fingpt-merged",
    tokenizer,
    save_method="merged_16bit",
)
print("Merged model saved to ./qwen3-8b-fingpt-merged")
print("Next: convert to GGUF using llama.cpp (see instructions)")