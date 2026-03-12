# Training Log — Decisions, Configs, and Lessons Learned

## Philosophy

The training approach here prioritizes:
1. **Specialization over generalization** — separate adapters per task
2. **Reproducibility** — fixed seeds, documented configs
3. **Hardware efficiency** — everything must fit on a 12GB consumer GPU
4. **Clean separation** — never mix incompatible tasks in one training run

---

## Hardware Setup

| Component | Spec |
|---|---|
| GPU | NVIDIA RTX 5070 Ti (12GB VRAM) |
| RAM | 32GB DDR5 |
| OS | Windows 11 |
| Framework | Unsloth + HuggingFace Transformers + PEFT |
| Precision | bf16 (native on Ada Lovelace architecture) |

---

## Round 1 — Sentiment Specialist

### Goal
Train a clean sentiment classification adapter as the first building block.

### Config
```
Base model:     unsloth/Qwen3-8B
Quantization:   4-bit (load_in_4bit=True)
LoRA rank:      r=8, alpha=16
Target modules: q_proj, k_proj, v_proj, o_proj,
                gate_proj, up_proj, down_proj
Dataset:        FinGPT/fingpt-sentiment-train (76.8K rows)
Max seq len:    512
Batch size:     1 (device) × 16 (grad accum) = 16 effective
Optimizer:      paged_adamw_8bit
Epochs:         1 (approx 10,131 steps)
Output:         ./qwen3-8b-fingpt-lora/adapter_model.safetensors
```

### Outcome
Adapter trained successfully and saved. Serves as the sentinel baseline
for comparison against Round 2 on sentiment tasks.

---

## Round 2 — Multi-Task Adapter

### Goal
Train a single adapter that handles sentiment, financial Q&A, and headline
classification — enough for the model to answer open-ended financial prompts
while remaining accurate on structured tasks.

### Key Decision: Start from Base, Not Round 1

Initially the Round 2 script loaded Round 1's weights before training.
This was intentionally removed before training began.

**Why:** Loading Round 1 weights and then training on new tasks causes
**catastrophic forgetting** — gradient updates from Q&A and headline tasks
shift the LoRA weights away from sentiment, partially erasing what Round 1
learned. By starting from a fresh LoRA scaffold on the frozen base, Round 2
becomes its own independent specialist trained cleanly from scratch.

Round 1's adapter is preserved untouched at `./qwen3-8b-fingpt-lora/`.

### Config
```
Base model:     unsloth/Qwen3-8B
Quantization:   4-bit (load_in_4bit=True)
LoRA rank:      r=16, alpha=16  ← higher than R1 for multi-task capacity
Target modules: q_proj, k_proj, v_proj, o_proj,
                gate_proj, up_proj, down_proj
Datasets:       FinGPT/fingpt-sentiment-train  (76.8K)
                FinGPT/fingpt-fiqa_qa           (17.1K)
                FinGPT/fingpt-headline          (82.2K)
                ─────────────────────────────────
                Total after filtering:         ~174K rows
Max seq len:    512
Batch size:     2 (device) × 8 (grad accum) = 16 effective
Warmup:         200 steps
Optimizer:      paged_adamw_8bit
Epochs:         1 (10,888 total steps)
Output:         ./qwen3-8b-round2-lora/adapter_model.safetensors
```

### Why r=16 for Round 2?

LoRA rank determines how many independent "directions" the adapter can learn.
With 3 tasks covering different output formats (classification, generation,
binary), rank 8 is too constrained — it would underfit the more complex Q&A
task while overfitting the simpler classification tasks.
Rank 16 gives the adapter enough capacity to represent all three tasks
without doubling compute cost.

### VRAM Optimizations Applied
- `expandable_segments:True` in CUDA allocator to reduce fragmentation
- Removed `max_split_size_mb:128` which was causing unnecessary fragmentation
- `packing=True` in SFTTrainer to pack short sequences together, improving
  GPU utilization on short sentiment samples
- `paged_adamw_8bit` keeps optimizer state in CPU RAM, not VRAM
- `gradient_checkpointing="unsloth"` trades compute for memory on forward pass

### Dataset Formatting

All three datasets are formatted into Qwen3's chat template:

```
<|im_start|>system
You are an expert financial analyst. Reason carefully...
<|im_start|>user
{instruction}\nInput: {input}
<|im_start|>assistant
{output}
```

Samples longer than 512 tokens are filtered out before training to prevent
truncation artifacts from corrupting gradient signals.

---

## Lesson: Why I Did NOT Merge All Tasks Into One Run

A natural first instinct is: "just train on everything at once."

The problem is output format collision. Sentiment classification trains the
model to output a single word. Q&A trains it to generate paragraphs.
When these gradients accumulate simultaneously, the model learns a
compromise — it outputs something that's neither a clean classification
nor a coherent paragraph. In practice this shows up as:

- Sentiment outputs like: "The sentiment is positive. The company has shown..."
  (instead of just "positive")
- Q&A outputs that trail off with a sentiment label at the end

Separate adapters avoid this entirely. Each adapter's training data is 100%
consistent in its expected output format.

---

## Planned: Round 3 — Relation Extraction

```
Dataset:  FinGPT/fingpt-finred (27.6K train, 5.1K test)
Task:     Entity-relation pair extraction
Output:   "relation: entity1, entity2; relation2: entity3, entity4"
Why sep:  Structural list output is incompatible with all other tasks
Config:   r=8 (simpler task, lower rank sufficient)
```

## Planned: Round 4 — Forecaster

```
Dataset:  FinGPT/fingpt-forecaster-dow30-202305-202405
Task:     Stock price movement prediction (next week)
Input:    Company intro + recent news + basic financials
Output:   [Positive Developments] / [Potential Concerns] / [Prediction]
Notes:    Will require max_seq_length >= 2048 due to long inputs
          May need to reduce batch size to 1
```
