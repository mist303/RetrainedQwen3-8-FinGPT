"""
FinGPT Orchestrator
====================
Routes user queries through the RAG layer and dispatches to the correct
LoRA agent, all sharing the same frozen Qwen3-8B base model.

The orchestrator reacts explicitly to every RAG status:
  OK             → proceed with enriched context
  EMPTY_INDEX    → warn once, proceed without context
  NO_MATCH       → proceed without context, note in result
  LOW_CONFIDENCE → log best score, proceed without context
  LIVE_FETCH     → yfinance data fetched on-the-fly, proceed with context
  ERROR          → log exception, flag in result, proceed without context

Conversation history is persisted to history/session.jsonl and loaded on
startup. Call reset_history() to clear between benchmark tasks or sessions.
"""

# unsloth MUST be the first non-stdlib import — it patches torch/transformers/peft
# internals at import time. Importing those libraries first disables optimizations.
from unsloth import FastLanguageModel  # noqa: E402

import gc
import json
import torch
import logging
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from safetensors.torch import load_file
from peft import set_peft_model_state_dict

from rag import RAGLayer
from rag.rag_layer import RAGStatus

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Cross-Platform CUDA Setup (Linux)
# ─────────────────────────────────────────────────────────────────────────────

OS_TYPE = platform.system()
if OS_TYPE == "Linux":
    # Ensure CUDA libraries are accessible on Linux
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    cuda_lib = os.path.join(cuda_home, "lib64")



# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

BASE_MODEL = "unsloth/Qwen3-8B"
MAX_LEN    = 4096

HISTORY_FILE      = Path("history/session.jsonl")
MAX_HISTORY       = 500   # prune to this many turns on load (file is append-only)
MAX_HISTORY_TURNS = 20    # max turns prepended into each prompt to stay within MAX_LEN

# Convert relative paths to absolute paths for robustness on all platforms
_PROJECT_ROOT = Path(__file__).parent

AGENTS = {
    "sentiment": {
        "path":        str(_PROJECT_ROOT / "qwen3-8b-fingpt-lora" / "adapter_model.safetensors"),
        "r":           8,
        "description": "Financial sentiment specialist (positive/neutral/negative)",
    },
    "multitask": {
        "path":        str(_PROJECT_ROOT / "qwen3-8b-round2-lora" / "adapter_model.safetensors"),
        "r":           16,
        "description": "Multi-task: sentiment, financial Q&A, headline classification",
    },
    # Uncomment after forecaster training:
    # "forecaster": {
    #     "path":        str(_PROJECT_ROOT / "qwen3-8b-forecaster-lora" / "adapter_model.safetensors"),
    #     "r":           16,
    #     "description": "Stock price movement forecasting",
    # },
}


# ─────────────────────────────────────────────────────────────────────────────
# Query classifier
# ─────────────────────────────────────────────────────────────────────────────

def classify_query(query: str) -> str:
    """
    Keyword-based router. Returns: "sentiment" | "multitask"
    Forecaster route is commented out until that adapter is trained.
    Intentionally simple — replace with a trained classifier if needed.
    """
    q = query.lower()

    # Forecaster keywords intentionally omitted — adapter not yet trained.
    # When ready: uncomment this block AND add "forecaster" entry to AGENTS.
    # if any(kw in q for kw in ["forecast", "predict", "next week", "price target",
    #                            "stock movement", "will rise", "will fall"]):
    #     return "forecaster"

    if any(kw in q for kw in ["sentiment", "bullish", "bearish",
                               "what is the tone", "classify this"]):
        return "sentiment"

    if any(kw in q for kw in ["headline", "price going up", "price going down",
                               "does this headline"]):
        return "multitask"

    return "multitask"   # safe default


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class Orchestrator:
    """
    Manages the base model, all LoRA agents, and the RAG layer.

    The base model is loaded once. LoRA adapters hot-swap per query with
    three-path logic to handle rank changes without crashing:
      - First load:    get_peft_model on clean base
      - Same rank:     reuse scaffold, overwrite weights only
      - Rank change:   strip via model.base_model.model, re-wrap, overwrite

    RAG status is always explicit — no silent fallbacks.
    Conversation history persists across sessions via history/session.jsonl.

    Args:
        load_in_4bit: Quantize base model (recommended, ~6GB VRAM).
        rag_top_k:    Number of context chunks to retrieve per query.
    """

    def __init__(self, load_in_4bit: bool = True, rag_top_k: int = 5):
        print("Initialising Orchestrator...")

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=BASE_MODEL,
            max_seq_length=MAX_LEN,
            dtype=None,
            load_in_4bit=load_in_4bit
        )
        self._current_agent:  str | None = None
        self._current_r:      int | None = None  # tracks live LoRA rank for swap logic
        self._empty_index_warned = False          # warn only once per session

        self.rag = RAGLayer(top_k=rag_top_k)

        # Load persisted conversation history on startup
        self._history: list[dict] = self._load_history()

        print(f"Orchestrator ready.")
        print(f"  RAG index:     {self.rag.index_size} chunks")
        print(f"  Agents:        {list(AGENTS.keys())}")
        print(f"  History turns: {len(self._history)}")

    # ── History persistence ───────────────────────────────────────────────────

    def _load_history(self) -> list[dict]:
        """
        Load conversation history from disk on startup.
        Prunes to MAX_HISTORY most recent turns — the file is append-only
        so oldest turns are at the top.
        """
        if not HISTORY_FILE.exists():
            return []
        try:
            turns = []
            with open(HISTORY_FILE, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        turns.append(json.loads(line))
            if len(turns) > MAX_HISTORY:
                turns = turns[-MAX_HISTORY:]
            log.info(f"Loaded {len(turns)} history turns from {HISTORY_FILE}")
            return turns
        except Exception as e:
            log.warning(f"Could not load history: {e}. Starting fresh.")
            return []

    def _append_history(self, role: str, content: str):
        """
        Append a single turn to in-memory history and immediately write
        to disk. Append-only write ensures no data is lost if the process
        crashes between turns.
        """
        ts   = datetime.now(timezone.utc).isoformat()
        turn = {"role": role, "content": content, "ts": ts}
        self._history.append(turn)
        try:
            HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(HISTORY_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(turn, ensure_ascii=False) + "\n")
        except Exception as e:
            log.warning(f"Could not persist history turn: {e}")

    def reset_history(self):
        """
        Clear in-memory history and delete the persisted file.
        Called by the benchmark between tasks to prevent prior turns
        contaminating subsequent task results.
        """
        self._history.clear()
        if HISTORY_FILE.exists():
            HISTORY_FILE.unlink()
        log.info("Conversation history cleared.")

    def _history_for_prompt(self) -> list[dict]:
        """
        Return the most recent MAX_HISTORY_TURNS turns stripped of the
        timestamp field — the model only needs role + content.
        Drops oldest turns first to stay within MAX_LEN token budget.
        """
        turns = self._history[-MAX_HISTORY_TURNS:]
        return [{"role": t["role"], "content": t["content"]} for t in turns]

    # ── Agent hot-swapping ────────────────────────────────────────────────────

    def _load_agent(self, agent_name: str):
        """
        Hot-swap the LoRA adapter with three-path logic:

        Path 1 — First load (_current_agent is None):
            The base model has no LoRA attached yet.
            Call get_peft_model to wrap it with a fresh scaffold.

        Path 2 — Same rank as current adapter:
            The scaffold already has the right shape.
            Skip get_peft_model, overwrite weights only via set_peft_model_state_dict.

        Path 3 — Different rank from current adapter:
            Unsloth refuses to re-wrap a model with a different r.
            Strip the PEFT wrapper via model.base_model.model (returns the
            underlying Qwen3ForCausalLM with Unsloth optimizations intact),
            then re-wrap with the new rank.
        """
        if self._current_agent == agent_name:
            return

        cfg  = AGENTS[agent_name]
        path = cfg["path"]

        if not Path(path).exists():
            raise FileNotFoundError(
                f"Agent '{agent_name}' adapter not found at {path}. "
                f"Train it first or update the path in AGENTS."
            )

        target_r = cfg["r"]
        print(f"\n[Orchestrator] Swapping to agent: {agent_name}  (r={target_r})")

        if self._current_agent is None:
            # Path 1: first load
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=target_r,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                lora_alpha=target_r,
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing=False,
                random_state=42,
            )
        elif target_r != self._current_r:
            # Path 3: rank change — strip wrapper first, then re-wrap
            self.model = self.model.base_model.model
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=target_r,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                lora_alpha=target_r,
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing=False,
                random_state=42,
            )
        # Path 2: same rank — fall through, existing scaffold reused

        set_peft_model_state_dict(self.model, load_file(path))
##        FastLanguageModel.for_inference(self.model)
        self._current_agent = agent_name
        self._current_r     = target_r
        print(f"  ✓ Agent '{agent_name}' loaded.")

    # ── RAG status handler ────────────────────────────────────────────────────

    def _handle_rag_status(self, rag_result: dict) -> str | None:
        """
        React to every RAG status explicitly.
        Returns a warning string for the result dict, or None if all is OK.
        """
        status = rag_result["rag_status"]
        msg    = rag_result["rag_status_msg"]

        if status == RAGStatus.OK:
            return None

        if status == RAGStatus.EMPTY_INDEX:
            if not self._empty_index_warned:
                print(
                    "\n[Orchestrator] ⚠ RAG index is empty. "
                    "Call orc.rag.add_documents() or use `index fetch <ticker>` "
                    "to index financial documents for context-aware answers."
                )
                self._empty_index_warned = True
            return "No documents indexed. Answer based on model knowledge only."

        if status == RAGStatus.NO_MATCH:
            log.info(f"[RAG] No match: {msg}")
            return "No relevant documents found in index. Answer based on model knowledge only."

        if status == RAGStatus.LOW_CONFIDENCE:
            log.warning(f"[RAG] Low confidence: {msg}")
            return f"Retrieved documents were not sufficiently relevant. {msg}"

        if status == RAGStatus.ERROR:
            log.error(f"[RAG] Retrieval error: {msg}")
            return f"RAG retrieval failed ({msg}). Answer based on model knowledge only."

        return None   # unknown status — safe fallback

    # ── Inference ─────────────────────────────────────────────────────────────

    def _generate(
        self,
        system_prompt:  str,
        user_prompt:    str,
        max_new_tokens: int = 512,
        history:        list[dict] | None = None,
    ) -> str:
        """
        Build the full message list (system → history → user), apply Qwen3
        chat template, generate, strip the <think> block, and return the
        clean final answer.
        """
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_prompt})

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=MAX_LEN,
            return_token_type_ids=False,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        raw = self.tokenizer.decode(
            outputs[0][input_len:], skip_special_tokens=True
        ).strip()

        # Strip Qwen3 thinking block — evaluate only the final answer
        idx = raw.find("</think>")
        return raw[idx + 8:].strip() if idx != -1 else raw

    # ── Main entry point ──────────────────────────────────────────────────────

    def query(
        self,
        user_query:     str,
        agent_override: str | None = None,
        max_new_tokens: int = 512,
        verbose:        bool = False,
        record_history: bool = True,
    ) -> dict:
        """
        Process a query end-to-end: classify → RAG → agent → generate.

        Returns:
            {
                "answer":           str         model's final answer (think stripped)
                "agent_used":       str         which agent handled the query
                "query_type":       str         classifier output
                "rag_used":         bool        whether context was injected
                "rag_status":       RAGStatus   exact RAG outcome enum value
                "rag_status_msg":   str         human-readable RAG explanation
                "rag_warning":      str|None    non-None if RAG failed/degraded
                "retrieved_chunks": list        raw retrieved chunk dicts
            }
        """
        # ── 1. Classify ───────────────────────────────────────────────────────
        query_type = agent_override or classify_query(user_query)

        if verbose:
            print(f"\n[Orchestrator] query_type={query_type}")

        # ── 2. RAG enrichment ─────────────────────────────────────────────────
        rag_result  = self.rag.enrich(user_query, task_hint=query_type)
        rag_warning = self._handle_rag_status(rag_result)

        if verbose:
            print(f"[RAG] status={rag_result['rag_status'].value}  {rag_result['rag_status_msg']}")
            if rag_result["rag_used"]:
                for c in rag_result["retrieved_chunks"]:
                    print(f"  chunk score={c['score']:.3f}  {c['text'][:80]}...")

        # ── 3. Prepend conversation history ───────────────────────────────────
        history = self._history_for_prompt()

        # ── 4. Load agent ─────────────────────────────────────────────────────
        self._load_agent(query_type)

        # ── 5. Generate ───────────────────────────────────────────────────────
        answer = self._generate(
            system_prompt  = rag_result["system_prompt"],
            user_prompt    = rag_result["user_prompt"],
            max_new_tokens = max_new_tokens,
            history        = history,
        )

        # ── 6. Persist both turns immediately after generation ────────────────
        # record_history=False is used by the benchmark to prevent history
        # accumulation within a task from truncating later prompts at MAX_LEN=512
        if record_history:
            self._append_history("user",      user_query)
            self._append_history("assistant", answer)

        return {
            "answer":           answer,
            "agent_used":       query_type,
            "query_type":       query_type,
            "rag_used":         rag_result["rag_used"],
            "rag_status":       rag_result["rag_status"],
            "rag_status_msg":   rag_result["rag_status_msg"],
            "rag_warning":      rag_warning,
            "retrieved_chunks": rag_result["retrieved_chunks"],
        }

    def __del__(self):
        try:
            del self.model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
