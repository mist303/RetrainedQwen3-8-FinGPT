"""
FinGPT Orchestrator
====================
Routes user queries through the RAG layer and dispatches to the correct
LoRA agent, all sharing the same frozen Qwen3-8B base model.

Architecture:
    User Query
         │
         ▼
    Orchestrator.query()
         │
         ├─ 1. classify_query()  →  "sentiment" | "multitask" | "forecaster"
         │
         ├─ 2. RAGLayer.enrich() →  inject retrieved context into system prompt
         │       └─ explicit RAGStatus returned; orchestrator reacts per-case
         │
         └─ 3. _load_agent()     →  hot-swap LoRA adapter (base model never reloads)
                └─ _generate()   →  run inference, strip <think> block, return answer

RAG status handling:
  OK             → proceed with enriched context
  EMPTY_INDEX    → warn once per session, proceed without context
  NO_MATCH       → note in result, proceed without context
  LOW_CONFIDENCE → log best score, proceed without context
  ERROR          → log exception, flag in result, proceed without context

See docs/orchestrator_requirements.md for the full specification.
"""

import gc
import torch
import logging
from pathlib import Path
from safetensors.torch import load_file
from peft import set_peft_model_state_dict
from unsloth import FastLanguageModel

from rag import RAGLayer
from rag.rag_layer import RAGStatus

log = logging.getLogger(__name__)

BASE_MODEL = "unsloth/Qwen3-8B"
MAX_LEN    = 512

AGENTS = {
    "sentiment": {
        "path":        "./qwen3-8b-fingpt-lora/adapter_model.safetensors",
        "r":           8,
        "description": "Financial sentiment specialist (positive/neutral/negative)",
    },
    "multitask": {
        "path":        "./qwen3-8b-round2-lora/adapter_model.safetensors",
        "r":           16,
        "description": "Multi-task: sentiment, financial Q&A, headline classification",
    },
    # Uncomment after Round 3 training:
    # "forecaster": {
    #     "path":        "./qwen3-8b-round3-lora/adapter_model.safetensors",
    #     "r":           16,
    #     "description": "Stock price movement forecasting",
    # },
}


def classify_query(query: str) -> str:
    """
    Keyword-based router → "sentiment" | "multitask" | "forecaster".
    Intentionally simple — upgrade to a trained classifier when needed.
    """
    q = query.lower()
    if any(kw in q for kw in ["forecast","predict","next week","price target",
                               "stock movement","will rise","will fall"]):
        return "forecaster"
    if any(kw in q for kw in ["sentiment","bullish","bearish",
                               "what is the tone","classify this"]):
        return "sentiment"
    if any(kw in q for kw in ["headline","price going up","price going down",
                               "does this headline"]):
        return "multitask"
    return "multitask"


class Orchestrator:
    """
    Manages the shared base model, all LoRA agents, and the RAG layer.

    Base model loads once. Adapters hot-swap per query (~seconds, no VRAM spike).
    RAG status is always explicit — no silent fallbacks.

    Args:
        load_in_4bit: Quantize base model (recommended, ~6 GB VRAM).
        rag_top_k:    Chunks to retrieve per query.
    """

    def __init__(self, load_in_4bit: bool = True, rag_top_k: int = 5):
        print("Initialising Orchestrator...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=BASE_MODEL, max_seq_length=MAX_LEN,
            dtype=None, load_in_4bit=load_in_4bit, local_files_only=True,
        )
        self._current_agent: str | None = None
        self._empty_index_warned = False
        self.rag = RAGLayer(top_k=rag_top_k)
        print(f"Orchestrator ready. RAG index: {self.rag.index_size} chunks | Agents: {list(AGENTS.keys())}")

    def _load_agent(self, agent_name: str):
        if self._current_agent == agent_name:
            return
        cfg = AGENTS[agent_name]
        if not Path(cfg["path"]).exists():
            raise FileNotFoundError(
                f"Agent '{agent_name}' not found at {cfg['path']}. "
                "Train it first or update AGENTS in orchestrator.py."
            )
        print(f"\n[Orchestrator] → {agent_name}  (r={cfg['r']})")
        self.model = FastLanguageModel.get_peft_model(
            self.model, r=cfg["r"],
            target_modules=["q_proj","k_proj","v_proj","o_proj",
                            "gate_proj","up_proj","down_proj"],
            lora_alpha=cfg["r"], lora_dropout=0, bias="none",
            use_gradient_checkpointing=False, random_state=42,
        )
        set_peft_model_state_dict(self.model, load_file(cfg["path"]))
        FastLanguageModel.for_inference(self.model)
        self._current_agent = agent_name

    def _handle_rag_status(self, rag_result: dict) -> str | None:
        status, msg = rag_result["rag_status"], rag_result["rag_status_msg"]
        if status == RAGStatus.OK:
            return None
        if status == RAGStatus.EMPTY_INDEX:
            if not self._empty_index_warned:
                print("\n[Orchestrator] ⚠ RAG index empty. Use `index fetch <ticker>` or `index add-file <path>`.")
                self._empty_index_warned = True
            return "No documents indexed. Answer based on model knowledge only."
        if status == RAGStatus.NO_MATCH:
            log.info(f"[RAG] No match: {msg}")
            return "No relevant documents found. Answer based on model knowledge only."
        if status == RAGStatus.LOW_CONFIDENCE:
            log.warning(f"[RAG] Low confidence: {msg}")
            return f"Retrieved documents not sufficiently relevant. {msg}"
        if status == RAGStatus.ERROR:
            log.error(f"[RAG] Error: {msg}")
            return f"RAG retrieval failed. Answer based on model knowledge only."
        return None

    def _generate(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 512) -> str:
        text = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt},
             {"role": "user",   "content": user_prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=MAX_LEN, return_token_type_ids=False,
        ).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        raw = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        idx = raw.find("</think>")
        return raw[idx + 8:].strip() if idx != -1 else raw

    def query(
        self,
        user_query:     str,
        agent_override: str | None = None,
        max_new_tokens: int = 512,
        verbose:        bool = False,
    ) -> dict:
        """
        classify → RAG enrich → agent dispatch → generate

        Returns:
            answer, agent_used, query_type, rag_used, rag_status,
            rag_status_msg, rag_warning, retrieved_chunks
        """
        query_type = agent_override or classify_query(user_query)
        if query_type == "forecaster" and "forecaster" not in AGENTS:
            print("[Orchestrator] Forecaster not yet trained — routing to multitask.")
            query_type = "multitask"

        rag_result  = self.rag.enrich(user_query, task_hint=query_type)
        rag_warning = self._handle_rag_status(rag_result)

        if verbose:
            print(f"[Orchestrator] type={query_type} rag={rag_result['rag_status'].value}")
            for c in rag_result.get("retrieved_chunks", []):
                print(f"  chunk score={c['score']:.3f}  {c['text'][:80]}...")

        self._load_agent(query_type)
        answer = self._generate(rag_result["system_prompt"], rag_result["user_prompt"], max_new_tokens)

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
            torch.cuda.empty_cache()
        except Exception:
            pass
