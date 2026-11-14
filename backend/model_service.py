"""Model loading and inference for BHARAT-AI.

generate() reuses the exact logic from the Colab generate_response():
same chat template, enable_thinking=False, greedy decoding, same slicing.
Timing is added around each step.
"""
import threading
import time
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class ModelNotReady(RuntimeError):
    pass


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _ms(since: float) -> float:
    return round((time.perf_counter() - since) * 1000, 2)


class ModelService:
    def __init__(self, model_name: str, dtype_name: str = "float16"):
        self.model_name = model_name
        self.dtype_name = dtype_name
        self.device = pick_device()
        self.tokenizer = None
        self.model = None
        self.state = "unloaded"  # unloaded | loading | loaded | failed
        self.error: Optional[str] = None
        self.load_seconds: Optional[float] = None
        self._gen_lock = threading.Lock()  # one generation at a time

    # ---------- loading ----------
    def load(self) -> None:
        self.state = "loading"
        self.error = None
        t0 = time.perf_counter()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name, dtype=DTYPES[self.dtype_name]
            )
            model.to(self.device)
            model.eval()
            self.model = model
            self.load_seconds = round(time.perf_counter() - t0, 2)
            self.state = "loaded"
        except Exception as exc:  # report, don't crash the server
            self.error = f"{type(exc).__name__}: {exc}"
            self.state = "failed"

    # ---------- monitoring ----------
    def _sync(self) -> None:
        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device == "mps":
            torch.mps.synchronize()

    def memory_info(self) -> dict:
        if self.device == "cuda":
            return {
                "allocated_mb": round(torch.cuda.memory_allocated() / 1e6, 1),
                "total_mb": round(
                    torch.cuda.get_device_properties(0).total_memory / 1e6, 1
                ),
            }
        if self.device == "mps":
            # Apple Silicon uses unified memory; no separate GPU total is exposed.
            return {
                "allocated_mb": round(torch.mps.current_allocated_memory() / 1e6, 1),
                "total_mb": None,
            }
        return {"allocated_mb": None, "total_mb": None}

    def status(self) -> dict:
        return {
            "state": self.state,
            "model_name": self.model_name,
            "device": self.device,
            "dtype": self.dtype_name,
            "load_seconds": self.load_seconds,
            "error": self.error,
            "memory": self.memory_info(),
            "gpu_utilization_percent": None,  # not exposed by PyTorch on MPS
        }

    # ---------- inference ----------
    def generate(self, prompt: str, max_new_tokens: int = 128) -> dict:
        if self.state != "loaded":
            raise ModelNotReady(f"Model state is '{self.state}'")

        with self._gen_lock:
            stages = []
            t_start = time.perf_counter()
            stages.append(dict(name="prompt_received", status="completed",
                               duration_ms=0.0, measurement="measured"))

            # Tokenization (apply_chat_template renders the template AND tokenizes)
            t = time.perf_counter()
            messages = [{"role": "user", "content": prompt}]
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_tensors="pt",
                return_dict=True,
            )
            stages.append(dict(
                name="tokenization", status="completed", duration_ms=_ms(t),
                measurement="measured",
                note="Chat-template rendering and tokenization happen in one call",
            ))

            # Input preparation (move tensors to the model device)
            t = time.perf_counter()
            input_ids = inputs["input_ids"].to(self.model.device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.model.device)
            self._sync()
            stages.append(dict(name="input_preparation", status="completed",
                               duration_ms=_ms(t), measurement="measured"))

            # Model inference + token generation (one call to model.generate)
            t = time.perf_counter()
            with torch.inference_mode():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            self._sync()
            inference_seconds = time.perf_counter() - t
            stages.append(dict(name="model_inference", status="completed",
                               duration_ms=round(inference_seconds * 1000, 2),
                               measurement="measured",
                               note="Includes prefill and all decoding steps"))
            stages.append(dict(name="token_generation", status="completed",
                               duration_ms=None, measurement="not_separable",
                               note="Needs streaming (Stage 2) to separate from inference"))

            # Decoding
            t = time.perf_counter()
            response_tokens = outputs[0][input_ids.shape[-1]:]
            response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            stages.append(dict(name="response_decoding", status="completed",
                               duration_ms=_ms(t), measurement="measured"))
            stages.append(dict(name="response_completed", status="completed",
                               duration_ms=None, measurement="measured"))

            n_out = int(response_tokens.shape[-1])  # may include the EOS token
            return {
                "response": response,
                "input_tokens": int(input_ids.shape[-1]),
                "generated_tokens": n_out,
                "inference_seconds": round(inference_seconds, 4),
                "total_seconds": round(time.perf_counter() - t_start, 4),
                "tokens_per_second": round(n_out / inference_seconds, 2)
                if n_out > 0 and inference_seconds > 0 else None,
                "stages": stages,
                "model_name": self.model_name,
                "device": self.device,
                "settings": {"max_new_tokens": max_new_tokens, "do_sample": False,
                             "dtype": self.dtype_name, "enable_thinking": False},
            }