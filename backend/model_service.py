"""Model loading and inference for BHARAT-AI.

generate() and generate_stream() use the same logic as the Colab
generate_response(): same chat template, enable_thinking=False, greedy decoding,
same slicing. generate_stream() adds real-time events and timing.
"""
import queue
import threading
import time
from typing import Iterator, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers.generation.streamers import BaseStreamer
except ImportError:  # generate() only calls put() and end(), so duck typing is enough
    BaseStreamer = object

DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

STAGE_ORDER = [
    "prompt_received",
    "tokenization",
    "input_preparation",
    "model_inference",
    "token_generation",
    "response_decoding",
    "response_completed",
]


class ModelNotReady(RuntimeError):
    pass


class _Cancelled(Exception):
    """Raised inside generate() when the client disconnects."""


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _ms(since: float) -> float:
    return round((time.perf_counter() - since) * 1000, 2)


def _stage(name, status, duration_ms=None, measurement="measured", note=None) -> dict:
    return dict(name=name, status=status, duration_ms=duration_ms,
                measurement=measurement, note=note)


class _EventStreamer(BaseStreamer):
    """Receives each generated token id from model.generate() and forwards it,
    with a timestamp, to a queue. Counting ids here gives exact token counts."""

    def __init__(self, q: "queue.Queue", cancel: threading.Event):
        self.q = q
        self.cancel = cancel
        self._prompt_seen = False

    def put(self, value):
        if self.cancel.is_set():
            raise _Cancelled()
        if not self._prompt_seen:  # the first call carries the prompt ids, not output
            self._prompt_seen = True
            return
        t = time.perf_counter()
        for tid in value.reshape(-1).tolist():
            self.q.put(("token", t, tid))

    def end(self):
        pass  # the worker thread signals completion itself


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

    # ---------- shared input preparation ----------
    def _tokenize(self, prompt: str):
        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True,
        )

    def _to_device(self, inputs):
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)
        self._sync()
        return input_ids, attention_mask

    # ---------- inference (non-streaming, unchanged behaviour) ----------
    def generate(self, prompt: str, max_new_tokens: int = 128) -> dict:
        if self.state != "loaded":
            raise ModelNotReady(f"Model state is '{self.state}'")

        with self._gen_lock:
            stages = []
            t_start = time.perf_counter()
            stages.append(_stage("prompt_received", "completed", 0.0))

            t = time.perf_counter()
            inputs = self._tokenize(prompt)
            stages.append(_stage(
                "tokenization", "completed", _ms(t),
                note="Chat-template rendering and tokenization happen in one call"))

            t = time.perf_counter()
            input_ids, attention_mask = self._to_device(inputs)
            stages.append(_stage("input_preparation", "completed", _ms(t)))

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
            stages.append(_stage(
                "model_inference", "completed", round(inference_seconds * 1000, 2),
                note="Includes prefill and all decoding steps"))
            stages.append(_stage(
                "token_generation", "completed", None, "not_separable",
                note="Use /api/generate/stream to separate this from inference"))

            t = time.perf_counter()
            response_tokens = outputs[0][input_ids.shape[-1]:]
            response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            stages.append(_stage("response_decoding", "completed", _ms(t)))
            stages.append(_stage("response_completed", "completed"))

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

    # ---------- inference (streaming) ----------
    def generate_stream(self, prompt: str, max_new_tokens: int = 128) -> Iterator[dict]:
        """Yield events: {"event": "stage"|"token"|"done"|"error", "data": {...}}.

        Measurement notes:
          model_inference  = time from calling generate() to the FIRST token
                             (prefill plus generate() setup).
          token_generation = first token to last token.
          response_decoding = the final full decode pass; per-token decoding for
                             the live text happens during token_generation.
        """
        if self.state != "loaded":
            raise ModelNotReady(f"Model state is '{self.state}'")

        stages = {n: _stage(n, "waiting") for n in STAGE_ORDER}

        def emit(name, status, duration_ms=None, measurement="measured", note=None):
            stages[name] = _stage(name, status, duration_ms, measurement, note)
            return {"event": "stage", "data": stages[name]}

        cancel = threading.Event()
        q: "queue.Queue" = queue.Queue()
        worker = None
        current = "prompt_received"

        with self._gen_lock:
            try:
                t_start = time.perf_counter()
                yield emit("prompt_received", "completed", 0.0)

                current = "tokenization"
                yield emit(current, "processing")
                t = time.perf_counter()
                inputs = self._tokenize(prompt)
                yield emit(current, "completed", _ms(t),
                           note="Chat-template rendering and tokenization happen in one call")

                current = "input_preparation"
                yield emit(current, "processing")
                t = time.perf_counter()
                input_ids, attention_mask = self._to_device(inputs)
                yield emit(current, "completed", _ms(t))

                current = "model_inference"
                yield emit(current, "processing", note="Prefill: waiting for the first token")

                streamer = _EventStreamer(q, cancel)
                errors: list = []

                def run():
                    try:
                        with torch.inference_mode():  # thread-local, so set it here
                            self.model.generate(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                max_new_tokens=max_new_tokens,
                                do_sample=False,
                                pad_token_id=self.tokenizer.eos_token_id,
                                streamer=streamer,
                            )
                        self._sync()
                    except _Cancelled:
                        pass
                    except Exception as exc:
                        errors.append(exc)
                    finally:
                        q.put(("end", time.perf_counter(), None))

                worker = threading.Thread(target=run, daemon=True)
                t_gen = time.perf_counter()
                worker.start()

                ids: list = []
                emitted = ""
                first_t = last_t = end_t = None

                while True:
                    kind, t, tid = q.get()
                    if kind == "end":
                        end_t = t
                        break
                    ids.append(tid)
                    last_t = t
                    if first_t is None:
                        first_t = t
                        yield emit("model_inference", "completed",
                                   round((t - t_gen) * 1000, 2),
                                   note="Time to first token (prefill plus generate() setup)")
                        current = "token_generation"
                        yield emit(current, "processing",
                                   note="Live token count is in the token events")

                    text = self.tokenizer.decode(ids, skip_special_tokens=True)
                    if text.endswith("\ufffd"):
                        continue  # incomplete multi-byte character; wait for more tokens
                    delta = text[len(emitted):]
                    emitted = text
                    yield {"event": "token", "data": {
                        "text": delta,
                        "count": len(ids),
                        "elapsed_ms": round((t - t_gen) * 1000, 1),
                    }}

                worker.join()
                if errors:
                    raise errors[0]
                if first_t is None:
                    raise RuntimeError("The model produced no tokens")

                n = len(ids)
                decode_ms = round((last_t - first_t) * 1000, 2)
                yield emit("token_generation", "completed", decode_ms,
                           note=f"{n} tokens streamed")

                current = "response_decoding"
                yield emit(current, "processing")
                t = time.perf_counter()
                response = self.tokenizer.decode(ids, skip_special_tokens=True)
                yield emit(current, "completed", _ms(t),
                           note="Final full decode pass; live decoding ran per token")

                current = "response_completed"
                yield emit(current, "completed")

                inference_seconds = end_t - t_gen
                yield {"event": "done", "data": {
                    "response": response,  # authoritative; replaces the streamed text
                    "input_tokens": int(input_ids.shape[-1]),
                    "generated_tokens": n,  # may include the EOS token
                    "inference_seconds": round(inference_seconds, 4),
                    "total_seconds": round(time.perf_counter() - t_start, 4),
                    "tokens_per_second": round(n / inference_seconds, 2)
                    if inference_seconds > 0 else None,
                    "time_to_first_token_ms": round((first_t - t_gen) * 1000, 2),
                    "decode_tokens_per_second": round((n - 1) / (last_t - first_t), 2)
                    if n > 1 and last_t > first_t else None,
                    "stages": [stages[s] for s in STAGE_ORDER],
                    "model_name": self.model_name,
                    "device": self.device,
                    "settings": {"max_new_tokens": max_new_tokens, "do_sample": False,
                                 "dtype": self.dtype_name, "enable_thinking": False},
                }}
            except Exception as exc:
                yield emit(current, "failed", note=f"{type(exc).__name__}: {exc}")
                yield {"event": "error", "data": {
                    "message": f"{type(exc).__name__}: {exc}", "stage": current}}
            finally:
                cancel.set()  # stops generation early if the client disconnected
                if worker is not None and worker.is_alive():
                    worker.join(timeout=30)