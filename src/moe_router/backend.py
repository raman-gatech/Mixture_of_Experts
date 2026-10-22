import json
import hashlib
import threading
import time
from pathlib import Path
from contextlib import nullcontext

from .settings import LABELS, SAFETY_LABELS
from .safety import SAFETY_SYSTEM, choose_expert


class BusyError(Exception):
    pass


class InputError(Exception):
    pass


def validate_artifacts(settings):
    """Fail before loading expensive weights; each checkpoint owns its label order."""
    for directory in (settings.router_dir, settings.base_dir):
        if not (directory / "config.json").is_file():
            raise FileNotFoundError(f"Missing config.json in {directory}")
        if not any(directory.glob("*.safetensors")):
            raise FileNotFoundError(f"Missing safetensors weights in {directory}")
    config = json.loads((settings.router_dir / "config.json").read_text())
    mapping = config.get("id2label", {})
    labels = tuple(str(mapping.get(str(i), "")).upper() for i in range(len(mapping)))
    if len(set(labels)) != len(labels) or set(labels) not in (set(LABELS), set(SAFETY_LABELS)):
        raise ValueError("Router requires explicit three- or four-expert id2label metadata")
    declared_bases = set()
    for label in labels:
        directory = settings.expert_dir(label)
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            if not (directory / name).is_file():
                raise FileNotFoundError(f"Missing {name} in {directory}")
        cfg = json.loads((directory / "adapter_config.json").read_text())
        declared_bases.add(cfg.get("base_model_name_or_path"))
    if len(declared_bases) != 1 or None in declared_bases:
        raise ValueError("Experts must declare the same base model")
    if "SAFETY" in labels and config.get("safety_validation_passed") is not True and not settings.allow_safety_candidate:
        raise ValueError("Safety router has not passed validation gates; use an isolated candidate evaluation")
    if "SAFETY" in labels and not settings.allow_safety_candidate:
        marker = settings.expert_dir("SAFETY") / "acceptance.json"
        if not marker.is_file():
            raise ValueError("Safety adapter lacks answer-review acceptance; use an isolated candidate evaluation")
        acceptance = json.loads(marker.read_text())
        if acceptance.get("accepted") is not True or not acceptance.get("review_evidence"):
            raise ValueError("Safety adapter has not passed answer-review acceptance")
        digest = hashlib.sha256()
        with (settings.expert_dir("SAFETY") / "adapter_model.safetensors").open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        if acceptance.get("adapter_sha256") != digest.hexdigest():
            raise ValueError("Safety answer-review acceptance does not match these adapter weights")
    return labels


class Backend:
    def __init__(self, settings):
        self.settings = settings
        self.lock = threading.Lock()
        self.slots = threading.BoundedSemaphore(settings.queue_capacity)
        self.ready = False

    def load(self):
        print("Loading inference libraries…", flush=True)
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
        from peft import PeftModel

        if not torch.cuda.is_available():
            raise RuntimeError("Real Llama inference requires an allocated CUDA GPU")
        self.labels = validate_artifacts(self.settings)
        router_config = json.loads((self.settings.router_dir / "config.json").read_text())
        self.router_max_length = int(router_config.get("router_max_length", 128))
        self.safety_threshold = float(router_config.get("safety_threshold", 0.5))
        if not 0 < self.safety_threshold <= 1 or not 32 <= self.router_max_length <= 512:
            raise ValueError("Invalid router window length or safety threshold")
        print("Loading router…", flush=True)
        self.torch = torch
        self.router_tokenizer = AutoTokenizer.from_pretrained(self.settings.router_dir, local_files_only=True)
        self.router = AutoModelForSequenceClassification.from_pretrained(
            self.settings.router_dir, local_files_only=True, output_loading_info=True
        )
        self.router, info = self.router
        if any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError(f"Incomplete router checkpoint: {info}")
        self.router.to(self.settings.router_device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.settings.base_dir, local_files_only=True)
        if not self.tokenizer.chat_template:
            raise ValueError("Base tokenizer has no chat template")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print("Loading quantized Instruct base…", flush=True)
        base = AutoModelForCausalLM.from_pretrained(
            self.settings.base_dir, local_files_only=True, torch_dtype=torch.float16,
            device_map={"": 0},
        )
        print("Loading named expert adapters…", flush=True)
        self.model = PeftModel.from_pretrained(base, self.settings.expert_dir("MATH"), adapter_name="MATH", is_trainable=False)
        for label in self.labels:
            if label == "MATH":
                continue
            self.model.load_adapter(self.settings.expert_dir(label), adapter_name=label, is_trainable=False)
        combined = self.settings.model_root / "combined_expert"
        if (combined / "adapter_config.json").is_file():
            self.model.load_adapter(combined, adapter_name="COMBINED", is_trainable=False)
        self.model.eval()
        self.ready = True
        print("All experts ready.", flush=True)

    def close(self):
        self.ready = False

    def generate(self, prompt, max_new_tokens=None, expert_override=None):
        if not self.ready:
            raise RuntimeError("Models are not ready")
        if expert_override is not None and expert_override not in (*getattr(self, "labels", LABELS), "BASE", "COMBINED"):
            raise ValueError("Invalid evaluation expert")
        if not self.slots.acquire(blocking=False):
            raise BusyError("The request queue is full; try again shortly")
        start = time.perf_counter()
        acquired = False
        try:
            acquired = self.lock.acquire(timeout=self.settings.queue_timeout)
            if not acquired:
                raise BusyError("The request timed out waiting for the model")
            if expert_override is None:
                return self._generate(prompt, max_new_tokens, start)
            return self._generate(prompt, max_new_tokens, start, expert_override)
        finally:
            if acquired:
                self.lock.release()
            self.slots.release()

    def _generate(self, prompt, max_new_tokens, start, expert_override=None):
        torch = self.torch
        raw = self.tokenizer(prompt, add_special_tokens=False, truncation=True,
                             max_length=self.settings.max_input_tokens + 1)
        if len(raw["input_ids"]) > self.settings.max_input_tokens:
            raise InputError(f"Input exceeds {self.settings.max_input_tokens} tokens")
        routing_start = time.perf_counter()
        if "SAFETY" in self.labels:
            encoded = self.router_tokenizer(prompt, max_length=self.router_max_length, truncation=True,
                                            stride=self.router_max_length // 4, return_overflowing_tokens=True,
                                            padding=True, return_tensors="pt")
            encoded.pop("overflow_to_sample_mapping", None)
        else:
            encoded = self.router_tokenizer(prompt, max_length=128, truncation=True, return_tensors="pt")
        encoded = encoded.to(self.settings.router_device)
        with torch.inference_mode():
            all_probs = torch.softmax(self.router(**encoded).logits, dim=-1)
        probs = all_probs[0].tolist()
        safety_probability = float(all_probs[:, self.labels.index("SAFETY")].max().item()) if "SAFETY" in self.labels else None
        label, router_expert, routing_mode, confidence = choose_expert(
            self.labels, probs, safety_probability, getattr(self, "safety_threshold", 0.5), expert_override
        )
        routing_ms = (time.perf_counter() - routing_start) * 1000
        # Every expert follows the common policy, including when the safety
        # classifier misses a prompt. The SAFETY adapter adds trained support
        # behavior; the instruction alone is not a safety guarantee.
        messages = [{"role": "system", "content": SAFETY_SYSTEM}]
        messages.append({"role": "user", "content": prompt})
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")
        if inputs.input_ids.shape[-1] > self.settings.max_input_tokens:
            raise InputError(f"Input exceeds {self.settings.max_input_tokens} tokens")
        inputs = inputs.to(self.model.device)
        if label != "BASE":
            self.model.set_adapter(label)
            if self.model.active_adapter != label:
                raise RuntimeError("Adapter selection failed")
        generation_start = time.perf_counter()
        adapter_context = self.model.disable_adapter() if label == "BASE" else nullcontext()
        with adapter_context, torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens or self.settings.max_new_tokens,
                do_sample=False, max_time=self.settings.generation_timeout,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        new_ids = output[0, inputs.input_ids.shape[-1]:]
        answer = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        if not answer:
            raise RuntimeError("The expert returned an empty answer")
        return {
            "response": answer, "expert": label,
            "confidence": round(confidence, 4) if confidence is not None else None,
            "router_expert": router_expert, "routing_mode": routing_mode,
            "active_adapter": "BASE" if label == "BASE" else self.model.active_adapter,
            "timings_ms": {"router": round(routing_ms, 1), "generation": round((time.perf_counter()-generation_start)*1000, 1), "total": round((time.perf_counter()-start)*1000, 1)},
            "generated_tokens": len(new_ids),
        }
