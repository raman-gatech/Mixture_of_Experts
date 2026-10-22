from dataclasses import dataclass
import os
from pathlib import Path

LABELS = ("MATH", "SCIENCE", "GENERAL")
SAFETY_LABELS = (*LABELS, "SAFETY")
_SOURCE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(os.getenv("MOE_PROJECT_ROOT", _SOURCE_ROOT if (_SOURCE_ROOT / "web/index.html").is_file() else Path.cwd())).resolve()


def load_local_env():
    path = Path(os.getenv("MOE_ENV_FILE", PROJECT_ROOT / ".env"))
    if path.is_file():
        for line in path.read_text().splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            name, separator, value = line.partition("=")
            if separator and name.strip().startswith("MOE_"):
                os.environ.setdefault(name.strip(), value.strip())


@dataclass(frozen=True)
class Settings:
    model_root: Path
    router_dir: Path
    base_dir: Path
    router_device: str = "cpu"
    max_prompt_chars: int = 12000
    max_input_tokens: int = 1024
    max_new_tokens: int = 200
    queue_capacity: int = 4
    queue_timeout: float = 20
    generation_timeout: float = 60
    api_token: str = ""
    allow_safety_candidate: bool = False
    active_profile: str = "external-v1"
    native_engine_url: str = "http://127.0.0.1:8001"
    native_engine_token_file: str = "runtime/native/engine.token"
    native_model_dir: str = "artifacts/internal/native/v1/merged"
    native_release: str = "native-internal-v1"
    native_context_limit: int = 2048
    native_output_limit: int = 256
    native_concurrency: int = 1

    def __post_init__(self):
        if min(self.max_prompt_chars, self.max_input_tokens, self.max_new_tokens, self.queue_capacity) < 1:
            raise ValueError("Input limits and queue capacity must be positive")
        if self.queue_timeout <= 0 or self.generation_timeout <= 0:
            raise ValueError("Timeouts must be positive")
        if self.active_profile not in ("external-v1", "native-internal-v1"):
            raise ValueError("Unavailable model profile")
        if not 1 <= self.native_concurrency <= 4 or not 1 <= self.native_output_limit < self.native_context_limit <= 4096:
            raise ValueError("Invalid native resource envelope")

    @classmethod
    def from_env(cls):
        load_local_env()
        root = Path(os.getenv("MOE_MODEL_ROOT", PROJECT_ROOT / "artifacts")).expanduser().resolve()
        default_router = root / ("router-earlier" if (root / "router-earlier/config.json").is_file() else "router")
        return cls(
            model_root=root,
            router_dir=Path(os.getenv("MOE_ROUTER_DIR", default_router)).expanduser().resolve(),
            base_dir=Path(os.getenv("MOE_BASE_DIR", root / "base")).expanduser().resolve(),
            router_device=os.getenv("MOE_ROUTER_DEVICE", "cpu"),
            max_prompt_chars=int(os.getenv("MOE_MAX_PROMPT_CHARS", "12000")),
            max_input_tokens=int(os.getenv("MOE_MAX_INPUT_TOKENS", "1024")),
            max_new_tokens=int(os.getenv("MOE_MAX_NEW_TOKENS", "200")),
            queue_capacity=int(os.getenv("MOE_QUEUE_CAPACITY", "4")),
            queue_timeout=float(os.getenv("MOE_QUEUE_TIMEOUT", "20")),
            generation_timeout=float(os.getenv("MOE_GENERATION_TIMEOUT", "60")),
            api_token=os.getenv("MOE_API_TOKEN", ""),
            allow_safety_candidate=os.getenv("MOE_ALLOW_SAFETY_CANDIDATE", "0") == "1",
            active_profile=os.getenv("MOE_ACTIVE_PROFILE", "external-v1"),
            native_engine_url=os.getenv("MOE_NATIVE_ENGINE_URL", "http://127.0.0.1:8001"),
            native_engine_token_file=os.getenv("MOE_ENGINE_TOKEN_FILE", "runtime/native/engine.token"),
            native_model_dir=os.getenv("MOE_NATIVE_MODEL_DIR", "artifacts/internal/native/v1/merged"),
            native_release=os.getenv("MOE_NATIVE_RELEASE", "native-internal-v1"),
            native_context_limit=int(os.getenv("MOE_NATIVE_CONTEXT_LIMIT", "2048")),
            native_output_limit=int(os.getenv("MOE_NATIVE_OUTPUT_LIMIT", "256")),
            native_concurrency=int(os.getenv("MOE_NATIVE_CONCURRENCY", "1")),
        )

    def expert_dir(self, label):
        if label not in SAFETY_LABELS:
            raise ValueError(f"Unknown expert: {label}")
        return self.model_root / f"{label.lower()}_expert"
