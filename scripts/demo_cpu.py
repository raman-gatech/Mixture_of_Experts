"""Run an explicitly simulated, loopback-only UI/API demo without model weights."""
import argparse
from pathlib import Path

import uvicorn

from moe_router.api import create_app
from moe_router.settings import LABELS, Settings


class SimulatedBackend:
    ready = False
    labels = LABELS

    def load(self):
        self.ready = True

    def close(self):
        self.ready = False

    def generate(self, prompt, max_new_tokens=None, expert_override=None):
        label = expert_override or "GENERAL"
        return {
            "response": "[SIMULATED CPU DEMO — NO MODEL INFERENCE] "
                        "The API, authentication and UI are working. "
                        "Use the documented GPU profile for a real model response.",
            "expert": label,
            "confidence": None,
            "active_adapter": "SIMULATED_" + label,
            "timings_ms": {"total": 0},
            "simulated": True,
        }


def build_app():
    # Intentionally do not load a research .env or any real model credentials.
    root = Path(__file__).resolve().parents[1]
    settings = Settings(root / "artifacts", root / "artifacts/router",
                        root / "artifacts/base", api_token="demo-local-only")
    app = create_app(settings, SimulatedBackend())
    app.title = "MoE simulated CPU demo — no model inference"

    @app.get("/demo-status")
    def demo_status():
        return {"simulated": True, "model_loaded": False, "production_accepted": False}

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8510)
    args = parser.parse_args()
    print("SIMULATED CPU DEMO: no weights, no inference; local token: demo-local-only", flush=True)
    uvicorn.run(build_app(), host="127.0.0.1", port=args.port, access_log=False)
