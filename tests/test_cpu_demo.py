import importlib.util
from pathlib import Path

from fastapi.testclient import TestClient


def test_cpu_demo_is_authenticated_and_explicitly_simulated():
    path = Path(__file__).resolve().parents[1] / "scripts/demo_cpu.py"
    spec = importlib.util.spec_from_file_location("cpu_demo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with TestClient(module.build_app()) as client:
        assert client.get("/demo-status").json()["model_loaded"] is False
        assert client.post("/chat", json={"prompt": "hello"}).status_code == 401
        response = client.post("/chat", headers={"Authorization": "Bearer demo-local-only"},
                               json={"prompt": "hello", "expert": "MATH"})
        assert response.status_code == 200
        answer = response.json()
        assert "SIMULATED CPU DEMO" in answer["response"]
        assert answer["active_adapter"] == "SIMULATED_MATH"
        assert answer["confidence"] is None
