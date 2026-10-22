from fastapi.testclient import TestClient
import pytest
from moe_router.api import create_app
from moe_router.backend import BusyError
from moe_router.settings import Settings


class ServiceDouble:
    ready = False
    def load(self): self.ready = True
    def close(self): self.ready = False
    def generate(self, prompt, max_new_tokens=None, expert_override=None):
        label = expert_override or "MATH"
        return {"response": "answer", "expert": label, "confidence": None if expert_override else .9, "active_adapter": label, "timings_ms": {"total": 1}}


@pytest.fixture
def settings(tmp_path):
    return Settings(tmp_path, tmp_path / "router", tmp_path / "base", api_token="secret")


def test_restricted_chat_and_compatibility(settings):
    with TestClient(create_app(settings, ServiceDouble())) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 401
        assert client.post("/chat", json={"prompt": "hello"}).status_code == 401
        auth = {"Authorization": "Bearer secret"}
        assert client.get("/ready", headers=auth).status_code == 200
        response = client.post("/chat", headers=auth, json={"prompt": "2+2?"})
        assert response.status_code == 200
        assert {"response", "expert", "confidence", "request_id"} <= response.json().keys()
        assert client.post("/chat", headers=auth, json={"prompt": " "}).status_code == 422
        assert client.post("/chat", headers=auth, json={"prompt": "q", "max_new_tokens": 201}).status_code == 422
        assert client.get("/").status_code == 200
        selected = client.post("/chat", headers=auth, json={"prompt": "q", "expert": "SCIENCE"}).json()
        assert selected["expert"] == selected["active_adapter"] == "SCIENCE"
        assert selected["confidence"] is None
        assert client.post("/chat", headers=auth, json={"prompt": "q", "expert": "BASE"}).status_code == 422


def test_full_queue_is_retryable(settings):
    class Full(ServiceDouble):
        def generate(self, *args): raise BusyError("busy")
    with TestClient(create_app(settings, Full())) as client:
        response = client.post("/chat", headers={"Authorization": "Bearer secret"}, json={"prompt": "hi"})
        assert response.status_code == 429
        assert response.headers["retry-after"] == "5"


def test_loader_failure_prevents_readiness(settings):
    class Broken(ServiceDouble):
        def load(self): raise RuntimeError("missing weights")
    with pytest.raises(RuntimeError, match="missing weights"):
        with TestClient(create_app(settings, Broken())): pass
