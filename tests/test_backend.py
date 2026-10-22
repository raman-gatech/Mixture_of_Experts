from concurrent.futures import ThreadPoolExecutor
import json
import threading
import time
import pytest
from moe_router.backend import Backend, BusyError, validate_artifacts
from moe_router.settings import Settings


def test_adapter_selection_and_generation_are_serialized(tmp_path):
    class ConcurrentBackend(Backend):
        def _generate(self, prompt, limit, start):
            self.active = prompt
            time.sleep(.03)
            assert self.active == prompt, "Another request changed the active adapter"
            return prompt
    backend = ConcurrentBackend(Settings(tmp_path, tmp_path, tmp_path))
    backend.ready = True
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(backend.generate, ["MATH", "SCIENCE", "GENERAL"]))
    assert results == ["MATH", "SCIENCE", "GENERAL"]


def test_queue_timeout_releases_capacity(tmp_path):
    backend = Backend(Settings(tmp_path, tmp_path, tmp_path, queue_capacity=1, queue_timeout=.01))
    backend.ready = True
    backend.lock.acquire()
    try:
        with pytest.raises(BusyError): backend.generate("q")
        assert backend.slots.acquire(blocking=False)
        backend.slots.release()
    finally: backend.lock.release()


def test_checkpoint_label_order_is_explicit(tmp_path):
    settings = Settings(tmp_path, tmp_path / "router", tmp_path / "base")
    for directory in [settings.router_dir, settings.base_dir]:
        directory.mkdir()
        (directory / "model.safetensors").touch()
        (directory / "config.json").write_text('{}')
    with pytest.raises(ValueError, match="id2label"): validate_artifacts(settings)
    mapping = {"0": "GENERAL", "1": "MATH", "2": "SCIENCE"}
    (settings.router_dir / "config.json").write_text(json.dumps({"id2label": mapping}))
    for name in ("MATH", "SCIENCE", "GENERAL"):
        directory = settings.expert_dir(name); directory.mkdir()
        (directory / "adapter_model.safetensors").touch()
        (directory / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "same/base"}))
    assert validate_artifacts(settings) == ("GENERAL", "MATH", "SCIENCE")
