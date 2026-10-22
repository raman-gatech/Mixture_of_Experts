from contextlib import asynccontextmanager
import asyncio
import hmac
import json
import logging
import uuid
from typing import Literal

from fastapi import FastAPI, HTTPException, Header, Depends, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .backend import Backend, BusyError, InputError
from .settings import Settings, PROJECT_ROOT, LABELS

logger = logging.getLogger(__name__)


class Query(BaseModel):
    model: Literal["external-v1"] = "external-v1"
    prompt: str = Field(min_length=1, max_length=12000)
    max_new_tokens: int | None = Field(default=None, ge=1, le=512)
    expert: Literal["MATH", "SCIENCE", "GENERAL", "SAFETY"] | None = None


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=12000)


class StreamQuery(BaseModel):
    model: Literal["native-internal-v1"]
    history_model: Literal["native-internal-v1"] | None = None
    prompt: str = Field(min_length=1, max_length=12000)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=20)
    mode: Literal["AUTO", "MATH", "SCIENCE", "GENERAL", "SAFETY"] = "AUTO"
    max_new_tokens: int = Field(default=256, ge=1, le=256)
    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex, pattern=r"^[a-f0-9]{32}$")


def create_app(settings=None, backend=None):
    settings = settings or Settings.from_env()
    if backend is None:
        if settings.active_profile == "native-internal-v1":
            from .backends.internal_vllm import NativeBackend
            backend = NativeBackend(settings)
        else:
            backend = Backend(settings)

    @asynccontextmanager
    async def lifespan(app):
        backend.load()
        yield
        backend.close()

    app = FastAPI(title="External MoE Router", version="0.1.0", lifespan=lifespan)
    app.state.backend = backend

    def authorize(authorization: str | None = Header(default=None)):
        if settings.api_token and not hmac.compare_digest(authorization or "", f"Bearer {settings.api_token}"):
            raise HTTPException(status_code=401, detail="Authentication required")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready", dependencies=[Depends(authorize)])
    def ready():
        if not backend.ready or (settings.active_profile == "native-internal-v1" and not backend.check_ready()):
            raise HTTPException(status_code=503, detail="Models are not ready")
        labels = list(getattr(backend, "labels", LABELS))
        return {"status": "ready", "model": settings.active_profile, "experts": labels,
                "production_accepted": False,
                "research_preview": settings.allow_safety_candidate and ("SAFETY" in labels or settings.active_profile == "native-internal-v1")}

    @app.get("/profiles", dependencies=[Depends(authorize)])
    def profiles():
        available = backend.ready and (settings.active_profile != "native-internal-v1" or backend.check_ready())
        return {"active": settings.active_profile, "profiles": [
            {"id": name, "available": name == settings.active_profile and available}
            for name in ("external-v1", "native-internal-v1", "custom-internal-v2")],
            "production_accepted": False, "research_preview": settings.allow_safety_candidate}

    @app.get("/metrics", dependencies=[Depends(authorize)])
    def metrics():
        return {"model": settings.active_profile, "pending": getattr(backend, "pending", None),
                "requests": getattr(backend, "metrics", {})}

    @app.post("/v1/cancel/{request_id}", dependencies=[Depends(authorize)])
    async def cancel(request_id: str):
        if settings.active_profile != "native-internal-v1":
            raise HTTPException(status_code=409, detail="Native profile is unavailable")
        return {"cancelled": backend.cancel(request_id)}

    @app.post("/v1/chat/stream", dependencies=[Depends(authorize)])
    async def stream(query: StreamQuery, request: Request):
        if query.model != settings.active_profile or not backend.ready:
            raise HTTPException(status_code=409, detail="Requested model profile is unavailable")
        if query.history and query.history_model != query.model:
            raise HTTPException(status_code=422, detail="History must belong to the selected model")
        if query.request_id in backend.cancellations:
            raise HTTPException(status_code=409, detail="Request ID is already active")
        if not query.prompt.strip() or len(query.prompt) > settings.max_prompt_chars or query.max_new_tokens > settings.native_output_limit:
            raise HTTPException(status_code=422, detail="Invalid prompt or output reservation")
        try:
            _, ids, trimmed = backend.prepare(query.prompt.strip(), [m.model_dump() for m in query.history],
                                               query.mode, query.max_new_tokens)
            event = asyncio.Event()
            async def watch_queue_disconnect():
                while not event.is_set():
                    if await request.is_disconnected():
                        event.set()
                        return
                    await asyncio.sleep(.1)
            watcher = asyncio.create_task(watch_queue_disconnect())
            try:
                await backend.acquire(query.request_id, event)
            finally:
                watcher.cancel()
                await asyncio.gather(watcher, return_exceptions=True)
        except InputError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except BusyError as error:
            raise HTTPException(status_code=429, detail=str(error), headers={"Retry-After":"5"}) from error

        async def events():
            try:
                async for name, data in backend.stream(query.request_id, ids, query.max_new_tokens, query.mode, event):
                    if name == "start":
                        data.update(history_trimmed_messages=trimmed, input_tokens=len(ids))
                    yield f"event: {name}\ndata: {json.dumps(data)}\n\n"
            finally:
                event.set()
                backend.release(query.request_id)
        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control":"no-store", "X-Accel-Buffering":"no"})

    @app.post("/chat", dependencies=[Depends(authorize)])
    def chat(query: Query):
        if settings.active_profile != "external-v1":
            raise HTTPException(status_code=409, detail="External profile is unavailable; select native streaming chat")
        prompt = query.prompt.strip()
        if not prompt or len(prompt) > settings.max_prompt_chars:
            raise HTTPException(status_code=422, detail="Prompt is empty or too long")
        if query.max_new_tokens and query.max_new_tokens > settings.max_new_tokens:
            raise HTTPException(status_code=422, detail=f"Output limit is {settings.max_new_tokens} tokens")
        if query.expert and query.expert not in getattr(backend, "labels", LABELS):
            raise HTTPException(status_code=422, detail="Requested expert is not installed")
        request_id = uuid.uuid4().hex
        try:
            if query.expert:
                result = backend.generate(prompt, query.max_new_tokens, expert_override=query.expert)
            else:
                result = backend.generate(prompt, query.max_new_tokens)
            result["request_id"] = request_id
            logger.info("request=%s expert=%s timings=%s", request_id, result["expert"], result["timings_ms"])
            return result
        except BusyError as error:
            raise HTTPException(status_code=429, detail=str(error), headers={"Retry-After": "5"}) from error
        except InputError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except Exception:
            logger.exception("Inference failed request=%s", request_id)
            raise HTTPException(status_code=503, detail=f"Inference failed. Reference: {request_id}")

    web = PROJECT_ROOT / "web"
    app.mount("/static", StaticFiles(directory=web), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(web / "index.html")

    return app


app = create_app()
