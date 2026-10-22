import os
import uvicorn
from .settings import load_local_env

load_local_env()

host = os.getenv("MOE_HOST", "127.0.0.1")
if host not in {"127.0.0.1", "localhost", "::1"} and not os.getenv("MOE_API_TOKEN"):
    raise SystemExit("Set MOE_API_TOKEN before binding to a non-loopback interface")
uvicorn.run("moe_router.api:app", host=host, port=int(os.getenv("MOE_PORT", "8510")), workers=1)
