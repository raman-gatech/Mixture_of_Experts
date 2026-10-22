"""vLLM ASGI middleware: require a server-only credential on every endpoint."""
import hmac
import os
from pathlib import Path

class EngineAuth:
    def __init__(self, app):
        self.app = app
        path = Path(os.environ['MOE_ENGINE_TOKEN_FILE'])
        if path.stat().st_mode & 0o077:
            raise RuntimeError('Engine token file must be private (0600)')
        self.token = path.read_text().strip()
        if len(self.token) < 32:
            raise RuntimeError('Engine token is missing or too short')

    async def __call__(self, scope, receive, send):
        if scope['type'] not in ('http','websocket'):
            return await self.app(scope, receive, send)
        headers = dict(scope.get('headers', []))
        expected = ('Bearer '+self.token).encode()
        if not hmac.compare_digest(headers.get(b'authorization', b''), expected):
            if scope['type']=='websocket':
                await send({'type':'websocket.close','code':1008})
            else:
                body = b'{"detail":"Authentication required"}'
                await send({'type':'http.response.start','status':401,
                            'headers':[(b'content-type',b'application/json')]})
                await send({'type':'http.response.body','body':body})
            return
        await self.app(scope, receive, send)
