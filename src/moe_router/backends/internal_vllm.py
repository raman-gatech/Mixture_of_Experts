"""Private native engine client with admission, disconnect cancellation, and SSE."""
import asyncio
import json
from pathlib import Path
import time
from urllib.parse import urlparse
import httpx
from ..backend import BusyError, InputError
from ..conversation import prepare_messages
from ..gptoss_format import completion_options, FinalChannelStream

class NativeBackend:
    labels = ()
    def __init__(self, settings, tokenizer=None, transport=None):
        self.settings = settings
        self.tokenizer = tokenizer
        self.transport = transport
        self.ready = False
        self.final_channel = False
        self.pending = 0
        self.active = asyncio.Semaphore(settings.native_concurrency)
        self.cancellations = {}
        self.metrics = dict(admitted=0, completed=0, cancelled=0, failed=0, generated_tokens=0)

    def load(self):
        s = self.settings
        if not s.api_token or not s.allow_safety_candidate:
            raise RuntimeError('Native research service requires authentication and explicit research-preview mode')
        parsed = urlparse(s.native_engine_url)
        if parsed.scheme != 'http' or parsed.hostname not in ('localhost','127.0.0.1','::1') or parsed.username or parsed.password:
            raise ValueError('Native engine must use private loopback HTTP')
        path = Path(s.native_engine_token_file)
        if path.stat().st_mode & 0o077:
            raise ValueError('Engine credential must be private')
        self.engine_token = path.read_text().strip()
        if len(self.engine_token)<32 or self.engine_token == s.api_token:
            raise ValueError('Engine needs a distinct server-only credential')
        if self.tokenizer is None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(s.native_model_dir, local_files_only=True)
        config_path = Path(s.native_model_dir)/'config.json'
        if config_path.exists():
            self.final_channel = json.loads(config_path.read_text()).get('model_type') == 'gpt_oss'
        if self.final_channel:
            completion_options(self.tokenizer)  # Validate channel token IDs before readiness.
        self.ready = True

    def check_ready(self):
        if not self.ready:
            return False
        try:
            with httpx.Client(base_url=self.settings.native_engine_url, timeout=3, transport=self.transport,
                              headers={'Authorization':'Bearer '+self.engine_token}) as client:
                r = client.get('/v1/models')
                return r.status_code==200 and self.settings.native_release in [m['id'] for m in r.json()['data']]
        except Exception:
            return False

    def close(self):
        self.ready = False
        for event in self.cancellations.values():
            event.set()

    def prepare(self, prompt, history, mode, output):
        return prepare_messages(self.tokenizer, prompt, history, mode,
                                self.settings.native_context_limit, output, final_channel=self.final_channel)

    async def acquire(self, request_id, event):
        if self.pending >= self.settings.native_concurrency+self.settings.queue_capacity:
            raise BusyError('The request queue is full; try again shortly')
        self.pending += 1
        self.cancellations[request_id] = event
        acquired = False
        waiter = asyncio.create_task(self.active.acquire())
        cancelled = asyncio.create_task(event.wait())
        try:
            done, _ = await asyncio.wait([waiter,cancelled], timeout=self.settings.queue_timeout,
                                         return_when=asyncio.FIRST_COMPLETED)
            if waiter in done:
                acquired = True
            if event.is_set():
                raise asyncio.CancelledError
            if not acquired:
                raise BusyError('The request timed out waiting for the model')
            self.metrics['admitted'] += 1
        except BaseException:
            waiter.cancel()
            await asyncio.gather(waiter,return_exceptions=True)
            if acquired or (not waiter.cancelled() and waiter.done() and waiter.exception() is None):
                self.active.release()
            self.pending -= 1
            self.cancellations.pop(request_id,None)
            raise
        finally:
            cancelled.cancel()
            await asyncio.gather(cancelled,return_exceptions=True)

    def release(self, request_id):
        if self.cancellations.pop(request_id,None) is not None:
            self.pending -= 1
            self.active.release()

    def cancel(self, request_id):
        event = self.cancellations.get(request_id)
        if event is not None:
            event.set()
            return True
        return False

    async def stream(self, request_id, ids, output, mode, event):
        start = time.perf_counter()
        # AUTO/support are buffered. Ordinary modes may stream in this explicitly
        # restricted preview; mode selection is not a validated safety classifier.
        buffered = mode in ('AUTO','SAFETY')
        fragments, first_ms, usage, finish = [], None, {}, None
        channel_guard = FinalChannelStream() if self.final_channel else None
        try:
            yield 'start', dict(request_id=request_id, model='native-internal-v1', release=self.settings.native_release,
                                backend_type='native_internal', mode=mode,
                                output_policy='buffered_unreviewed' if buffered else 'streamed_unreviewed',
                                production_accepted=False)
            async with asyncio.timeout(self.settings.generation_timeout) if hasattr(asyncio,'timeout') else _Timeout(self.settings.generation_timeout):
                async with httpx.AsyncClient(base_url=self.settings.native_engine_url, transport=self.transport,
                    timeout=httpx.Timeout(self.settings.generation_timeout),
                    headers={'Authorization':'Bearer '+self.engine_token,'X-Request-Id':request_id}) as client:
                    async with client.stream('POST','/v1/completions',json=dict(model=self.settings.native_release,
                        prompt=ids, max_tokens=output, temperature=0, stream=True,
                        stream_options={'include_usage':True},
                        **(completion_options(self.tokenizer) if self.final_channel else {}))) as response:
                        if response.status_code!=200:
                            raise RuntimeError('Private inference engine unavailable')
                        iterator = response.aiter_lines().__aiter__()
                        while not event.is_set():
                            line_task = asyncio.create_task(iterator.__anext__())
                            cancel_task = asyncio.create_task(event.wait())
                            try:
                                done,_ = await asyncio.wait([line_task,cancel_task],return_when=asyncio.FIRST_COMPLETED)
                                if event.is_set():
                                    break
                                try:
                                    line = line_task.result()
                                except StopAsyncIteration:
                                    break
                            finally:
                                for task in (line_task,cancel_task):
                                    if not task.done(): task.cancel()
                                await asyncio.gather(line_task,cancel_task,return_exceptions=True)
                            if not line.startswith('data: '): continue
                            data = line[6:]
                            if data=='[DONE]': break
                            obj = json.loads(data)
                            if obj.get('usage'): usage=obj['usage']
                            for choice in obj.get('choices',[]):
                                fragment=choice.get('text','')
                                finish=choice.get('finish_reason') or finish
                                if channel_guard and fragment:
                                    fragment=channel_guard.feed(fragment)
                                if fragment:
                                    if first_ms is None: first_ms=(time.perf_counter()-start)*1000
                                    fragments.append(fragment)
                                    if not buffered: yield 'delta', {'text':fragment}
            if event.is_set():
                self.metrics['cancelled'] += 1
                yield 'cancelled', {'request_id':request_id}
                return
            if channel_guard:
                tail=channel_guard.finish()
                if tail:
                    fragments.append(tail)
                    if not buffered: yield 'delta', {'text':tail}
            answer=''.join(fragments)
            if not answer.strip() or finish is None:
                raise RuntimeError('Incomplete engine response')
            if buffered: yield 'delta', {'text':answer}
            self.metrics['completed'] += 1
            self.metrics['generated_tokens'] += usage.get('completion_tokens',0)
            yield 'done', dict(request_id=request_id, finish_reason=finish, usage=usage,
                               timings_ms={'first_engine_token':first_ms,'total':(time.perf_counter()-start)*1000})
        except asyncio.CancelledError:
            self.metrics['cancelled'] += 1
            raise
        except Exception:
            self.metrics['failed'] += 1
            yield 'error', {'request_id':request_id,'message':'Inference failed or timed out; please retry.'}
        finally:
            # Closing the upstream HTTP stream invokes vLLM disconnect abort.
            self.release(request_id)

class _Timeout:
    """Python 3.10 total deadline, separate from the HTTP read timeout."""
    def __init__(self, seconds): self.seconds=seconds
    async def __aenter__(self):
        self.task=asyncio.current_task()
        self.expired=False
        def expire():
            self.expired=True
            self.task.cancel()
        self.handle=asyncio.get_running_loop().call_later(self.seconds,expire)
    async def __aexit__(self, typ, value, tb):
        self.handle.cancel()
        if typ is asyncio.CancelledError and self.expired:
            raise TimeoutError('Generation deadline reached')
