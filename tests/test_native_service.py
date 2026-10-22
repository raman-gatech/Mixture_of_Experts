import asyncio
from dataclasses import replace
import json
import pytest
import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from moe_router.api import create_app
from moe_router.backend import InputError, BusyError
from moe_router.backends.internal_vllm import NativeBackend
from moe_router.conversation import prepare_messages
from moe_router.engine_auth import EngineAuth
from moe_router.settings import Settings

class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return list(range(sum(len(m['content'].split())+2 for m in messages)))

@pytest.fixture
def native_settings(tmp_path):
    key=tmp_path/'engine.token'
    key.write_text('e'*48)
    key.chmod(0o600)
    return Settings(tmp_path,tmp_path/'router',tmp_path/'base',api_token='app-secret',
        allow_safety_candidate=True,active_profile='native-internal-v1',native_engine_token_file=str(key))

def transport(request):
    assert request.headers['authorization']=='Bearer '+'e'*48
    if request.url.path=='/v1/models':
        return httpx.Response(200,json={'data':[{'id':'native-internal-v1'}]})
    data=json.loads(request.content)
    assert data['model']=='native-internal-v1' and isinstance(data['prompt'],list)
    chunks=[{'choices':[{'text':'An answer','finish_reason':None}]},
            {'choices':[{'text':'','finish_reason':'stop'}]},
            {'choices':[],'usage':{'prompt_tokens':12,'completion_tokens':2}}]
    return httpx.Response(200,text=''.join('data: '+json.dumps(c)+'\n\n' for c in chunks)+'data: [DONE]\n\n')

def test_native_auth_contract_history_and_unavailable_profiles(native_settings):
    backend=NativeBackend(native_settings,Tokenizer(),httpx.MockTransport(transport))
    with TestClient(create_app(native_settings,backend)) as client:
        auth={'Authorization':'Bearer app-secret'}
        assert client.get('/profiles').status_code==401
        assert client.get('/metrics').status_code==401
        assert client.post('/v1/cancel/'+'a'*32).status_code==401
        assert client.post('/v1/chat/stream',json={'model':'native-internal-v1','prompt':'hi'}).status_code==401
        assert client.get('/ready',headers=auth).json()['research_preview']
        assert client.post('/chat',headers=auth,json={'prompt':'hi'}).status_code==409
        query=dict(model='native-internal-v1',prompt='hi',mode='MATH')
        response=client.post('/v1/chat/stream',headers=auth,json=query)
        assert response.status_code==200
        assert 'event: delta' in response.text and 'event: done' in response.text
        assert 'native_internal' in response.text and 'production_accepted' in response.text
        assert backend.pending==0
        query['history']=[{'role':'user','content':'q'},{'role':'assistant','content':'a'}]
        assert client.post('/v1/chat/stream',headers=auth,json=query).status_code==422
        query['history_model']='native-internal-v1'
        assert client.post('/v1/chat/stream',headers=auth,json=query).status_code==200
        query['history'][0]['role']='system'
        assert client.post('/v1/chat/stream',headers=auth,json=query).status_code==422
        assert not client.get('/profiles',headers=auth).json()['profiles'][0]['available']

def test_context_reserves_output_and_trims_complete_turns():
    tokenizer=Tokenizer()
    history=[{'role':'user','content':'long '*100},{'role':'assistant','content':'old '*100}]
    _,ids,trimmed=prepare_messages(tokenizer,'hi',history,'MATH',256,64)
    assert trimmed==2 and len(ids)+64<=256
    with pytest.raises(InputError):
        prepare_messages(tokenizer,'huge '*300,[],'MATH',256,64)
    with pytest.raises(InputError):
        prepare_messages(tokenizer,'hi',[{'role':'assistant','content':'bad'}],'MATH',256,64)

def test_native_requires_research_flag_and_private_distinct_engine_key(native_settings):
    with pytest.raises(RuntimeError):
        NativeBackend(replace(native_settings,allow_safety_candidate=False),Tokenizer()).load()
    with pytest.raises(RuntimeError):
        NativeBackend(replace(native_settings,api_token=''),Tokenizer()).load()
    with pytest.raises(ValueError):
        NativeBackend(replace(native_settings,native_engine_url='http://0.0.0.0:8001'),Tokenizer()).load()

def test_all_engine_paths_require_server_credential(native_settings,monkeypatch):
    monkeypatch.setenv('MOE_ENGINE_TOKEN_FILE',native_settings.native_engine_token_file)
    app=FastAPI()
    @app.get('/{path:path}')
    def route(path): return {'ok':True}
    app.add_middleware(EngineAuth)
    with TestClient(app) as client:
        for path in ['/health','/metrics','/v1/models','/tokenize','/generate','/docs','/openapi.json']:
            assert client.get(path).status_code==401
            assert client.get(path,headers={'Authorization':'Bearer app-secret'}).status_code==401
            assert client.get(path,headers={'Authorization':'Bearer '+'e'*48}).status_code==200

def test_admission_timeout_cancel_and_release(native_settings):
    async def run():
        backend=NativeBackend(replace(native_settings,queue_capacity=1,queue_timeout=.02),Tokenizer())
        first,second=asyncio.Event(),asyncio.Event()
        await backend.acquire('first',first)
        with pytest.raises(BusyError):
            await backend.acquire('second',second)
        assert backend.pending==1
        task=asyncio.create_task(backend.acquire('cancel',second))
        await asyncio.sleep(0)
        backend.cancel('cancel')
        with pytest.raises(asyncio.CancelledError): await task
        assert backend.pending==1
        backend.release('first')
        assert backend.pending==0 and backend.active._value==1
    asyncio.run(run())

def test_buffered_support_does_not_emit_partial_text(native_settings):
    async def run():
        backend=NativeBackend(native_settings,Tokenizer(),httpx.MockTransport(transport))
        backend.load()
        event=asyncio.Event()
        await backend.acquire('a',event)
        events=[e async for e in backend.stream('a',[1,2],32,'SAFETY',event)]
        assert [name for name,data in events]==['start','delta','done']
        assert events[0][1]['output_policy']=='buffered_unreviewed'
        assert backend.pending==0
    asyncio.run(run())
