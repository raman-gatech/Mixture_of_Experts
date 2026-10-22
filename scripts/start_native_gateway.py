"""Start the authenticated native gateway; never print credentials."""
import os
import uvicorn
from moe_router.settings import Settings

if __name__=='__main__':
    settings=Settings.from_env()
    if settings.active_profile!='native-internal-v1' or not settings.api_token or not settings.allow_safety_candidate:
        raise SystemExit('Native startup requires private token, native profile, explicit research-preview setting')
    uvicorn.run('moe_router.api:app',host='127.0.0.1',port=int(os.getenv('MOE_PORT','8512')),
                workers=1,access_log=False)
