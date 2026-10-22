"""Actual desktop/mobile native UI checks against the private allocated gateway."""
import argparse
import json
import os
from pathlib import Path
from playwright.sync_api import sync_playwright

p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
a.output.mkdir(parents=True,mode=0o700)
# Read only the app credential, never the engine credential, and never log it.
config=Path(os.environ.get('MOE_ENV_FILE','/path/to/ice-home/scratch/moe_router_app/.env'))
token=os.environ.get('MOE_API_TOKEN')
if not token:
    for line in config.read_text().splitlines():
        name,sep,value=line.partition('=')
        if name.strip()=='MOE_API_TOKEN' and sep:token=value.strip()
assert token
url='http://127.0.0.1:'+os.environ.get('MOE_PORT','8512')
report=dict(production_accepted=False,viewports=[])
with sync_playwright() as playwright:
    browser=playwright.chromium.launch(headless=True,args=['--disable-dev-shm-usage'])
    for width,height in [(1440,1000),(390,844)]:
        page=browser.new_page(viewport={'width':width,'height':height})
        errors=[]
        page.on('pageerror',lambda error:errors.append(type(error).__name__))
        page.goto(url)
        page.locator('summary').click()
        page.locator('#token').fill(token)
        page.locator('#recheck').click()
        page.wait_for_function("document.querySelector('#status').textContent==='Model ready'")
        assert page.locator('#model').input_value()=='native-internal-v1'
        assert page.locator('#research-note').is_visible()
        page.locator('#expert').select_option('MATH')
        page.locator('#prompt').fill('What is 23 times 17?')
        page.locator('#send').click()
        page.wait_for_function("!document.querySelector('#send').disabled",timeout=120000)
        answer=page.locator('.message.assistant').last.inner_text()
        assert '391' in answer
        assert 'confidence' not in answer.casefold()
        page.locator('#clear').click()
        assert page.locator('.message').count()==0
        page.locator('#expert').select_option('SAFETY')
        page.locator('#prompt').fill('Give a long detailed study plan for the semester.')
        page.locator('#send').click()
        page.locator('#cancel').wait_for(state='visible')
        page.locator('#cancel').click()
        page.wait_for_function("!document.querySelector('#send').disabled",timeout=30000)
        assert 'cancelled' in page.locator('#feedback').inner_text().casefold()
        # Screenshots exclude access settings (and its password field).
        page.locator('summary').click()
        overflow=page.evaluate('document.documentElement.scrollWidth > window.innerWidth')
        page.screenshot(path=str(a.output/f'ui-{width}.png'),full_page=True)
        report['viewports'].append(dict(width=width,height=height,overflow=overflow,javascript_errors=errors,
            answer=answer,passed=not overflow and not errors))
        page.close()
    browser.close()
report['passed']=all(v['passed'] for v in report['viewports'])
(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(browser_checks_passed=report['passed'],output=str(a.output))),flush=True)
if not report['passed']:raise SystemExit(1)
