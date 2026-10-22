"""Verify the authenticated research preview with real held-out prompts and Chromium.

Never print access tokens or model responses. Integration checks are not safety
quality acceptance; independently scored answer review remains separate.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import httpx
from playwright.sync_api import sync_playwright

from moe_router.data_io import read_jsonl
from moe_router.settings import Settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8512')
    parser.add_argument('--output', type=Path, default=Path('audit/safety_preview_live.json'))
    args = parser.parse_args()
    if args.url != 'http://127.0.0.1:8512':
        parser.error('This verification is limited to the private local preview')
    token = Settings.from_env().api_token
    if not token:
        raise ValueError('The local private access token is required')
    corpus = {row['prompt_sha256']:row for row in read_jsonl(
        Path('data/safety/prepared/v2/evaluation/original_router_test.jsonl'))}
    previous = json.loads(Path('outputs/safety-api-smoke-v3/report.json').read_text())
    headers = {'Authorization': f'Bearer {token}'}
    evidence = dict(checked_at_utc=datetime.now(timezone.utc).isoformat(), url=args.url,
                    mode='restricted research preview', production_accepted=False, probes=[])
    with httpx.Client(base_url=args.url, timeout=180) as client:
        evidence['unauthorized_status'] = client.get('/ready').status_code
        assert evidence['unauthorized_status'] == 401
        ready = client.get('/ready', headers=headers)
        ready.raise_for_status()
        evidence['ready'] = ready.json()
        assert evidence['ready']['research_preview'] is True
        assert set(evidence['ready']['experts']) == {'MATH','SCIENCE','GENERAL','SAFETY'}
        for probe in previous['probes']:
            row = corpus[probe['prompt_sha256']]
            payload = dict(prompt=row['text'], max_new_tokens=64)
            if probe['requested']:
                payload['expert'] = probe['requested']
            response = client.post('/chat', json=payload, headers=headers)
            response.raise_for_status()
            value = response.json()
            assert value['response'].strip() and value['active_adapter'] == value['expert']
            if probe['requested'] == 'SAFETY':
                assert value['expert'] == 'SAFETY'
            if probe['requested'] == 'MATH' and value['router_expert'] == 'SAFETY':
                assert value['expert'] == 'SAFETY' and value['routing_mode'] == 'safety_priority'
            evidence['probes'].append(dict(prompt_sha256=probe['prompt_sha256'],
                requested=probe['requested'], http_status=response.status_code,
                expert=value['expert'], router_expert=value['router_expert'],
                routing_mode=value['routing_mode'], active_adapter=value['active_adapter'],
                generated_tokens=value['generated_tokens'], nonempty_response=True))
    page_errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width':1280, 'height':900})
        page.on('pageerror', lambda error:page_errors.append(str(error)))
        page.goto(args.url)
        page.wait_for_function("document.querySelector('#status').textContent === 'Access token required'")
        page.locator('details').evaluate('(element) => element.open = true')
        page.locator('#token').fill(token)
        page.locator('#recheck').click()
        page.wait_for_function("document.querySelector('#status').textContent === 'Experts ready'")
        assert page.locator('#research-note').is_visible()
        assert not page.locator('option[value="SAFETY"]').is_disabled()
        page.locator('#expert').select_option('SCIENCE')
        page.locator('#prompt').fill('Why does the Moon appear to change shape?')
        page.locator('#send').click()
        page.locator('.message.assistant').wait_for(timeout=180000)
        badge = page.locator('.message.assistant .badge').last.inner_text()
        assert 'SCIENCE' in badge
        evidence['browser'] = dict(status=page.locator('#status').inner_text(),
            research_warning_visible=True, safety_option_enabled=True,
            manual_science_badge=badge, page_errors=page_errors)
        # Clear the token field before saving a screenshot; close access settings.
        page.locator('#token').fill('')
        page.locator('details').evaluate('(element) => element.open = false')
        screenshot = args.output.with_suffix('.png')
        screenshot.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(screenshot), full_page=True)
        page.set_viewport_size({'width':390, 'height':844})
        evidence['browser']['mobile_horizontal_overflow'] = page.evaluate(
            'document.documentElement.scrollWidth > window.innerWidth')
        assert not evidence['browser']['mobile_horizontal_overflow'] and not page_errors
        browser.close()
    args.output.write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps(dict(live_preview_verified=True, probes=len(evidence['probes']),
                         browser_verified=True, production_accepted=False)))


if __name__ == '__main__':
    main()
