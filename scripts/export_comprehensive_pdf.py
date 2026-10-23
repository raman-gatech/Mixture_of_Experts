"""Print the standalone report using installed Playwright Chromium, without inference."""
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    from playwright.sync_api import sync_playwright
    root=Path(__file__).resolve().parents[1]/'reports/comprehensive-20261022'
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--disable-dev-shm-usage'])
        page=browser.new_page(viewport={'width':1280,'height':1000})
        errors=[]
        page.on('pageerror',lambda error:errors.append(type(error).__name__))
        page.goto((root/'COMPREHENSIVE_PROJECT_REPORT.html').as_uri(),wait_until='load')
        count=page.locator('img').count()
        assert count==18
        assert page.evaluate('Array.from(document.images).every(x=>x.complete && x.naturalWidth>0)')
        assert not errors
        page.pdf(path=str(root/'COMPREHENSIVE_PROJECT_REPORT.pdf'),print_background=True,prefer_css_page_size=True,
            display_header_footer=True,header_template='<div></div>',
            footer_template='<div style="font-size:8px;text-align:center;width:100%;color:#64748b">MoE Routing and Adaptation · Research report · <span class="pageNumber"></span> / <span class="totalPages"></span></div>')
        page.screenshot(path=str(root/'report-cover-preview.png'))
        browser.close()
    manifest=json.loads((root/'report-manifest.json').read_text())
    pdf=root/'COMPREHENSIVE_PROJECT_REPORT.pdf'
    manifest.update(pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),pdf_bytes=pdf.stat().st_size,
        html_images_verified=count,html_javascript_errors=[])
    info=subprocess.check_output(['pdfinfo',str(pdf)],text=True)
    manifest['pdf_pages']=int(next(line.split(':')[1] for line in info.splitlines() if line.startswith('Pages:')))
    (root/'report-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'pdf_pages':manifest['pdf_pages'],'pdf_bytes':manifest['pdf_bytes'],'images_verified':count}))


if __name__=='__main__':
    main()
