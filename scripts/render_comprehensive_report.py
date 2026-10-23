"""Render the full Markdown report as standalone HTML (Markdown==3.8.2)."""
import base64
import hashlib
import html
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT/'docs/COMPREHENSIVE_PROJECT_REPORT_2026-10-22.md'
OUT = ROOT/'reports/comprehensive-20261022'


def main():
    import markdown
    text = DOC.read_text()
    body = markdown.markdown(text, extensions=['tables', 'fenced_code', 'toc'],
                             extension_configs={'toc': {'toc_depth': '2', 'title': 'Contents'}})
    def image(match):
        source=match.group(1)
        path=(DOC.parent/source).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError('Missing or non-project figure: '+source)
        mime='image/svg+xml' if path.suffix=='.svg' else 'image/png'
        return 'src="data:'+mime+';base64,'+base64.b64encode(path.read_bytes()).decode()+'"'
    body=re.sub(r'src="([^"]+)"',image,body)
    def figure(match):
        tag=match.group(1)
        alt=re.search(r'alt="([^"]*)"',tag)
        return '<figure>'+tag+'<figcaption>'+html.escape(html.unescape(alt.group(1)) if alt else '')+'</figcaption></figure>'
    body=re.sub(r'<p>\s*(<img [^>]+>)\s*</p>',figure,body)
    css='''
    :root{color-scheme:light}*{box-sizing:border-box}body{max-width:1120px;margin:44px auto;padding:0 34px;color:#182339;font:15px/1.6 "DejaVu Sans",Arial,sans-serif}
    h1{font-size:34px;line-height:1.18;border-bottom:5px solid #2563eb;padding-bottom:20px;color:#10264a}h2{font-size:25px;line-height:1.25;margin-top:44px;color:#17345d;border-bottom:1px solid #cbd5e1;padding-bottom:9px}h3{font-size:19px;margin-top:28px;line-height:1.35}
    h1,h2,h3{break-after:avoid}p,li{orphans:3;widows:3}a{color:#1d4ed8;text-decoration:none}code{font-size:.83em;overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f5f9;padding:14px;border-left:3px solid #94a3b8}
    table{border-collapse:collapse;width:100%;font-size:12px;line-height:1.5;margin:20px 0;table-layout:auto}thead{display:table-header-group}th,td{vertical-align:top;text-align:left;border:1px solid #dbe2ea;padding:8px;overflow-wrap:anywhere}th{background:#edf3fb}tr{break-inside:avoid}
    figure{margin:26px auto;break-inside:avoid;text-align:center}figure img{max-width:100%;max-height:850px;object-fit:contain}figcaption{font-size:12px;color:#475569;margin-top:9px;text-align:center}
    .toc{padding:18px 24px;background:#f8fafc;border:1px solid #dbe2ea;margin:30px 0}.toc ul{list-style:none;padding-left:18px}.toctitle{font-size:23px;font-weight:bold}.toc li{margin:5px 0}
    @page{size:A4;margin:19mm 16mm 18mm 16mm}
    @media print{body{max-width:none;margin:0;padding:0;font-size:9.5pt;line-height:1.48}h1{font-size:25pt}h2{font-size:18pt;break-before:page;margin-top:0}h3{font-size:12.5pt}table{font-size:7.8pt;line-height:1.38}th,td{padding:5px}pre{font-size:8pt}figure img{max-height:185mm}figcaption{font-size:8pt}.toc{break-after:page;background:white;border:none;padding:0}.toc li{font-size:9pt}.toc a{color:#182339}a{color:inherit}}
    '''
    payload='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MoE Routing and Adaptation — Comprehensive Project Report</title><style>'+css+'</style></head><body>'+body+'</body></html>'
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'COMPREHENSIVE_PROJECT_REPORT.html';path.write_text(payload)
    manifest=json.loads((OUT/'report-manifest.json').read_text())
    manifest['html_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    manifest['html_bytes']=path.stat().st_size
    manifest['self_contained_html']=True
    (OUT/'report-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'html':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,'embedded_images':body.count('src="data:')}))


if __name__=='__main__':
    main()
