"""Build a standalone HTML report from public Markdown and aggregate figures.

Optional documentation dependency: Markdown==3.8.2. Print to PDF with a browser.
"""
import argparse
import base64
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    import markdown
    root = args.root.resolve()
    sections = ['RESULTS.md', 'CASE_STUDY.md', 'EVALUATION.md', 'SYSTEM_CARD.md']
    bodies = []
    for name in sections:
        text = (root / 'docs' / name).read_text()
        bodies.append('<section id="' + Path(name).stem.lower() + '">' + markdown.markdown(text, extensions=['tables', 'fenced_code']) + '</section>')
    chart = base64.b64encode((root / 'docs/assets/frozen-comparison.png').read_bytes()).decode()
    css = '''body{max-width:980px;margin:40px auto;padding:0 30px;color:#172033;font:16px/1.55 system-ui,sans-serif}
    h1,h2,h3{line-height:1.2;break-after:avoid}h1{font-size:28px}h2{font-size:21px;margin-top:30px}
    table{border-collapse:collapse;width:100%;font-size:13px;margin:18px 0}th,td{border-bottom:1px solid #cbd5e1;padding:9px;text-align:left}
    th{background:#eff6ff}code{font-size:11px;overflow-wrap:anywhere}a{color:#1d4ed8}img{width:100%}
    .cover{border-bottom:4px solid #2563eb;margin-bottom:30px}.status{padding:14px;background:#fff7ed}
    section{margin-top:36px}p,li{orphans:3;widows:3}@page{size:A4;margin:18mm}
    @media print{body{margin:0;padding:0;font-size:11px}h1{font-size:22px}h2{font-size:17px}table{font-size:10px}section{break-before:page}code{font-size:8px}}'''
    html = '<!doctype html><html lang="en"><meta charset="utf-8"><title>MoE Routing and Adaptation — Research Report</title>'
    html += '<style>' + css + '</style><body><div class="cover"><h1>MoE Routing and Adaptation</h1><p>Research and engineering report · October 22, 2026</p></div>'
    html += '<p class="status"><strong>Research release.</strong> Experiments complete and frozen. Stage 3 not promoted. Answer-quality and production acceptance false; independent human review pending.</p>'
    html += '<p>A study of request-level routing, native expert/gate adaptation, checkpoint recovery, numerical export checks and authenticated serving.</p>'
    html += '<img alt="Frozen automatic proxy comparison; not independently reviewed accuracy" src="data:image/png;base64,' + chart + '">'
    html += '<p>Aggregate evidence and original artifact identities accompany this report in the source release. A successful export or operating check is distinct from answer quality.</p>'
    html += ''.join(bodies) + '</body></html>'
    # The HTML is self-contained; rewrite relative documentation anchors to
    # in-document sections so it does not depend on a private filesystem.
    import re
    html = re.sub(r'href="(?:\.\./)?(?:docs/)?(RESULTS|CASE_STUDY|EVALUATION|SYSTEM_CARD)\.md"',
                  lambda match: 'href="#' + match.group(1).lower() + '"', html)
    (root / 'reports').mkdir(exist_ok=True)
    (root / 'reports/research-report.html').write_text(html)
    print('Wrote standalone reports/research-report.html')


if __name__ == '__main__':
    main()
