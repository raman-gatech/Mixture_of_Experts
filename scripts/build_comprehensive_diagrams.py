"""Render the full project pipeline using Graphviz; no model execution."""
from pathlib import Path
import subprocess
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/comprehensive-20261022/figures'


def graph(name, title, nodes, edges, direction='TB'):
    def quote(value):
        return json.dumps(value)
    lines = ['digraph G {', f'rankdir={direction};', 'graph [bgcolor="white",pad="0.2",nodesep="0.25",ranksep="0.35",fontname="DejaVu Sans",fontsize=14];',
             'node [shape=box,style="rounded,filled",fillcolor="#eff6ff",color="#64748b",fontname="DejaVu Sans",fontsize=10,margin="0.14,0.10"];',
             'edge [color="#475569",fontname="DejaVu Sans",fontsize=9];', f'label={quote(title)};labelloc=t;']
    for key, value in nodes.items():
        label, color = value if isinstance(value, tuple) else (value, '#eff6ff')
        lines.append(f'{key} [label={quote(label)},fillcolor={quote(color)}];')
    for edge in edges:
        a, b, *label = edge
        lines.append(f'{a} -> {b}' + (f' [label={quote(label[0])}]' if label else '') + ';')
    lines.append('}')
    OUT.mkdir(parents=True, exist_ok=True)
    source = OUT / (name + '.dot')
    source.write_text('\n'.join(lines))
    for extension in ['svg', 'png']:
        # Graphviz 2.44 applies non-72 SVG DPI to the graph transform without
        # expanding its viewBox, which clips nodes even when the image fits a page.
        dpi = '72' if extension == 'svg' else '140'
        subprocess.run(['dot', '-T' + extension, '-Gdpi=' + dpi, str(source), '-o', str(OUT / (name + '.' + extension))], check=True)


def main():
    graph('01-lifecycle', 'End-to-end project lifecycle', {
        'a':'Recover source, weights and provenance','b':'Stage 1: request router + named LoRA experts',
        'c':'Source-pinned data, grouped splits and holdout isolation','d':'Stage 2: native attention adaptation + calibration',
        'e':'Verified export, archive, serving and evaluation','f':'Stage 3: expert-only / joint-gate pilots',
        'g':'Fixed export calibration + frozen three-arm comparison',
        'h':('No held-out proxy gain: retain candidate for research','#fff7ed'),
        'i':'Freeze experimentation; report, package and restricted demo'},
        [('a','b'),('b','c'),('c','d'),('d','e'),('e','f'),('f','g'),('g','h'),('h','i')])
    graph('02-system', 'System design and trust boundaries', {
        'u':'Browser / API client','t':'SSH tunnel to allocated compute node','a':'FastAPI gateway\napplication bearer authentication',
        'e':'External backend\nbounded queue + model lock','r':'DeBERTa classifier\nlabel metadata + safety threshold',
        'l':'Shared Llama base\nselected named LoRA adapter','n':'Native backend\nadmission, context, history, SSE',
        'v':'Private loopback vLLM engine\ndistinct server-only credential','m':'GPT-OSS native token router\nand sparse FFN experts',
        'p':'Persistent scratch\narchives, hashes and reports'},
        [('u','t'),('t','a'),('a','e','external profile'),('e','r'),('r','l'),('a','n','native profile'),('n','v'),('v','m'),('p','l','verified weights'),('p','m','restore to node-local disk')])
    graph('03-external', 'External profile: one request and one active adapter', {
        'a':'Authenticate and validate request','b':'Reserve queue slot; acquire exclusive model lock','c':'Tokenize request and router windows',
        'd':'Compute per-window probabilities\nmaximum safety probability across windows',
        'e':('Safety threshold met?','#fef3c7'),'f':'SAFETY adapter priority','g':'Manual domain or automatic ordinary domain',
        'h':'Set and verify named active adapter','i':'Apply common policy + model chat template',
        'j':'Deterministic generation; decode generated tokens only','k':'Return route/adapter/timing; release lock and slot'},
        [('a','b'),('b','c'),('c','d'),('d','e'),('e','f','yes'),('e','g','no'),('f','h'),('g','h'),('h','i'),('i','j'),('j','k')])
    graph('04-native-request', 'Native profile request flow', {
        'a':'Authenticate; require active native profile','b':'Validate prompt, output reservation and history profile',
        'c':'Reject system-role injection; trim oldest complete turns','d':'Encode final-channel prompt within context budget',
        'e':('Admission available?','#fef3c7'),'q':('429 + Retry-After / queue timeout','#fee2e2'),
        'f':'Acquire semaphore; register request cancellation event','g':'Stream from authenticated private engine',
        'h':'Guard fragmented private-channel delimiters','i':'Buffer AUTO/SAFETY; stream ordinary task modes',
        'j':'done / cancelled / error; close upstream and free admission'},
        [('a','b'),('b','c'),('c','d'),('d','e'),('e','q','no'),('e','f','yes'),('f','g'),('g','h'),('h','i'),('i','j')])
    graph('05-data', 'Data preparation and leakage controls', {
        'a':'Pinned published sources + revisions + file hashes','b':'Verify raw and inherited safety manifests',
        'c':'Canonical prompt/context keys; retain source IDs and annotations','d':'Protect earlier heldouts; quarantine overlap/conflicting targets',
        'e':'Group shared passages, conversations and duplicate instructions','f':'Preserve official splits; deterministic split for eligible groups',
        'g':'Cap training tasks; retain validation/test independently','h':'Verify zero exact cross-split prompt/context overlap',
        'i':'Whole-example token filtering + response-only labels','j':'Manifest: hashes, counts, exclusions, provenance and limitations'},
        [('a','b'),('b','c'),('c','d'),('d','e'),('e','f'),('f','g'),('g','h'),('h','i'),('i','j')])
    graph('06-stage2', 'Stage 2: attention adaptation and selection', {
        'a':'Qualify pinned GPT-OSS base and final-channel formatting','b':'Freeze FFN experts/gates; attach attention LoRA',
        'c':'Response-only SFT; checkpoint optimizer/scheduler/RNG/cursor','d':'Resume with same data/config/source identity',
        'e':'Full run: 1,489 steps; best checkpoint 300','f':'Correct symmetric final-answer extraction; preserve legacy scores',
        'g':'Declared scales 0.5 / 0.25 / 0.125\n128 fixed validation answers',
        'h':'Select nonzero 0.125 update passing per-domain tolerance','i':'Numerical export, immutable archive and operational qualification'},
        [('a','b'),('b','c'),('c','d','allocation ends'),('d','c'),('c','e','completed'),('e','f'),('f','g'),('g','h'),('h','i')])
    graph('07-stage3', 'Stage 3: actual native FFN and gate updates', {
        'a':'Verified Stage 2 parent','b':'Freeze base and native biases; add FP32 low-rank expert factors',
        'c':'Native router scores and top-4 selection among 32 experts',
        'd':'Expert-only: train expert factors; gates frozen','e':'Joint: same expert factors + existing native gate parameters',
        'f':'200-step matched pilots; fixed 1,000/80 row pool',
        'g':'Best/latest state + native usage telemetry','h':'Fixed export calibration; select using development only'},
        [('a','b'),('b','c'),('c','d'),('c','e'),('d','f'),('e','f'),('f','g'),('g','h')])
    graph('08-export', 'Export qualification and persistence', {
        'a':'Eligible identity-bound trained checkpoint','b':'Merge local updates with precision-aware checks',
        'c':'Original merge-KL and cached/full numerical probes',
        'd':('All original numerical gates pass?','#fef3c7'),'x':('Retain failed report; do not mint eligible release','#fee2e2'),
        'e':'Check frozen tensors and runtime buffers','f':'Serialize standard model; verify all saved tensors',
        'g':'Fresh reload: logits, cache and generation checks','h':'Compress to persistent scratch archive + SHA256',
        'i':'Independent restore and file/weight verification','j':'Eligible research export; quality acceptance remains separate'},
        [('a','b'),('b','c'),('c','d'),('d','x','no'),('d','e','yes'),('e','f'),('f','g'),('g','h'),('h','i'),('i','j')])
    graph('09-evaluation', 'Frozen comparison: selection precedes held-out collection', {
        'a':'Development-only export selection frozen','b':'Reserve 500 prompts; exact project overlap checks',
        'c':'Parent baseline: 256 output budget','d':'Parent precision: 512 output budget','e':'Stage 3 precision: 512 output budget',
        'f':'Verify 1,500 unique nonempty outputs + identities',
        'g':'All 500 paired parent/candidate token hashes equal',
        'h':'Numeric/phrase proxies + truncation + attributed error analysis',
        'i':('Candidate has no proxy advantage; no promotion','#fff7ed'),'j':('Independent human/intent review pending','#fef3c7')},
        [('a','b'),('b','c'),('b','d'),('b','e'),('c','f'),('d','f'),('e','f'),('f','g'),('g','h'),('h','i'),('h','j')])
    graph('10-recovery', 'Serving lifecycle and recovery', {
        'a':'Finite normal-QoS GPU allocation','b':'Verify archive; inspect node-local storage; restore',
        'c':'Start engine and tokenizer/gateway in parallel','d':'Authenticate exact-release readiness; record whole cold-start clock',
        'e':'API/operations/browser checks','f':'If full qualification: fixed-workload soak',
        'g':'Stop native children; restart same native profile','h':'Stop native; load external rollback and verify real answer',
        'i':'Restore native; verify again','j':'Intentional stop/allocation signal; stop owned children and mark stopped'},
        [('a','b'),('b','c'),('c','d'),('d','e'),('e','f','full qualification'),('f','g'),('g','h'),('h','i'),('i','j'),('e','j','restricted demo after serving window')])
    graph('11-storage', 'Storage roles and preservation', {
        'h':'NFS home\nsmall settings, credentials, OnDemand databases',
        's':'Persistent scratch\nenvironments, data, checkpoints, archives, logs, caches',
        'n':'Allocated node-local temporary disk\nrestored model and hot working data',
        'p':'Public source export\ncode, aggregate reports and selected evidence only',
        'b':'Durable backups / existing external artifact storage\nsubject to access and redistribution terms'},
        [('s','n','stage immutable hot inputs'),('n','s','persist required outputs before shutdown'),('s','p','curate, sanitize, scan; exclude weights/data/secrets'),('s','b','verified backups'),('h','s','bulk relocation with hashes; retain needed aliases')])
    graph('12-gates', 'Separate acceptance gates', {
        'a':'Training and artifact integrity','b':'Numerical export and reload qualification',
        'c':'Operating qualification: startup, API, load, recovery',
        'd':'Answer quality and benign-sensitive behavior','e':'Independent reviewer/intent assessment',
        'f':('Research source release with explicit limitations','#dcfce7'),
        'g':('Production acceptance blocked','#fee2e2')},
        [('a','b'),('b','c'),('c','d'),('d','e'),('b','f','engineering evidence'),('d','g','failed quality gates'),('e','g','review pending')])
    print('Rendered 12 DOT/SVG/PNG project diagrams')


if __name__ == '__main__':
    main()
