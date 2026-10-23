"""Add source/evidence appendices and derive figures from frozen saved artifacts."""
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/comprehensive-20261022'
DOC = ROOT / 'docs/COMPREHENSIVE_PROJECT_REPORT_2026-10-22.md'
EXTERNAL = ROOT.parent / 'moe_router_app'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    figures = OUT / 'figures'
    figures.mkdir(exist_ok=True)
    sources = [
        ('E01', 'External preservation inventory', ROOT/'audit/releases/external-v1-ice.json'),
        ('E02', 'Latest external/source preservation verification', ROOT/'audit/completion-batch-20261021/preserved-external-artifacts-verified-20261022.json'),
        ('E03', 'External safety extension, v3 routing and comparison ledger', EXTERNAL/'audit/safety_improvement_v2.json'),
        ('E04', 'Native prepared-data manifest', ROOT/'data/native/prepared/v1/manifest.json'),
        ('E05', 'Rejected OLMoE candidate outcome', ROOT/'audit/native/olmoe-v1-candidate-outcome.json'),
        ('E06', 'One/two-GPU numerical/scaling comparison', ROOT/'outputs/internal/native/gptoss/ddp-qualification-6329565/comparison.json'),
        ('E07', 'Stage 2 archive restoration', ROOT/'artifacts/internal/native/gptoss/v1/archive-20261018-r4-scored-calibrated/archive-verification.json'),
        ('E08', 'Four-trial calibration and selected Stage 3 archive', ROOT/'audit/completion-batch-20261021/verified-calibration-results.json'),
        ('E09', 'Real native expert qualification', ROOT/'outputs/internal/native/gptoss/qualify-20261020-native-eval-v2/report.json'),
        ('E10', 'Frozen completion plan and selection identity', ROOT/'audit/completion-batch-20261021/plan.json'),
        ('E11', 'Expert-only routing telemetry', ROOT/'outputs/internal/native/gptoss/moe-e2-f36368c989-expert-only/expert-usage.json'),
        ('E12', 'Joint-gate routing telemetry', ROOT/'outputs/internal/native/gptoss/moe-e2-f36368c989-joint-gates/expert-usage.json'),
        ('E13', 'Earlier 526-answer attributed assessment', ROOT/'audit/native/gptoss-r4-answer-assessment-20261019.json'),
        ('E14', 'Final 1,500-answer verified comparison', ROOT/'outputs/internal/evaluation/finish-20261021/FINAL_RESEARCH_RESULTS.json'),
        ('E15', 'Post-hoc qualitative sample manifest', ROOT/'audit/completion-batch-20261021/qualitative-review-sample.json'),
        ('E16', 'Historical passing cold start', ROOT/'audit/native/archive-cold-start-6403909.json'),
        ('E17', 'Historical full Stage 2 soak', ROOT/'audit/native/load-soak-6403909/summary.json'),
        ('E18', 'Historical native restart and external rollback', ROOT/'audit/native/recovery-6403909.json'),
        ('E19', 'Latest real engine/gateway checks', ROOT/'audit/native/live-6479705.json'),
        ('E20', 'Latest real cancellation/history/overload checks', ROOT/'audit/native/live-6479705-operations.json'),
        ('E21', 'Latest desktop/mobile browser checks', ROOT/'audit/native/browser-6479705/report.json'),
        ('E22', 'Latest failed cold-start gate', ROOT/'audit/native/archive-cold-start-6479705.json'),
        ('E23', 'Latest stopped deployment record', ROOT/'audit/native/deployment-6479705.json'),
        ('E24', 'Home/scratch cleanup verification', ROOT/'audit/completion-batch-20261021/final-storage-verification-20261022.json'),
        ('E25', 'Storage relocation and cleanup narrative', ROOT/'docs/STORAGE_CLEANUP_2026-10-21.md'),
        ('E26', 'Release-tree CPU test results', ROOT/'audit/release-20261022/release-tests.xml'),
        ('E27', 'Fresh local clone quick-start test results', ROOT/'audit/release-20261022/fresh-clone-tests.xml'),
        ('E28', 'Preserved GPT-OSS inference overlay inventory', ROOT/'audit/completion-batch-20261021/inference-environment-supplement-20261022.json'),
        ('E29', 'Parent qualitative review', ROOT/'audit/completion-batch-20261021/parent-qualitative-review-20261022.json'),
        ('E30', 'Candidate qualitative review', ROOT/'audit/completion-batch-20261021/candidate-qualitative-review-20261022.json'),
        ('E31', 'Original immutable final research report', ROOT/'docs/FINAL_RESEARCH_REPORT_2026-10-21.md'),
        ('E32', 'Expert-only completed pilot', ROOT/'outputs/internal/native/gptoss/moe-e2-f36368c989-expert-only/report.json'),
        ('E33', 'Joint-gate completed pilot', ROOT/'outputs/internal/native/gptoss/moe-e2-f36368c989-joint-gates/report.json'),
        ('E34', 'Stage 2 training and calibrated provenance', ROOT/'outputs/internal/native/gptoss/20261018-r4-scored-calibrated/sft/provenance.json'),
        ('E35', 'Final independent artifact audit', ROOT/'audit/completion-batch-20261021/FINAL_COMPLETION_AUDIT_20261022.json'),
        ('E36', 'Actual joint-gate pilot configuration and environment', ROOT/'outputs/internal/native/gptoss/moe-e2-f36368c989-joint-gates/metadata-6429777.json'),
        ('E37', 'Pinned native data sources', ROOT/'configs/native_sources.json'),
        ('E38', 'Pinned original safety data sources', ROOT/'configs/safety_sources.json'),
        ('E39', 'Pinned support/toxicity extension sources', ROOT/'configs/safety_sources_v2.json'),
        ('E40', 'Historical external safety experiment report', ROOT/'docs/SAFETY_RESULTS.md'),
    ]
    catalog = []
    for eid, purpose, path in sources:
        if not path.is_file():
            raise FileNotFoundError(path)
        label = 'external-workspace/' + str(path.relative_to(EXTERNAL)) if path.is_relative_to(EXTERNAL) else str(path.relative_to(ROOT))
        catalog.append(dict(id=eid, purpose=purpose, path=label, sha256=sha(path), bytes=path.stat().st_size,
                            availability='Preserved private research workspace; not necessarily bundled with public source'))
    final = read(ROOT/'outputs/internal/evaluation/finish-20261021/FINAL_RESEARCH_RESULTS.json')
    for arm, metrics in final['arms'].items():
        path = ROOT/'outputs/internal/evaluation/finish-20261021'/arm/'answers.jsonl'
        assert sha(path) == metrics['answers_sha256'], arm
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        assert len(rows) == metrics['answers'] == 500
    usage, pilots = {}, {}
    for variant in ['expert-only', 'joint-gates']:
        directory = ROOT/'outputs/internal/native/gptoss'/('moe-e2-f36368c989-'+variant)
        pilots[variant] = read(directory/'report.json')
        values = list(read(directory/'expert-usage.json').values())
        usage[variant] = dict(groups=len(values), min_used_experts=min(v['used_experts'] for v in values),
            max_used_experts=max(v['used_experts'] for v in values),
            mean_entropy_over_ln32=statistics.mean(v['entropy']/math.log(32) for v in values),
            scope='Descriptive saved telemetry, not an independent experiment or proof of semantic specialization')
    cold_ids = ['6377096','6402339','6403909','6479705']
    cold = {job: read(ROOT/'audit/native'/('archive-cold-start-'+job+'.json')) for job in cold_ids}
    metric_export = dict(built_at_utc=datetime.now(timezone.utc).isoformat(), frozen_answers=final['arms'],
        paired_comparison=final['paired_comparison'], saved_usage_summary=usage,
        cold_start={job:{k:r[k] for k in ['archive_boot_to_ready_seconds','maximum_seconds','passed']} for job,r in cold.items()},
        production_accepted=False,answer_quality_accepted=False,independent_human_review='pending',
        new_training_or_generation=False)
    (OUT/'verified-report-metrics.json').write_text(json.dumps(metric_export,indent=2)+'\n')
    (OUT/'evidence-catalog.json').write_text(json.dumps(catalog,indent=2)+'\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,ax = plt.subplots(figsize=(8,3.8),layout='constrained')
    values = [cold[job]['archive_boot_to_ready_seconds'] for job in cold_ids]
    bars=ax.bar(cold_ids, values, color=['#b91c1c' if value>600 else '#2563eb' for value in values])
    ax.axhline(600,color='#111827',linestyle='--',label='600-second target')
    ax.bar_label(bars,fmt='%.2f s',padding=4)
    ax.set_ylim(0,900);ax.set_xlabel('Slurm job ID');ax.set_ylabel('Seconds: full archive-to-ready clock')
    ax.set_title('Cold-start history: latest demo misses unchanged target');ax.legend(loc='upper right')
    fig.savefig(figures/'cold-start-history.png',dpi=170);fig.savefig(figures/'cold-start-history.svg');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,3.8),layout='constrained')
    for variant,color in [('expert-only','#2563eb'),('joint-gates','#d97706')]:
        directory=ROOT/'outputs/internal/native/gptoss'/('moe-e2-f36368c989-'+variant)
        rows=[json.loads(x) for x in (directory/'metrics.jsonl').read_text().splitlines()]
        x=[r['step'] for r in rows];y=[r['loss'] for r in rows]
        axes[0].plot(x,y,color=color,alpha=.18,linewidth=.8)
        axes[0].plot(x[9:],np.convolve(y,np.ones(10)/10,mode='valid'),color=color,label=variant)
        axes[1].plot(x,[r['tokens_per_second'] for r in rows],color=color,alpha=.6,label=variant)
    axes[0].set_title('Training loss: raw + 10-step moving mean');axes[0].set_ylabel('Logged supervised loss')
    axes[1].set_title('Logged pilot throughput');axes[1].set_ylabel('Supervised tokens / second')
    for ax in axes:ax.set_xlabel('Optimizer step');ax.legend()
    fig.savefig(figures/'pilot-training-curves.png',dpi=170);fig.savefig(figures/'pilot-training-curves.svg');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,7),layout='constrained')
    for ax,variant in zip(axes,['expert-only','joint-gates']):
        raw=read(ROOT/'outputs/internal/native/gptoss'/('moe-e2-f36368c989-'+variant)/'expert-usage.json')
        matrix=np.zeros((24,32))
        for key,value in raw.items():
            layer=int(key.split('layers.')[1].split('.')[0])
            for expert,count in value['counts'].items():matrix[layer,int(expert)]+=count
        matrix=matrix/matrix.sum(axis=1,keepdims=True)
        im=ax.imshow(matrix,cmap='Blues',aspect='auto',vmin=0,vmax=.15)
        ax.set_title(variant);ax.set_xlabel('Native expert ID');ax.set_ylabel('Layer')
    fig.colorbar(im,ax=axes,shrink=.6,label='Recorded assignment share pooled within layer')
    fig.suptitle('Saved routing telemetry: descriptive, not semantic-specialization evidence')
    fig.savefig(figures/'expert-usage.png',dpi=160);fig.savefig(figures/'expert-usage.svg');plt.close(fig)
    shutil.copyfile(ROOT/'release/docs/assets/frozen-comparison.png',figures/'frozen-comparison.png')
    shutil.copyfile(ROOT/'audit/native/browser-6479705/ui-1440.png',figures/'native-ui-desktop.png')
    shutil.copyfile(ROOT/'audit/native/browser-6479705/ui-390.png',figures/'native-ui-mobile.png')

    directories=['src','web','training','evaluation','scripts','configs','tests','deploy']
    inventory=[]
    for directory in directories:
        for path in sorted((ROOT/directory).rglob('*')):
            if not path.is_file() or path.is_symlink() or '__pycache__' in path.parts or any(p.endswith('.egg-info') for p in path.parts):continue
            if path.suffix not in {'.py','.js','.css','.html','.json','.sbatch','.sh','.txt','.yaml','.yml'} and path.name!='Dockerfile':continue
            purpose='Configuration or deployment/template asset'
            functions=[]
            if path.suffix=='.py':
                tree=ast.parse(path.read_text())
                purpose=(ast.get_docstring(tree) or '').split('\n')[0]
                functions=[n.name for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))]
                if not purpose:
                    purpose=('Tests: ' if directory=='tests' else 'Definitions: ')+', '.join(functions[:4])
            elif path.suffix=='.sbatch':purpose='Site-specific Slurm template: '+path.stem.replace('ice_','').replace('_',' ')
            elif path.suffix=='.sh':purpose='Shell environment/startup utility'
            elif path.suffix=='.js':purpose='Browser event, request, history and stream behavior'
            elif path.suffix in {'.html','.css'}:purpose='Browser UI markup/styles'
            inventory.append(dict(path=str(path.relative_to(ROOT)),purpose=purpose,bytes=path.stat().st_size,
                                  sha256=sha(path),definitions=functions))
    (OUT/'source-inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    report=DOC.read_text().split('<!-- GENERATED_APPENDICES -->')[0]
    extra='''<!-- GENERATED_APPENDICES -->

## Appendix A. Selected experiment and operational ledger

This is the outcome ledger for major decision points, not a complete billing log.
Original raw failed reports/checkpoints remain preserved. Historical status refers
to the cited artifact outcome, not a claim that the allocation is still active.

| Job | Purpose | Verified outcome |
|---|---|---|
| 6222580 | External support-aware router | Training complete; acceptance still failed |
| 6222389 | SAFETY LoRA continuation | Training complete; independent answer review pending |
| 6223631 | Router formatting correction v3 | Completed; formatting diagnostic improved; broader routing gates failed |
| 6238243 | Historical four-expert preview restart | API/browser verified then; allocation expired |
| 6313020 | OLMoE application qualification | Operations passed; quality screening failed; stopped |
| 6329565 | Native one/two-H100 benchmark | Numerical equivalence and measured scaling passed |
| 6362669 | Attention run continuation | Full epoch completed at 1,489 steps |
| 6376731 | Earlier calibration/base reproduction | Failed before adapter trials; scorer issue recorded |
| 6377094 | Corrected attention calibration | Selected nonzero scale 0.125 |
| 6377095 | Standard attention export/archive | Passed numerical/persistence qualification |
| 6377096 | First full candidate qualification | Cold start failed; other checks completed; old signal exit 143 |
| 6402339 | Startup rerun | Cold start passed; soak stopped at quota exhaustion |
| 6403909 | Full operating rerun | Cold start, soak, restart and external rollback passed |
| 6421946 | Native expert training qualification | Five steps and split 3→5 continuation passed |
| 6429776 | Expert-only pilot | 200 steps; best trained checkpoint 200 |
| 6429777 | Joint-gate pilot | 200 steps; best trained checkpoint 200 |
| 6447695 | Fixed four-trial export calibration | Reports complete; selected joint-gate scale 0.5 |
| 6447696 | Two parent comparison arms | Completed 1,000 valid outputs |
| 6447697 | Initial candidate allocation | CUDA access failed before generation; no valid candidate outputs |
| 6447698 | Initial automatic report | Correctly failed incomplete collection; history retained |
| 6477680 | Candidate-only recovery | Completed 500 valid outputs without retuning |
| 6477681 | Final CPU result verification | Complete 1,500-output report |
| 6479723 | Public-export CPU model-logic tests | 136 passed; five deliberate dependency/data skips |
| 6479705 | Latest restricted release demo | API/browser/operations passed; cold start failed; now stopped |

## Appendix B. Saved training and routing figures

![Pilot training loss and throughput from saved step logs](../reports/comprehensive-20261022/figures/pilot-training-curves.png)

The faint traces are raw per-step training loss and the darker loss curves are
10-step moving means. These are training traces, not held-out learning curves.
Recorded throughput includes the measured step path and is not an end-to-end
service benchmark. Curves are derived from the two saved 200-row metric files;
no missing historical Stage 2 learning curve is fabricated.

![Layer/expert usage from saved hook counts](../reports/comprehensive-20261022/figures/expert-usage.png)

Each layer's counts pool the five recorded sources and normalize by its observed
assignments. Similar heatmaps do not establish equal quality, causal gate effects,
semantic expert labels or general load balancing. Raw telemetry and derivation
identities are listed in the evidence catalog.

![Actual mobile native UI after the recorded cancellation probe](../reports/comprehensive-20261022/figures/native-ui-mobile.png)

The screenshot is historical real-model integration evidence. Its “ready” badge
is not a current readiness claim. The allocation subsequently stopped.

## Appendix C. Complete maintained source inventory

This enumerates maintained code, tests, configuration and launch assets inspected
for the report. Bulk data/weights, environments, caches and private archives are
deliberately outside this source inventory. File descriptions use module docstrings
or actual definitions; template presence does not mean a deployment was executed.
The accompanying JSON retains each full SHA256 and definition list.

'''
    for directory in directories:
        entries=[x for x in inventory if x['path'].split('/')[0]==directory]
        extra+='### '+directory+' — '+str(len(entries))+' files\n\n| Path | Responsibility / definitions |\n|---|---|\n'
        for item in entries:
            desc=item['purpose'].replace('|','/').replace('\n',' ')[:185]
            extra+='| `'+item['path']+'` | '+desc+' |\n'
        extra+='\n'
    extra+='''## Appendix D. Evidence catalog and original identities

References [E01]–[E40] identify the preserved records actually read for this report.
Paths are relative to the research workspace unless prefixed `external-workspace/`,
which denotes the preserved external application checkout. Raw artifacts may remain
private or access-restricted; hashes are not a claim that those files ship with
public source. The PDF/HTML is self-contained for interpretation of its results.

| ID | Record | Preserved source path | SHA256 prefix |
|---|---|---|---|
'''
    for e in catalog:
        extra+=f"| {e['id']} | {e['purpose']} | `{e['path']}` | `{e['sha256'][:16]}` |\n"
    extra+='''
Full 64-character digests and byte counts are in `evidence-catalog.json` beside the
rendered report. `verified-report-metrics.json` contains the checked answer hashes,
cold-start outcomes and descriptive telemetry summaries. DOT/SVG/PNG figure sources
are supplied in `figures/`; diagrams represent implemented paths and declared
gates, not unsupported product capabilities.

## Appendix E. Reproduction and reporting boundaries

The source CPU quick start is `python -m pip install -e '.[dev]'` followed by the
targeted API/demo/data/packaging tests. The release also provides an exact CPU-demo
dependency lock. Real GPU inference requires the matching upstream base,
compatible tokenizer/adapters, separate model access, a qualified environment and
an actual allocation. Full past research requires identity-matched prepared data
and archived artifacts; a source-only clone does not supply them.

Read-only status verification uses `squeue`, `sacct`, exact job-bound reports and
artifact checksums. Do not execute historical training/export commands merely
because they are documented: experimentation is frozen. To rebuild this report,
run `scripts/build_comprehensive_diagrams.py` and
`scripts/complete_comprehensive_report.py`, then render the Markdown with the
report renderer. Rebuilding uses saved evidence and creates no model experiment.

Some underlying metrics were generated by preexisting imperfect automatic scorers;
the report preserves their limitations and versions. It does not fabricate human
labels, silently relax thresholds, merge incompatible cohorts or convert a clean
scheduler exit into a quality pass. Updated publication/deployment evidence should
be appended with a new verification timestamp, preserving prior failed records.
'''
    extra+='''
## Appendix F. Documentation and repository entry-point index

Older documentation is retained for provenance and may contain historical queue
states or superseded plans. This comprehensive report and job-bound artifacts
govern current status. Original reports are not silently rewritten to hide failed
experiments. The following index locates the rest of the project documentation.

| Path | Document / entry point |
|---|---|
'''
    docpaths=sorted((ROOT/'docs').rglob('*.md'))
    docpaths += [ROOT/name for name in ['README.md','README_STAGE_1_2.md','AGENTS.md','PROJECT_COMPLETION_PLAN.md','LICENSE','pyproject.toml','requirements-api.lock','.gitignore','.env.example','.github/workflows/checks.yml'] if (ROOT/name).is_file()]
    for path in docpaths:
        if path==DOC:continue
        lines=path.read_text().splitlines()
        heading=next((line.lstrip('# ').strip() for line in lines if line.startswith('# ')),path.name)
        extra+='| `'+str(path.relative_to(ROOT))+'` | '+heading.replace('|','/')+' |\n'
    extra+='''
## Appendix G. Requirement-to-evidence traceability

| Requirement | Implementation/evidence | Current decision |
|---|---|---|
| Preserve existing specialists | Frozen weight inventory and original source archive, E01–E02 | Verified |
| Correct request routing | Named adapter selection, checkpoint label metadata and lock | Implemented and integrated |
| Add support/safety coverage | Published-data extension and v3 ledger, E03 | Integration complete; routing gates failed |
| Avoid fabricated data/reviews | Source manifests, review importer and blank masked forms | No synthetic native training examples; human review pending |
| Prevent exact project split leakage | Grouped source preparation and protected keys, E04 | Checks passed; semantic contamination unknown |
| Train native model reproducibly | Provenance/config/checkpoint identity, E34 | Stage 2 completed; quality failed |
| Justify multi-GPU work | Fixed-batch numerical/scaling comparison, E06 | Qualified only within measured benchmark |
| Export actual learned updates | Projection/frozen-state/reload probes and archives, E07–E10 | Selected research exports qualified |
| Train actual FFN experts/gates | Qualified resume, changed-state and pilot reports, E09/E32/E33 | Both pilots completed |
| Compare selected model fairly | Frozen protocol, 500 paired input hashes, E14 | No demonstrated held-out proxy gain |
| Operate private app reliably | API/browser/load/recovery, E16–E23 | Historical full pass; latest cold start failed |
| Keep resources/storage bounded | QoS policy, scratch helper, cleanup proof, E24–E25 | Implemented policy; quotas require ongoing checks |
| Provide reproducible public source | Export manifests and CPU/fresh-clone checks, E26–E27 | Local candidate prepared; remote publication pending |
| Establish independent quality | Masked review sheet and reviewer protocol | Pending; not replaced by AI assessments |
| Accept production deployment | All relevant integrity, operations, quality and review gates | False; research scope only |
'''
    provenance=read(ROOT/'outputs/internal/native/gptoss/20261018-r4-scored-calibrated/sft/provenance.json')
    pilot_meta=read(ROOT/'outputs/internal/native/gptoss/moe-e2-f36368c989-joint-gates/metadata-6429777.json')
    extra+='''
## Appendix H. Full declared configurations and pinned source registry

These are the saved configuration objects, not newly chosen hyperparameters.
Stage 2 provenance also records its post-training calibration. Stage 3's runtime
variant flag chooses expert-only versus joint gates while sharing the configuration
below. Paths refer to the prepared research workspace, whose required artifacts
are not included by a source-only clone. [E34, E36]

### Stage 2 recorded configuration

```json
'''+json.dumps(provenance['identity']['config'],indent=2)+'''\n```

### Stage 3 recorded configuration

```json
'''+json.dumps(pilot_meta['identity']['config'],indent=2)+'''\n```

### Actual pilot environment

The joint-gate metadata records CUDA '''+str(pilot_meta['cuda'])+''' and the following
package versions. A CUDA version recorded by Torch is not a claim that every
system driver/toolkit component has the same version.

| Package | Recorded version |
|---|---|
'''
    for name,value in pilot_meta['versions'].items():extra+='| '+name+' | '+str(value)+' |\n'
    extra+='''
### Pinned source identities

These source registries distinguish upstream revision identity from a locally
prepared split hash. Recorded terms are provenance, not blanket redistribution
permission. The source files and inherited manifests must also pass integrity
checks before reproducing the original preparation. [E37–E39]

| Registry / source | Repository or provider | Pinned revision | Recorded terms |
|---|---|---|---|
'''
    for registry in ['native_sources.json','safety_sources.json','safety_sources_v2.json']:
        for name,item in read(ROOT/'configs'/registry).items():
            extra+='| '+registry+' / '+name+' | '+str(item.get('repository',item.get('provider','recorded source')))+' | `'+str(item.get('revision','See source registry'))+'` | '+str(item.get('license','See source registry'))+' |\n'
    DOC.write_text(report+extra)
    final_manifest=dict(built_at_utc=datetime.now(timezone.utc).isoformat(),
        report_markdown_sha256=sha(DOC), word_count=len(DOC.read_text().split()),
        source_files=len(inventory), evidence_records=len(catalog), diagram_count=12,
        answer_hashes_reverified=True, new_training=False, new_generation=False,
        latest_demo_job='6479705',latest_demo_live=False,latest_cold_start_passed=False,
        independent_human_review='pending',production_accepted=False,answer_quality_accepted=False)
    (OUT/'report-manifest.json').write_text(json.dumps(final_manifest,indent=2)+'\n')
    print(json.dumps(final_manifest,indent=2))


if __name__ == '__main__':
    main()
