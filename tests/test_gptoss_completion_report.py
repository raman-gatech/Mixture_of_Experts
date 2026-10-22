import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import pytest

spec=importlib.util.spec_from_file_location('completion',Path(__file__).resolve().parents[1]/'scripts/summarize_gptoss_completion.py')
completion=importlib.util.module_from_spec(spec)
spec.loader.exec_module(completion)


def test_completed_collection_requires_matched_inputs_and_selected_checkpoint(tmp_path,monkeypatch):
    suite=tmp_path/'suite';suite.mkdir()
    rows=[dict(id=str(i),prompt='fixture '+str(i),task='MATH' if i<2 else 'GENERAL') for i in range(500)]
    text=''.join(json.dumps(r)+'\n' for r in rows);(suite/'prompts.jsonl').write_text(text)
    protocol=tmp_path/'protocol.json';protocol.write_text(json.dumps(dict(suite_sha256=hashlib.sha256(text.encode()).hexdigest())))
    ph=hashlib.sha256(protocol.read_bytes()).hexdigest()
    plan=tmp_path/'plan.json';plan.write_text(json.dumps(dict(confirmatory_suite=str(suite),protocol=str(protocol),protocol_sha256=ph)))
    root=tmp_path/'results';root.mkdir()
    (root/'selected-candidate.json').write_text(json.dumps(dict(calibration_completed=True,kind='unmerged_checkpoint',checkpoint_sha256='fixture-state')))
    for arm in ['parent-baseline-256','parent-precision-512','candidate-precision-512']:
        p=root/arm;p.mkdir();policy,budget=('baseline',256) if arm=='parent-baseline-256' else ('precision',512)
        (p/'answers.jsonl').write_text(''.join(json.dumps(dict(r,answer='fixture',input_ids_sha256='same',policy=policy,max_new_tokens=budget,math_numeric_proxy=True))+'\n' for r in rows))
        (p/'metadata.json').write_text(json.dumps(dict(identity=dict(protocol_sha256=ph,cells=[[policy,budget]],training_checkpoint_sha256='fixture-state'))))
    monkeypatch.setattr(sys,'argv',['summary','--root',str(root),'--plan',str(plan)])
    completion.main()
    assert json.loads((root/'FINAL_RESEARCH_RESULTS.json').read_text())['experiment_execution_complete']
    assert json.loads((root/'FINAL_RESEARCH_RESULTS.json').read_text())['production_accepted'] is False
    p=root/'candidate-precision-512/answers.jsonl';changed=[json.loads(l) for l in p.open()];changed[0]['input_ids_sha256']='different'
    p.write_text(''.join(json.dumps(r)+'\n' for r in changed))
    with pytest.raises(SystemExit):completion.main()
    assert json.loads((root/'FINAL_RESEARCH_RESULTS.json').read_text())['experiment_execution_complete'] is False
