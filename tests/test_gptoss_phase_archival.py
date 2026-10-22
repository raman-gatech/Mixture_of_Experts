import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from gptoss_distributed import archive_previous_results,digest


def test_interim_answers_status_and_adapter_are_preserved_before_resume(tmp_path):
    (tmp_path/'adapter').mkdir();(tmp_path/'adapter/weights').write_bytes(b'prior learned tensor fixture')
    (tmp_path/'training-status.json').write_text('{"completed":false}')
    (tmp_path/'answer-validation-selected-final.json').write_text('{"step":69}')
    (tmp_path/'answer-validation-0.json').write_text('immutable baseline')
    expected=digest(tmp_path/'adapter/weights')
    archive=Path(archive_previous_results(tmp_path))
    assert digest(archive/'adapter/weights')==expected
    assert json.loads((archive/'archival-manifest.json').read_text())['files_sha256']['adapter/weights']==expected
    assert (tmp_path/'answer-validation-0.json').read_text()=='immutable baseline'
    assert not (tmp_path/'training-status.json').exists()
    assert archive_previous_results(tmp_path) is None
