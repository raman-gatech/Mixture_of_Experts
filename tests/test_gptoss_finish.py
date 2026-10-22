"""Development selection must not turn failed exports into accepted candidates."""
import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('finish',Path(__file__).resolve().parents[1]/'scripts/finish_gptoss_research.py')
finish=importlib.util.module_from_spec(spec)
spec.loader.exec_module(finish)


def test_select_only_qualified_strictly_improved_trials_and_predeclared_ties():
    trials=[dict(passed=False,nll=.1,variant='expert-only',scale=.5),
            dict(passed=True,nll=3.,variant='expert-only',scale=.5)]
    assert finish.choose_trial(trials,2.5) is None
    trials += [dict(passed=True,nll=2.,variant='joint-gates',scale=.5),
               dict(passed=True,nll=2.,variant='expert-only',scale=.25),
               dict(passed=True,nll=2.,variant='expert-only',scale=.5)]
    assert finish.choose_trial(trials,2.5) is trials[-1]
    assert finish.choose_trial([dict(passed=True,nll=float('nan'),variant='expert-only',scale=.5)],2.5) is None
