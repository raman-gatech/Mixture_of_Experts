"""Operational input counts must match the actual gateway template and reservation."""
import importlib.util
from pathlib import Path
import pytest
from transformers import AutoTokenizer
from moe_router.conversation import prepare_messages


def test_gptoss_load_count_equals_actual_gateway_count():
    base=Path('artifacts/internal/native/candidates/gpt-oss-20b/base')
    if not (base/'tokenizer.json').exists(): pytest.skip('Pinned tokenizer unavailable')
    spec=importlib.util.spec_from_file_location('native_load',Path('evaluation/load_native_product.py'))
    load=importlib.util.module_from_spec(spec); spec.loader.exec_module(load)
    tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True)
    prompt,n,source=load.load_prompt(tokenizer,True)
    assert abs(n-512)<=3 and source['source_id']
    for index in [0,10,10000000]:
        varied,count=load.varied_prompt(tokenizer,prompt,index,True)
        _,actual,_=prepare_messages(tokenizer,varied,[],'GENERAL',2048,128,final_channel=True)
        assert count==len(actual) and 509<=count<=512
