"""Two-rank continuation of the identity-pinned trainer, with explicit topology lineage."""
import argparse
from collections import Counter
from datetime import timedelta
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time

from gptoss_distributed import archive_previous_results,digest,latest_complete,trainer_class,validate_checkpoint
from run_gptoss_checkpointed import CheckpointPolicy


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--distributed-stop-step',type=int)
    parser.add_argument('--distributed-resume-latest',action='store_true')
    extra,remaining=parser.parse_known_args()
    output=Path(remaining[remaining.index('--output')+1])
    if extra.distributed_resume_latest:
        if '--resume' in remaining:raise ValueError('Choose explicit or latest resume')
        remaining+=['--resume',str(latest_complete(output))]
    if '--resume' not in remaining:raise ValueError('Explicit migrated or complete distributed checkpoint required')
    checkpoint=Path(remaining[remaining.index('--resume')+1]);validate_checkpoint(checkpoint)
    lineage=json.loads((output/'distributed-lineage.json').read_text())
    if any(digest(Path('training')/name)!=expected for name,expected in lineage['source_sha256'].items()):
        raise ValueError('Distributed execution source changed; create an attributed new phase')
    import torch
    import torch.distributed as dist
    import train_gptoss as native
    world=int(os.environ.get('WORLD_SIZE','1'));rank=int(os.environ.get('RANK','0'));local=int(os.environ.get('LOCAL_RANK','0'))
    if world!=2 or not os.getenv('SLURM_JOB_ID'):raise ValueError('Qualified allocated two-rank execution required')
    torch.cuda.set_device(local)
    dist.init_process_group('nccl',device_id=torch.device('cuda',local),timeout=timedelta(minutes=10))
    policy=CheckpointPolicy();signal.signal(signal.SIGUSR1,policy.request_stop)
    prior_results=archive_previous_results(output) if rank==0 else None
    dist.barrier()
    original_model=native.AutoModelForCausalLM;original_tokenizer=native.AutoTokenizer;original_write=native.write_json
    class RankModel:
        @staticmethod
        def from_pretrained(*args,**kwargs):
            kwargs['device_map']={'':local};return original_model.from_pretrained(*args,**kwargs)
    class RankTokenizer:
        @staticmethod
        def from_pretrained(*args,**kwargs):
            tokenizer=original_tokenizer.from_pretrained(*args,**kwargs);save=tokenizer.save_pretrained
            tokenizer.save_pretrained=lambda *a,**k:save(*a,**k) if rank==0 else ()
            return tokenizer
    native.AutoModelForCausalLM=RankModel;native.AutoTokenizer=RankTokenizer
    native.write_json=lambda *a,**k:original_write(*a,**k) if rank==0 else None
    original_validation=native.AnswerValidation
    class DistributedAnswers(original_validation):
        def assess(self,model,step):
            previous=model.config.use_cache;was_training=model.training;model.eval();model.config.use_cache=True
            answers=[];began=time.perf_counter()
            try:
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    for row in self.rows[rank::world]:
                        ids=torch.tensor([native.prompt_ids(self.tokenizer,row,'AUTO')],device='cuda')
                        if ids.shape[1]+256>2048:raise ValueError('Validation context overflow')
                        generated=model.generate(ids,attention_mask=torch.ones_like(ids),max_new_tokens=256,do_sample=False,use_cache=True,pad_token_id=self.tokenizer.eos_token_id)
                        raw=self.tokenizer.decode(generated[0,ids.shape[1]:],skip_special_tokens=False);answer=native.final_text(raw)
                        answers.append(dict(source=row['source'],source_id=row['source_id'],prompt_sha256=row['prompt_sha256'],reference=row['output'],answer=answer,
                            automatic_match=native.match_answer(row,answer),generated_tokens=int(generated.shape[1]-ids.shape[1]),truncated=int(generated.shape[1]-ids.shape[1])>=256,answer_sha256=hashlib.sha256(answer.encode()).hexdigest()))
            finally:model.config.use_cache=previous;model.train(was_training)
            gathered=[None]*world;dist.all_gather_object(gathered,answers)
            lookup={(a['source'],a['source_id']):a for part in gathered for a in part}
            if len(lookup)!=len(self.rows):raise ValueError('Duplicate/missing distributed validation answers')
            ordered=[lookup[(r['source'],r['source_id'])] for r in self.rows]
            counts=Counter(a['source'] for a in ordered);correct=Counter(a['source'] for a in ordered if a['automatic_match'])
            accuracy={source:correct[source]/counts[source] for source in counts}
            score,passed=native.product_score(accuracy,self.baseline or accuracy)
            elapsed=torch.tensor(time.perf_counter()-began,device='cuda');dist.all_reduce(elapsed,op=dist.ReduceOp.MAX)
            report=dict(step=step,mode='AUTO',final_prefill=True,counts=dict(counts),accuracy=accuracy,product_score=score,nonregression_passed=passed,baseline=self.baseline,
                seconds=float(elapsed),answers=ordered,world_size=world,production_accepted=False,limitation='Answer-key proxies only; full heldout General/support/safety and semantic acceptance remain required')
            native.write_json(self.output/f'answer-validation-{step}.json',report);self.last=report
            if rank==0:print(json.dumps({k:v for k,v in report.items() if k!='answers'}),flush=True)
            return report
    native.AnswerValidation=DistributedAnswers
    native.Trainer=trainer_class(native.Trainer,policy,extra.distributed_stop_step)
    metadata=dict(job=os.environ['SLURM_JOB_ID'],rank=rank,world_size=world,hostname=socket.gethostname(),started_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),command=sys.argv,
        checkpoint=str(checkpoint),previous_results_archived=prior_results,imported_topology=lineage['rng_policy'],python_version=sys.version,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
        versions={name:importlib.metadata.version(name) for name in ['torch','transformers','peft','accelerate']},
        source_sha256={name:digest(Path('training')/name) for name in ['gptoss_distributed.py','run_gptoss_distributed.py','launch_gptoss_distributed.py']},git_commit=None,
        seed=json.loads((output/'provenance.json').read_text())['identity']['config']['seed'],checkpoint_interval_seconds=1200,production_accepted=False)
    with (output/f'distributed-launch-{os.environ["SLURM_JOB_ID"]}-{rank}-{time.time_ns()}.json').open('x') as stream:json.dump(metadata,stream,indent=2);stream.write('\n')
    sys.argv=[sys.argv[0],*remaining]
    try:native.main()
    finally:dist.destroy_process_group()


if __name__=='__main__':
    os.umask(0o077);main()
