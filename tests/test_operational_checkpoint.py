"""Wall-clock saves must preserve the Trainer's existing evaluation decisions."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location(
    'checkpoint_runtime', Path(__file__).resolve().parents[1] / 'training/run_gptoss_checkpointed.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_periodic_save_and_preemption_preserve_evaluation_decisions():
    now = [0]
    policy = module.CheckpointPolicy(clock=lambda: now[0])
    control = SimpleNamespace(should_save=False, should_training_stop=False, should_evaluate=True)
    now[0] = 1199
    assert not policy.step(control).should_save
    now[0] = 1200
    assert policy.step(control).should_save
    assert control.should_evaluate and not control.should_training_stop
    policy.saved()
    control.should_save = False
    now[0] = 1201
    assert not policy.step(control).should_save
    policy.request_stop(None, None)
    assert policy.step(control).should_save and control.should_training_stop
    assert control.should_evaluate


def test_real_trainer_saves_optimizer_rng_and_resumes_without_changing_best(tmp_path):
    import json
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments

    torch.set_num_threads(1)
    config = GPT2Config(vocab_size=16, n_embd=8, n_layer=1, n_head=1,
                       n_positions=16, n_ctx=16, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)
    data = [{'input_ids': torch.tensor([1, 2, 3, 4]),
             'labels': torch.tensor([1, 2, 3, 4])} for _ in range(4)]
    def arguments(steps):
        return TrainingArguments(output_dir=str(tmp_path), use_cpu=True,
            max_steps=steps, per_device_train_batch_size=1, per_device_eval_batch_size=1,
            eval_strategy='steps', eval_steps=2, save_strategy='steps', save_steps=2,
            logging_steps=1, report_to='none', disable_tqdm=True,
            load_best_model_at_end=True, metric_for_best_model='eval_loss', save_total_limit=3)

    # Time-based callback requests saves at non-evaluation steps 1 and 3.
    policy = module.CheckpointPolicy(interval=0)
    trainer = Trainer(model=GPT2LMHeadModel(config), args=arguments(3),
        train_dataset=data, eval_dataset=data, callbacks=[module.checkpoint_callback(policy)])
    trainer.train()
    latest = tmp_path / 'checkpoint-3'
    for filename in ['optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'trainer_state.json']:
        assert (latest / filename).is_file()
    state = json.loads((latest / 'trainer_state.json').read_text())
    assert state['global_step'] == 3
    assert state['best_model_checkpoint'].endswith('checkpoint-2')
    assert trainer.state.best_model_checkpoint.endswith('checkpoint-2')

    # Preserve real optimizer/scheduler/RNG restoration rather than restarting at zero.
    resumed = Trainer(model=GPT2LMHeadModel(config), args=arguments(4),
        train_dataset=data, eval_dataset=data,
        callbacks=[module.checkpoint_callback(module.CheckpointPolicy(interval=0))])
    resumed.train(resume_from_checkpoint=str(latest))
    assert resumed.state.global_step == 4
    assert any(row.get('step') == 3 for row in resumed.state.log_history)
