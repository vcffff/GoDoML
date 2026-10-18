"""Bounded four-example memorization test; never installs a serving adapter."""
import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_cash_sft import read, sha, write
from scripts.train_cash_lora import batch, load_base, loss_forward


def metrics(rows):
    return {**{key: sum(r['expected'] == truth and r['prediction'] == pred for r in rows)
               for key, truth, pred in [('TP', 'violation', 'violation'),
                                       ('FP', 'no_violation', 'violation'),
                                       ('TN', 'no_violation', 'no_violation'),
                                       ('FN', 'violation', 'no_violation')]},
            'unknown': sum(r['prediction'] == 'unknown' for r in rows),
            'errors': sum(r['prediction'] is None for r in rows),
            'n': len(rows), 'correct': sum(r['expected'] == r['prediction'] for r in rows)}


def main(args):
    import torch
    from lmformatenforcer import CharacterLevelParserConfig, JsonSchemaParser
    from lmformatenforcer.integrations.transformers import (
        build_token_enforcer_tokenizer_data,
        build_transformers_prefix_allowed_tokens_fn,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    root = ROOT / 'outputs/cash-diagnostic'
    if args.run_name:
        if Path(args.run_name).name != args.run_name or args.run_name in {'.', '..'}:
            raise ValueError('run-name must be a single directory name')
        root = root / 'runs' / args.run_name
    out = root / args.mode
    if (out / 'result.json').exists():
        raise FileExistsError('Preserve completed results: use a new --run-name for both modes')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'executed_script.py').write_text(Path(__file__).read_text(encoding='utf-8'), encoding='utf-8')
    examples_path = ROOT / 'outputs/cash-diagnostic/examples.json'
    examples = read(examples_path)
    train = [e for e in examples if e['split'] == 'train']
    val = [e for e in examples if e['split'] == 'validation']
    fingerprint = {'data': sha(examples_path), 'script': sha(Path(__file__)), 'max_side': 336}
    if args.mode == 'experiment':
        gate = read(out.parent / 'memorize/result.json')
        if not gate.get('memorization_passed') or gate['fingerprint'] != fingerprint:
            raise RuntimeError('Matching successful memorization gate required')
    write(out / 'config.json', {**vars(args), 'fingerprint': fingerprint,
          'learning_rate': 3e-4, 'rank': 8, 'accumulation': 4,
          'deployment_allowed': False, 'validation_used_for_training': False})
    torch.manual_seed(123)
    random.seed(123)
    _, base = load_base(ROOT / 'models/Qwen3-VL-2B-Instruct')
    from transformers import AutoProcessor
    # Set limits at initialization: mutating image_processor alone leaves
    # tokenizer.init_kwargs with the old max_pixels used by chat templates.
    processor = AutoProcessor.from_pretrained(ROOT / 'models/Qwen3-VL-2B-Instruct',
        local_files_only=True, min_pixels=32 * 32, max_pixels=336 * 336)
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True,
                                           gradient_checkpointing_kwargs={'use_reentrant': False})
    base.model.visual.to(dtype=torch.bfloat16)
    model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, lora_dropout=0,
        target_modules=r'.*language_model\.layers\.\d+\.self_attn\.(q_proj|v_proj)$',
        task_type='CAUSAL_LM', bias='none'))
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert params and all('lora_' in n and 'visual' not in n for n in params)
    initial = {n: p.detach().cpu().clone() for n, p in params.items()}
    optimizer = torch.optim.AdamW(params.values(), lr=3e-4)
    tokenizer_data = build_token_enforcer_tokenizer_data(processor.tokenizer)
    schema = {'type': 'object', 'properties': {'event': {'type': 'string',
              'enum': ['violation', 'no_violation', 'unknown']}},
              'required': ['event'], 'additionalProperties': False}
    visual_calls = []
    hook = model.base_model.model.model.visual.register_forward_hook(
        lambda module, inputs, output: visual_calls.append(1))

    def predict(items, phase, constrained=True, blank=False, swapped=False):
        model.eval()
        rows = []
        for source in items:
            example = copy.deepcopy(source)
            if swapped:
                opposite = next(x for x in train if x['target']['event'] != source['target']['event'])
                example['images'] = opposite['images']
            started = time.monotonic()
            inputs = batch(processor, example, target=False, blank=blank)
            decoding = {}
            if constrained:
                parser = JsonSchemaParser(schema, config=CharacterLevelParserConfig(
                    alphabet=tokenizer_data.tokenizer_alphabet))
                decoding['prefix_allowed_tokens_fn'] = build_transformers_prefix_allowed_tokens_fn(
                    tokenizer_data, parser)
            with torch.no_grad():
                tokens = model.generate(**inputs, do_sample=False, max_new_tokens=48,
                    use_cache=True, **decoding)
            raw = processor.decode(tokens[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            prediction = None
            try:
                parsed = json.loads(raw)
                if set(parsed) == {'event'} and parsed['event'] in ['violation', 'no_violation', 'unknown']:
                    prediction = parsed['event']
            except (ValueError, TypeError):
                pass
            rows.append({'id': source['id'], 'phase': phase, 'expected': source['target']['event'],
                         'prediction': prediction, 'raw': raw,
                         'elapsed_sec': time.monotonic()-started})
            del inputs, tokens
        write(out / f'{phase}.json', {'metrics': metrics(rows), 'rows': rows})
        print(json.dumps({'phase': phase, **metrics(rows)}), flush=True)
        return rows

    try:
        # Record real input tensors/mask sizes and initial image ablation.
        model.eval()
        with torch.no_grad():
            inputs = batch(processor, train[0])
            audited = {'input_tokens': inputs['input_ids'].shape[1],
                'assistant_tokens': int((inputs['labels'] != -100).sum()),
                'masked_tokens': int((inputs['labels'] == -100).sum()),
                'image_grid_thw': inputs['image_grid_thw'].tolist(),
                'pixel_values_shape': list(inputs['pixel_values'].shape),
                'initial_real_loss': float(loss_forward(model, inputs)),
                'initial_blank_loss': float(loss_forward(model, batch(processor, train[0], blank=True)))}
            del inputs
        write(out / 'tensor_audit.json', audited)
        with model.disable_adapter():
            predict(train, 'base_train')
            if args.mode == 'experiment':
                predict(val, 'base_validation')
        # Every update contains both positive and both negative training videos.
        # Explicit checkpoints/early stop, not an uncontrolled epoch increase.
        passed = False
        smoke_done = False
        started = time.monotonic()
        max_steps = 48 if args.mode == 'memorize' else 12
        check_steps = {4, 12, 24, 48} if args.mode == 'memorize' else {12}
        torch.cuda.reset_peak_memory_stats()
        for step in range(1, max_steps + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for e in train:
                inputs = batch(processor, e)
                loss = loss_forward(model, inputs)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite loss')
                losses.append(float(loss.detach()))
                (loss / len(train)).backward()
                del inputs, loss
            norm = torch.nn.utils.clip_grad_norm_(list(params.values()), 1)
            assert math.isfinite(float(norm)) and float(norm) > 0
            optimizer.step()
            row = {'step': step, 'loss': sum(losses)/len(losses), 'gradient_norm': float(norm),
                   'visual_forward_calls': len(visual_calls),
                   'peak_vram_mib': torch.cuda.max_memory_allocated()/2**20,
                   'elapsed_sec': time.monotonic()-started}
            with (out / 'training.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row)+'\n')
            print(json.dumps(row), flush=True)
            if step == 3:
                delta = max(float((p.detach().cpu()-initial[n]).abs().max()) for n, p in params.items())
                assert delta > 0 and len(visual_calls) >= 12
                model.save_pretrained(out / 'smoke-adapter')
                model.eval()
                with torch.no_grad():
                    real = float(loss_forward(model, batch(processor, train[0])))
                    blank = float(loss_forward(model, batch(processor, train[0], blank=True)))
                assert abs(real-blank) > 1e-6
                # Verify serialization against actual adapter tensors without changing trainability.
                from peft.utils.save_and_load import get_peft_model_state_dict
                from safetensors.torch import load_file
                saved = load_file(out / 'smoke-adapter/adapter_model.safetensors')
                current = get_peft_model_state_dict(model)
                assert all(torch.equal(saved[k], v.detach().cpu()) for k, v in current.items())
                write(out / 'smoke.json', {'passed': True, 'steps': step, 'delta': delta,
                      'real_loss': real, 'blank_loss': blank, 'serialized_tensors_equal': True,
                      'peak_vram_mib': row['peak_vram_mib']})
                smoke_done = True
            if step > 3:
                assert smoke_done
            if step in check_steps:
                predictions = predict(train, f'train_step_{step}')
                model.save_pretrained(out / f'checkpoint-{step}')
                if metrics(predictions)['correct'] == len(train):
                    passed = True
                    if args.mode == 'memorize':
                        break
        model.save_pretrained(out / 'adapter')
        model.load_adapter(str(out / 'adapter'), adapter_name='reloaded', is_trainable=False)
        model.set_adapter('reloaded')
        reloaded = predict(train, 'reloaded_train')
        passed = passed and metrics(reloaded)['correct'] == len(train)
        predict(train, 'unconstrained_train', constrained=False)
        predict(train, 'blank_train', blank=True)
        predict(train, 'swapped_train', swapped=True)
        if args.mode == 'experiment':
            predict(val, 'adapter_validation')
        write(out / 'result.json', {'fingerprint': fingerprint, 'completed': True,
              'memorization_passed': passed, 'optimizer_steps': step,
              'train_examples': len(train), 'validation_examples': len(val),
              'visual_forward_calls': len(visual_calls),
              'peak_vram_mib': torch.cuda.max_memory_allocated()/2**20,
              'deployment_allowed': False, 'validation_positive_examples': 0,
              'limitation': 'Memorization only. Short diagnostic event protocol differs from API. '
                            'No positive validation example for this precise event; recall unmeasurable.'})
    except Exception as exc:
        write(out / 'failure.json', {'completed': False, 'type': type(exc).__name__, 'error': str(exc)})
        raise
    finally:
        hook.remove()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['memorize', 'experiment'], default='memorize')
    parser.add_argument('--run-name', default='', help='New named run; same name for memorize and experiment')
    main(parser.parse_args())
