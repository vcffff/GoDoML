"""Local Qwen3-VL QLoRA SFT. Assistant-only loss; audited smoke gate."""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_cash_sft import read, sha, write

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def load_base(model_dir):
    import torch
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen3VLForConditionalGeneration,
    )
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no fake CPU training fallback')
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                              bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=torch.bfloat16,
                              llm_int8_skip_modules=['visual', 'lm_head'])
    processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True,
                                              min_pixels=32 * 32, max_pixels=224 * 224)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(model_dir), local_files_only=True, quantization_config=quant,
        device_map={'': 0}, dtype=torch.bfloat16, attn_implementation='sdpa')
    return processor, model


def batch(processor, example, target=True, blank=False):
    import torch
    from PIL import Image
    content = []
    for i, (path, timestamp) in enumerate(zip(example['images'], example['timestamps'], strict=True)):
        with Image.open(ROOT / path) as source:
            picture = source.convert('RGB')
        if blank:
            picture = Image.new('RGB', picture.size)
        content.extend([{'type': 'text', 'text': f'Frame {i}, time {timestamp:.3f} seconds:'},
                        {'type': 'image', 'image': picture}])
    content.append({'type': 'text', 'text': example['prompt']})
    messages = [{'role': 'user', 'content': content}]
    prefix = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors='pt')
    if not target:
        return prefix.to('cuda')
    messages.append({'role': 'assistant', 'content': [
        {'type': 'text', 'text': json.dumps(example['target'], ensure_ascii=False)}]})
    full = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
                                         return_dict=True, return_tensors='pt')
    length = prefix['input_ids'].shape[1]
    if not torch.equal(prefix['input_ids'], full['input_ids'][:, :length]):
        raise RuntimeError('Chat template prefix mismatch; assistant mask would be wrong')
    labels = full['input_ids'].clone()
    labels[:, :length] = -100
    labels[full['attention_mask'] == 0] = -100
    assert (labels != -100).any() and (labels[:, :length] == -100).all()
    assert full['pixel_values'].numel() > 0
    assert full['image_grid_thw'].shape[0] == len(example['images'])
    full['labels'] = labels
    return full.to('cuda')


def loss_forward(model, inputs):
    # Only materialize logits on supervised positions (vocabulary is 152k).
    # All prompt and image tokens still pass through the full transformer.
    import torch
    labels = inputs['labels']
    indices = torch.where(labels[0] != -100)[0]
    positions = indices - 1
    kwargs = {k: v for k, v in inputs.items() if k != 'labels'}
    output = model(**kwargs, logits_to_keep=positions, use_cache=False)
    return torch.nn.functional.cross_entropy(output.logits[0].float(), labels[0, indices])


def validation_loss(model, processor, examples):
    import torch
    model.eval()
    values = []
    with torch.no_grad():
        for example in examples:
            value = float(loss_forward(model, batch(processor, example)).item())
            if not math.isfinite(value):
                raise RuntimeError('Non-finite validation loss')
            values.append(value)
    model.train()
    return sum(values) / len(values)


def main(args):
    import torch
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    out = ROOT / args.output
    examples_path = out / 'examples.json'
    examples = read(examples_path)
    train = [e for e in examples if e['split'] == 'train']
    val = [e for e in examples if e['split'] == 'validation']
    assert train and val
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    fingerprint = {'examples_sha256': sha(examples_path), 'rank': args.rank,
                   'base_config_sha256': sha(ROOT / args.model / 'config.json'),
                   'script_sha256': sha(Path(__file__)), 'precision': 'nf4-bfloat16',
                   'gradient_accumulation': args.accumulation}
    if args.mode == 'train':
        gate = read(out / 'smoke_report.json')
        if gate.get('passed') is not True or gate['fingerprint'] != fingerprint:
            raise RuntimeError('Run a successful matching smoke first')
    write(out / f'{args.mode}_config.json', {**vars(args), 'fingerprint': fingerprint})
    processor, base = load_base(ROOT / args.model)
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True,
                                          gradient_checkpointing_kwargs={'use_reentrant': False})
    # PEFT promotes non-quantized weights to fp32; restore the frozen vision tower to bf16.
    base.model.visual.to(dtype=torch.bfloat16)
    config = LoraConfig(r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.0,
                        target_modules=r'.*language_model\.layers\.\d+\.self_attn\.(q_proj|v_proj)$',
                        bias='none', task_type='CAUSAL_LM')
    model = get_peft_model(base, config)
    model.config.use_cache = False
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all('lora_' in n and 'visual' not in n for n in trainable)
    optimizer = torch.optim.AdamW(list(trainable.values()), lr=args.lr)
    initial = {n: p.detach().cpu().clone() for n, p in trainable.items()}
    write(out / 'hardware.json', {'gpu': torch.cuda.get_device_name(),
          'vram_bytes': torch.cuda.get_device_properties(0).total_memory,
          'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
          'trainable_parameters': sum(p.numel() for p in trainable.values()),
          'base_and_visual_frozen': True})
    torch.cuda.reset_peak_memory_stats()
    visual_calls = []
    hook = model.base_model.model.model.visual.register_forward_hook(
        lambda module, inputs, output: visual_calls.append(True))
    report = {'fingerprint': fingerprint, 'passed': False}
    logs = out / f'{args.mode}_log.jsonl'
    best = float('inf')
    started = time.monotonic()
    model.train()
    steps = args.steps if args.mode == 'smoke' else math.ceil(len(train) * args.epochs / args.accumulation)
    order = list(train)
    random.shuffle(order)
    used = set()
    try:
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for micro in range(args.accumulation):
                example = order[(step * args.accumulation + micro) % len(order)]
                inputs = batch(processor, example)
                loss = loss_forward(model, inputs)
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite training loss')
                losses.append(float(loss.item()))
                (loss / args.accumulation).backward()
                used.add(example['id'])
                del inputs, loss
            norm = torch.nn.utils.clip_grad_norm_(list(trainable.values()), 1.0)
            if not torch.isfinite(norm) or norm.item() <= 0:
                raise RuntimeError('LoRA gradients must be finite and nonzero')
            optimizer.step()
            entry = {'step': step + 1, 'loss': sum(losses) / len(losses),
                     'gradient_norm': float(norm), 'visual_forward_calls': len(visual_calls),
                     'peak_vram_mib': torch.cuda.max_memory_allocated() / 2**20,
                     'elapsed_sec': time.monotonic() - started}
            if args.mode == 'train' and ((step + 1) % args.eval_every == 0 or step == steps - 1):
                entry['validation_loss'] = validation_loss(model, processor, val)
                checkpoint = out / f'checkpoint-{step + 1}'
                model.save_pretrained(checkpoint)
                torch.save({'optimizer': optimizer.state_dict(), 'step': step + 1,
                            'rng': torch.get_rng_state()}, checkpoint / 'training_state.pt')
                if entry['validation_loss'] < best:
                    best = entry['validation_loss']
                    model.save_pretrained(out / 'best-adapter')
                    write(out / 'best-adapter/training_contract.json',
                          {'protocol': 'cash-rule-v1', 'staged_context': True,
                           'rules_sha256': sha(ROOT / 'data/rules_v2.json'),
                           'frames': len(train[0]['images']), 'max_side': 224,
                           'validation_loss': best, 'step': step + 1})
            with logs.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(entry) + '\n')
            print(json.dumps(entry), flush=True)
        delta = max(float((p.detach().cpu() - initial[n]).abs().max()) for n, p in trainable.items())
        assert delta > 0 and visual_calls
        if args.mode == 'smoke':
            saved = out / 'smoke-adapter'
            model.save_pretrained(saved)
            model.eval()
            with torch.no_grad():
                real_loss = float(loss_forward(model, batch(processor, train[0])))
                blank_loss = float(loss_forward(model, batch(processor, train[0], blank=True)))
            assert abs(real_loss - blank_loss) > 1e-6, 'Visual ablation had no effect'
            # Reload into the same base with a second adapter and compare trained tensor values.
            model.load_adapter(str(saved), adapter_name='reload_check', is_trainable=False)
            model.set_adapter('reload_check')
            with torch.no_grad():
                reload_loss = float(loss_forward(model, batch(processor, train[0])))
            assert abs(real_loss - reload_loss) < 1e-3
            report.update(passed=True, optimizer_steps=steps, adapter_max_delta=delta,
                          visual_forward_calls=len(visual_calls), real_image_loss=real_loss,
                          blank_image_loss=blank_loss, reloaded_loss=reload_loss,
                          adapter_reloaded=True, actual_train_examples=len(used))
            write(out / 'smoke_report.json', report)
        else:
            write(out / 'training_result.json', {'completed': True, 'optimizer_steps': steps,
                  'actual_train_examples': len(used), 'validation_examples': len(val),
                  'best_validation_loss': best, 'elapsed_sec': time.monotonic() - started,
                  'adapter_max_delta': delta, 'peak_vram_mib': torch.cuda.max_memory_allocated() / 2**20})
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error=str(exc), completed=False)
        write(out / f'{args.mode}_failure.json', report)
        raise
    finally:
        hook.remove()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['smoke', 'train'], required=True)
    parser.add_argument('--output', default='outputs/cash-lora')
    parser.add_argument('--model', default='models/Qwen3-VL-2B-Instruct')
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--accumulation', type=int, default=4)
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval-every', type=int, default=8)
    main(parser.parse_args())
