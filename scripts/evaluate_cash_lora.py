"""Paired generation on the frozen validation queries, using API evidence checks."""
import argparse
import html
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_cash_sft import read, write


def main(args):
    import torch
    from peft import PeftModel
    from PIL import Image

    from cashml.config import Settings
    from cashml.pipeline import Analyzer
    from cashml.training_protocol import load_rules
    from cashml.video import Frame
    from scripts.train_cash_lora import load_base

    out = ROOT / args.output
    report_name = 'comparison_constrained' if args.constrained else 'comparison'
    examples = [e for e in read(out / 'examples.json') if e['split'] == 'validation']
    processor, base = load_base(ROOT / args.model)
    model = PeftModel.from_pretrained(base, out / 'best-adapter', is_trainable=False).eval()
    rules = load_rules(ROOT / 'data/rules_v2.json')
    analyzer = Analyzer(Settings(mode='local', max_new_tokens=args.max_new_tokens,
                                  staged_context=1, max_side=224, constrained_json=int(args.constrained)))
    analyzer.model = model
    analyzer.processor = processor
    analyzer.ready = True
    rows = []
    for phase in ['base', 'adapter']:
        for example in examples:
            frames = []
            for path, stamp in zip(example['images'], example['timestamps'], strict=True):
                with Image.open(ROOT / path) as source:
                    frames.append(Frame(stamp, source.convert('RGB')))
            key = example['audit']['rule_id']
            analyzer.rules = {key: rules[key]}
            expected = example['target']['assessments'][0]['m']
            row = {'phase': phase, 'id': example['id'], 'rule_id': key,
                   'expected_criterion': expected, 'source_legal_match': example['audit']['legal_match'],
                   'semantic_action_correct': None, 'manual_rule_explanation_correct': None}
            started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            try:
                import contextlib
                with model.disable_adapter() if phase == 'base' else contextlib.nullcontext():
                    result = analyzer._window(frames, time.monotonic() + args.timeout)
                item = result[0]
                prediction = True if analyzer._positive(item) else (
                    'unknown' if analyzer._unknown(item) else False)
                row.update(outcome='success', prediction=prediction,
                           assessment=item.model_dump(), correct=prediction == expected,
                           false_positive=prediction is True and expected is not True)
                marked_time = example['audit']['timecode_sec']
                if prediction is True and expected is True and marked_time is not None:
                    row['first_evidence_time_abs_error_sec'] = abs(
                        min(item.supporting_timestamps_sec) - marked_time)
            except Exception as exc:  # noqa: BLE001 -- record failed generation without a negative verdict
                row.update(outcome='error', error_type=type(exc).__name__, prediction=None)
            row['elapsed_sec'] = time.perf_counter() - started
            row['peak_vram_mib'] = torch.cuda.max_memory_allocated() / 2**20
            write(out / f'{report_name}_wire/{phase}-{example["id"]}.json', analyzer.wire_audit[-1:])
            rows.append(row)
            write(out / f'{report_name}_rows.json', rows)
            print(json.dumps({k: v for k, v in row.items() if k != 'assessment'}), flush=True)
    summary = {}
    for phase in ['base', 'adapter']:
        subset = [r for r in rows if r['phase'] == phase]
        good = [r for r in subset if r['outcome'] == 'success']
        summary[phase] = {'queries': len(subset), 'errors': len(subset) - len(good),
            'criterion_correct': sum(r['correct'] for r in good),
            'false_positives': sum(r['false_positive'] for r in good),
            'uncertain': sum(r['prediction'] == 'unknown' for r in good),
            'mean_elapsed_sec': sum(r['elapsed_sec'] for r in subset) / len(subset),
            'confirmed_rule_hits': sum(r['prediction'] is True and r['expected_criterion'] is True for r in good),
            'confirmed_rule_total': sum(r['expected_criterion'] is True for r in subset)}
        summary[phase]['false_positives_on_negative'] = sum(
            r['prediction'] is True and r['expected_criterion'] is False for r in good)
        summary[phase]['unsupported_positives_on_uncertain'] = sum(
            r['prediction'] is True and r['expected_criterion'] == 'unknown' for r in good)
        summary[phase]['binary_event_counts'] = {
            name: sum(r['expected_criterion'] is actual and r['prediction'] is predicted for r in good)
            for name, actual, predicted in [('TP', True, True), ('TN', False, False),
                                             ('FP', False, True), ('FN', True, False)]}
    summary['limitations'] = [
        'Evaluation is conditional on the queried rule; not unrestricted rule retrieval.',
        'Semantic action/explanation accuracy needs manual review of comparison_rows.json.',
        'Validation used for checkpoint selection; not an independent final test.',
        'Base and adapter use identical NF4 base, frames, rules, prompt and decoding.',
        'Errors and unknown answers are reported separately, never true negatives.']
    summary['decoding'] = 'schema + repetition_penalty 1.05 + no_repeat_ngram 12' if args.constrained else 'greedy'
    write(out / f'{report_name}.json', summary)
    sections = ['<!doctype html><meta charset="utf-8"><title>Local paired validation</title>',
                '<style>body{font:16px sans-serif;margin:24px}img{width:160px}pre{white-space:pre-wrap}</style>',
                '<h1>Local paired validation</h1><p>Semantic review is pending; these are model outputs.</p>']
    for example in examples:
        sections.append('<h2>' + html.escape(example['id']) + '</h2>')
        for path, stamp in zip(example['images'], example['timestamps'], strict=True):
            relative = Path(path).relative_to(Path(args.output)).as_posix()
            sections.append(f'<img src="{html.escape(relative)}" alt="{stamp:.3f}s" title="{stamp:.3f}s">')
        sections.append('<h3>Annotated target</h3><pre>' + html.escape(
            json.dumps(example['target'], ensure_ascii=False, indent=2)) + '</pre>')
        for row in rows:
            if row['id'] == example['id']:
                sections.append('<h3>' + row['phase'] + '</h3><pre>' + html.escape(
                    json.dumps(row, ensure_ascii=False, indent=2)) + '</pre>')
    (out / f'{report_name}_review.html').write_text('\n'.join(sections), encoding='utf-8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='outputs/cash-lora')
    parser.add_argument('--model', default='models/Qwen3-VL-2B-Instruct')
    parser.add_argument('--max-new-tokens', type=int, default=768)
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--constrained', action='store_true')
    main(parser.parse_args())
