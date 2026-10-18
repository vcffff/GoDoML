"""Build a local, reviewable diagnostic report from measured artifacts."""
import argparse
import csv
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_cash_sft import read, write


def main(run_name=''):
    out = ROOT / 'outputs/cash-diagnostic'
    runs = out / 'runs' / run_name if run_name else out
    counts = read(out / 'distribution.json')
    states = ['violation', 'no_violation', 'unknown']
    with (out / 'distribution.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['rule_id'] + [f'{split}_{state}' for split in ['train', 'validation'] for state in states])
        for row in counts:
            writer.writerow([row['rule_id']] + [row[split][state] for split in ['train', 'validation'] for state in states])
    totals = {split: {state: sum(row[split][state] for row in counts) for state in states}
              for split in ['train', 'validation']}
    summary = {'original_target_counts': totals, 'deployment_changed': False,
               'visibility_human_verified': False, 'scope': read(out / 'preparation.json')}
    for mode in ['memorize', 'experiment']:
        if (runs / mode / 'result.json').exists():
            summary[mode] = read(runs / mode / 'result.json')
            summary[mode]['tensor_audit'] = read(runs / mode / 'tensor_audit.json')
            summary[mode]['processor_note'] = (
                'Initial run: source frames max_side336, inherited max_pixels50176; '
                'see actual grid. Corrected runs initialize the processor with max_pixels112896.'
                if not run_name else 'Processor initialized with max_pixels112896; see actual grid.')
            for phase in ['base_train', 'reloaded_train', 'base_validation', 'adapter_validation',
                          'unconstrained_train', 'blank_train', 'swapped_train']:
                path = runs / mode / f'{phase}.json'
                if path.exists():
                    data = read(path)
                    summary[mode][phase] = {**data['metrics'],
                        'mean_elapsed_sec': sum(r['elapsed_sec'] for r in data['rows'])/len(data['rows'])}
    if (out / 'loss_mask_reference.json').exists():
        summary['mask_reference_check'] = read(out / 'loss_mask_reference.json')
    summary['run_name'] = run_name or 'initial_pixel_cap'
    suffix = '-' + run_name if run_name else ''
    write(out / f'summary{suffix}.json', summary)
    sections = ['<!doctype html><meta charset="utf-8"><title>Cash diagnostic</title>',
        ('<style>body{font:16px sans-serif;margin:24px}td,th{border:1px solid #aaa;padding:6px}'
         'table{border-collapse:collapse}pre{white-space:pre-wrap}img{max-width:100%}</style>'),
        '<h1>Cash diagnostic — local files only</h1>',
        ('<p>No model deployed. Unknown is a separate class. '
         'Memorization is NOT a generalization test. Human visibility review is pending.</p>'),
        '<a href="contact-sheets.html">All 39 original contact sheets and annotated phase coverage</a>',
        '<h2>Original target distribution: train V / N / U; validation V / N / U</h2>',
        '<table><tr><th>rule_id</th><th>Train</th><th>Validation</th></tr>']
    for row in counts:
        sections.append('<tr><td>' + html.escape(row['rule_id']) + '</td>' + ''.join(
            '<td>' + ' / '.join(str(row[split][s]) for s in states) + '</td>'
            for split in ['train', 'validation']) + '</tr>')
    sections.extend(['</table>', '<h2>Measured results</h2><pre>' + html.escape(
        json.dumps(summary, ensure_ascii=False, indent=2)) + '</pre>',
        '<h2>Context sequences (16 frames, 336 max side)</h2>',
        '<p>Source video names/labels below are audit metadata only; they are absent from model prompts.</p>'])
    triage = read(out / 'local_visibility_triage.json') if (out / 'local_visibility_triage.json').exists() else []
    for e in read(out / 'examples.json'):
        sections.extend(['<h3>' + html.escape(e['audit']['video'] + ' / ' + e['split']) + '</h3>',
            f'<img src="contact-sheets/context/{e["id"]}.jpg">',
            '<pre>' + html.escape(json.dumps({'target': e['target'],
                'sampling': e['audit']['sampling_policy'],
                'source_timestamps': e['audit']['source_timestamps']}, ensure_ascii=False)) + '</pre>'])
        for row in triage:
            if row['id'] == e['id']:
                sections.append('<h4>Unverified local base-model description: ' + row['variant'] + '</h4><pre>'
                                + html.escape('Processor max_pixels: ' + str(row.get('processor_max_pixels'))
                                + '; variant refers to source image size.\n' + row['model_description']) + '</pre>')
    (out / f'report{suffix}.html').write_text('\n'.join(sections), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', default='')
    main(parser.parse_args().run_name)
