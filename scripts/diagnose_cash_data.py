"""Local-only label/frame audit and fixed-split single-event diagnostic data."""
import copy
import html
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_cash_sft import read, sha, write

OUT = ROOT / 'outputs/cash-diagnostic'
RULE = 'CASH_PREVIOUS_MONEY'
STATES = ['violation', 'no_violation', 'unknown']


def state(value):
    return 'violation' if value is True else 'no_violation' if value is False else 'unknown'


def sheet(example, path):
    from PIL import Image, ImageDraw
    canvas = Image.new('RGB', (4 * 352, 4 * 378), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, (source, stamp) in enumerate(zip(example['images'], example['timestamps'], strict=True)):
        with Image.open(ROOT / source) as image:
            image = image.convert('RGB')
            image.thumbnail((336, 336))
            x, y = (i % 4) * 352, (i // 4) * 378
            canvas.paste(image, (x, y + 26))  # Preserve actual pixel size, never upscale 224.
            draw.text((x, y + 5), f'frame {i}: {stamp:.3f}s ({image.width}x{image.height})', fill='black')
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main():
    from cashml.video import Video
    examples = read(ROOT / 'outputs/cash-lora/examples.json')
    annotations = read(ROOT / 'data/annotations_v2.json')
    records = {r['video']: r for r in annotations['records']}
    manifest = read(ROOT / 'outputs/cash-lora/split_manifest.json')
    write(OUT / 'split_manifest.json', manifest)
    counts = []
    for rule in sorted({e['audit']['rule_id'] for e in examples}):
        row = {'rule_id': rule}
        for split in ['train', 'validation']:
            c = Counter(state(e['target']['assessments'][0]['m']) for e in examples
                        if e['split'] == split and e['audit']['rule_id'] == rule)
            row[split] = {s: c[s] for s in STATES}
        counts.append(row)
    write(OUT / 'distribution.json', counts)
    audit = []
    page = ['<!doctype html><meta charset="utf-8"><title>Local frame audit</title>',
            '<style>body{font:16px sans-serif}img{max-width:100%}pre{white-space:pre-wrap}</style>',
            '<h1>Local frame audit — original 16 / 224</h1>',
            ('<p>Labels below are annotations, not independent visual verification. '
             'One sampled timestamp inside a phase does not prove the action is visible.</p>')]
    for e in examples:
        record = records[e['audit']['video']]
        phases = []
        for obs in record['observations']:
            hits = [i for i, t in enumerate(e['timestamps']) if obs['start_sec'] <= t <= obs['end_sec']]
            phases.append({**obs, 'sampled_indices': hits, 'sample_count': len(hits)})
        row = {'id': e['id'], 'video': record['video'], 'split': e['split'],
               'target': state(e['target']['assessments'][0]['m']), 'phases': phases,
               'largest_gap_sec': max(b-a for a, b in zip(e['timestamps'], e['timestamps'][1:])),
               'visual_semantics_verified': False}
        audit.append(row)
        relative = f'contact-sheets/original/{e["id"]}.jpg'
        sheet(e, OUT / relative)
        page.extend(['<h2>' + html.escape(record['video'] + ' / ' + e['split']) + '</h2>',
                     f'<img src="{relative}">', '<pre>' + html.escape(
                         json.dumps(row, ensure_ascii=False, indent=2)) + '</pre>'])
    write(OUT / 'frame_audit.json', audit)
    (OUT / 'contact-sheets.html').write_text('\n'.join(page), encoding='utf-8')

    # Precisely the customer-transition event, not every event sharing this legal rule.
    positive = {'videos/Х-4.mov', 'videos/Х-38.mov'}
    negative = {'videos/Х-5.mov', 'videos/Х-39.mov', 'videos/Х-19.mov'}
    selected = []
    for original in examples:
        name = original['audit']['video']
        if original['audit']['rule_id'] != RULE or name not in positive | negative:
            continue
        e = copy.deepcopy(original)
        r = records[name]
        assert e['split'] == manifest['assignments'][name]
        with Video(ROOT / 'data' / name) as video:
            if e['split'] == 'validation':
                # Strictly blind to annotation times: whole video, uniform fixed count.
                frames = video.sample(0, video.duration - 1/video.fps, 16, 336)
                policy = 'uniform_whole_video_no_annotation_access'
            else:
                # Equal frame budget per phase includes receipt, transition and aftermath.
                phases = r['observations']
                times = []
                for j, obs in enumerate(phases):
                    n = 16 // len(phases) + int(j < 16 % len(phases))
                    start = max(0, obs['start_sec'])
                    end = min(obs['end_sec'], video.duration - 1/video.fps)
                    times.extend(start + (end-start)*(i+.5)/n for i in range(n))
                frames = [video.frame(t, 336) for t in sorted(times)]
                policy = 'train_only_phase_stratified_full_event_context'
        paths = []
        for i, f in enumerate(frames):
            p = OUT / 'frames' / e['id'] / f'{i:02d}.jpg'
            p.parent.mkdir(parents=True, exist_ok=True)
            f.image.save(p, quality=95)
            paths.append(str(p.relative_to(ROOT)))
        e['images'] = paths
        # Relative ordinary timestamps only; no event marker, source filename or annotations.
        e['timestamps'] = [f.timestamp - frames[0].timestamp for f in frames]
        e['audit']['source_timestamps'] = [f.timestamp for f in frames]
        e['audit']['sampling_policy'] = policy
        e['prompt'] = (
            'Контекст: постановочная учебная кассовая сцена. Рассмотри кадры по порядку. '
            'Проверь одно событие CASH_PREVIOUS_MONEY: деньги первого клиента остаются '
            'на столе, когда кассир начинает обслуживать следующего клиента. '
            'violation: видны получение первых денег, переход к следующему клиенту '
            'и оставшаяся первая пачка. no_violation: видно, что первая пачка убрана '
            'перед следующим клиентом. unknown: последовательность или предметы неразличимы. '
            'Не считай просто присутствие денег нарушением. '
            'Верни только JSON с единственным полем event и одним из трёх значений: '
            'violation, no_violation, unknown. Это диагностика видимого события, '
            'не вывод о юридической ответственности.')
        e['target'] = {'event': 'violation' if name in positive else 'no_violation'}
        selected.append(e)
        sheet(e, OUT / 'contact-sheets/context' / f'{e["id"]}.jpg')
    assert Counter(e['split'] for e in selected) == {'train': 4, 'validation': 1}
    assert len({e['audit']['split_group'] for e in selected}) == len(selected)
    write(OUT / 'examples.json', selected)
    write(OUT / 'preparation.json', {
        'original_examples_sha256': sha(ROOT / 'outputs/cash-lora/examples.json'),
        'train': {'violation': 2, 'no_violation': 2, 'unknown': 0},
        'validation': {'violation': 0, 'no_violation': 1, 'unknown': 0},
        'validation_positive_unavailable': True,
        'excluded_other_event_same_rule': ['videos/Х-8.mov', 'videos/Х-26.mov', 'videos/Х-27.mov'],
        'excluded_conditional': ['videos/Х-15.mov', 'videos/Х-18.mov'],
        'visual_semantics_verified': False,
        'limitation': 'Contact sheets stay local. Human visibility review is pending. '
                      'Diagnostic memorization cannot establish event recognition or generalization.',
        'protocol': 'diagnostic-event-v1; intentionally NOT deployable in Analyzer',
        'sampling': '16 frames, side336; train phase-stratified; validation full-video uniform'})
    print('Prepared distribution, 39 original contact sheets, 5 context sheets, 4 train / 1 validation.')


if __name__ == '__main__':
    main()
