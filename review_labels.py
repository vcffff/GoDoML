"""Build a private human-review report from labels; never used by the service."""
import argparse
import datetime
import html
import json
import re
import time
from pathlib import Path

from openpyxl import load_workbook

from cashml.config import ROOT, Settings
from cashml.pipeline import Analyzer
from cashml.video import Video


def seconds(value):
    if value is None or str(value).strip() == '':
        return None
    if isinstance(value, datetime.time):
        return value.hour * 3600 + value.minute * 60 + value.second
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    if ':' in text:
        parts = [float(p.replace(',', '.')) for p in text.split(':')]
        if len(parts) not in (2, 3):
            raise ValueError('Unsupported timestamp')
        return sum(v * 60**i for i, v in enumerate(reversed(parts)))
    numbers = [float(p.replace(',', '.')) for p in re.findall(r'\d+(?:[.,]\d+)?', text)]
    if len(numbers) == 2 and 'мин' in text:
        return numbers[0] * 60 + numbers[1]
    if len(numbers) == 1:
        return numbers[0] * 60 if 'мин' in text else numbers[0]
    raise ValueError('Unsupported timestamp')


def normalized(value):
    return Path(str(value)).stem.casefold().replace('x', 'х')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=0, help='0 = all timestamped rows')
    parser.add_argument('--frames', type=int, default=4)
    parser.add_argument('--radius', type=float, default=3)
    args = parser.parse_args()
    settings = Settings.from_env()
    settings.mode = 'local'
    out = ROOT / 'results/label_review'
    out.mkdir(parents=True, exist_ok=True)
    workbooks = sorted(settings.labels_dir.rglob('*.xlsx'))
    if len(workbooks) != 1:
        raise SystemExit('Expected exactly one XLSX; select labels folder via ML_LABELS_DIR')
    workbook = load_workbook(workbooks[0], read_only=True, data_only=True)
    sheet = workbook['labels']
    rows = list(sheet.values)
    workbook.close()
    headers = [str(v).strip().lower() for v in rows[0]]
    expected = ['наименование файла', 'таймстемп', 'нарушение / без нарушения']
    if headers != expected:
        raise SystemExit('Unexpected XLSX headers; update explicit column mapping')
    videos = {normalized(p.name): p for p in settings.videos_dir.iterdir() if p.is_file()}
    jobs = []
    counts = {'rows': len(rows) - 1, 'without_timestamp': 0, 'errors': 0, 'completed': 0}
    for index, row in enumerate(rows[1:], start=2):
        try:
            timestamp = seconds(row[1])
            if timestamp is None:
                counts['without_timestamp'] += 1
                continue
            jobs.append((index, row, timestamp))
        except ValueError:
            counts['errors'] += 1
    if args.limit:
        jobs = jobs[:args.limit]
    analyzer = Analyzer(settings)
    started = time.monotonic()
    analyzer.load()
    sections = []
    manifest = []
    for index, row, timestamp in jobs:
        case = out / f'row_{index:03d}'
        case.mkdir(exist_ok=True)
        try:
            path = videos[normalized(row[0])]
            with Video(path, settings.max_duration_sec) as video:
                if not 0 <= timestamp < video.duration:
                    raise ValueError('Label timestamp outside video')
                begin = max(0.0, timestamp - args.radius)
                end = min(video.duration - 1 / video.fps, timestamp + args.radius)
                frames = video.sample(begin, end, args.frames, settings.max_side)
            text = analyzer.infer(
                frames, 'Describe only visible actions across these chronological sampled frames. '
                'Do not identify people. Do not infer account entries, intent or unseen actions. '
                'Do not classify legal violations. State uncertainty. Write in Russian.',
                time.monotonic() + settings.timeout_sec)
            (case / 'description.txt').write_text(text, encoding='utf-8')
            figures = []
            for i, frame in enumerate(frames):
                frame.image.save(case / f'frame_{i}.jpg', quality=85)
                figures.append(f'<figure><img src="{case.name}/frame_{i}.jpg">'
                               f'<figcaption>{frame.timestamp:.3f} s</figcaption></figure>')
            # Native browser playback uses original local video, never uploads anywhere.
            uri = html.escape(path.resolve().as_uri(), quote=True)
            sections.append(f'<section><h2>{html.escape(str(row[0]))}</h2>'
                            f'<p>Разметка: {html.escape(str(row[2]))}; таймкод {timestamp:.3f} s</p>'
                            f'<video controls preload="none" src="{uri}#t={begin:.3f},{end:.3f}"></video>'
                            + ''.join(figures) + '<pre>' + html.escape(text) + '</pre>'
                            '<p>Ваши заметки для подтверждения категории:</p>'
                            f'<textarea data-key="row-{index}" rows="4"></textarea></section>')
            manifest.append({'row': index, 'video': str(path), 'label': row[2],
                             'label_time_sec': timestamp,
                             'sample_timestamps_sec': [f.timestamp for f in frames],
                             'status': 'ok'})
            counts['completed'] += 1
        except Exception as exc:  # noqa: BLE001 -- privacy boundary, details only local
            (case / 'error.txt').write_text(str(exc), encoding='utf-8')
            manifest.append({'row': index, 'status': 'error', 'error_type': type(exc).__name__})
            counts['errors'] += 1
        # Checkpoint report after each row; a failed case never hides the rest.
        document = ('<!doctype html><html lang="ru"><meta charset="utf-8">'
                    '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
                    'img-src \'self\' file:; media-src file:; style-src \'unsafe-inline\'; '
                    'script-src \'unsafe-inline\'">'
                    '<title>Локальная проверка категорий</title>'
                    '<style>body{font:16px system-ui;max-width:1100px;margin:30px auto}'
                    'section{border-top:1px solid #aaa;margin:30px 0}figure{display:inline-block;margin:8px}'
                    'img{max-width:320px}video{display:block;max-width:700px}pre{white-space:pre-wrap}'
                    'textarea{width:95%}</style>'
                    '<h1>Проверка категорий человеком</h1><p>Ответы локальной Qwen описывают '
                    'наблюдения. Они не являются подтверждёнными категориями или юридическими выводами. '
                    'Отчёт использует разметку только для выбора фрагментов. Другие видео оставлены '
                    'для независимой проверки. Если MOV не воспроизводится в браузере, используйте '
                    'кадры и оригинал в локальном плеере. Заметки сохраняются в этом браузере.</p>'
                    + ''.join(sections) + '<script>document.querySelectorAll("textarea").forEach(t=>{'
                    'try{t.value=localStorage.getItem(t.dataset.key)||"";'
                    't.oninput=()=>localStorage.setItem(t.dataset.key,t.value)}catch(e){}})</script></html>')
        (out / 'report.html').write_text(document, encoding='utf-8')
        (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        counts['seconds'] = round(time.monotonic() - started, 2)
        (out / 'metrics.json').write_text(json.dumps(counts, indent=2), encoding='utf-8')
        print(json.dumps(counts), flush=True)
    import torch
    counts['peak_vram_mib'] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
    (out / 'metrics.json').write_text(json.dumps(counts, indent=2), encoding='utf-8')
    print('Report: results/label_review/report.html')


if __name__ == '__main__':
    main()
