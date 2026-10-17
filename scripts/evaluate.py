"""Post-hoc evaluation. Model responses and labels stay in local artifacts."""
import argparse
import csv
import hashlib
import html
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openpyxl import load_workbook

from cashml.config import Settings
from cashml.schema import AnalysisResult
from review_labels import normalized, seconds


def parse_label(value):
    text = ' '.join(str(value).strip().casefold().split())
    if text in ('без нарушения', 'без нарушений', 'no_violation', '0', 'нет'):
        return 'no_violation'
    if text in ('нарушение', 'нарушения', 'violation', '1', 'да', 'с нарушением'):
        return 'violation'
    raise ValueError('Unknown ground-truth label')


def calculate(rows, tolerance=5.0):
    """One ground-truth timestamp per video; temporal matches make no type claim."""
    confusion = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
    errors = {'violation': 0, 'no_violation': 0}
    pending = {'violation': 0, 'no_violation': 0}
    counts = {'violation': 0, 'no_violation': 0}
    matches = []
    nearest = []
    requests = []
    unmatched_gold = 0
    unmatched_predictions = 0
    positive_without_timestamp = 0
    successful = 0
    for row in rows:
        truth = row['true_label']
        counts[truth] += 1
        record = row['request']
        requests.append({key: record.get(key) for key in
                         ('case_id', 'outcome', 'http_status', 'elapsed_sec', 'sla_pass', 'peak_vram_mib')})
        prediction = row.get('prediction')
        if record.get('outcome') in ('not_started', 'in_flight_unknown'):
            pending[truth] += 1
            continue
        if prediction not in ('violation', 'no_violation') or record.get('outcome') != 'success':
            errors[truth] += 1
            if truth == 'violation' and row['true_time_sec'] is not None:
                unmatched_gold += 1
            continue
        successful += 1
        positive = truth == 'violation'
        predicted_positive = prediction == 'violation'
        key = ('TP' if predicted_positive else 'FN') if positive else ('FP' if predicted_positive else 'TN')
        confusion[key] += 1
        times = row.get('predicted_times_sec', [])
        unmatched_predictions += len(times)
        timestamp = row['true_time_sec']
        if positive and timestamp is None:
            positive_without_timestamp += 1
        if positive and timestamp is not None:
            if times:
                error = min(abs(value - timestamp) for value in times)
                if predicted_positive:
                    nearest.append({'case_id': row['case_id'], 'absolute_error_sec': round(error, 3)})
                if error <= tolerance:
                    matches.append({'case_id': row['case_id'], 'absolute_error_sec': round(error, 3)})
                    unmatched_predictions -= 1
                else:
                    unmatched_gold += 1
            else:
                unmatched_gold += 1
    tp, fp, fn = confusion['TP'], confusion['FP'], confusion['FN']
    vram = []
    for request in requests:
        try:
            value = float(request['peak_vram_mib'])
            if math.isfinite(value):
                vram.append(value)
        except (TypeError, ValueError):
            pass
    return {
        'quality_validated_by_human': False,
        'scope': 'provisional video-level predictions; event categories not evaluated',
        'listed_cases': len(rows), 'evaluated_cases': len(rows) - sum(pending.values()),
        'successful_cases': successful,
        'truth_class_counts': counts, 'errors_by_true_class': errors,
        'pending_by_true_class': pending, 'pending_cases': sum(pending.values()),
        'completed_cases': len(rows) - sum(pending.values()),
        'successful_by_true_class': {key: counts[key] - errors[key] - pending[key] for key in counts},
        'confusion_successful_requests_only': confusion,
        'precision': tp / (tp + fp) if tp + fp else None,
        'recall_successful_requests_only': tp / (tp + fn) if tp + fn else None,
        'end_to_end_recall_lower_bound': tp / counts['violation'] if counts['violation'] else None,
        'recall_lower_bound_definition': 'TP / all known positive videos; pending and errors remain unknown, not FN',
        'temporal_match_tolerance_sec': tolerance,
        'temporal_matching': 'nearest predicted timestamp to one gold timestamp per video; no type verification',
        'matched_absolute_errors_sec': matches,
        'nearest_absolute_errors_successful_TP_sec': nearest,
        'unmatched_gold_timestamps': unmatched_gold,
        'unmatched_predicted_timestamps': unmatched_predictions,
        'successful_positive_cases_without_gold_timestamp': positive_without_timestamp,
        'max_peak_vram_mib': max(vram) if vram else None,
        'missing_vram_measurements': sum(request.get('peak_vram_mib') is None for request in requests),
        'sla_failures': sum(request.get('sla_pass') is False for request in requests),
        'requests': requests,
    }


def read_labels(settings):
    files = sorted(settings.labels_dir.rglob('*.xlsx'))
    if len(files) != 1:
        raise ValueError('Expected exactly one XLSX')
    workbook = load_workbook(files[0], read_only=True, data_only=True)
    try:
        iterator = iter(workbook['labels'].values)
        headers = [str(value).strip().lower() for value in next(iterator)]
        if headers != ['наименование файла', 'таймстемп', 'нарушение / без нарушения']:
            raise ValueError('Unexpected label headers')
        result = {}
        for name, timestamp, label in iterator:
            if name is None:
                continue
            key = normalized(name)
            if key in result:
                raise ValueError('Duplicate label video')
            result[key] = {'true_label': parse_label(label), 'true_time_sec': seconds(timestamp)}
        return result
    finally:
        workbook.close()


def collect(destination, labels, partial=False):
    manifest = json.loads((destination / 'manifest.json').read_text(encoding='utf-8'))
    rows = []
    for case in manifest['cases']:
        path = destination / case['case_id'] / 'request.json'
        if not path.exists():
            if not partial:
                raise ValueError('Incomplete benchmark: use --partial for explicit partial assessment')
            record = {'case_id': case['case_id'], 'outcome': 'not_started'}
        else:
            record = json.loads(path.read_text(encoding='utf-8'))
            if record.get('outcome') == 'in_flight_unknown' and not partial:
                raise ValueError('In-flight benchmark: use --partial for explicit partial assessment')
        truth = labels[normalized(Path(case['video_path']).name)]
        row = {**case, **truth, 'request': record, 'prediction': None,
               'predicted_times_sec': [], 'response': None}
        if record.get('outcome') == 'success':
            response = AnalysisResult.model_validate_json(
                (path.parent / 'response.json').read_bytes()).model_dump()
            row.update(prediction=response['verdict'], response=response,
                       predicted_times_sec=[event['timecode_sec'] for event in response['violations']])
        rows.append(row)
    return rows, len(manifest['cases'])


def write_manual(destination, rows):
    fieldnames = ['case_id', 'video', 'true_label', 'prediction', 'true_time_sec',
                  'predicted_times_sec', 'description_correct', 'category_confirmed', 'response_file']
    csv_path = destination / 'manual_review.csv'
    judgments = {}
    if csv_path.exists():
        with csv_path.open(newline='', encoding='utf-8-sig') as stream:
            judgments = {entry['case_id']: {
                key: entry.get(key, '') for key in ('description_correct', 'category_confirmed')
            } for entry in csv.DictReader(stream)}
    temporary = csv_path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({'case_id': row['case_id'], 'video': Path(row['video_path']).name,
                             'true_label': row['true_label'],
                             'prediction': row['prediction'] or (
                                 'pending' if row['request'].get('outcome') in ('not_started', 'in_flight_unknown')
                                 else 'error / unknown'),
                             'true_time_sec': row['true_time_sec'],
                             'predicted_times_sec': '; '.join(map(str, row['predicted_times_sec'])),
                             'description_correct': judgments.get(row['case_id'], {}).get('description_correct', ''),
                             'category_confirmed': judgments.get(row['case_id'], {}).get('category_confirmed', ''),
                             'response_file': f"{row['case_id']}/response.json"})
    temporary.replace(csv_path)
    escape = lambda value: html.escape(str(value), quote=True)
    sections = []
    for row in rows:
        case_id = row['case_id']
        revision = hashlib.sha256(json.dumps(row['request'], sort_keys=True).encode()).hexdigest()[:16]
        details = []
        for index, event in enumerate((row['response'] or {}).get('violations', []), 1):
            images = ''.join('<figure><img src="data:image/jpeg;base64,' + escape(frame['image_base64'])
                             + '"><figcaption>' + escape(frame['timestamp_sec']) + ' s</figcaption></figure>'
                             for frame in event['evidence_frames'])
            details.append(f'<h3>Событие {index}: {escape(event["type"])}</h3>'
                           f'<p>{escape(event["description"])}</p>{images}')
        sections.append(
            f'<tr><td><a href="#{case_id}">{escape(Path(row["video_path"]).name)}</a></td>'
            f'<td>{escape(row["true_label"])}</td><td>{escape(row["prediction"] or ("pending" if row["request"].get("outcome") in ("not_started", "in_flight_unknown") else "error / unknown"))}</td>'
            f'<td>{escape(row["true_time_sec"])}</td><td>{escape(row["predicted_times_sec"])}</td>'
            f'<td><select data-key="{case_id}-{revision}-description"><option value="">Не проверено</option>'
            '<option>Верно</option><option>Неверно</option></select></td>'
            f'<td><select data-key="{case_id}-{revision}-category"><option value="">Не проверено</option>'
            '<option>Подтверждена</option><option>Не подтверждена</option></select></td></tr>')
        row['_details_html'] = (
            f'<section id="{case_id}"><h2>{escape(Path(row["video_path"]).name)}</h2>'
            f'<p><a href="{case_id}/response.json">Полный локальный ответ</a></p>'
            + (f'<p><a href="{case_id}/assessments.json">Структурированные оценки критериев и локальный аудит</a></p>'
               if (destination / case_id / 'assessments.json').exists() else '')
            +
            f'<video controls preload="none" src="{escape(Path(row["video_path"]).as_uri())}"></video>'
            + ''.join(details) + '</section>')
    document = (
        '<!doctype html><html lang="ru"><meta charset="utf-8">'
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
        'img-src data:; media-src file:; style-src \'unsafe-inline\'; script-src \'unsafe-inline\'">'
        '<title>Ручная проверка полного анализа</title><style>'
        'body{font:15px system-ui;margin:24px}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:6px}'
        'section{border-top:1px solid #aaa;margin-top:24px}video{max-width:800px}img{max-width:300px}'
        'figure{display:inline-block}p{white-space:pre-wrap}</style>'
        '<h1>Ручная проверка полного анализа</h1><p>Качество и категории не подтверждены. '
        'Видео и ответы остаются локальными. Ошибки не считаются «без нарушения». '
        'Выбор сохраняется в этом браузере; CSV содержит пустые поля для ручного заполнения. '
        'Для MOV при отсутствии воспроизведения используйте локальный видеоплеер.</p>'
        '<table><thead><tr><th>Видео</th><th>Истинная метка</th><th>Предсказание</th>'
        '<th>Истинный таймкод, с</th><th>Предсказанные таймкоды, с</th>'
        '<th>Описание верно / неверно</th><th>Категория подтверждена / не подтверждена</th>'
        '</tr></thead><tbody>' + ''.join(sections) + '</tbody></table>'
        + ''.join(row.pop('_details_html') for row in rows)
        + '<script>document.querySelectorAll("select").forEach(e=>{try{'
        'const key=location.pathname+":"+e.dataset.key;e.value=localStorage.getItem(key)||"";'
        'e.onchange=()=>localStorage.setItem(key,e.value)}catch(_){}})</script></html>')
    (destination / 'manual_review.html').write_text(document, encoding='utf-8')


def write_request_metrics(destination, rows):
    fields = ['case_id', 'http_status', 'outcome', 'elapsed_sec', 'peak_vram_mib', 'sla_pass']
    with (destination / 'request_metrics.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row['request'].get(key) for key in fields} for row in rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    try:
        destination = args.run_dir.resolve()
        # Labels are accessed only now, after the independent HTTP runner has produced responses.
        rows, expected = collect(destination, read_labels(Settings.from_env()), args.partial)
        snapshot = json.loads((destination / 'config_snapshot.json').read_text(encoding='utf-8'))
        summary = calculate(rows, snapshot['temporal_match_tolerance_sec'])
        summary.update(expected_cases=expected, partial=summary['pending_cases'] > 0)
        (destination / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        write_manual(destination, rows)
        write_request_metrics(destination, rows)
        # Safe aggregate output: no source filenames, video contents or model descriptions.
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except Exception as exc:  # noqa: BLE001 -- privacy boundary: no response contents in errors
        print(f'Evaluation stopped: {type(exc).__name__}; inspect local configuration.')
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
