"""Frozen development-only comparison; private video/model contents never printed."""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cashml.config import ROOT, Settings

DEST = ROOT / 'results/fixed_subset_20261017'
BASE = ROOT / 'results/benchmark/verification_20261017'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def prepare(destination=DEST, baseline=BASE):
    """Offline evaluation design only: freeze labels separately from inference inputs."""
    if (destination / 'split_manifest.json').exists():
        return read(destination / 'preparation_summary.json')
    from review_labels import normalized
    from scripts.evaluate import read_labels

    labels = read_labels(Settings.from_env())
    cases = read(baseline / 'manifest.json')['cases']
    identities = [str(Path(case['video_path']).resolve()).casefold() for case in cases]
    if len(identities) != len(set(identities)):
        raise ValueError('Duplicate video identity')
    records = {}
    for case in cases:
        path = baseline / case['case_id'] / 'request.json'
        records[case['case_id']] = read(path) if path.exists() else {}
    successful = [case for case in cases if records[case['case_id']].get('outcome') == 'success']
    truth = lambda case: labels[normalized(Path(case['video_path']).name)]
    longest = max(cases, key=lambda case: case['duration_sec'])
    selected = [longest]
    for label in ('violation', 'no_violation'):
        candidates = sorted(cases, key=lambda case: (
            case not in successful, case['duration_sec'], case['video_path']))
        for case in candidates:
            if sum(truth(item)['true_label'] == label for item in selected) >= 3:
                break
            if case not in selected and truth(case)['true_label'] == label:
                selected.append(case)
    if len(selected) != 6:
        raise ValueError('Need three videos per class')
    selected.sort(key=lambda case: (-case['duration_sec'], case['case_id']))
    selected_ids = {case['case_id'] for case in selected}
    development = [{**case, **truth(case)} for case in selected]
    holdout = [{**case, **truth(case)} for case in cases if case['case_id'] not in selected_ids]
    destination.mkdir(parents=True, exist_ok=True)
    write(destination / 'split_manifest.json', {
        'method': 'Frozen before new predictions: longest mandatory; prefer existing successful baseline, then shortest duration per class.',
        'holdout_warning': 'Previously previewed dataset and partial old benchmark; not a pristine unseen dataset. Do not run holdout or tune against holdout predictions.',
        'development': development, 'holdout': holdout})
    write(destination / 'inference_manifest.json', {'cases': [
        {key: case[key] for key in ('case_id', 'video_path', 'duration_sec')} for case in selected]})
    copied = {'violation': 0, 'no_violation': 0}
    missing = []
    for case in selected:
        if case in successful:
            source = baseline / case['case_id']
            target = destination / 'before' / case['case_id']
            target.mkdir(parents=True, exist_ok=False)
            for name in ('response.json', 'request.json'):
                shutil.copy2(source / name, target / name)
            copied[truth(case)['true_label']] += 1
        else:
            missing.append(case['case_id'])
    write(destination / 'missing_baseline_manifest.json', {'cases': [
        {key: case[key] for key in ('case_id', 'video_path', 'duration_sec')}
        for case in selected if case['case_id'] in missing]})
    summary = {'development_cases': len(selected), 'holdout_cases': len(holdout),
               'existing_baselines_by_class': copied, 'missing_baselines': len(missing),
               'longest_in_development': True, 'longest_duration_sec': longest['duration_sec']}
    write(destination / 'preparation_summary.json', summary)
    return summary


def run(destination, phase, url, only_case=None):
    """Inference reads only opaque case inputs; no labels or evaluation manifest."""
    import httpx

    from cashml.schema import AnalysisResult

    if urlparse(url).hostname not in ('127.0.0.1', 'localhost', '::1'):
        raise ValueError('Local service required')
    filename = 'inference_manifest.json' if phase == 'after' else 'missing_baseline_manifest.json'
    cases = read(destination / filename)['cases']
    if only_case is not None:
        cases = [case for case in cases if case['case_id'] == only_case]
        if not cases:
            raise ValueError('Probe case must belong to frozen development inputs')
    with httpx.Client(timeout=330, trust_env=False) as client:
        health = client.get(url.rstrip('/') + '/health').json()
        if health.get('mode') != 'local' or health.get('mock') is not False or not health.get('ready'):
            raise ValueError('Ready local non-mock model required')
        if (health.get('assessment_schema_version') == 2) != (phase == 'after'):
            raise ValueError('Assessment schema does not match comparison phase')
        write(destination / phase / 'health.json', health)
        for case in cases:
            folder = destination / phase / case['case_id']
            record_path = folder / 'request.json'
            if record_path.exists():
                continue  # Never retry an uncertain request automatically.
            record = {'case_id': case['case_id'], 'outcome': 'in_flight_unknown'}
            write(record_path, record)
            started = time.perf_counter()
            try:
                with Path(case['video_path']).open('rb') as source:
                    response = client.post(url.rstrip('/') + '/analyze', files={
                        'file': ('input.video', source, 'application/octet-stream')})
                elapsed = time.perf_counter() - started
                (folder / 'response.json').write_bytes(response.content)
                record.update(http_status=response.status_code, elapsed_sec=round(elapsed, 3),
                              sla_pass=elapsed <= 300, mode=response.headers.get('X-ML-Mode'),
                              peak_vram_mib=response.headers.get('X-ML-Peak-VRAM-MiB'))
                record['outcome'] = 'http_error'
                if response.status_code == 200 and record['mode'] == 'local':
                    AnalysisResult.model_validate_json(response.content)
                    record['outcome'] = 'success'
                elif response.status_code == 500:
                    detail = response.json().get('detail')
                    if isinstance(detail, dict) and detail.get('code') == 'analysis_inconclusive':
                        record['outcome'] = 'inconclusive'
                audit = response.headers.get('X-ML-Assessment-File')
                if audit:
                    path = (ROOT / audit).resolve()
                    if not path.is_relative_to((ROOT / 'results/assessments').resolve()):
                        raise ValueError('Invalid audit path')
                    shutil.copy2(path, folder / 'assessments.json')
            except Exception as exc:  # noqa: BLE001 -- keep private contents out of logs
                record.update(outcome='request_error_unknown', error_type=type(exc).__name__,
                              elapsed_sec=round(time.perf_counter() - started, 3))
            write(record_path, record)
            print(json.dumps(record), flush=True)


def comparison_rows(destination, phase, development):
    rows = []
    for case in development:
        folder = destination / phase / case['case_id']
        path = folder / 'request.json'
        record = read(path) if path.exists() else {'case_id': case['case_id'], 'outcome': 'not_started'}
        row = {**case, 'request': record, 'prediction': None, 'predicted_times_sec': [], 'response': None}
        if record.get('outcome') == 'success':
            response = read(folder / 'response.json')
            if response.get('verdict') in ('violation', 'no_violation'):
                row.update(prediction=response['verdict'], response=response,
                           predicted_times_sec=[event['timecode_sec'] for event in response['violations']])
        rows.append(row)
    return rows


def compare(destination=DEST):
    from review_labels import normalized
    from scripts.evaluate import (
        calculate,
        read_labels,
        write_manual,
        write_request_metrics,
    )

    split = read(destination / 'split_manifest.json')
    # Post-hoc evaluation alone joins XLSX; inference never opens it.
    labels = read_labels(Settings.from_env())
    for case in split['development']:
        truth = labels[normalized(Path(case['video_path']).name)]
        if any(case[key] != truth[key] for key in ('true_label', 'true_time_sec')):
            raise ValueError('Ground truth changed after split freeze')
    summaries = {}
    for phase in ('before', 'after'):
        rows = comparison_rows(destination, phase, split['development'])
        summaries[phase] = calculate(rows)
        summaries[phase]['inconclusive_cases'] = sum(
            row['request'].get('outcome') == 'inconclusive' for row in rows)
        summaries[phase]['processing_error_cases'] = sum(
            row['request'].get('outcome') in ('http_error', 'request_error_unknown', 'mode_error')
            for row in rows)
        coverage = []
        for row in rows:
            audit = destination / phase / row['case_id'] / 'assessments.json'
            coverage.append(bool(read(audit).get('coverage_complete')) if audit.exists()
                            else row['request'].get('outcome') == 'success')
        summaries[phase]['full_video_coverage_cases'] = sum(coverage)
        folder = destination / phase
        folder.mkdir(exist_ok=True)
        write(folder / 'summary.json', summaries[phase])
        write_manual(folder, rows)
        write_request_metrics(folder, rows)
    result = {'quality_validated_by_human': False, 'development_cases': len(split['development']),
              'holdout_cases_not_run': len(split['holdout']), 'before': summaries['before'], 'after': summaries['after']}
    paired = []
    before_rows = comparison_rows(destination, 'before', split['development'])
    after_rows = comparison_rows(destination, 'after', split['development'])
    for before, after in zip(before_rows, after_rows, strict=True):
        paired.append({
            'case_id': before['case_id'],
            'true_label': before['true_label'], 'true_time_sec': before['true_time_sec'],
            'before_outcome': before['request'].get('outcome'),
            'after_outcome': after['request'].get('outcome'),
            'before_prediction': before['prediction'], 'after_prediction': after['prediction'],
            'before_nearest_absolute_error_sec': min(
                (abs(value - before['true_time_sec']) for value in before['predicted_times_sec']), default=None)
            if before['true_time_sec'] is not None else None,
            'after_nearest_absolute_error_sec': min(
                (abs(value - after['true_time_sec']) for value in after['predicted_times_sec']), default=None)
            if after['true_time_sec'] is not None else None,
            'elapsed_delta_sec': round(after['request']['elapsed_sec'] - before['request']['elapsed_sec'], 3)
            if before['request'].get('elapsed_sec') is not None and after['request'].get('elapsed_sec') is not None else None,
        })
    result['paired_requests'] = paired
    result['metric_deltas_after_minus_before'] = {
        key: summaries['after']['confusion_successful_requests_only'][key]
        - summaries['before']['confusion_successful_requests_only'][key] for key in ('TP', 'FP', 'TN', 'FN')}
    result['comparison_warning'] = 'Confusion excludes unknown/error/pending outcomes. Decreased FP or FN is not improvement if coverage decreases. Categories require human review.'
    write(destination / 'comparison.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare', 'run-after', 'run-before-missing', 'compare'):
        modes.add_argument('--' + name, action='store_true')
    parser.add_argument('--destination', type=Path, default=DEST)
    parser.add_argument('--url', default='http://127.0.0.1:8001')
    parser.add_argument('--only-case', help='Protocol probe restricted to one frozen development case')
    args = parser.parse_args()
    try:
        if args.prepare:
            print(json.dumps(prepare(args.destination), indent=2))
        elif args.compare:
            print(json.dumps(compare(args.destination), indent=2))
        else:
            run(args.destination, 'after' if args.run_after else 'before', args.url, args.only_case)
    except Exception as exc:  # noqa: BLE001 -- privacy boundary
        print(f'Fixed subset stopped: {type(exc).__name__}; inspect local artifacts.')
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
