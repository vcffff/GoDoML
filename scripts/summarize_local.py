"""Join frozen labels after local inference; emit only safe evaluation aggregates."""
import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cashml.schema import AnalysisResult
from scripts.evaluate import calculate

PENDING = {'not_started', 'in_flight_unknown'}
OUTCOMES = PENDING | {'success', 'inconclusive', 'timeout', 'processing_error'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def finite_seconds(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) and number >= 0 else None
    except (TypeError, ValueError, OverflowError):
        return None


def selected_cases(split, subset):
    if subset not in ('development', 'holdout', 'all'):
        raise ValueError('Unknown subset')
    all_cases = split['development'] + split['holdout']
    ids = [case['case_id'] for case in all_cases]
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r'case_\d{3,}', value) for value in ids):
        raise ValueError('Frozen split requires distinct opaque case ids')
    for case in all_cases:
        if case['true_label'] not in ('violation', 'no_violation'):
            raise ValueError('Unknown frozen label')
        timestamp = case['true_time_sec']
        if timestamp is not None and finite_seconds(timestamp) is None:
            raise ValueError('Invalid frozen timestamp')
    return all_cases if subset == 'all' else split[subset]


def collect_local(run, cases, partial=False):
    rows = []
    for case in cases:
        folder = run / case['case_id']
        request_path = folder / 'request.json'
        if request_path.exists():
            record = read(request_path)
            if record.get('case_id') != case['case_id']:
                raise ValueError('Request identity differs from frozen case')
            if record.get('mode') != 'local' or record.get('timing_scope') != 'local_pipeline_only':
                raise ValueError('Expected real local-pipeline inference artifacts')
        else:
            record = {'case_id': case['case_id'], 'outcome': 'not_started'}
        outcome = record.get('outcome')
        if outcome not in OUTCOMES:
            raise ValueError('Unknown local request outcome')
        if outcome in PENDING and not partial:
            raise ValueError('Incomplete local evaluation requires --partial')
        row = {'case_id': case['case_id'], 'true_label': case['true_label'],
               'true_time_sec': finite_seconds(case['true_time_sec']),
               'request': record, 'prediction': None, 'predicted_times_sec': [], 'rule_ids': []}
        if outcome == 'success':
            response = AnalysisResult.model_validate_json((folder / 'response.json').read_bytes())
            row['prediction'] = response.verdict
            row['predicted_times_sec'] = [event.timecode_sec for event in response.violations]
            for event in response.violations:
                rule_id = getattr(event, 'rule_id', None)
                row['rule_ids'].append(rule_id if isinstance(rule_id, str)
                                       and re.fullmatch(r'[A-Za-z0-9_.:-]{1,128}', rule_id) else None)
        rows.append(row)
    return rows


def summarize(run, split_path, subset, partial=False):
    run, split_path = Path(run), Path(split_path)
    snapshot_path = run / 'config_snapshot.json'
    snapshot = read(snapshot_path)
    baseline = snapshot.get('evaluation_phase') == 'baseline_all42'
    frozen = snapshot.get('candidate_frozen_before_holdout') is True
    if subset in ('holdout', 'all') and not (baseline or frozen):
        raise ValueError('Freeze the candidate before exposing holdout evaluation')
    if snapshot.get('labels_read_by_inference') is not False:
        raise ValueError('Inference label isolation is not recorded')
    cases = selected_cases(read(split_path), subset)
    rows = collect_local(run, cases, partial)
    # calculate's HTTP fields are deliberately omitted: elapsed_sec here measures
    # the local pipeline, not upload/response latency or an API SLA.
    metric_rows = [{**row, 'request': {key: row['request'].get(key) for key in
                   ('case_id', 'outcome', 'peak_vram_mib')}} for row in rows]
    result = calculate(metric_rows, tolerance=5.0)
    for key in ('request_sla_sec', 'sla_failures', 'missing_sla_measurements', 'requests'):
        result.pop(key)
    outcomes = Counter(row['request']['outcome'] for row in rows)
    timings = [finite_seconds(row['request'].get('elapsed_sec')) for row in rows
               if row['request']['outcome'] not in PENDING]
    measured = [value for value in timings if value is not None]
    runtime_path = run / 'runtime.json'
    runtime = read(runtime_path) if runtime_path.exists() else {}
    rules = Counter()
    rule_videos = defaultdict(set)
    unmapped_events = 0
    examples = {'violation': [], 'no_violation': []}
    for row in rows:
        for rule_id in row['rule_ids']:
            if rule_id is None:
                unmapped_events += 1
            else:
                rules[rule_id] += 1
                rule_videos[rule_id].add(row['case_id'])
        if row['prediction'] == row['true_label'] and len(examples[row['true_label']]) < 5:
            examples[row['true_label']].append({
                'case_id': row['case_id'],
                'rule_ids': sorted({rule_id for rule_id in row['rule_ids'] if rule_id is not None}),
                'event_category_verified': False,
            })
    gold_times = sum(row['true_label'] == 'violation' and row['true_time_sec'] is not None for row in rows)
    matched = len(result['matched_absolute_errors_sec'])
    result.update({
        'subset': subset, 'expected_cases': len(cases), 'partial': result['pending_cases'] > 0,
        'outcome_counts': dict(sorted(outcomes.items())),
        'uncertain_cases': outcomes['inconclusive'], 'timeout_cases': outcomes['timeout'],
        'processing_error_cases': outcomes['processing_error'],
        'processing_failure_cases': outcomes['timeout'] + outcomes['processing_error'],
        'errors_by_true_class_definition': 'Nonbinary completed outcomes, including uncertainty and processing failures',
        'full_video_coverage_cases': sum(row['request'].get('coverage_complete') is True for row in rows),
        'unknown_video_coverage_cases': sum(not isinstance(row['request'].get('coverage_complete'), bool)
                                            for row in rows),
        'timing_scope': 'local pipeline only; includes decoding and generation; excludes model load and HTTP',
        'pipeline_timing_sec': {
            'measured_cases': len(measured), 'missing_or_invalid_completed_cases': len(timings) - len(measured),
            'total': sum(measured) if measured else None,
            'mean': statistics.mean(measured) if measured else None,
            'median': statistics.median(measured) if measured else None,
            'maximum': max(measured) if measured else None,
        },
        'model_load_sec': finite_seconds(runtime.get('model_load_sec')),
        'matched_timestamp_cases': matched, 'gold_timestamp_cases': gold_times,
        'localization_recall_lower_bound': matched / gold_times if gold_times else None,
        'localization_definition': 'Matched positive predictions within 5 seconds / all timestamped positive videos; event type unverified',
        'rule_detection_counts': {rule_id: {'predicted_events': count, 'predicted_videos': len(rule_videos[rule_id])}
                                  for rule_id, count in sorted(rules.items())},
        'rule_detection_definition': 'Events in validated successful responses only; predicted counts are not per-rule accuracy',
        'events_without_source_rule_id': unmapped_events,
        'video_label_agreement_examples': examples,
        'example_definition': 'Binary video-label agreement only; category labels are absent and event descriptions are unverified',
        'provenance': {
            'baseline_all42': baseline, 'candidate_frozen_before_holdout': frozen,
            'labels_read_by_inference': False,
            'config_snapshot_sha256': hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
            'frozen_split_sha256': hashlib.sha256(split_path.read_bytes()).hexdigest(),
            'warnings': [
                'The dataset was previously previewed; the all-42 baseline is known. Holdout is not a pristine unseen dataset.',
                'Holdout or all-case candidate results are exposed only after the candidate is frozen; do not tune on these results.',
                'Small frozen subsets and video-level labels do not establish category accuracy or generalization.',
            ],
        },
    })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--subset', choices=('development', 'holdout', 'all'), required=True)
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    try:
        result = summarize(args.run, args.split, args.subset, args.partial)
        # Never replace the runner's summary or an earlier evaluation artifact.
        output = args.run / f'evaluation_{args.subset}.json'
        with output.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except Exception as exc:  # noqa: BLE001 -- private payloads and paths never enter logs
        print(f'Local evaluation stopped: {type(exc).__name__}; inspect local configuration.')
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
