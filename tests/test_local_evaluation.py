"""Post-hoc local evaluation preserves uncertainty, provenance and private data."""
import base64
import io
import json

import pytest
from PIL import Image
from pydantic import ValidationError

from scripts.summarize_local import main, summarize


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def prepare(tmp_path, labels, holdout=0, baseline=False, frozen=False):
    cases = [{'case_id': f'case_{index:03}', 'true_label': label,
              'true_time_sec': 10 if label == 'violation' else None,
              'video_path': 'private-source-name-keep-local.video'}
             for index, label in enumerate(labels, 1)]
    boundary = len(cases) - holdout
    split = tmp_path / 'split.json'
    write(split, {'development': cases[:boundary], 'holdout': cases[boundary:]})
    run = tmp_path / 'run'
    write(run / 'config_snapshot.json', {
        'case_ids': [case['case_id'] for case in cases], 'labels_read_by_inference': False,
        'evaluation_phase': 'baseline_all42' if baseline else 'candidate',
        'candidate_frozen_before_holdout': frozen,
        'settings': {'model_dir': 'private-model-path-keep-local'},
    })
    write(run / 'runtime.json', {'model_load_sec': 7.5})
    return run, split, cases


def record(run, index, outcome='success', verdict='no_violation', times=(), rule_id='rule_7'):
    folder = run / f'case_{index:03}'
    write(folder / 'request.json', {
        'case_id': folder.name, 'mode': 'local', 'timing_scope': 'local_pipeline_only',
        'outcome': outcome, 'elapsed_sec': 350, 'peak_vram_mib': 4000,
        'coverage_complete': outcome in ('success', 'inconclusive'),
    })
    if outcome == 'success':
        image = io.BytesIO()
        Image.new('RGB', (2, 2)).save(image, format='JPEG')
        events = [{
            'type': 'private-type-keep-local', 'description': 'private-observation-keep-local',
            'timecode_sec': time, 'rule': {'title': 'private-law-title', 'section': 'private-section',
                                         'url': 'urn:private-law'},
            'evidence_frames': [{'timestamp_sec': time,
                                 'image_base64': base64.b64encode(image.getvalue()).decode()}],
            **({'rule_id': rule_id} if rule_id is not None else {}),
        } for time in times]
        write(folder / 'response.json', {'verdict': verdict, 'violations': events})


def test_local_metrics_separate_binary_quality_uncertainty_and_failures(tmp_path):
    run, split, _ = prepare(tmp_path, [
        'violation', 'no_violation', 'no_violation', 'violation',
        'violation', 'no_violation', 'violation', 'no_violation'])
    record(run, 1, verdict='violation', times=(12, 25))
    record(run, 2)
    record(run, 3, verdict='violation', times=(2,))
    record(run, 4)
    record(run, 5, outcome='inconclusive')
    record(run, 6, outcome='timeout')
    record(run, 7, outcome='processing_error')
    result = summarize(run, split, 'development', partial=True)
    assert result['confusion_successful_requests_only'] == {'TP': 1, 'FP': 1, 'TN': 1, 'FN': 1}
    assert result['accuracy_successful_requests_only'] == 0.5
    assert result['f1_successful_requests_only'] == 0.5
    assert result['decision_coverage'] == 0.5
    assert result['end_to_end_accuracy_lower_bound'] == 0.25
    assert result['uncertain_cases'] == 1
    assert result['processing_failure_cases'] == 2
    assert result['pending_cases'] == 1
    assert result['partial'] is True
    assert result['localization_recall_lower_bound'] == 0.25
    assert result['matched_timestamp_cases'] == 1
    assert result['temporal_match_tolerance_sec'] == 5
    assert result['rule_detection_counts'] == {'rule_7': {'predicted_events': 3, 'predicted_videos': 2}}
    assert result['video_label_agreement_examples'] == {
        'violation': [{'case_id': 'case_001', 'rule_ids': ['rule_7'], 'event_category_verified': False}],
        'no_violation': [{'case_id': 'case_002', 'rule_ids': [], 'event_category_verified': False}],
    }
    assert result['pipeline_timing_sec']['measured_cases'] == 7
    assert result['pipeline_timing_sec']['maximum'] == 350
    assert result['model_load_sec'] == 7.5
    assert 'sla_failures' not in result
    assert 'request_sla_sec' not in result
    assert 'requests' not in result
    assert 'private-' not in json.dumps(result)
    assert 'image_base64' not in json.dumps(result)


def test_all_uncertain_returns_undefined_quality_without_negative_predictions(tmp_path):
    run, split, _ = prepare(tmp_path, ['violation', 'no_violation'])
    record(run, 1, outcome='inconclusive')
    record(run, 2, outcome='inconclusive')
    result = summarize(run, split, 'development')
    assert result['accuracy_successful_requests_only'] is None
    assert result['f1_successful_requests_only'] is None
    assert result['precision'] is None
    assert result['decision_coverage'] == 0
    assert result['end_to_end_accuracy_lower_bound'] == 0
    assert result['uncertain_cases'] == 2
    assert result['processing_failure_cases'] == 0
    assert result['confusion_successful_requests_only'] == {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}


@pytest.mark.parametrize('outcome', [None, 'in_flight_unknown'])
def test_partial_flag_is_required_for_missing_or_inflight_results(tmp_path, outcome):
    run, split, _ = prepare(tmp_path, ['violation'])
    if outcome:
        record(run, 1, outcome=outcome)
    with pytest.raises(ValueError, match='--partial'):
        summarize(run, split, 'development')
    result = summarize(run, split, 'development', partial=True)
    assert result['pending_cases'] == 1
    assert result['errors_by_true_class']['violation'] == 0
    assert result['pipeline_timing_sec']['measured_cases'] == 0


def test_saved_success_must_validate_complete_response_schema(tmp_path):
    run, split, _ = prepare(tmp_path, ['violation'])
    record(run, 1, verdict='no_violation', times=(10,))
    with pytest.raises(ValidationError):
        summarize(run, split, 'development')


@pytest.mark.parametrize('subset', ['all', 'holdout'])
def test_holdout_results_are_not_read_before_candidate_freeze(tmp_path, monkeypatch, subset):
    from scripts import summarize_local

    run, split, _ = prepare(tmp_path, ['no_violation'], holdout=1)
    monkeypatch.setattr(summarize_local, 'collect_local', lambda *args: pytest.fail('Read heldout results'))
    with pytest.raises(ValueError, match='Freeze the candidate'):
        summarize(run, split, subset)


@pytest.mark.parametrize('baseline, frozen', [(True, False), (False, True)])
def test_known_baseline_and_frozen_candidate_allow_separate_holdout_report(tmp_path, baseline, frozen):
    run, split, _ = prepare(tmp_path, ['violation', 'no_violation'], holdout=1,
                            baseline=baseline, frozen=frozen)
    record(run, 2)
    result = summarize(run, split, 'holdout')
    assert result['expected_cases'] == 1
    assert result['successful_cases'] == 1
    assert result['provenance']['baseline_all42'] is baseline
    assert result['provenance']['candidate_frozen_before_holdout'] is frozen
    assert result['provenance']['warnings']


def test_missing_rule_ids_are_counted_without_using_private_event_type(tmp_path):
    run, split, _ = prepare(tmp_path, ['violation'])
    record(run, 1, verdict='violation', times=(10,), rule_id=None)
    result = summarize(run, split, 'development')
    assert result['events_without_source_rule_id'] == 1
    assert result['rule_detection_counts'] == {}
    assert 'private-' not in json.dumps(result)


def test_invalid_local_timing_remains_missing_and_does_not_become_http_sla(tmp_path):
    run, split, _ = prepare(tmp_path, ['no_violation'])
    record(run, 1)
    path = run / 'case_001/request.json'
    request = json.loads(path.read_text())
    request['elapsed_sec'] = 'invalid'
    write(path, request)
    result = summarize(run, split, 'development')
    assert result['pipeline_timing_sec']['measured_cases'] == 0
    assert result['pipeline_timing_sec']['missing_or_invalid_completed_cases'] == 1
    assert result['pipeline_timing_sec']['total'] is None
    assert 'sla_failures' not in result


def test_cli_writes_new_summary_without_replacing_runner_or_existing_evaluation(tmp_path, monkeypatch, capsys):
    run, split, _ = prepare(tmp_path, ['no_violation'])
    record(run, 1)
    write(run / 'summary.json', {'runner': 'unchanged'})
    monkeypatch.setattr('sys.argv', ['summarize_local', '--run', str(run), '--split', str(split),
                                   '--subset', 'development'])
    main()
    result = json.loads(capsys.readouterr().out)
    assert result['successful_cases'] == 1
    assert json.loads((run / 'summary.json').read_text()) == {'runner': 'unchanged'}
    output = run / 'evaluation_development.json'
    saved = output.read_bytes()
    with pytest.raises(SystemExit):
        main()
    assert output.read_bytes() == saved
    assert 'FileExistsError' in capsys.readouterr().out
