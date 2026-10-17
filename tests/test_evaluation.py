import csv
import json

import pytest

from scripts.evaluate import (
    calculate,
    collect,
    parse_label,
    write_manual,
    write_request_metrics,
)


def case(case_id, truth, prediction, timestamp=None, times=(), outcome='success'):
    return {'case_id': case_id, 'true_label': truth, 'prediction': prediction,
            'true_time_sec': timestamp, 'predicted_times_sec': list(times),
            'request': {'case_id': case_id, 'outcome': outcome,
                        'elapsed_sec': 20, 'sla_pass': True, 'peak_vram_mib': '2500'}}


def test_confusion_excludes_failed_requests_and_reports_coverage():
    rows = [case('a', 'violation', 'violation', 10, [12]),
            case('b', 'no_violation', 'violation', None, [20]),
            case('c', 'no_violation', 'no_violation'),
            case('d', 'violation', 'no_violation', 10),
            case('e', 'no_violation', None, outcome='request_error_unknown'),
            case('f', 'violation', None, 5, outcome='http_error')]
    result = calculate(rows)
    assert result['confusion_successful_requests_only'] == {'TP': 1, 'FP': 1, 'TN': 1, 'FN': 1}
    assert result['errors_by_true_class'] == {'violation': 1, 'no_violation': 1}
    assert result['precision'] == 0.5
    assert result['recall_successful_requests_only'] == 0.5
    assert result['unmatched_gold_timestamps'] == 2
    assert result['unmatched_predicted_timestamps'] == 1
    assert result['matched_absolute_errors_sec'] == [{'case_id': 'a', 'absolute_error_sec': 2}]


def test_matching_is_nearest_and_single_gold_has_at_most_one_match():
    result = calculate([case('a', 'violation', 'violation', 10, [5, 9, 11])])
    assert result['matched_absolute_errors_sec'][0]['absolute_error_sec'] == 1
    assert result['unmatched_predicted_timestamps'] == 2
    result = calculate([case('b', 'violation', 'violation', 10, [15.001])])
    assert result['matched_absolute_errors_sec'] == []
    assert result['nearest_absolute_errors_successful_TP_sec'][0]['absolute_error_sec'] == 5.001
    assert result['unmatched_gold_timestamps'] == 1


def test_five_second_boundary_is_inclusive():
    result = calculate([case('a', 'violation', 'violation', 10, [15])])
    assert len(result['matched_absolute_errors_sec']) == 1


def test_zero_denominators_are_unknown_and_error_never_becomes_true_negative():
    result = calculate([case('a', 'no_violation', None, outcome='http_error')])
    assert result['precision'] is None
    assert result['recall_successful_requests_only'] is None
    assert result['confusion_successful_requests_only']['TN'] == 0
    assert result['successful_cases'] == 0


@pytest.mark.parametrize('value, expected', [('Без нарушения', 'no_violation'),
                                            ('Нарушение', 'violation'), (' нет ', 'no_violation')])
def test_labels(value, expected):
    assert parse_label(value) == expected


def test_unknown_label_is_not_guessed():
    with pytest.raises(ValueError):
        parse_label('unreviewed')


def test_incomplete_run_requires_explicit_partial(tmp_path):
    (tmp_path / 'manifest.json').write_text(json.dumps({'cases': [
        {'case_id': 'case_001', 'video_path': str(tmp_path / 'synthetic.avi')}
    ]}), encoding='utf-8')
    with pytest.raises(ValueError):
        collect(tmp_path, {})
    rows, expected = collect(tmp_path, {'synthetic': {'true_label': 'violation', 'true_time_sec': 2}}, partial=True)
    assert len(rows) == 1
    assert rows[0]['request']['outcome'] == 'not_started'
    assert expected == 1


def test_pending_is_neither_failure_nor_negative_and_lower_bound_uses_all_positives():
    rows = [case('a', 'violation', 'violation', 10, [12]),
            case('b', 'violation', None, 2, outcome='not_started'),
            case('c', 'no_violation', None, outcome='in_flight_unknown'),
            case('d', 'violation', None, 5, outcome='http_error')]
    result = calculate(rows)
    assert result['pending_cases'] == 2
    assert result['errors_by_true_class'] == {'violation': 1, 'no_violation': 0}
    assert result['confusion_successful_requests_only'] == {'TP': 1, 'FP': 0, 'TN': 0, 'FN': 0}
    assert result['recall_successful_requests_only'] == 1
    assert result['end_to_end_recall_lower_bound'] == pytest.approx(1 / 3)
    assert result['unmatched_gold_timestamps'] == 1


def test_technical_metrics_exclude_private_fields(tmp_path):
    row = case('case_001', 'violation', 'violation', 10, [12])
    row['video_path'] = 'private-name.avi'
    write_request_metrics(tmp_path, [row])
    with (tmp_path / 'request_metrics.csv').open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ['case_id', 'http_status', 'outcome', 'elapsed_sec', 'peak_vram_mib', 'sla_pass']
        assert next(reader)['case_id'] == 'case_001'


def test_manual_review_leaves_human_judgments_unanswered(tmp_path):
    row = case('case_001', 'violation', 'violation', 10, [12])
    row.update(video_path=str(tmp_path / 'synthetic.avi'), response={
        'violations': [{'type': 'hypothesis', 'description': '<unsafe>synthetic</unsafe>',
                        'evidence_frames': []}]})
    write_manual(tmp_path, [row])
    document = (tmp_path / 'manual_review.html').read_text(encoding='utf-8')
    assert '&lt;unsafe&gt;synthetic&lt;/unsafe&gt;' in document
    assert '<unsafe>' not in document
    assert 'https://' not in document
    with (tmp_path / 'manual_review.csv').open(encoding='utf-8-sig', newline='') as stream:
        entry = next(csv.DictReader(stream))
    assert entry['description_correct'] == ''
    assert entry['category_confirmed'] == ''


def test_manual_review_refresh_preserves_only_human_judgments(tmp_path):
    row = case('case_001', 'violation', 'no_violation', 10)
    row.update(video_path=str(tmp_path / 'synthetic.avi'), response={'violations': []})
    write_manual(tmp_path, [row])
    csv_path = tmp_path / 'manual_review.csv'
    with csv_path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        entries = list(reader)
    entries[0].update(description_correct='Верно', category_confirmed='Не подтверждена',
                      true_label='edited truth', prediction='edited prediction')
    with csv_path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(entries)
    row.update(prediction='violation', predicted_times_sec=[12])
    write_manual(tmp_path, [row])
    with csv_path.open(encoding='utf-8-sig', newline='') as stream:
        refreshed = next(csv.DictReader(stream))
    assert refreshed['description_correct'] == 'Верно'
    assert refreshed['category_confirmed'] == 'Не подтверждена'
    assert refreshed['true_label'] == 'violation'
    assert refreshed['prediction'] == 'violation'
    assert refreshed['predicted_times_sec'] == '12'
