"""CPU-only regression checks for frozen-subset evaluation boundaries."""
import json
from pathlib import Path

import pytest

from scripts.evaluate import calculate
from scripts.fixed_subset import comparison_rows, write


def case(case_id, label, timestamp=None):
    return {'case_id': case_id, 'video_path': str(Path('opaque.video').resolve()),
            'duration_sec': 30, 'true_label': label, 'true_time_sec': timestamp}


def completed(tmp_path, case_id, verdict, times):
    folder = tmp_path / 'after' / case_id
    write(folder / 'request.json', {'case_id': case_id, 'outcome': 'success',
                                  'elapsed_sec': 2, 'peak_vram_mib': '512'})
    write(folder / 'response.json', {'verdict': verdict, 'violations': [
        {'timecode_sec': value} for value in times]})


def test_unknown_error_and_pending_are_not_negative(tmp_path):
    cases = [case('a', 'violation', 10), case('b', 'no_violation'),
             case('c', 'no_violation')]
    write(tmp_path / 'after/a/request.json', {'outcome': 'inconclusive'})
    write(tmp_path / 'after/b/request.json', {'outcome': 'http_error'})
    result = calculate(comparison_rows(tmp_path, 'after', cases))
    assert result['confusion_successful_requests_only'] == {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
    assert result['successful_cases'] == 0
    assert result['pending_cases'] == 1
    assert result['errors_by_true_class'] == {'violation': 1, 'no_violation': 1}


def test_comparison_confusion_localization_and_telemetry(tmp_path):
    cases = [case('tp', 'violation', 10), case('fn', 'violation', 20),
             case('fp', 'no_violation'), case('tn', 'no_violation')]
    completed(tmp_path, 'tp', 'violation', [12, 25])
    completed(tmp_path, 'fn', 'no_violation', [])
    completed(tmp_path, 'fp', 'violation', [1])
    completed(tmp_path, 'tn', 'no_violation', [])
    result = calculate(comparison_rows(tmp_path, 'after', cases))
    assert result['confusion_successful_requests_only'] == {'TP': 1, 'FP': 1, 'TN': 1, 'FN': 1}
    assert result['precision'] == 0.5
    assert result['recall_successful_requests_only'] == 0.5
    assert result['matched_absolute_errors_sec'] == [{'case_id': 'tp', 'absolute_error_sec': 2}]
    assert result['nearest_absolute_errors_successful_TP_sec'] == [{'case_id': 'tp', 'absolute_error_sec': 2}]
    assert len(result['requests']) == 4


def test_inference_opaque_upload_inconclusive_and_no_retry(tmp_path, monkeypatch):
    import httpx

    from scripts.fixed_subset import run

    video = tmp_path / 'private_filename.video'
    video.write_bytes(b'fixture-only')
    write(tmp_path / 'inference_manifest.json', {'cases': [{
        'case_id': 'opaque001', 'video_path': str(video), 'duration_sec': 1}]})
    uploads = []

    def handler(request):
        if request.url.path == '/health':
            return httpx.Response(200, json={'mode': 'local', 'mock': False,
                                            'ready': True, 'assessment_schema_version': 2})
        uploads.append(request.read())
        return httpx.Response(500, json={'detail': {'code': 'analysis_inconclusive'}})

    original_client = httpx.Client
    monkeypatch.setattr(httpx, 'Client', lambda **kwargs: original_client(
        transport=httpx.MockTransport(handler), **kwargs))
    run(tmp_path, 'after', 'http://127.0.0.1:8001')
    run(tmp_path, 'after', 'http://127.0.0.1:8001')
    assert len(uploads) == 1
    assert b'filename="input.video"' in uploads[0]
    assert b'private_filename' not in uploads[0]
    assert b'true_label' not in uploads[0]
    record = json.loads((tmp_path / 'after/opaque001/request.json').read_text())
    assert record['outcome'] == 'inconclusive'


def test_holdout_cannot_be_selected_for_development_probe(tmp_path):
    from scripts.fixed_subset import run

    write(tmp_path / 'inference_manifest.json', {'cases': [{
        'case_id': 'development001', 'video_path': 'fixture.video', 'duration_sec': 1}]})
    with pytest.raises(ValueError, match='frozen development'):
        run(tmp_path, 'after', 'http://127.0.0.1:8001', only_case='holdout001')
