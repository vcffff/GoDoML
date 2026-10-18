"""Official API contract tests with synthetic local video and stub inference."""

import asyncio
import base64
import io
import threading
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from test_api import video as video  # noqa: PLC0414 -- shared synthetic fixture

from app import create_app
from cashml.config import Settings
from cashml.pipeline import (
    AnalysisError,
    AnalysisInconclusive,
    AnalysisTimeout,
    Analyzer,
)
from cashml.video import InvalidVideo

TEST_KEY = 'synthetic-test-key-not-a-deployment-secret'
HEADERS = {'X-API-Key': TEST_KEY}


@pytest.fixture
def settings(tmp_path):
    return Settings(api_key=TEST_KEY, temp_dir=tmp_path / 'uploads')


def completed_job(client, job_id):
    for _ in range(200):
        response = client.get(f'/api/v1/jobs/{job_id}', headers=HEADERS)
        assert response.status_code == 200
        result = response.json()
        if result['status'] != 'processing':
            return result
        assert result == {'job_id': job_id, 'status': 'processing'}
        time.sleep(0.01)
    pytest.fail('Synthetic job did not finish')


def test_official_done_shape_preserves_jpeg_evidence_and_source(settings, video):
    with TestClient(create_app(settings)) as client:
        response = client.post('/api/v1/analyze', headers=HEADERS, files={'file': ('input.mov', video)})
        assert response.status_code == 202
        assert set(response.json()) == {'job_id'}
        job_id = response.json()['job_id']
        result = completed_job(client, job_id)
        assert set(result) == {'job_id', 'status', 'verdict', 'violations'}
        assert result['status'] == 'done'
        assert result['verdict'] == 'violation'
        event = result['violations'][0]
        assert set(event) == {'type', 'timecode_sec', 'description', 'evidence_frames'}
        assert 'urn:cashml:test-fixture:mock-v1' in event['description']
        assert 1 <= len(event['evidence_frames']) <= 3
        for evidence in event['evidence_frames']:
            assert set(evidence) == {'timestamp_sec', 'image_base64'}
            with Image.open(io.BytesIO(base64.b64decode(evidence['image_base64'], validate=True))) as image:
                assert image.format == 'JPEG'
                image.verify()
        repeated = client.get(f'/api/v1/jobs/{job_id}', headers=HEADERS)
        assert repeated.json() == result
        assert repeated.headers['X-ML-Mode'] == 'mock'
        assert float(repeated.headers['X-ML-Request-Seconds']) >= 0
        assert not list(settings.temp_dir.iterdir())


@pytest.mark.parametrize(('error', 'code'), [
    (AnalysisInconclusive('private model response'), 'analysis_inconclusive'),
    (AnalysisError('private malformed model response'), 'analysis_error'),
    (AnalysisTimeout('private timeout detail'), 'analysis_timeout'),
    (TimeoutError('private timeout detail'), 'analysis_timeout'),
    (RuntimeError('private inference detail'), 'inference_error'),
    (InvalidVideo('private path detail'), 'invalid_video'),
])
def test_failed_jobs_poll_as_200_without_fabricated_negative(settings, video, error, code):
    analyzer = Analyzer(settings)

    def fail(*args):
        raise error

    analyzer.analyze_video = fail
    with TestClient(create_app(settings, analyzer)) as client:
        response = client.post('/api/v1/analyze', headers=HEADERS, files={'file': ('input.mov', video)})
        assert response.status_code == 202
        result = completed_job(client, response.json()['job_id'])
        assert set(result) == {'job_id', 'status', 'error'}
        assert result['status'] == 'failed'
        assert result['error']['code'] == code
        assert 'private' not in str(result)
        assert 'no_violation' not in str(result)
        assert not list(settings.temp_dir.iterdir())


@pytest.mark.parametrize('path', ['/api/v1/analyze', '/api/v1/jobs/missing', '/analyze'])
def test_authentication_rejects_before_reading_body(settings, path):
    settings.mode = 'local'
    app = create_app(settings, Analyzer(settings))
    sent = []

    async def receive():
        pytest.fail('Unauthorized request body was read')

    async def send(message):
        sent.append(message)

    scope = {'type': 'http', 'http_version': '1.1', 'method': 'POST', 'scheme': 'http',
             'path': path, 'raw_path': path.encode(), 'query_string': b'', 'root_path': '',
             'headers': [], 'server': ('test', 80), 'client': ('test', 1234)}
    asyncio.run(app(scope, receive, send))
    assert sent[0]['status'] == 401
    assert TEST_KEY.encode() not in b''.join(item.get('body', b'') for item in sent)


def test_official_auth_required_for_submit_and_poll_but_health_is_public(settings, video):
    with TestClient(create_app(settings)) as client:
        assert client.get('/health').status_code == 200
        for headers in ({}, {'X-API-Key': 'incorrect'}, [('X-API-Key', TEST_KEY)] * 2):
            assert client.post('/api/v1/analyze', headers=headers,
                               files={'file': ('input.mov', video)}).status_code == 401
            assert client.get('/api/v1/jobs/missing', headers=headers).status_code == 401
        assert client.get('/api/v1/jobs/missing', headers=HEADERS).status_code == 404
        assert not list(settings.temp_dir.iterdir())


def test_local_startup_requires_private_key_and_repr_hides_it(settings):
    assert TEST_KEY not in repr(settings)
    settings.mode = 'local'
    settings.api_key = ''
    with pytest.raises(ValueError, match='ML_API_KEY'), TestClient(create_app(settings, Analyzer(settings))):
        pass


@pytest.mark.parametrize('first_official', [False, True])
def test_legacy_and_official_share_one_analyzer_and_gpu_reservation(settings, video, first_official):
    analyzer = Analyzer(settings)
    original = analyzer.analyze_video
    entered, release = threading.Event(), threading.Event()
    calls = []

    def slow(*args):
        calls.append(True)
        entered.set()
        assert release.wait(5)
        return original(*args)

    analyzer.analyze_video = slow
    app = create_app(settings, analyzer)
    assert app.state.analyzer is analyzer
    with TestClient(app) as client:
        responses = []
        thread = None
        if first_official:
            response = client.post('/api/v1/analyze', headers=HEADERS,
                                   files={'file': ('input.mov', video)})
            assert response.status_code == 202
        else:
            thread = threading.Thread(target=lambda: responses.append(client.post(
                '/analyze', files={'file': ('input.mov', video)})))
            thread.start()
        try:
            assert entered.wait(5)
            assert client.get('/health').json()['busy'] is True
            other = '/analyze' if first_official else '/api/v1/analyze'
            assert client.post(other, headers=HEADERS, files={'file': ('input.mov', video)}).status_code == 503
            assert len(calls) == 1
        finally:
            release.set()
            if thread:
                thread.join(5)
        if first_official:
            assert completed_job(client, response.json()['job_id'])['status'] == 'done'
        else:
            assert responses[0].status_code == 200
        assert client.get('/health').json()['busy'] is False
        assert not list(settings.temp_dir.iterdir())


def test_official_upload_rejects_non_mov_empty_and_oversized_files(settings, video):
    settings.max_upload_mb = 1
    with TestClient(create_app(settings)) as client:
        for name, payload in [('input.mp4', video), ('input.mov', b''), ('input.mov', b'x' * 1200000)]:
            response = client.post('/api/v1/analyze', headers=HEADERS, files={'file': (name, payload)})
            assert response.status_code == 400
            assert 'job_id' not in response.json()
        assert not list(settings.temp_dir.iterdir())
        assert client.get('/health').json()['busy'] is False


def test_job_deadline_cannot_return_late_negative(settings, video):
    settings.timeout_sec = 0.02
    analyzer = Analyzer(settings)

    def late(path, deadline):
        assert deadline <= time.monotonic() + settings.timeout_sec
        time.sleep(0.06)
        return {'verdict': 'no_violation', 'violations': []}

    analyzer.analyze_video = late
    with TestClient(create_app(settings, analyzer)) as client:
        response = client.post('/api/v1/analyze', headers=HEADERS, files={'file': ('input.mov', video)})
        assert response.status_code == 202
        result = completed_job(client, response.json()['job_id'])
        assert result['status'] == 'failed'
        assert result['error']['code'] == 'analysis_timeout'
        assert 'verdict' not in result


def test_deadline_poll_fails_without_releasing_a_still_running_worker(settings, video):
    settings.timeout_sec = 0.05
    analyzer = Analyzer(settings)
    entered, release = threading.Event(), threading.Event()

    def blocked(path, deadline):
        entered.set()
        assert release.wait(5)
        assert path.exists()
        return {'verdict': 'no_violation', 'violations': []}

    analyzer.analyze_video = blocked
    with TestClient(create_app(settings, analyzer)) as client:
        response = client.post('/api/v1/analyze', headers=HEADERS, files={'file': ('input.mov', video)})
        assert response.status_code == 202
        try:
            assert entered.wait(2)
            time.sleep(0.07)
            result = client.get(f"/api/v1/jobs/{response.json()['job_id']}", headers=HEADERS)
            assert result.status_code == 200
            assert result.json()['error']['code'] == 'analysis_timeout'
            assert 'verdict' not in result.json()
            assert client.get('/health').json()['busy'] is True
            assert len(list(settings.temp_dir.iterdir())) == 1
        finally:
            release.set()
    assert not list(settings.temp_dir.iterdir())


def test_official_transport_accepts_29_mb_with_default_limit(settings):
    analyzer = Analyzer(settings)
    sizes = []

    def inspect(path, deadline):
        sizes.append(path.stat().st_size)
        return {'verdict': 'no_violation', 'violations': []}

    analyzer.analyze_video = inspect
    payload = b'0' * (29 * 1024 * 1024)
    with TestClient(create_app(settings, analyzer)) as client:
        response = client.post('/api/v1/analyze', headers=HEADERS,
                               files={'file': ('transport-only.mov', payload)})
        assert response.status_code == 202
        result = completed_job(client, response.json()['job_id'])
        assert result == {'job_id': response.json()['job_id'], 'status': 'done',
                          'verdict': 'no_violation', 'violations': []}
    assert sizes == [len(payload)]
    assert not list(settings.temp_dir.iterdir())
