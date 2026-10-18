import asyncio
import base64
import io
import threading
import time

import cv2
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import create_app
from cashml.config import Settings
from cashml.pipeline import (
    AnalysisError,
    AnalysisInconclusive,
    AnalysisTimeout,
    Analyzer,
)
from cashml.schema import AnalysisResult


@pytest.fixture
def video(tmp_path):
    path = tmp_path / 'synthetic.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (64, 64))
    assert writer.isOpened()
    for i in range(20):
        writer.write(np.full((64, 64, 3), (i * 10, 50, 120), dtype=np.uint8))
    writer.release()
    return path.read_bytes()


@pytest.fixture
def settings(tmp_path):
    return Settings(temp_dir=tmp_path / 'uploads')


def test_mock_valid_jpeg_repeat_cleanup(settings, video):
    with TestClient(create_app(settings)) as client:
        assert client.get('/health').json()['mock'] is True
        for _ in range(2):
            response = client.post('/analyze', files={'file': ('x.mov', video)})
            assert response.status_code == 200
            assert response.headers['X-ML-Mode'] == 'mock'
            assert float(response.headers['X-ML-Request-Seconds']) >= 0
            result = AnalysisResult.model_validate(response.json())
            event = result.violations[0]
            assert event.type == 'MOCK_TEST_EVENT'
            assert 0 <= event.timecode_sec < 2
            assert event.rule.url.startswith('urn:')
            evidence = event.evidence_frames[0]
            image = Image.open(io.BytesIO(base64.b64decode(evidence.image_base64)))
            assert image.format == 'JPEG'
            assert image.size == (64, 64)
            # Real source content rather than a fabricated JPEG.
            assert abs(image.getpixel((32, 32))[0] - 120) < 10
            assert list(settings.temp_dir.iterdir()) == []


def test_missing_bad_and_empty(settings):
    with TestClient(create_app(settings)) as client:
        assert client.post('/analyze').status_code == 400
        for payload in (b'', b'not a video', b'#EXTM3U\nhttp://127.0.0.1/private.ts'):
            assert client.post('/analyze', files={'file': ('bad.mp4', payload)}).status_code == 400
        assert list(settings.temp_dir.iterdir()) == []


def test_multiple_files_rejected(settings, video):
    with TestClient(create_app(settings)) as client:
        response = client.post('/analyze', files=[('file', ('a', video)), ('extra', ('b', video))])
        assert response.status_code == 400
        assert not list(settings.temp_dir.iterdir())


def test_duration_size(settings, video):
    settings.max_duration_sec = 1
    with TestClient(create_app(settings)) as client:
        assert client.post('/analyze', files={'file': ('x', video)}).status_code == 400
    settings.max_upload_mb = 1
    with TestClient(create_app(settings)) as client:
        assert client.post('/analyze', files={'file': ('x', b'x' * 1200000)}).status_code == 400
    assert not list(settings.temp_dir.iterdir())


@pytest.mark.parametrize('error,status', [(AnalysisError('private'), 500),
                                        (RuntimeError('private'), 500),
                                        (AnalysisTimeout(), 500)])
def test_failure_never_success(settings, video, error, status):
    analyzer = Analyzer(settings)

    def fail(*args):
        raise error

    analyzer.analyze_video = fail
    with TestClient(create_app(settings, analyzer)) as client:
        response = client.post('/analyze', files={'file': ('x', video)})
        assert response.status_code == status
        assert 'private' not in response.text
        assert 'no_violation' not in response.text
    assert not list(settings.temp_dir.iterdir())


def test_busy(settings, video):
    analyzer = Analyzer(settings)
    original = analyzer.analyze_video
    entered, release = threading.Event(), threading.Event()

    def slow(*args):
        entered.set()
        assert release.wait(5)
        return original(*args)

    analyzer.analyze_video = slow
    with TestClient(create_app(settings, analyzer)) as client:
        results = []
        thread = threading.Thread(target=lambda: results.append(
            client.post('/analyze', files={'file': ('x', video)}).status_code))
        thread.start()
        assert entered.wait(5)
        try:
            assert client.post('/analyze', files={'file': ('x', video)}).status_code == 503
        finally:
            release.set()
            thread.join(5)
        assert results == [200]
    assert not list(settings.temp_dir.iterdir())


def test_deadline(settings, video):
    analyzer = Analyzer(settings)
    analyzer.load()
    path = settings.temp_dir.parent / 'test.avi'
    path.write_bytes(video)
    with pytest.raises(AnalysisTimeout):
        analyzer.analyze_video(path, time.monotonic() - 1)


def test_repeated_cancellation_keeps_gpu_and_file_until_worker_finishes(settings, video):
    analyzer = Analyzer(settings)
    original = analyzer.analyze_video
    entered, release = threading.Event(), threading.Event()

    def slow(path, deadline):
        entered.set()
        assert release.wait(5)
        assert path.exists()
        return original(path, deadline)

    analyzer.analyze_video = slow
    app = create_app(settings, analyzer)

    async def scenario():
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client,
        ):
            task = asyncio.create_task(client.post('/analyze', files={'file': ('x', video)}))
            assert await asyncio.to_thread(entered.wait, 3)
            try:
                task.cancel()
                await asyncio.sleep(0.02)
                task.cancel()
                await asyncio.sleep(0.02)
                assert (await client.get('/health')).json()['busy'] is True
                assert len(list(settings.temp_dir.iterdir())) == 1
            finally:
                release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert (await client.get('/health')).json()['busy'] is False
            assert not list(settings.temp_dir.iterdir())

    asyncio.run(scenario())


def test_late_success_cannot_escape_as_no_violation(settings, video):
    settings.timeout_sec = 0.1
    analyzer = Analyzer(settings)

    def ignore_deadline(*args):
        time.sleep(0.15)
        return {'verdict': 'no_violation', 'violations': []}

    analyzer.analyze_video = ignore_deadline
    with TestClient(create_app(settings, analyzer)) as client:
        response = client.post('/analyze', files={'file': ('x', video)})
        assert response.status_code == 500
        assert 'no_violation' not in response.text
        assert response.headers['X-ML-Mode'] == 'mock'
        assert not list(settings.temp_dir.iterdir())


def test_upload_time_is_included_in_deadline(settings):
    settings.timeout_sec = 0.05
    analyzer = Analyzer(settings)
    calls = []
    analyzer.analyze_video = lambda *args: calls.append(True)
    app = create_app(settings, analyzer)

    async def slow_body():
        yield b'--boundary\r\nContent-Disposition: form-data; name="file"; filename="input.video"\r\n\r\n'
        await asyncio.sleep(0.15)
        yield b'body\r\n--boundary--\r\n'

    async def scenario():
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client,
        ):
            response = await client.post('/analyze', content=slow_body(),
                                         headers={'Content-Type': 'multipart/form-data; boundary=boundary'})
            assert response.status_code == 500
            assert 'no_violation' not in response.text
            assert not calls
            assert not list(settings.temp_dir.iterdir())

    asyncio.run(scenario())


def test_unknown_has_explicit_inconclusive_error(settings, video):
    analyzer = Analyzer(settings)
    issues = [{'rule_id': 'fixture', 'start_sec': 0.0, 'end_sec': 1.0,
               'pass': 'coarse', 'unknown_fields': ['criterion_met']}]

    def unknown(*args):
        raise AnalysisInconclusive('Private explanation must remain local', issues=issues)

    analyzer.analyze_video = unknown
    with TestClient(create_app(settings, analyzer)) as client:
        response = client.post('/analyze', files={'file': ('input.video', video)})
        assert response.status_code == 500
        assert response.json()['detail']['code'] == 'analysis_inconclusive'
        assert response.json()['detail']['stage'] == 'verdict_aggregation'
        assert response.json()['detail']['issues'] == issues
        assert 'no_violation' not in response.text
        assert 'Private explanation' not in response.text
        assert not list(settings.temp_dir.iterdir())


def wait_job(client, job_id):
    for _ in range(200):
        response = client.get(f'/api/v1/jobs/{job_id}')
        assert response.status_code == 200
        job = response.json()
        if job['status'] != 'processing':
            return job
        time.sleep(0.01)
    pytest.fail('Job did not finish')


def test_async_job_upload_poll_result_and_cleanup(settings, video):
    with TestClient(create_app(settings)) as client:
        response = client.post('/api/v1/analyze', files={'file': ('x.mov', video)})
        assert response.status_code == 202
        job = wait_job(client, response.json()['job_id'])
        assert job['status'] == 'completed'
        AnalysisResult.model_validate(job['result'])
        assert not list(settings.temp_dir.iterdir())
        assert client.get('/api/v1/jobs/missing').status_code == 404


def test_async_inconclusive_preserves_diagnostics(settings, video):
    analyzer = Analyzer(settings)
    issues = [{'rule_id': 'fixture', 'unknown_fields': ['criterion_met']}]

    def unknown(*args):
        raise AnalysisInconclusive('Private explanation', issues=issues)

    analyzer.analyze_video = unknown
    with TestClient(create_app(settings, analyzer)) as client:
        response = client.post('/api/v1/analyze', files={'file': ('x.mov', video)})
        job = wait_job(client, response.json()['job_id'])
        assert job['status'] == 'failed'
        assert job['error']['issues'] == issues
        assert job['error']['stage'] == 'verdict_aggregation'
        assert 'Private explanation' not in str(job)
        assert not list(settings.temp_dir.iterdir())


def test_protocol_failure_has_correct_status_and_job_error(settings, video):
    from cashml.pipeline import AnalysisProtocolError

    analyzer = Analyzer(settings)

    def malformed(*args):
        raise AnalysisProtocolError('Private model output')

    analyzer.analyze_video = malformed
    with TestClient(create_app(settings, analyzer)) as client:
        response = client.post('/analyze', files={'file': ('x.mov', video)})
        assert response.status_code == 500
        assert response.json()['detail']['code'] == 'model_protocol_error'
        response = client.post('/api/v1/analyze', files={'file': ('x.mov', video)})
        job = wait_job(client, response.json()['job_id'])
        assert job['status'] == 'failed'
        assert job['error']['code'] == 'model_protocol_error'
        assert 'Private model output' not in str(job)


def test_async_busy_and_shutdown_wait_for_worker(settings, video):
    analyzer = Analyzer(settings)
    entered, release = threading.Event(), threading.Event()

    def slow(*args):
        entered.set()
        assert release.wait(5)
        return {'verdict': 'no_violation', 'violations': []}

    analyzer.analyze_video = slow
    with TestClient(create_app(settings, analyzer)) as client:
        response = client.post('/api/v1/analyze', files={'file': ('x.mov', video)})
        try:
            assert response.status_code == 202
            assert entered.wait(2)
            assert client.get('/health').json()['busy']
            assert client.post('/api/v1/analyze', files={'file': ('x', video)}).status_code == 503
            assert client.post('/analyze', files={'file': ('x', video)}).status_code == 503
        finally:
            release.set()
    assert not list(settings.temp_dir.iterdir())


def test_api_key_protects_sync_and_job_routes(settings, video, monkeypatch):
    monkeypatch.setenv('JURY_API_KEY', 'test-key')
    with TestClient(create_app(settings)) as client:
        assert client.get('/health').status_code == 200
        for path in ('/analyze', '/api/v1/analyze'):
            assert client.post(path, files={'file': ('x', video)}).status_code == 401
        assert client.get('/api/v1/jobs/missing').status_code == 401
        assert client.get('/api/v1/jobs/missing', headers={'X-API-Key': 'test-key'}).status_code == 404
        response = client.post('/api/v1/analyze', files={'file': ('x', video)},
                               headers={'X-API-Key': 'test-key'})
        assert response.status_code == 202
