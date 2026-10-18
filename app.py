import asyncio
import json
import logging
import secrets
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.formparsers import MultiPartException

from cashml.config import Settings
from cashml.pipeline import (
    AnalysisError,
    AnalysisInconclusive,
    AnalysisTimeout,
    Analyzer,
)
from cashml.schema import AnalysisResult
from cashml.video import InvalidVideo

log = logging.getLogger('cashml')


class UploadLimit(MultiPartException):
    pass


class RequestLimits:
    def __init__(self, app, settings):
        self.app, self.settings = app, settings

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        official = scope['path'].startswith('/api/v1/')
        protected = official or (scope['path'] == '/analyze' and self.settings.mode == 'local')
        if protected:
            keys = [value for name, value in scope.get('headers', []) if name.lower() == b'x-api-key']
            valid = (len(keys) == 1 and bool(self.settings.api_key)
                     and secrets.compare_digest(keys[0], self.settings.api_key.encode('utf-8')))
            if not valid:
                return await JSONResponse({'detail': 'Invalid API key'}, status_code=401)(scope, receive, send)
        if scope['path'] not in ('/analyze', '/api/v1/analyze'):
            return await self.app(scope, receive, send)
        started = time.monotonic()
        scope['analysis_deadline'] = started + self.settings.timeout_sec
        total = 0
        replaced = False

        async def measured_send(message):
            nonlocal replaced
            if message['type'] == 'http.response.start':
                elapsed = time.monotonic() - started
                if message['status'] == 200 and time.monotonic() >= scope['analysis_deadline']:
                    replaced = True
                    error_headers = {'X-ML-Mode': self.settings.mode,
                                     'X-ML-Request-Seconds': f'{elapsed:.6f}'}
                    if scope.get('peak_vram_mib') is not None:
                        error_headers['X-ML-Peak-VRAM-MiB'] = f'{scope["peak_vram_mib"]:.3f}'
                    error = JSONResponse({'detail': 'Analysis time limit exceeded'}, status_code=500,
                                         headers=error_headers)
                    await error(scope, receive, send)
                    return
                headers = list(message.get('headers', []))
                headers.extend([(b'x-ml-mode', self.settings.mode.encode()),
                                (b'x-ml-request-seconds', f'{elapsed:.6f}'.encode())])
                peak = scope.get('peak_vram_mib')
                if peak is not None:
                    headers.append((b'x-ml-peak-vram-mib', f'{peak:.3f}'.encode()))
                if scope.get('assessment_file'):
                    headers.append((b'x-ml-assessment-file', scope['assessment_file'].encode()))
                message = {**message, 'headers': headers}
            if not replaced:
                await send(message)

        async def limited_receive():
            nonlocal total
            remaining = scope['analysis_deadline'] - time.monotonic()
            if remaining <= 0:
                scope['upload_timed_out'] = True
                raise MultiPartException('Upload deadline exceeded')
            try:
                message = await asyncio.wait_for(receive(), remaining)
            except TimeoutError:
                scope['upload_timed_out'] = True
                raise MultiPartException('Upload deadline exceeded') from None
            total += len(message.get('body', b''))
            if total > self.settings.max_upload_mb * 1024 * 1024 + 65536:
                raise UploadLimit('Upload exceeds limit')
            return message

        await self.app(scope, limited_receive, measured_send)
        log.info('mode=%s request_seconds=%.3f', self.settings.mode, time.monotonic() - started)


def create_app(settings=None, analyzer=None):
    settings = settings or Settings.from_env()
    analyzer = analyzer or Analyzer(settings)
    lock = asyncio.Lock()
    jobs = {}  # Deliberately in memory: restarting the service removes job history.
    job_deadlines = {}
    job_metadata = {}
    pending_jobs = set()

    @asynccontextmanager
    async def lifespan(app):
        if settings.mode == 'local' and not settings.api_key.strip():
            raise ValueError('ML_API_KEY is required for local API startup')
        settings.temp_dir.mkdir(parents=True, exist_ok=True)
        analyzer.load()
        log.warning('ML mode=%s ready=%s', settings.mode, analyzer.ready)
        try:
            yield
        finally:
            if pending_jobs:
                await asyncio.gather(*pending_jobs, return_exceptions=True)

    app = FastAPI(title='GoDo local ML', lifespan=lifespan)
    app.add_middleware(RequestLimits, settings=settings)
    app.state.analyzer = analyzer

    @app.get('/health', summary='Check ML service readiness',
             description=('Public endpoint; no API key or request body required. '
                          'Does not run inference. A busy model remains healthy; '
                          'check ready=true and busy=false before submitting a video. '
                          'Readiness does not confirm classification quality.'),
             responses={200: {'description': 'Model loaded; inspect busy for availability.'},
                        503: {'description': 'Model is not ready; ready is false.'}})
    async def health():
        return JSONResponse({'mode': settings.mode, 'ready': analyzer.ready,
                             'busy': lock.locked(), 'mock': settings.mode == 'mock',
                             'classification_validated': False,
                             'inference_only': True, 'weights_updated': False,
                             'assessment_schema_version': 2,
                             'verdict_semantics': 'candidate_for_human_review'},
                            status_code=200 if analyzer.ready else 503)

    async def reserve_model():
        if not analyzer.ready:
            raise HTTPException(503, 'ML service not ready')
        if lock.locked():
            raise HTTPException(503, 'ML service busy')
        await lock.acquire()

    def cleanup(path):
        try:
            if path is not None:
                path.unlink(missing_ok=True)
        finally:
            lock.release()

    async def save_upload(request, *, official=False):
        path = None
        try:
            deadline = request.scope['analysis_deadline']
            try:
                async with request.form(max_files=1, max_fields=2) as form:
                    upload = form.get('file')
                    if not isinstance(upload, UploadFile):
                        raise HTTPException(400, 'Multipart field file is required')
                    if official and Path(upload.filename or '').suffix.lower() != '.mov':
                        raise HTTPException(400, 'A .mov video is required')
                    with tempfile.NamedTemporaryFile(dir=settings.temp_dir, suffix='.video', delete=False) as out:
                        path = Path(out.name)
                        size = 0
                        while chunk := await upload.read(1024 * 1024):
                            size += len(chunk)
                            if size > settings.max_upload_mb * 1024 * 1024:
                                raise HTTPException(400, 'Video exceeds size limit')
                            if time.monotonic() >= deadline:
                                raise AnalysisTimeout()
                            out.write(chunk)
                    if not size:
                        raise HTTPException(400, 'Empty video')
            except (UploadLimit, ValueError):
                if request.scope.get('upload_timed_out'):
                    raise AnalysisTimeout() from None
                raise HTTPException(400, 'Invalid or oversized multipart upload') from None
            except StarletteHTTPException:
                if request.scope.get('upload_timed_out') or time.monotonic() >= deadline:
                    raise AnalysisTimeout() from None
                raise HTTPException(400, 'Invalid multipart upload') from None
            return path
        except BaseException:
            if path is not None:
                path.unlink(missing_ok=True)
            raise

    async def execute_model(path, scope):
        deadline = scope['analysis_deadline']

        def measured_analysis():
            if settings.mode != 'local':
                return analyzer.analyze_video(path, deadline)
            import torch
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            try:
                return analyzer.analyze_video(path, deadline)
            finally:
                torch.cuda.synchronize()
                scope['peak_vram_mib'] = torch.cuda.max_memory_allocated() / 1024**2
                settings.audit_dir.mkdir(parents=True, exist_ok=True)
                audit = settings.audit_dir / f'{path.stem}.json'
                audit.write_text(json.dumps({'schema_version': 2, 'inference_only': True,
                                             'coverage_complete': analyzer.coverage_complete,
                                             'windows': analyzer.assessment_audit,
                                             'unparsed_windows': analyzer.unparsed_audit,
                                             'protocol_normalizations': analyzer.protocol_audit,
                                             'model_wire_outputs': analyzer.wire_audit,
                                             'action_screenings': getattr(analyzer, 'action_audit', []),
                                             'stage_metrics': getattr(analyzer, 'stage_metrics', {})},
                                            ensure_ascii=False), encoding='utf-8')
                scope['assessment_file'] = audit.relative_to(Path(__file__).parent).as_posix()

        # Keep the GPU reservation and upload until the actual worker has finished.
        task = asyncio.create_task(asyncio.to_thread(measured_analysis))
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
        if cancelled:
            task.exception()
            raise asyncio.CancelledError()
        result = task.result()
        if time.monotonic() >= deadline:
            raise AnalysisTimeout('Analysis time limit exceeded')
        return AnalysisResult.model_validate(result).model_dump(exclude_none=True)

    @app.post('/analyze', response_model=AnalysisResult, response_model_exclude_none=True,
              response_model_exclude={'uncertain_rule_ids': True,
                                      'violations': {'__all__': {'rule_id'}}}, openapi_extra={
        'requestBody': {'required': True, 'content': {'multipart/form-data': {'schema': {
            'type': 'object', 'required': ['file'],
            'properties': {'file': {'type': 'string', 'format': 'binary'}}}}}}})
    async def analyze(request: Request):
        await reserve_model()
        path = None
        try:
            path = await save_upload(request)
            return await execute_model(path, request.scope)
        except InvalidVideo:
            raise HTTPException(400, 'Invalid or unsupported video') from None
        except (AnalysisTimeout, TimeoutError):
            raise HTTPException(500, 'Analysis time limit exceeded') from None
        except AnalysisInconclusive:
            raise HTTPException(500, {'code': 'analysis_inconclusive',
                                     'message': 'Visual criteria remain unknown; human review required'}) from None
        except HTTPException:
            raise
        except Exception:  # noqa: BLE001 -- API privacy boundary, return safe 500
            # Never log model output, file contents, or exceptions containing private data.
            log.error('ML processing failed')
            raise HTTPException(500, 'Video processing failed') from None
        finally:
            cleanup(path)

    async def run_job(job_id, path, scope):
        try:
            result = await execute_model(path, scope)
            violations = []
            for event in result['violations']:
                source = event['rule']
                violations.append({key: event[key] for key in
                                   ('type', 'timecode_sec', 'description', 'evidence_frames')})
                violations[-1]['description'] += (
                    f"\nИсточник: {source['title']}; {source['section']}; {source['url']}")
            jobs[job_id] = {'job_id': job_id, 'status': 'done',
                            'verdict': result['verdict'], 'violations': violations}
        except Exception as exc:  # noqa: BLE001 -- job privacy boundary
            if isinstance(exc, (AnalysisTimeout, TimeoutError)) or time.monotonic() >= scope['analysis_deadline']:
                code, message = 'analysis_timeout', 'Analysis time limit exceeded'
            elif isinstance(exc, AnalysisInconclusive):
                code, message = 'analysis_inconclusive', 'Visual evidence is insufficient; human review required'
            elif isinstance(exc, InvalidVideo):
                code, message = 'invalid_video', 'Invalid or unsupported video'
            elif isinstance(exc, AnalysisError):
                code, message = 'analysis_error', 'Analysis could not be completed'
            else:
                code, message = 'inference_error', 'Video processing failed'
            jobs[job_id] = {'job_id': job_id, 'status': 'failed',
                            'error': {'code': code, 'message': message}}
        finally:
            elapsed = time.monotonic() - (scope['analysis_deadline'] - settings.timeout_sec)
            metadata = {'X-ML-Mode': settings.mode, 'X-ML-Request-Seconds': f'{elapsed:.6f}'}
            if scope.get('peak_vram_mib') is not None:
                metadata['X-ML-Peak-VRAM-MiB'] = f'{scope["peak_vram_mib"]:.3f}'
            if scope.get('assessment_file'):
                metadata['X-ML-Assessment-File'] = scope['assessment_file']
            job_metadata[job_id] = metadata
            cleanup(path)

    @app.post('/api/v1/analyze', status_code=202,
              description='Submit a .mov video. Job history is in memory and is lost on restart.',
              openapi_extra={'requestBody': {'required': True, 'content': {
                  'multipart/form-data': {'schema': {'type': 'object', 'required': ['file'],
                      'properties': {'file': {'type': 'string', 'format': 'binary'}}}}}}})
    async def submit_job(request: Request):
        await reserve_model()
        path = None
        accepted = False
        try:
            path = await save_upload(request, official=True)
            job_id = secrets.token_hex(16)
            jobs[job_id] = {'job_id': job_id, 'status': 'processing'}
            job_deadlines[job_id] = request.scope['analysis_deadline']
            job_scope = {'analysis_deadline': request.scope['analysis_deadline']}
            task = asyncio.create_task(run_job(job_id, path, job_scope))
            pending_jobs.add(task)
            task.add_done_callback(pending_jobs.discard)
            accepted = True
            return {'job_id': job_id}
        except (AnalysisTimeout, TimeoutError):
            raise HTTPException(408, 'Upload time limit exceeded') from None
        finally:
            if not accepted:
                cleanup(path)

    @app.get('/api/v1/jobs/{job_id}')
    async def get_job(job_id: str):
        if job_id not in jobs:
            raise HTTPException(404, 'Unknown job')
        if jobs[job_id]['status'] == 'processing' and time.monotonic() >= job_deadlines[job_id]:
            # Report the deadline even if a native decoder/GPU operation is still unwinding.
            # The worker retains its reservation and upload until it actually exits.
            jobs[job_id] = {'job_id': job_id, 'status': 'failed', 'error': {
                'code': 'analysis_timeout', 'message': 'Analysis time limit exceeded'}}
        return JSONResponse(jobs[job_id], headers=job_metadata.get(job_id, {}))

    return app


app = create_app()
