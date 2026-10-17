import asyncio
import json
import logging
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
from cashml.pipeline import AnalysisInconclusive, AnalysisTimeout, Analyzer
from cashml.schema import AnalysisResult
from cashml.video import InvalidVideo

log = logging.getLogger('cashml')


class UploadLimit(MultiPartException):
    pass


class RequestLimits:
    def __init__(self, app, settings):
        self.app, self.settings = app, settings

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope['path'] != '/analyze':
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

    @asynccontextmanager
    async def lifespan(app):
        settings.temp_dir.mkdir(parents=True, exist_ok=True)
        analyzer.load()
        log.warning('ML mode=%s ready=%s', settings.mode, analyzer.ready)
        yield

    app = FastAPI(title='GoDo local ML', lifespan=lifespan)
    app.add_middleware(RequestLimits, settings=settings)

    @app.get('/health')
    async def health():
        return JSONResponse({'mode': settings.mode, 'ready': analyzer.ready,
                             'busy': lock.locked(), 'mock': settings.mode == 'mock',
                             'classification_validated': False,
                             'inference_only': True, 'weights_updated': False,
                             'assessment_schema_version': 2,
                             'verdict_semantics': 'candidate_for_human_review'},
                            status_code=200 if analyzer.ready else 503)

    @app.post('/analyze', response_model=AnalysisResult, openapi_extra={
        'requestBody': {'required': True, 'content': {'multipart/form-data': {'schema': {
            'type': 'object', 'required': ['file'],
            'properties': {'file': {'type': 'string', 'format': 'binary'}}}}}}})
    async def analyze(request: Request):
        if lock.locked():
            raise HTTPException(503, 'ML service busy')
        await lock.acquire()
        path = None
        try:
            if not analyzer.ready:
                raise HTTPException(503, 'ML service not ready')
            deadline = request.scope['analysis_deadline']
            try:
                async with request.form(max_files=1, max_fields=2) as form:
                    upload = form.get('file')
                    if not isinstance(upload, UploadFile):
                        raise HTTPException(400, 'Multipart field file is required')
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
            # Await the actual worker even if client disconnects/cancels: keep GPU lock and file.
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
                    request.scope['peak_vram_mib'] = torch.cuda.max_memory_allocated() / 1024**2
                    settings.audit_dir.mkdir(parents=True, exist_ok=True)
                    audit = settings.audit_dir / f'{path.stem}.json'
                    audit.write_text(json.dumps({'schema_version': 2, 'inference_only': True,
                                                 'coverage_complete': analyzer.coverage_complete,
                                                 'windows': analyzer.assessment_audit,
                                                 'unparsed_windows': analyzer.unparsed_audit,
                                                 'protocol_normalizations': analyzer.protocol_audit,
                                                 'model_wire_outputs': analyzer.wire_audit},
                                                ensure_ascii=False), encoding='utf-8')
                    request.scope['assessment_file'] = audit.relative_to(Path(__file__).parent).as_posix()

            task = asyncio.create_task(asyncio.to_thread(measured_analysis))
            cancelled = False
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    cancelled = True
            if cancelled:
                task.exception()  # Retrieve any failure without exposing private details.
                raise asyncio.CancelledError()
            return task.result()
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
            try:
                if path is not None:
                    path.unlink(missing_ok=True)
            finally:
                lock.release()

    return app


app = create_app()
