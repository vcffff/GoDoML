"""Run an actual local model through upload/job polling and retain diagnostics."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from fastapi.testclient import TestClient

    from app import create_app
    from cashml.config import Settings
    from cashml.pipeline import Analyzer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/real-video-debug.json')
    args = parser.parse_args()
    settings = Settings.from_env()
    settings.mode = 'local'
    analyzer = Analyzer(settings)
    started = time.monotonic()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TestClient(create_app(settings, analyzer)) as client:
        print(json.dumps({'stage': 'loaded', 'rules': list(analyzer.rules),
                          'adapter_loaded': analyzer.adapter_loaded,
                          'timeout_sec': settings.timeout_sec}), flush=True)
        import os
        key = os.getenv('JURY_API_KEY') or os.getenv('ML_API_KEY') or os.getenv('API_KEY')
        headers = {'X-API-Key': key} if key else {}
        with args.video.open('rb') as source:
            response = client.post('/api/v1/analyze', headers=headers,
                                   files={'file': (args.video.name, source, 'video/quicktime')})
        response.raise_for_status()
        job_id = response.json()['job_id']
        print(json.dumps({'stage': 'upload', 'http_status': response.status_code,
                          'job_id': job_id}), flush=True)
        last_report = time.monotonic()
        while True:
            job = client.get(f'/api/v1/jobs/{job_id}', headers=headers).json()
            if job['status'] != 'processing':
                break
            if time.monotonic() - last_report > 20:
                print(json.dumps({'stage': 'processing',
                                  'elapsed_sec': round(time.monotonic() - started, 1),
                                  'completed_windows': len(analyzer.assessment_audit)}), flush=True)
                last_report = time.monotonic()
            time.sleep(0.2)
        report = {'settings': {k: str(v) if isinstance(v, Path) else v
                               for k, v in vars(settings).items()},
                  'health': client.get('/health').json(),
                  'elapsed_sec': time.monotonic() - started, 'job': job,
                  'coverage_complete': analyzer.coverage_complete,
                  'windows': analyzer.assessment_audit,
                  'unparsed_windows': analyzer.unparsed_audit,
                  'model_wire_outputs': analyzer.wire_audit}
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'stage': 'finished', 'status': job['status'],
                          'error': job.get('error'), 'report': str(args.output)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
