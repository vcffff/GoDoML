"""Private sequential HTTP benchmark. No labels are imported or read here."""
import argparse
import dataclasses
import datetime
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import httpx

from cashml.config import ROOT, Settings
from cashml.schema import AnalysisResult


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def prepare(settings, destination, url):
    cases = []
    for path in settings.videos_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in {'.mov', '.mp4', '.avi', '.mkv', '.webm'}:
            continue
        cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = count / fps if fps > 0 else 0
            if not cap.isOpened() or not math.isfinite(duration) or duration <= 0:
                raise ValueError('Invalid metadata')
        finally:
            cap.release()
        cases.append({'video_path': str(path.resolve()), 'duration_sec': duration,
                      'bytes': path.stat().st_size})
    cases.sort(key=lambda case: -case['duration_sec'])
    if not cases:
        raise ValueError('No videos')
    for index, case in enumerate(cases, 1):
        case['case_id'] = f'case_{index:03d}'
    destination.mkdir(parents=True, exist_ok=False)
    sources = [ROOT / 'app.py', *sorted((ROOT / 'cashml').glob('*.py')), settings.rules_file]
    snapshot = {
        'created_at_utc': datetime.datetime.now(datetime.UTC).isoformat(),
        'url': url,
        'settings': {key: str(value) if isinstance(value, Path) else value
                     for key, value in dataclasses.asdict(settings).items() if key != 'api_key'},
        'provenance_only_sha256': {str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path):
                                   hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        'order': 'descending metadata duration, longest first',
        'inference_metadata': 'opaque multipart filename input.video; no labels or hashes sent',
        'temporal_match_tolerance_sec': 5.0,
        'request_sla_sec': 300.0,
    }
    write_json(destination / 'config_snapshot.json', snapshot)
    write_json(destination / 'manifest.json', {'cases': cases})
    return cases


def run(destination, cases, url):
    parsed = urlparse(url)
    if parsed.hostname not in ('127.0.0.1', 'localhost', '::1'):
        raise ValueError('Benchmark must use local service')
    with httpx.Client(timeout=httpx.Timeout(330.0), trust_env=False) as client:
        health = client.get(url.rstrip('/') + '/health').json()
        if health.get('mode') != 'local' or health.get('mock') is not False or health.get('ready') is not True:
            raise ValueError('Local ready non-mock service required')
        write_json(destination / 'health.json', health)
        for index, case in enumerate(cases, 1):
            folder = destination / case['case_id']
            folder.mkdir(exist_ok=True)
            record_path = folder / 'request.json'
            if record_path.exists():
                # Never retry automatically: even an interrupted request may have run server-side.
                print(f"{case['case_id']} {index}/{len(cases)} skipped_existing", flush=True)
                continue
            record = {'case_id': case['case_id'], 'outcome': 'in_flight_unknown',
                      'started_at_utc': datetime.datetime.now(datetime.UTC).isoformat()}
            write_json(record_path, record)
            started = time.perf_counter()
            try:
                with Path(case['video_path']).open('rb') as source:
                    response = client.post(url.rstrip('/') + '/analyze',
                                           files={'file': ('input.video', source, 'application/octet-stream')})
                elapsed = time.perf_counter() - started
                record.update(http_status=response.status_code, elapsed_sec=round(elapsed, 3),
                              sla_pass=elapsed <= 300,
                              mode=response.headers.get('X-ML-Mode'),
                              peak_vram_mib=response.headers.get('X-ML-Peak-VRAM-MiB'))
                (folder / 'response.json').write_bytes(response.content)
                if response.status_code == 200 and record['mode'] == 'local':
                    AnalysisResult.model_validate_json(response.content)
                    record['outcome'] = 'success'
                else:
                    record['outcome'] = 'http_error' if response.status_code != 200 else 'mode_error'
            except Exception as exc:  # noqa: BLE001 -- privacy boundary: never print model text
                record.update(outcome='request_error_unknown', error_type=type(exc).__name__,
                              elapsed_sec=round(time.perf_counter() - started, 3), sla_pass=False)
            write_json(record_path, record)
            print(f"{case['case_id']} {index}/{len(cases)} status={record.get('http_status', 'unknown')} "
                  f"outcome={record['outcome']} elapsed={record['elapsed_sec']}s "
                  f"vram={record.get('peak_vram_mib', 'unknown')}MiB", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8001')
    parser.add_argument('--run-id', default=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    try:
        if args.resume:
            destination = args.resume.resolve()
            cases = json.loads((destination / 'manifest.json').read_text(encoding='utf-8'))['cases']
            saved = json.loads((destination / 'config_snapshot.json').read_text(encoding='utf-8'))
            if args.url != saved['url']:
                raise ValueError('Resume URL differs from snapshot')
        else:
            if Path(args.run_id).name != args.run_id or args.run_id in ('.', '..'):
                raise ValueError('Invalid run ID')
            destination = ROOT / 'results/benchmark' / args.run_id
            cases = prepare(Settings.from_env(), destination, args.url)
        print(f"Prepared {len(cases)} cases; longest duration={cases[0]['duration_sec']:.3f}s; "
              f"run={destination.relative_to(ROOT)}", flush=True)
        if not args.prepare_only:
            run(destination, cases, args.url)
    except Exception as exc:  # noqa: BLE001 -- privacy boundary: never print model text
        print(f'Benchmark stopped: {type(exc).__name__}; inspect local configuration.', flush=True)
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
