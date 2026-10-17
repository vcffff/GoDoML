"""Replay captured structured output locally on sampled frames, without loading Qwen."""
import argparse
import json
import time
from pathlib import Path

from PIL import Image

from cashml.config import Settings
from cashml.pipeline import AnalysisError, Analyzer
from cashml.video import Frame, Video

root = Path('results/fixed_subset_20261017')
parser = argparse.ArgumentParser()
parser.add_argument('--phase', default='after_protocol_v2')
args = parser.parse_args()
manifest = json.loads((root / 'inference_manifest.json').read_text(encoding='utf-8'))
settings = Settings()
analyzer = Analyzer(settings)
analyzer.load()
for case in manifest['cases']:
    path = root / args.phase / case['case_id'] / 'assessments.json'
    if not path.exists():
        continue
    audit = json.loads(path.read_text(encoding='utf-8'))
    wires = audit.get('model_wire_outputs', [])
    if wires:
        wire = wires[-1]
        frames = [Frame(timestamp, Image.new('RGB', (16, 16)))
                  for timestamp in wire['sample_timestamps_sec']]
        analyzer.infer = lambda *args, raw=wire['raw_response']: raw
        try:
            analyzer._window(frames, time.monotonic() + 10)
            print(json.dumps({'case_id': case['case_id'], 'replay': 'last_captured_window_accepted',
                              'coverage_complete': audit['coverage_complete']}))
        except AnalysisError as exc:
            print(json.dumps({'case_id': case['case_id'], 'replay': 'rejected', 'reason': str(exc)}))
        continue
    for failure in audit.get('unparsed_windows', []):
        with Video(case['video_path']) as video:
            frames = video.sample(0, min(12, video.duration - 1 / video.fps), 4, 64)
        analyzer.infer = lambda *args, raw=failure['raw_response']: raw
        try:
            analyzer._window(frames, time.monotonic() + 10)
            print(json.dumps({'case_id': case['case_id'], 'replay': 'accepted'}))
        except AnalysisError as exc:
            # Only authored static messages; no raw output, descriptions, or validators' inputs.
            print(json.dumps({'case_id': case['case_id'], 'replay': 'rejected', 'reason': str(exc)}))
