"""Local decoder diagnostics without exporting images."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cashml.video import Video

if __name__ == '__main__':
    records = json.loads((ROOT / 'data/annotations_v2.json').read_text(encoding='utf-8'))['records']
    for record in records:
        with Video(ROOT / 'data' / record['video']) as video:
            for i in range(16):
                timestamp = (video.duration - 1 / video.fps) * i / 15
                try:
                    video.frame(timestamp, 224)
                except ValueError as exc:
                    print(json.dumps({'video': record['video'], 'index': i,
                        'timestamp': timestamp, 'count': video.count, 'fps': video.fps,
                        'error': type(exc).__name__}), flush=True)
                    import cv2
                    capture = cv2.VideoCapture(str(ROOT / 'data' / record['video']))
                    count, pts = 0, 0
                    while True:
                        ok, _ = capture.read()
                        if not ok:
                            break
                        count += 1
                        pts = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
                    capture.release()
                    print(json.dumps({'decoded_count': count, 'last_pts': pts}), flush=True)
