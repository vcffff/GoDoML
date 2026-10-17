"""Private local smoke report. Never print model responses or video contents."""
import argparse
import html
import json
import time
from pathlib import Path

from cashml.config import ROOT, Settings
from cashml.pipeline import Analyzer
from cashml.video import Video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=Path)
    parser.add_argument('--second', type=float, default=11)
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()
    settings = Settings.from_env()
    settings.mode = 'local'
    candidates = [p for p in settings.videos_dir.iterdir()
                  if p.stem.replace('X', 'Х').casefold() == 'х-46']
    path = args.video or (candidates[0] if candidates else None)
    if path is None:
        raise SystemExit('Smoke video not found; pass --video')
    out = ROOT / 'results/smoke'
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    import torch
    torch.cuda.reset_peak_memory_stats()
    analyzer = Analyzer(settings)
    try:
        analyzer.load()
        loaded = time.monotonic()
        with Video(path, settings.max_duration_sec) as video:
            center = min(args.second, video.duration - 1 / video.fps)
            frames = video.sample(max(0, center - 1), min(video.duration - 1 / video.fps, center + 1),
                                  3, settings.max_side)
        text = analyzer.infer(frames,
                             'Describe only directly visible actions in these chronological frames. '
                             'Do not identify people, infer intent, account entries or legal violations. '
                             'Mention uncertainty and limitations of sampling.',
                             time.monotonic() + settings.timeout_sec)
        (out / 'description.txt').write_text(text, encoding='utf-8')
        images = ''.join(f'<figure><img src="data:image/jpeg;base64,{f.evidence()["image_base64"]}">'
                         f'<figcaption>{f.timestamp:.3f} s</figcaption></figure>' for f in frames)
        (out / 'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Local ML review</title>'
                                        '<h1>Local smoke — visual actions, no legal conclusion</h1>'
                                        + images + '<pre style="white-space:pre-wrap">' + html.escape(text)
                                        + '</pre>', encoding='utf-8')
        metrics = {'status': 'ok', 'load_seconds': round(loaded - started, 2),
                   'total_seconds': round(time.monotonic() - started, 2),
                   'peak_vram_mib': round(torch.cuda.max_memory_allocated() / 1024**2, 1)}
        if args.full:
            begin = time.monotonic()
            original_infer = analyzer.infer
            calls = []

            def recorded_infer(*call_args, **kwargs):
                raw = original_infer(*call_args, **kwargs)
                calls.append(raw)
                (out / 'window_outputs.json').write_text(json.dumps(calls, ensure_ascii=False), encoding='utf-8')
                return raw

            analyzer.infer = recorded_infer
            result = analyzer.analyze_video(path)
            (out / 'analysis.json').write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')
            metrics['analysis_seconds'] = round(time.monotonic() - begin, 2)
        (out / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        print(json.dumps(metrics))
        print('Report: results/smoke/report.html')
    except Exception as exc:  # noqa: BLE001 -- prevent private input/output from entering console
        metrics = {'status': 'error', 'error_type': type(exc).__name__,
                   'seconds': round(time.monotonic() - started, 2)}
        (out / 'metrics.json').write_text(json.dumps(metrics), encoding='utf-8')
        # Detailed technical exception stays local; it may include processor inputs.
        (out / 'error.txt').write_text(str(exc), encoding='utf-8')
        print(json.dumps(metrics))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
