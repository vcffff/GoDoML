import base64
import math
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
from PIL import Image


class InvalidVideo(ValueError):
    pass


@dataclass
class Frame:
    timestamp: float
    image: Image.Image

    def evidence(self):
        import io
        buffer = io.BytesIO()
        self.image.save(buffer, format='JPEG', quality=85)
        return {'timestamp_sec': float(self.timestamp),
                'image_base64': base64.b64encode(buffer.getvalue()).decode('ascii')}


class Video:
    def __init__(self, path, max_duration=150):
        try:
            with Path(path).open('rb') as source:
                header = source.read(16)
        except OSError:
            raise InvalidVideo('Video file is unavailable') from None
        # Reject playlists, URLs, images and text before invoking a decoder.
        avi = header[:4] == b'RIFF' and header[8:12] == b'AVI '
        mov = header[4:8] in (b'ftyp', b'moov', b'mdat', b'wide', b'free', b'skip')
        mkv = header[:4] == b'\x1a\x45\xdf\xa3'
        if len(header) < 16 or not (avi or mov or mkv):
            raise InvalidVideo('Expected MOV/MP4, AVI or MKV/WebM container')
        # FFmpeg protocol option blocks external network resources in nested inputs.
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'protocol_whitelist;file'
        self._path = str(path)
        self.cap = cv2.VideoCapture(self._path, cv2.CAP_FFMPEG)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if (not self.cap.isOpened() or not math.isfinite(count) or count <= 0
                or not math.isfinite(self.fps) or not 0 < self.fps <= 240):
            self.close()
            raise InvalidVideo('Video cannot be decoded')
        self.count = int(count)
        self.duration = self.count / self.fps
        if not 0 < self.duration <= max_duration:
            self.close()
            raise InvalidVideo('Video duration exceeds limit')
        try:
            self.frame(0)
        except Exception:
            self.close()
            raise

    def frame(self, second, max_side=448):
        index = min(self.count - 1, max(0, round(second * self.fps)))
        sought = self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, bgr = self.cap.read() if sought else (False, None)
        if not ok:
            # FPS-based seeking can overshoot valid tail frames in variable-rate files.
            # Reopen and decode the exact index; never substitute an earlier frame.
            sequential = cv2.VideoCapture(self._path, cv2.CAP_FFMPEG)
            try:
                for _ in range(index + 1):
                    ok, bgr = sequential.read()
                    if not ok:
                        raise InvalidVideo('Video frame cannot be decoded')
            except Exception:
                sequential.release()
                raise
            self.cap.release()
            self.cap = sequential
        # Decoded frame PTS where supported, frame-index/FPS fallback for CFR codecs.
        pts = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if not 0 <= pts < self.duration or (index > 0 and pts == 0):
            pts = index / self.fps
        image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        image.thumbnail((max_side, max_side))
        return Frame(pts, image)

    def sample(self, start, end, count, max_side):
        return [self.frame(start + (end - start) * i / max(1, count - 1), max_side)
                for i in range(count)]

    def close(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
