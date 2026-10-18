"""Synthetic decoder regressions without reading private videos."""

import cv2
import numpy as np
import pytest

from cashml.video import InvalidVideo, Video


class SyntheticCapture:
    def __init__(self, *, seek_failure=None, unreadable_at=None, count=5):
        self.seek_failure = seek_failure
        self.unreadable_at = unreadable_at
        self.count = count
        self.position = 0
        self.last_index = 0
        self.released = False
        self.read_count = 0

    def isOpened(self):
        return True

    def get(self, property_id):
        return {cv2.CAP_PROP_FPS: 10.0, cv2.CAP_PROP_FRAME_COUNT: self.count,
                cv2.CAP_PROP_POS_MSEC: self.last_index * 110.0}[property_id]

    def set(self, property_id, value):
        assert property_id == cv2.CAP_PROP_POS_FRAMES
        if value == 4 and self.seek_failure == 'set':
            return False
        self.position = 5 if value == 4 and self.seek_failure == 'read' else int(value)
        return True

    def read(self):
        self.read_count += 1
        if self.position >= 5 or self.position == self.unreadable_at:
            return False, None
        self.last_index = self.position
        self.position += 1
        return True, np.full((16, 16, 3), self.last_index * 40, dtype=np.uint8)

    def release(self):
        self.released = True


@pytest.fixture
def video_path(tmp_path):
    path = tmp_path / 'synthetic.avi'
    path.write_bytes(b'RIFF\x00\x00\x00\x00AVI \x00\x00\x00\x00')
    return path


def install_captures(monkeypatch, *captures):
    pending = iter(captures)
    opened = []

    def capture(path, backend):
        assert backend == cv2.CAP_FFMPEG
        result = next(pending)
        opened.append(result)
        return result

    monkeypatch.setattr(cv2, 'VideoCapture', capture)
    return opened


@pytest.mark.parametrize('seek_failure', ['set', 'read'])
def test_failed_tail_seek_recovers_exact_frame_by_sequential_decode(
        monkeypatch, video_path, seek_failure):
    primary = SyntheticCapture(seek_failure=seek_failure)
    fallback = SyntheticCapture()
    opened = install_captures(monkeypatch, primary, fallback)

    with Video(video_path) as video:
        frame = video.frame(0.4)

    assert frame.timestamp == pytest.approx(0.44)
    assert frame.image.getpixel((0, 0)) == (160, 160, 160)
    assert fallback.read_count == 5
    assert len(opened) == 2
    assert primary.released and fallback.released


def test_sequential_fallback_does_not_skip_midstream_decode_failure(monkeypatch, video_path):
    primary = SyntheticCapture(seek_failure='read')
    fallback = SyntheticCapture(unreadable_at=2)
    install_captures(monkeypatch, primary, fallback)

    with Video(video_path) as video, pytest.raises(InvalidVideo):
        video.frame(0.4)

    assert fallback.read_count == 3
    assert primary.released and fallback.released


def test_overreported_frame_count_still_fails_instead_of_clamping_to_last_frame(monkeypatch, video_path):
    primary = SyntheticCapture(count=6)
    fallback = SyntheticCapture(count=6)
    install_captures(monkeypatch, primary, fallback)

    with Video(video_path) as video, pytest.raises(InvalidVideo):
        video.frame(0.5)

    assert fallback.read_count == 6
    assert primary.released and fallback.released


def test_successful_seek_does_not_trigger_sequential_redecode(monkeypatch, video_path):
    primary = SyntheticCapture()
    opened = install_captures(monkeypatch, primary)

    with Video(video_path) as video:
        frame = video.frame(0.2)

    assert frame.timestamp == pytest.approx(0.22)
    assert frame.image.getpixel((0, 0)) == (80, 80, 80)
    assert len(opened) == 1
