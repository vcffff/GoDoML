import json
import time

import cv2
import numpy as np
import pytest
from PIL import Image

from cashml.config import ROOT, Settings
from cashml.pipeline import AnalysisError, Analyzer
from cashml.schema import AnalysisResult
from cashml.video import Frame
from review_labels import seconds


def configured():
    settings = Settings(mode='mock', rules_file=ROOT / 'rules.json',
                        window_sec=1, stride_sec=0.5, frames_per_window=3)
    analyzer = Analyzer(settings)
    analyzer.load()
    settings.mode = 'local'  # Test pipeline with stub inference, no GPU/model load.
    return analyzer


@pytest.mark.parametrize('raw', ['not json', '{"events":', '{"events":null}',
                               '{"events":[{"type_id":"unknown","frame_index":0,"description":"x"}]}',
                               '{"events":[{"type_id":"camera_phone_in_cash_area","frame_index":99,"description":"x"}]}'])
def test_bad_model_output_raises(raw):
    analyzer = configured()
    analyzer.infer = lambda *args: raw
    with pytest.raises(AnalysisError):
        analyzer._window([Frame(0.0, Image.new('RGB', (16, 16)))], time.monotonic() + 1)


def test_whole_video_coverage_dedup_real_evidence(tmp_path):
    path = tmp_path / 'synthetic.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (64, 64))
    for i in range(30):
        writer.write(np.full((64, 64, 3), i * 5, np.uint8))
    writer.release()
    analyzer = configured()
    observed = []

    def infer(frames, *args):
        observed.append([f.timestamp for f in frames])
        return json.dumps({'assessments': [{
            'type_id': key, 'subject_id': 'phone_left' if key == 'camera_phone_in_cash_area' else 'none',
            'observed': key == 'camera_phone_in_cash_area',
            'criterion_met': key == 'camera_phone_in_cash_area',
            'supporting_timestamps_sec': [frames[1].timestamp] if key == 'camera_phone_in_cash_area' else [],
            'explanation': 'Synthetic test',
            'applicability': {condition: True for condition in rule['applicability_conditions']}}
            for key, rule in analyzer.rules.items()]})

    analyzer.infer = infer
    result = AnalysisResult.model_validate(analyzer.analyze_video(path))
    assert len(observed) == 5
    assert observed[0][0] == 0
    assert observed[-1][-1] == pytest.approx(2.9)
    assert len(result.violations) == 1
    assert len(result.violations[0].evidence_frames) == 3
    assert all(0 <= f.timestamp_sec < 3 for f in result.violations[0].evidence_frames)
    analyzer.infer = lambda *args: json.dumps({'assessments': [{
        'type_id': key, 'subject_id': 'none', 'observed': False, 'criterion_met': False,
        'supporting_timestamps_sec': [], 'explanation': 'Absent synthetic target',
        'applicability': {condition: 'unknown' for condition in rule['applicability_conditions']}}
        for key, rule in analyzer.rules.items()]})
    assert analyzer.analyze_video(path) == {'verdict': 'no_violation', 'violations': []}


@pytest.mark.parametrize('value,expected', [('11 секунд', 11), ('1минут 2 секунд', 62),
                                          ('1:02', 62), (None, None), ('5 секунд', 5)])
def test_label_timestamp_parser(value, expected):
    assert seconds(value) == expected
