"""Synthetic refinement regressions; no private videos or model output are used."""

import json

import pytest
from PIL import Image
from test_criteria import analyzer as analyzer  # noqa: PLC0414 -- re-export shared pytest fixture
from test_criteria import assessment

from cashml.pipeline import AnalysisInconclusive
from cashml.video import Frame, Video


@pytest.fixture
def sampled_video(monkeypatch):
    class SyntheticVideo:
        duration = 12.1
        fps = 10
        sample = Video.sample

        def __init__(self, *args):
            pass

        def frame(self, second, max_side):
            return Frame(round(second * self.fps) / self.fps,
                         Image.new('RGB', (16, 16)))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr('cashml.pipeline.Video', SyntheticVideo)
    return SyntheticVideo


@pytest.mark.parametrize(('coarse_frames', 'refine_frames'), [(4, 6), (5, 16)])
def test_refinement_retains_coarse_evidence_within_frame_budget(
        analyzer, sampled_video, coarse_frames, refine_frames):
    analyzer.settings.window_sec = 12
    analyzer.settings.frames_per_window = coarse_frames
    analyzer.settings.refine_frames = refine_frames
    sampled_times = []
    expected_times = [12.0 * i / (coarse_frames - 1) for i in range(coarse_frames)]
    evidence_times = expected_times[1:3]

    def infer(frames, *args):
        times = [frame.timestamp for frame in frames]
        sampled_times.append(times)
        present = set(evidence_times) <= set(times)
        item = assessment(observed=present, criterion_met=present,
                          timestamps=evidence_times if present else [],
                          subject='object_a' if present else 'none')
        return json.dumps({'assessments': [item]})

    analyzer.infer = infer
    result = analyzer.analyze_video('synthetic')

    assert result['verdict'] == 'violation'
    assert sampled_times[0] == expected_times
    assert set(sampled_times[0]) <= set(sampled_times[1])
    assert len(sampled_times[1]) == refine_frames
    assert sampled_times[1] == sorted(set(sampled_times[1]))
    assert result['violations'][0]['timecode_sec'] == evidence_times[0]
    assert [item['timestamp_sec'] for item in result['violations'][0]['evidence_frames']] == evidence_times


def test_refinement_deduplicates_actual_decoded_timestamps(
        analyzer, sampled_video, monkeypatch):
    analyzer.settings.window_sec = 12
    analyzer.settings.frames_per_window = 4
    analyzer.settings.refine_frames = 6
    sampled_times = []

    def sample(self, start, end, count, max_side):
        times = [0.0, 0.0, 4.0, 8.0] if count == 4 else [0.0, 0.0, 4.0, 4.0, 8.0, 12.0]
        return [Frame(timestamp, Image.new('RGB', (16, 16))) for timestamp in times]

    def infer(frames, *args):
        sampled_times.append([frame.timestamp for frame in frames])
        return json.dumps({'assessments': [assessment(timestamps=[4.0])]})

    monkeypatch.setattr(sampled_video, 'sample', sample)
    analyzer.infer = infer
    assert analyzer.analyze_video('synthetic')['verdict'] == 'violation'
    assert sampled_times[1] == [0.0, 4.0, 8.0, 12.0]
    assert len(sampled_times[1]) <= analyzer.settings.refine_frames


@pytest.mark.parametrize('refined_subject', ['none', 'object_b'])
def test_refinement_cannot_silently_drop_a_confirmed_subject(
        analyzer, sampled_video, refined_subject):
    analyzer.settings.window_sec = 12
    analyzer.settings.frames_per_window = 4
    analyzer.settings.refine_frames = 6
    calls = []

    def infer(frames, *args):
        calls.append(len(frames))
        coarse = len(calls) == 1
        present = coarse or refined_subject != 'none'
        item = assessment(observed=present, criterion_met=present,
                          timestamps=[frames[0].timestamp] if present else [],
                          subject='object_a' if coarse else refined_subject)
        return json.dumps({'assessments': [item]})

    analyzer.infer = infer
    with pytest.raises(AnalysisInconclusive):
        analyzer.analyze_video('synthetic')

    assert calls == [4, 6]
    assert analyzer.coverage_complete is True
    assert [record['pass'] for record in analyzer.assessment_audit] == ['coarse', 'refined']
    assert analyzer.assessment_audit[0]['assessments'][0]['subject_id'] == 'object_a'
    assert analyzer.assessment_audit[1]['assessments'][0]['subject_id'] == refined_subject
