"""CPU-only synthetic regression fixtures; no dataset or model output is read."""

import json
import time

import cv2
import numpy as np
import pytest
from PIL import Image

from cashml.config import Settings
from cashml.pipeline import AnalysisError, Analyzer
from cashml.video import Frame


def configured():
    settings = Settings(mode='mock', window_sec=1, stride_sec=1,
                        frames_per_window=3, merge_gap_sec=4)
    analyzer = Analyzer(settings)
    analyzer.load()
    settings.mode = 'local'
    return analyzer


def test_legacy_negative_description_cannot_create_event():
    analyzer = configured()
    analyzer.infer = lambda *args: json.dumps({'events': [{
        'type_id': 'personal_bag_in_cash_area', 'frame_index': 0,
        'description': 'There is no personal bag visible.'}]})
    with pytest.raises(AnalysisError):
        analyzer._window([Frame(0.0, Image.new('RGB', (16, 16)))],
                         time.monotonic() + 10)


@pytest.fixture
def analyzer():
    instance = configured()
    instance.rules = {'fixture': {
        'verified': True, 'event_title': 'Synthetic fixture',
        'criteria': 'A synthetic object satisfies the synthetic visual condition.',
        'applicability_conditions': {'test_area': 'Inside the synthetic test area.'},
        'title': 'TEST FIXTURE, not a regulation', 'section': 'fixture',
        'url': 'urn:cashml:test-fixture:criteria'}}
    return instance


def assessment(*, observed=True, criterion_met=True, timestamps=None,
               applicable=True, subject='object_a', type_id='fixture'):
    return {'type_id': type_id, 'observed': observed, 'criterion_met': criterion_met,
            'supporting_timestamps_sec': [0.5] if timestamps is None else timestamps,
            'explanation': 'Synthetic test fixture explanation.',
            'applicability': {'test_area': applicable}, 'subject_id': subject}


def window(analyzer, assessments):
    analyzer.infer = lambda *args: json.dumps({'assessments': assessments})
    frames = [Frame(t, Image.new('RGB', (16, 16))) for t in (0.0, 0.5, 1.0)]
    return analyzer._window(frames, time.monotonic() + 10)


@pytest.fixture
def synthetic_video(tmp_path):
    path = tmp_path / 'synthetic.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (64, 64))
    assert writer.isOpened()
    for i in range(30):
        writer.write(np.full((64, 64, 3), i * 5, np.uint8))
    writer.release()
    return path


def test_structured_explicit_absence_is_not_positive(analyzer, synthetic_video):
    item = assessment(observed=False, criterion_met=False, timestamps=[], subject='none')
    item['explanation'] = 'The synthetic bag is not present; the bag is absent.'
    analyzer.infer = lambda *args: json.dumps({'assessments': [item]})
    assert analyzer.analyze_video(synthetic_video) == {
        'verdict': 'no_violation', 'violations': []}


@pytest.mark.parametrize('changes', [
    {'observed': False, 'criterion_met': True, 'supporting_timestamps_sec': []},
    {'observed': 'true'}, {'criterion_met': 'false'}, {'observed': 1},
    {'supporting_timestamps_sec': []}, {'supporting_timestamps_sec': [0.25]},
    {'supporting_timestamps_sec': [99.0]}, {'type_id': 'unconfigured'},
    {'applicability': {}},
    {'subject_id': ''},
])
def test_invalid_semantics_rejected(analyzer, changes):
    item = assessment()
    item.update(changes)
    with pytest.raises(AnalysisError):
        window(analyzer, [item])


def test_missing_criterion_assessment_rejected(analyzer):
    with pytest.raises(AnalysisError):
        window(analyzer, [])


def test_negative_assessment_may_cite_frames_without_creating_event(analyzer, synthetic_video):
    def infer(frames, *args):
        item = assessment(observed=False, criterion_met=False,
                          timestamps=[frames[1].timestamp], subject='none')
        return json.dumps({'assessments': [item]})

    analyzer.infer = infer
    assert analyzer.analyze_video(synthetic_video) == {'verdict': 'no_violation', 'violations': []}


def test_visible_object_not_meeting_criterion_needs_no_event_identity(analyzer, synthetic_video):
    def infer(frames, *args):
        return json.dumps({'assessments': [assessment(observed=True, criterion_met=False,
                           timestamps=[frames[1].timestamp], subject='none')]})

    analyzer.infer = infer
    assert analyzer.analyze_video(synthetic_video) == {'verdict': 'no_violation', 'violations': []}


@pytest.mark.parametrize('context', ['true', 'false', 'not enough information', None, 1])
def test_untrusted_applicability_is_unknown_never_truthy_or_false(analyzer, synthetic_video, context):
    from cashml.pipeline import AnalysisInconclusive

    def infer(frames, *args):
        return json.dumps({'assessments': [assessment(
            timestamps=[frames[1].timestamp], applicable=context)]})

    analyzer.infer = infer
    with pytest.raises(AnalysisInconclusive):
        analyzer.analyze_video(synthetic_video)
    assert analyzer.coverage_complete
    assert all(record['assessments'][0]['applicability']['test_area'] == 'unknown'
               for record in analyzer.assessment_audit)
    assert len(analyzer.protocol_audit) == 3


@pytest.mark.parametrize('changes', [
    {'observed': 'unknown', 'criterion_met': 'unknown', 'supporting_timestamps_sec': []},
    {'criterion_met': 'unknown', 'supporting_timestamps_sec': []},
    {'applicability': {'test_area': 'unknown'}},
])
def test_unknown_is_inconclusive_after_full_coverage(analyzer, synthetic_video, changes):
    from cashml.pipeline import AnalysisInconclusive

    calls = []

    def infer(frames, *args):
        calls.append(frames[0].timestamp)
        item = assessment(timestamps=[frames[1].timestamp])
        item.update(changes)
        return json.dumps({'assessments': [item]})

    analyzer.infer = infer
    with pytest.raises(AnalysisInconclusive):
        analyzer.analyze_video(synthetic_video)
    assert len(calls) == 3


def test_inapplicable_rule_blocks_event(analyzer, synthetic_video):
    def infer(frames, *args):
        return json.dumps({'assessments': [assessment(
            timestamps=[frames[1].timestamp], applicable=False)]})

    analyzer.infer = infer
    assert analyzer.analyze_video(synthetic_video)['verdict'] == 'no_violation'


def test_visible_object_without_visual_criterion_is_not_violation(analyzer, synthetic_video):
    def infer(frames, *args):
        return json.dumps({'assessments': [assessment(
            criterion_met=False, timestamps=[frames[1].timestamp])]})

    analyzer.infer = infer
    assert analyzer.analyze_video(synthetic_video)['verdict'] == 'no_violation'


def test_unknown_prevents_partial_positive_success(analyzer, synthetic_video):
    from cashml.pipeline import AnalysisInconclusive

    def infer(frames, *args):
        unknown = frames[0].timestamp > 1.5
        return json.dumps({'assessments': [assessment(
            observed='unknown' if unknown else True,
            criterion_met='unknown' if unknown else True,
            timestamps=[] if unknown else [frames[-1].timestamp])]})

    analyzer.infer = infer
    with pytest.raises(AnalysisInconclusive):
        analyzer.analyze_video(synthetic_video)


def test_inconclusive_reports_exact_window_rule_and_field(analyzer, synthetic_video):
    from cashml.pipeline import AnalysisInconclusive

    def infer(frames, *args):
        return json.dumps({'assessments': [assessment(
            timestamps=[frames[1].timestamp],
            applicable='unknown' if frames[0].timestamp > 1.5 else True)]})

    analyzer.infer = infer
    with pytest.raises(AnalysisInconclusive) as failure:
        analyzer.analyze_video(synthetic_video)
    assert failure.value.issues == [{
        'rule_id': 'fixture', 'start_sec': 2.0, 'end_sec': 2.9,
        'pass': 'coarse', 'unknown_fields': ['applicability.test_area']}]
    assert analyzer.coverage_complete


def test_inconclusive_diagnostics_exclude_irrelevant_context(analyzer, synthetic_video):
    from cashml.pipeline import AnalysisInconclusive

    def infer(frames, *args):
        return json.dumps({'assessments': [assessment(
            observed='unknown', criterion_met='unknown', timestamps=[], applicable='unknown')]})

    analyzer.infer = infer
    with pytest.raises(AnalysisInconclusive) as failure:
        analyzer.analyze_video(synthetic_video)
    assert len(failure.value.issues) == 3
    assert all(issue['unknown_fields'] == ['observed', 'criterion_met']
               for issue in failure.value.issues)


def test_confirmed_frame_controls_timecode_and_adjacent_duplicates_merge(
        analyzer, synthetic_video):
    def infer(frames, *args):
        return json.dumps({'assessments': [assessment(timestamps=[frames[-1].timestamp])]})

    analyzer.infer = infer
    events = analyzer.analyze_video(synthetic_video)['violations']
    assert len(events) == 1
    assert events[0]['timecode_sec'] == pytest.approx(1.0)
    evidence_times = [frame['timestamp_sec'] for frame in events[0]['evidence_frames']]
    assert evidence_times == pytest.approx([1.0, 2.0, 2.9])


def test_known_negative_window_breaks_continuous_event(analyzer, synthetic_video):
    def infer(frames, *args):
        negative = 0.9 < frames[0].timestamp < 1.1
        return json.dumps({'assessments': [assessment(
            observed=not negative, criterion_met=not negative,
            timestamps=[] if negative else [frames[-1].timestamp])]})

    analyzer.infer = infer
    events = analyzer.analyze_video(synthetic_video)['violations']
    assert len(events) == 2
    assert [event['timecode_sec'] for event in events] == pytest.approx([1.0, 2.9])


def test_separate_objects_receive_separate_supporting_evidence(analyzer, synthetic_video):
    def infer(frames, *args):
        return json.dumps({'assessments': [
            assessment(timestamps=[frames[0].timestamp], subject='object_a'),
            assessment(timestamps=[frames[-1].timestamp], subject='object_b')]})

    analyzer.infer = infer
    events = analyzer.analyze_video(synthetic_video)['violations']
    assert len(events) == 2
    assert events[0]['timecode_sec'] == pytest.approx(0)
    assert events[1]['timecode_sec'] == pytest.approx(1)
    times_a = [frame['timestamp_sec'] for frame in events[0]['evidence_frames']]
    times_b = [frame['timestamp_sec'] for frame in events[1]['evidence_frames']]
    assert times_a == pytest.approx([0, 1, 2])
    assert times_b == pytest.approx([1, 2, 2.9])


def test_root_assessment_array_explicit_absence_is_negative(analyzer, synthetic_video):
    item = assessment(observed=False, criterion_met=False, timestamps=[], subject='none')
    analyzer.infer = lambda *args: json.dumps([item])
    assert analyzer.analyze_video(synthetic_video) == {
        'verdict': 'no_violation', 'violations': []}


@pytest.mark.parametrize('payload', [
    [],
    [assessment(observed='true')],
    [{'type_id': 'fixture', 'frame_index': 0, 'description': 'No object present.'}],
])
def test_root_array_does_not_bypass_strict_validation(analyzer, payload):
    analyzer.infer = lambda *args: json.dumps(payload)
    frames = [Frame(t, Image.new('RGB', (16, 16))) for t in (0.0, 0.5, 1.0)]
    with pytest.raises(AnalysisError):
        analyzer._window(frames, time.monotonic() + 10)


def test_refinement_cannot_erase_other_criterion_unknown(analyzer, synthetic_video):
    from cashml.pipeline import AnalysisInconclusive

    analyzer.rules['fixture_second'] = dict(analyzer.rules['fixture'])
    analyzer.settings.refine_frames = 5
    calls = []

    def infer(frames, *args):
        calls.append(len(frames))
        if len(frames) == analyzer.settings.frames_per_window:
            items = [assessment(timestamps=[frames[1].timestamp]),
                     assessment(type_id='fixture_second', observed='unknown',
                                criterion_met='unknown', timestamps=[], subject='none')]
        else:
            items = [assessment(type_id=type_id, observed=False,
                                criterion_met=False, timestamps=[], subject='none')
                     for type_id in analyzer.rules]
        return json.dumps({'assessments': items})

    analyzer.infer = infer
    with pytest.raises(AnalysisInconclusive):
        analyzer.analyze_video(synthetic_video)
    assert calls == [3, 5, 3, 5, 3, 5]
    assert analyzer.coverage_complete is True
    coarse = [item for item in analyzer.assessment_audit if item['pass'] == 'coarse']
    refined = [item for item in analyzer.assessment_audit if item['pass'] == 'refined']
    assert len(coarse) == len(refined) == 3
    assert all(record['assessments'][1]['observed'] == 'unknown' for record in coarse)
    assert all(item['observed'] is False
               for record in refined for item in record['assessments'])


def index_assessment(indices, **kwargs):
    item = assessment(**kwargs)
    item.pop('supporting_timestamps_sec')
    item['supporting_frame_indices'] = indices
    return item


def test_frame_indices_resolve_to_real_sample_timestamps(analyzer):
    parsed = window(analyzer, [index_assessment([1, 2])])
    assert parsed[0].supporting_timestamps_sec == [0.5, 1.0]


def test_index_supported_event_uses_selected_frame_and_audits_actual_pts(
        analyzer, synthetic_video):
    analyzer.infer = lambda *args: json.dumps({
        'assessments': [index_assessment([1, 2])]})
    events = analyzer.analyze_video(synthetic_video)['violations']
    assert len(events) == 1
    assert events[0]['timecode_sec'] == pytest.approx(0.5)
    assert events[0]['evidence_frames'][0]['timestamp_sec'] == pytest.approx(0.5)
    assert analyzer.assessment_audit[0]['assessments'][0][
        'supporting_timestamps_sec'] == pytest.approx([0.5, 1.0])


@pytest.mark.parametrize('indices', [[-1], [3], ['1'], [True], [1.0], []])
def test_invalid_supporting_frame_indices_rejected(analyzer, indices):
    with pytest.raises(AnalysisError):
        window(analyzer, [index_assessment(indices)])


def test_frame_indices_and_explicit_timestamps_must_agree(analyzer):
    item = index_assessment([1, 2])
    item['supporting_timestamps_sec'] = [0.0, 0.5]
    with pytest.raises(AnalysisError):
        window(analyzer, [item])


def test_matching_frame_indices_and_timestamps_remain_valid(analyzer):
    item = index_assessment([1, 2])
    item['supporting_timestamps_sec'] = [0.5, 1.0]
    parsed = window(analyzer, [item])
    assert parsed[0].supporting_timestamps_sec == [0.5, 1.0]


def test_absence_can_cite_frame_indices_without_becoming_positive(analyzer, synthetic_video):
    analyzer.infer = lambda *args: json.dumps({'assessments': [index_assessment(
        [1, 2], observed=False, criterion_met=False, subject='none')]})
    assert analyzer.analyze_video(synthetic_video) == {
        'verdict': 'no_violation', 'violations': []}
    assert analyzer.assessment_audit[0]['assessments'][0][
        'supporting_timestamps_sec'] == pytest.approx([0.5, 1.0])
