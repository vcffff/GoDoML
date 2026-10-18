"""Synthetic integration coverage for action screening and catalog assessment."""

import json
import time
from copy import deepcopy

import pytest
from PIL import Image
from test_criteria import synthetic_video as synthetic_video  # noqa: PLC0414 -- shared fixture

from cashml.config import Settings
from cashml.pipeline import (
    AnalysisError,
    AnalysisInconclusive,
    Analyzer,
    InputBudgetExceeded,
)
from cashml.schema import AnalysisResult
from cashml.video import Frame


@pytest.fixture
def analyzer():
    instance = Analyzer(Settings(mode='local', window_sec=3, stride_sec=3,
                                 frames_per_window=3, refine_frames=0))
    instance.ready = True
    instance.rules = {
        source_id: {
            'rule_id': source_id, 'type_id': source_id, 'source_rule': {'rule_id': source_id},
            'event_title': f'Exact event {index}',
            'criteria': f'Visible synthetic action sequence {index}.',
            'required_visual_observations': [f'Visible synthetic action sequence {index}.'],
            'applicability_conditions': {'condition_0': 'Synthetic visible context.'},
            'do_not_flag_when': [], 'non_visual_or_unobservable_limits': [],
            'assessment_mode': 'visual_candidate',
            'evidence_requirements': {'sequence_required': True,
                                      'absence_requires_complete_relevant_interval': True},
            'title': 'Synthetic source document', 'section': 'Synthetic section',
            'url': 'urn:test:synthetic-catalog',
        }
        for index, source_id in enumerate(('SOURCE_ALPHA', 'SOURCE_BETA', 'SOURCE_GAMMA'))
    }
    return instance


@pytest.fixture
def frames():
    return [Frame(timestamp, Image.new('RGB', (16, 16))) for timestamp in (0.0, 0.5, 1.0)]


def screen(candidates=(0,), *, clear=None, uncertain=()):
    return {'actions': ([{'description': 'Left person transfers notes to right person.',
                          'frames': [0, 2], 'rules': list(candidates)}] if candidates else []),
            'clear': ([index for index in range(3)
                       if index not in candidates and index not in uncertain]
                      if clear is None else list(clear)),
            'uncertain': list(uncertain)}


def detailed(*, o=True, m=True, f=(0, 2), a=(True,), b='action'):
    states = {True: 'Y', False: 'N', 'unknown': 'U'}
    return {'checks': ''.join(states[value] for value in (o, m, *a)),
            'frames': ''.join('1' if index in f else '0' for index in range(3)),
            'basis': {'action': 'A', 'absence': 'O', 'unknown': 'U'}[b]}


def responses(analyzer, *payloads):
    pending = iter(payloads)
    calls = []

    def infer(frames, prompt, deadline, response_shape=None):
        calls.append({'timestamps': [frame.timestamp for frame in frames], 'prompt': prompt,
                      'response_shape': response_shape})
        return json.dumps(next(pending))

    analyzer.infer = infer
    return calls


def test_catalog_routes_every_candidate_and_returns_unchanged_source_ids(analyzer, synthetic_video):
    calls = responses(analyzer, screen((0, 1, 2)),
                      detailed(), detailed(), detailed())

    result = analyzer.analyze_video(synthetic_video)

    assert result['verdict'] == 'violation'
    assert [event['rule_id'] for event in result['violations']] == list(analyzer.rules)
    assert [event['type'] for event in result['violations']] == ['Exact event 0', 'Exact event 1',
                                                               'Exact event 2']
    assert len(calls) == 4
    assert [call['response_shape'] for call in calls] == [None, (3, 3), (3, 3), (3, 3)]
    assert all(event['timecode_sec'] == calls[0]['timestamps'][0] for event in result['violations'])
    assert all([frame['timestamp_sec'] for frame in event['evidence_frames']]
               == [calls[0]['timestamps'][0], calls[0]['timestamps'][2]]
               for event in result['violations'])


@pytest.mark.parametrize(('exception_excluded', 'outcome'), [
    (True, 'violation'), (False, 'inconclusive'), ('unknown', 'inconclusive')])
def test_exceptions_must_be_ruled_out_before_a_positive(
        analyzer, synthetic_video, exception_excluded, outcome):
    analyzer.rules['SOURCE_ALPHA']['do_not_flag_when'] = ['Authorized synthetic exception.']
    calls = responses(analyzer, screen(),
                      detailed(a=[True, exception_excluded]))

    if outcome == 'inconclusive':
        with pytest.raises(AnalysisInconclusive):
            analyzer.analyze_video(synthetic_video)
    else:
        assert analyzer.analyze_video(synthetic_video)['verdict'] == outcome

    assert 'Authorized synthetic exception.' in calls[1]['prompt']
    item = next(item for item in analyzer.assessment_audit[0]['assessments']
                if item['type_id'] == 'SOURCE_ALPHA')
    assert item['applicability']['exception_excluded_0'] == exception_excluded


def test_external_confirmation_cannot_be_supplied_by_model_claim(analyzer, synthetic_video):
    analyzer.rules['SOURCE_ALPHA']['assessment_mode'] = 'requires_external_confirmation'
    calls = responses(analyzer, screen())

    with pytest.raises(AnalysisInconclusive):
        analyzer.analyze_video(synthetic_video)

    item = next(item for item in analyzer.assessment_audit[0]['assessments']
                if item['type_id'] == 'SOURCE_ALPHA')
    assert item['applicability']['external_confirmation'] == 'unknown'
    assert analyzer.action_audit[0]['actions'][0]['description'].startswith('Left person transfers')
    assert analyzer.coverage_complete is True
    assert len(calls) == 1


def test_unknown_applicability_preserves_visible_action_audit(analyzer, synthetic_video):
    responses(analyzer, screen(), detailed(a=['unknown']))
    with pytest.raises(AnalysisInconclusive):
        analyzer.analyze_video(synthetic_video)
    assert analyzer.action_audit[0]['actions'][0]['supporting_timestamps_sec'] == pytest.approx([0.0, 2.9])
    item = next(item for item in analyzer.assessment_audit[0]['assessments']
                if item['type_id'] == 'SOURCE_ALPHA')
    assert item['observed'] is item['criterion_met'] is True


@pytest.mark.parametrize('indices', [[0], [0, 0]])
def test_sequence_needs_two_distinct_evidence_times(analyzer, frames, indices):
    responses(analyzer, screen(), detailed(f=indices))
    items = analyzer._window(frames, time.monotonic() + 10)
    item = next(item for item in items if item.type_id == 'SOURCE_ALPHA')
    assert item.criterion_met == 'unknown'
    assert analyzer._positive(item) is False


def test_duplicate_decoded_frames_cannot_establish_a_sequence(analyzer, frames):
    frames[2].timestamp = frames[0].timestamp
    responses(analyzer, screen(), detailed(f=[0, 2]))
    items = analyzer._window(frames, time.monotonic() + 10)
    item = next(item for item in items if item.type_id == 'SOURCE_ALPHA')
    assert item.supporting_timestamps_sec == [0.0]
    assert item.criterion_met == 'unknown'


@pytest.mark.parametrize('basis', ['absence', 'unknown'])
def test_sparse_absence_or_unknown_basis_cannot_establish_positive(analyzer, synthetic_video, basis):
    responses(analyzer, screen(), detailed(b=basis))
    with pytest.raises(AnalysisInconclusive):
        analyzer.analyze_video(synthetic_video)


def test_complete_clear_screen_returns_no_violation_without_detail_call(analyzer, synthetic_video):
    calls = responses(analyzer, screen(()))
    assert analyzer.analyze_video(synthetic_video) == {'verdict': 'no_violation', 'violations': []}
    assert len(calls) == 1


def test_complete_uncertain_screen_is_inconclusive(analyzer, synthetic_video):
    calls = responses(analyzer, screen((), uncertain=(0, 1, 2)))
    with pytest.raises(AnalysisInconclusive):
        analyzer.analyze_video(synthetic_video)
    assert len(calls) == 1
    assert analyzer.coverage_complete is True


@pytest.mark.parametrize('payload', [
    screen((0,), clear=(1,)),
    screen((0,), clear=(1, 2, 3)),
])
def test_screen_missing_or_unknown_rule_is_a_technical_error(analyzer, frames, payload):
    responses(analyzer, payload)
    with pytest.raises(AnalysisError, match='action screening'):
        analyzer._window(frames, time.monotonic() + 10)


def test_invalid_second_rule_cannot_silently_drop_screened_candidate(analyzer, frames):
    calls = responses(analyzer, screen((0, 1)), detailed(), {})
    with pytest.raises(AnalysisInconclusive, match='rule assessment could not be validated'):
        analyzer._window(frames, time.monotonic() + 10)
    assert len(calls) == 3
    assert analyzer.unparsed_audit[-1]['rule_ids'] == ['SOURCE_BETA']


def test_generic_description_without_support_cannot_create_event(analyzer, synthetic_video):
    screening = screen()
    screening['actions'][0]['description'] = 'A person is sitting.'
    responses(analyzer, screening, detailed(f=[]))
    with pytest.raises(AnalysisInconclusive, match='rule assessment could not be validated'):
        analyzer.analyze_video(synthetic_video)


def test_visible_person_without_noncompliance_cannot_create_event(analyzer, synthetic_video):
    screening = screen()
    screening['actions'][0]['description'] = 'A person is sitting.'
    responses(analyzer, screening, detailed(m=False, f=[]))
    assert analyzer.analyze_video(synthetic_video) == {'verdict': 'no_violation', 'violations': []}


def test_incomplete_exception_checks_are_rejected(analyzer, frames):
    analyzer.rules['SOURCE_ALPHA']['do_not_flag_when'] = ['Authorized synthetic exception.']
    responses(analyzer, screen(), detailed(a=[True]))
    with pytest.raises(AnalysisInconclusive, match='rule assessment could not be validated'):
        analyzer._window(frames, time.monotonic() + 10)


def test_all_31_candidates_survive_single_rule_assessment(analyzer, synthetic_video):
    template = analyzer.rules['SOURCE_ALPHA']
    analyzer.rules = {}
    for index in range(31):
        source_id = f'ARBITRARY_SOURCE_{index}'
        rule = deepcopy(template)
        rule.update(rule_id=source_id, type_id=source_id, event_title=f'Exact catalog event {index}',
                    source_rule={'rule_id': source_id})
        analyzer.rules[source_id] = rule
    screening = screen(tuple(range(31)), clear=())
    calls = responses(analyzer, screening, *(detailed() for _ in range(31)))

    result = analyzer.analyze_video(synthetic_video)

    assert [event['rule_id'] for event in result['violations']] == list(analyzer.rules)
    assert len(calls) == 32
    assert [entry['rule_ids'] for entry in analyzer.wire_audit
            if entry['stage'] == 'rule_assessment'] == [[key] for key in analyzer.rules]


def test_detailed_prompt_preserves_nonvisual_limits(analyzer, frames):
    limit = 'Synthetic ownership cannot be established from appearance.'
    analyzer.rules['SOURCE_ALPHA']['non_visual_or_unobservable_limits'] = [limit]
    calls = responses(analyzer, screen(), detailed())
    analyzer._window(frames, time.monotonic() + 10)
    assert limit in calls[1]['prompt']


def test_supported_rule_is_returned_when_an_independent_rule_is_uncertain(analyzer, synthetic_video):
    responses(analyzer, screen((0,), uncertain=(1,)), detailed())

    result = analyzer.analyze_video(synthetic_video)

    assert result['verdict'] == 'violation'
    assert [event['rule_id'] for event in result['violations']] == ['SOURCE_ALPHA']
    assert result['uncertain_rule_ids'] == ['SOURCE_BETA']
    assert analyzer.coverage_complete is True


def test_negative_result_cannot_carry_unresolved_rules():
    with pytest.raises(ValueError, match='Unresolved rules'):
        AnalysisResult.model_validate({'verdict': 'no_violation', 'violations': [],
                                       'uncertain_rule_ids': ['SOURCE_BETA']})


@pytest.mark.parametrize('exception_excluded', [False, 'unknown'])
def test_other_supported_rule_cannot_override_a_rules_own_exception(
        analyzer, synthetic_video, exception_excluded):
    analyzer.rules['SOURCE_ALPHA']['do_not_flag_when'] = ['Authorized synthetic exception.']
    responses(analyzer, screen((0, 1)), detailed(a=[True, exception_excluded]), detailed())

    result = analyzer.analyze_video(synthetic_video)

    assert result['verdict'] == 'violation'
    assert [event['rule_id'] for event in result['violations']] == ['SOURCE_BETA']
    assert result['uncertain_rule_ids'] == ['SOURCE_ALPHA']


def test_source_constraints_and_relevant_transition_interpretation_reach_detail(analyzer, frames):
    constraints = ['Synthetic root constraint: object presence alone proves no procedural breach.',
                   'Synthetic root constraint: preserve the source exception scope.']
    transition = 'Synthetic transition: earlier procedure remains permitted during the stated interval.'
    for rule in analyzer.rules.values():
        rule['interpretation_constraints'] = constraints
    analyzer.rules['SOURCE_ALPHA']['transitional_provisions'] = [{
        'interpretation': transition, 'official_transition_text': 'Synthetic source transition.',
        'effective_until_exclusive': '2099-01-01', 'affected_sections': ['SYNTHETIC_SECTION']}]
    analyzer.rules['SOURCE_ALPHA']['legal_basis_section_ids'] = ['SYNTHETIC_SECTION']
    calls = responses(analyzer, screen((0, 1)),
                      detailed(), detailed())

    analyzer._window(frames, time.monotonic() + 10)

    assert all(constraint in calls[1]['prompt'] for constraint in constraints)
    assert transition in calls[1]['prompt']
    assert 'SYNTHETIC_SECTION' in calls[1]['prompt']


def test_legacy_source_metadata_alone_does_not_enable_catalog_dispatch(analyzer, synthetic_video):
    for rule in analyzer.rules.values():
        rule.pop('assessment_mode')
    records = [{'c': index, 's': 'left_counter', 'o': True, 'm': True,
                'f': [0, 2], 'a': [True], 'e': 'Left person transfers notes to right person.'}
               for index in range(3)]
    calls = responses(analyzer, {'assessments': records})

    result = analyzer.analyze_video(synthetic_video)

    assert result['verdict'] == 'violation'
    assert len(result['violations']) == 3
    assert all('rule_id' not in event for event in result['violations'])
    assert len(calls) == 1
    assert analyzer.action_audit == []


@pytest.mark.parametrize('scope', [False, True, 'unknown'])
def test_untyped_condition_cannot_certify_negative_with_unknown_visual_criterion(
        analyzer, synthetic_video, scope):
    analyzer.rules['SOURCE_ALPHA']['applicability_conditions']['condition_0'] = (
        'The complete relevant interval is visible.')
    responses(analyzer, screen(), detailed(
        o='unknown', m='unknown', f=[], a=[scope], b='unknown'))

    with pytest.raises(AnalysisInconclusive):
        analyzer.analyze_video(synthetic_video)


def test_failed_context_check_cannot_certify_negative_after_sparse_absence_downgrade(
        analyzer, synthetic_video):
    responses(analyzer, screen(), detailed(a=[False], b='absence'))

    with pytest.raises(AnalysisInconclusive):
        analyzer.analyze_video(synthetic_video)
    item = next(item for item in analyzer.assessment_audit[0]['assessments']
                if item['type_id'] == 'SOURCE_ALPHA')
    assert item['criterion_met'] == 'unknown'
    assert item['applicability']['condition_0'] is False


@pytest.mark.parametrize('exception_excluded', [False, True, 'unknown'])
def test_untyped_exception_check_cannot_certify_negative_with_unknown_visual_criterion(
        analyzer, synthetic_video, exception_excluded):
    analyzer.rules['SOURCE_ALPHA']['do_not_flag_when'] = ['Ownership cannot be established visually.']
    responses(analyzer, screen(), detailed(
        m='unknown', f=[], a=[True, exception_excluded], b='unknown'))

    with pytest.raises(AnalysisInconclusive):
        analyzer.analyze_video(synthetic_video)


@pytest.mark.parametrize('context', [False, True, 'unknown'])
def test_observed_compliance_can_be_negative_despite_unresolved_context(
        analyzer, synthetic_video, context):
    screening = screen()
    screening['actions'][0]['description'] = 'The visible check is completed before transfer.'
    responses(analyzer, screening, detailed(o=True, m=False, a=[context]))

    assert analyzer.analyze_video(synthetic_video) == {'verdict': 'no_violation', 'violations': []}


def test_single_rule_prompts_preserve_candidates_condition_order_and_exceptions(
        analyzer, synthetic_video):
    for index, rule in enumerate(analyzer.rules.values()):
        rule['applicability_conditions']['condition_1'] = f'Additional exact condition {index}.'
        rule['do_not_flag_when'] = [f'First original exception {index}.',
                                    f'Second original exception {index}.']
    original = deepcopy(analyzer.rules)
    detailed_calls = []

    def infer(frames, prompt, deadline, response_shape=None):
        if 'RULE CATALOG:\n' in prompt:
            return json.dumps(screen((0, 1, 2)))
        criterion = json.loads(prompt.split('\n', 1)[1].split('\nchecks length=', 1)[0])
        detailed_calls.append(criterion)
        assert response_shape == (6, 3)
        return json.dumps(detailed(a=[True, True, True, True]))

    analyzer.infer = infer
    result = analyzer.analyze_video(synthetic_video)

    assert [event['rule_id'] for event in result['violations']] == list(original)
    assert [item['rule_id'] for item in detailed_calls] == list(original)
    for index, item in enumerate(detailed_calls):
        checks = item['checks']
        assert [check['index'] for check in checks] == list(range(6))
        assert 'directly visible' in checks[0]['condition']
        assert 'NONCOMPLIANCE' in checks[1]['condition']
        assert [check['condition'] for check in checks[2:4]] == list(
            original[item['rule_id']]['applicability_conditions'].values())
        assert checks[4]['condition'].endswith(f'First original exception {index}.')
        assert checks[5]['condition'].endswith(f'Second original exception {index}.')
    assert all(entry['check_order'] == ['observed', 'criterion_met', 'condition_0', 'condition_1',
                                        'exception_excluded_0', 'exception_excluded_1']
               for entry in analyzer.wire_audit if entry['stage'] == 'rule_assessment')
    assert analyzer.rules == original


@pytest.mark.parametrize('candidates', [(0,), (0, 1)])
def test_single_rule_exceeding_budget_propagates_error_without_dropping_it(
        analyzer, synthetic_video, candidates):
    detailed_calls = []

    def infer(frames, prompt, deadline, response_shape=None):
        if 'RULE CATALOG:\n' in prompt:
            return json.dumps(screen(candidates))
        detailed_calls.append(prompt)
        raise InputBudgetExceeded('Synthetic singleton exceeds token limit')

    analyzer.infer = infer
    with pytest.raises(InputBudgetExceeded, match='singleton exceeds'):
        analyzer.analyze_video(synthetic_video)

    assert len(detailed_calls) == 1
    assert analyzer.coverage_complete is False
