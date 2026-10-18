"""Synthetic catalog and frame fixtures for the generic action screen."""

import copy
import json

import pytest
from PIL import Image

from cashml.actions import parse_screen, screen_prompt
from cashml.video import Frame


@pytest.fixture
def rules():
    return {
        f'synthetic_{index}': {
            'event_title': f'Exact synthetic event {index}',
            'source_rule': {
                'required_visual_observations': [f'Required sequence {index}']}}
        for index in range(31)
    }


@pytest.fixture
def frames():
    return [Frame(timestamp, Image.new('RGB', (16, 16)))
            for timestamp in (1.25, 3.5, 9.75)]


def valid_screen():
    return {
        'actions': [{'description': 'Left person passes visible notes to right person.',
                     'frames': [0, 2], 'rules': [2, 30]}],
        'clear': [0, 1],
        'uncertain': list(range(3, 30)),
    }


def test_prompt_includes_every_exact_original_event_title(rules):
    prompt = screen_prompt(rules)
    catalog = json.loads(prompt.split('RULE CATALOG:\n', 1)[1])
    assert len(catalog) == 31
    for index, (entry, rule) in enumerate(zip(catalog, rules.values(), strict=True)):
        assert entry == [index, rule['event_title']]
    assert 'Unknown applicability must not hide' in prompt
    assert 'not a violation verdict' in prompt


def test_screen_omits_duplicate_observation_text_without_modifying_source_rules(rules):
    rules['synthetic_0']['required_visual_observations'] = ['Normalized observation']
    original = copy.deepcopy(rules)
    prompt = screen_prompt(rules)
    assert 'Normalized observation' not in prompt
    assert 'Required sequence 0' not in prompt
    assert rules == original
    assert 'ONE JSON object with exactly e,f,c,n,u' in prompt
    assert 'max 20 words' in prompt
    assert 'Each listed index appears once across c,n,u' in prompt
    assert 'Omitted indices remain unknown, never clear' in prompt
    assert 'Visible counting alone does NOT make access, transport, packaging, or document checks candidates' in prompt


def test_unseen_required_compliance_behavior_does_not_instruct_a_clear_result(rules):
    requirement = 'The operator visibly checks each container before moving it.'
    rules['synthetic_0']['required_visual_observations'] = [requirement]

    prompt = screen_prompt(rules)

    assert requirement not in prompt
    assert 'Required visual observations may describe compliant behavior or equipment' in prompt
    assert 'Missing expected behavior or equipment must remain a candidate or uncertain' in prompt
    assert 'never clear solely because a required observation is missing' in prompt
    assert 'clearly absent required visual behavior' not in prompt


def test_static_equipment_rules_do_not_require_motion_to_be_screened(rules):
    requirement = 'The fixed barrier visibly has an intact safety latch.'
    rules['synthetic_0']['required_visual_observations'] = [requirement]

    prompt = screen_prompt(rules)

    assert requirement not in prompt
    assert 'static equipment or layout states' in prompt
    assert 'Motion is not required for state-based rules' in prompt
    assert 'generic description such as "a person is sitting"' in prompt


def test_static_state_candidate_still_requires_real_frame_support(rules, frames):
    payload = valid_screen()
    payload['actions'][0].update(
        description='The foreground safety barrier has a visibly broken latch.', frames=[1])

    result = parse_screen(json.dumps(payload), frames, rules)

    assert result['actions'][0]['supporting_timestamps_sec'] == [frames[1].timestamp]
    payload['actions'][0]['frames'] = []
    with pytest.raises(ValueError):
        parse_screen(json.dumps(payload), frames, rules)


def test_screen_uses_only_actual_evidence_timestamps(rules, frames):
    result = parse_screen(json.dumps(valid_screen()), frames, rules)
    assert result['actions'][0] == {
        **valid_screen()['actions'][0], 'supporting_timestamps_sec': [1.25, 9.75]}
    assert result['clear'] == [0, 1]
    assert result['uncertain'] == list(range(3, 30))


def test_candidates_can_repeat_across_separate_observed_actions(rules, frames):
    payload = valid_screen()
    payload['actions'].append({
        'description': 'Person at rear counter hands notes across the counter.',
        'frames': [1, 2], 'rules': [2]})
    result = parse_screen(json.dumps(payload), frames, rules)
    assert len(result['actions']) == 2
    assert result['actions'][1]['supporting_timestamps_sec'] == [3.5, 9.75]


def test_clear_or_uncertain_complete_screen_needs_no_invented_actions(rules, frames):
    for disposition in ('clear', 'uncertain'):
        payload = {'actions': [], 'clear': [], 'uncertain': []}
        payload[disposition] = list(range(31))
        assert parse_screen(json.dumps(payload), frames, rules) == payload


def test_fenced_json_is_supported(rules, frames):
    raw = '```json\n' + json.dumps(valid_screen()) + '\n```'
    assert parse_screen(raw, frames, rules)['actions'][0]['rules'] == [2, 30]


@pytest.mark.parametrize('changes', [
    {'description': ''}, {'description': ' \n\t '}, {'description': 1},
    {'frames': []}, {'frames': [True]}, {'frames': ['0']}, {'frames': [0.0]},
    {'frames': [-1]}, {'frames': [3]}, {'frames': [0, 0]},
    {'frames': list(range(17))},
    {'rules': []}, {'rules': [True]}, {'rules': ['2']}, {'rules': [2.0]},
    {'rules': [-1]}, {'rules': [31]}, {'rules': [2, 2, 30]},
    {'supporting_timestamps_sec': [999.0]}, {'unknown': False},
])
def test_invalid_actions_are_rejected(rules, frames, changes):
    payload = valid_screen()
    payload['actions'][0].update(changes)
    with pytest.raises(ValueError):
        parse_screen(json.dumps(payload), frames, rules)


@pytest.mark.parametrize('field', ['description', 'frames', 'rules'])
def test_missing_action_fields_are_rejected(rules, frames, field):
    payload = valid_screen()
    payload['actions'][0].pop(field)
    with pytest.raises(ValueError):
        parse_screen(json.dumps(payload), frames, rules)


@pytest.mark.parametrize(('field', 'values'), [
    ('clear', [0]), ('clear', [0, 1, 2]), ('clear', [0, 1, 3]),
    ('clear', [0, 1, 31]), ('clear', [0, 1, -1]), ('clear', [0, 1, 1]),
    ('clear', [False, 1]), ('clear', ['0', 1]), ('clear', [0.0, 1]),
    ('uncertain', list(range(3, 30)) + [3]),
    ('uncertain', list(range(3, 30)) + [2]),
    ('uncertain', [True] + list(range(3, 30))),
    ('uncertain', ['3'] + list(range(4, 30))),
    ('uncertain', [3.0] + list(range(4, 30))),
])
def test_invalid_or_incomplete_rule_dispositions_are_rejected(rules, frames, field, values):
    payload = valid_screen()
    payload[field] = values
    with pytest.raises(ValueError):
        parse_screen(json.dumps(payload), frames, rules)


@pytest.mark.parametrize('field', ['actions', 'clear', 'uncertain'])
def test_missing_screen_fields_are_rejected(rules, frames, field):
    payload = valid_screen()
    payload.pop(field)
    with pytest.raises(ValueError):
        parse_screen(json.dumps(payload), frames, rules)


def test_unknown_screen_keys_and_excess_actions_are_rejected(rules, frames):
    extra = valid_screen()
    extra['verdict'] = 'no_violation'
    excessive = valid_screen()
    excessive['actions'] = [copy.deepcopy(excessive['actions'][0]) for _ in range(17)]
    for payload in (extra, excessive):
        with pytest.raises(ValueError):
            parse_screen(json.dumps(payload), frames, rules)


@pytest.mark.parametrize('raw', ['not JSON', '[]', 'null', '{}'])
def test_non_screen_payloads_are_rejected(rules, frames, raw):
    with pytest.raises(ValueError):
        parse_screen(raw, frames, rules)


def compact_screen():
    original = valid_screen()
    action = original['actions'][0]
    return {'e': action['description'], 'f': action['frames'], 'c': action['rules'],
            'n': original['clear'], 'u': original['uncertain']}


def test_compact_screen_normalizes_to_existing_action_and_timestamp_contract(rules, frames):
    result = parse_screen(json.dumps(compact_screen()), frames, rules)
    assert result.pop('omitted_rule_indices') == []
    assert result == parse_screen(json.dumps(valid_screen()), frames, rules)
    assert result['actions'][0]['supporting_timestamps_sec'] == [1.25, 9.75]


@pytest.mark.parametrize(('description', 'indices'), [('', []), ('Visible foreground counter.', [1])])
def test_compact_screen_without_candidates_allows_optional_observation(rules, frames, description, indices):
    payload = {'e': description, 'f': indices, 'c': [], 'n': [0, 1], 'u': list(range(2, 31))}
    assert parse_screen(json.dumps(payload), frames, rules) == {
        'actions': [], 'clear': [0, 1], 'uncertain': list(range(2, 31)),
        'omitted_rule_indices': []}


@pytest.mark.parametrize('field', ['e', 'f', 'c', 'n', 'u'])
def test_compact_screen_missing_keys_are_rejected(rules, frames, field):
    payload = compact_screen()
    payload.pop(field)
    with pytest.raises(ValueError):
        parse_screen(json.dumps(payload), frames, rules)


@pytest.mark.parametrize('changes', [
    {'e': ''}, {'e': ' \n '}, {'e': False}, {'f': []},
    {'f': [False, 2]}, {'f': ['0', 2]}, {'f': [0.0, 2]}, {'f': [-1, 2]},
    {'f': [0, 3]}, {'f': [0, 0]}, {'f': list(range(17))},
    {'c': [True, 30]}, {'c': ['2', 30]}, {'c': [2.0, 30]}, {'c': [-1, 30]},
    {'c': [2, 31]}, {'c': [2, 2, 30]},
    {'n': [False, 1]}, {'n': ['0', 1]}, {'n': [0.0, 1]}, {'n': [0, 1, 1]},
    {'n': [0, 1, 2]}, {'n': [0, 1, 3]}, {'n': [0, 1, 31]},
    {'u': [True] + list(range(4, 30))}, {'u': ['3'] + list(range(4, 30))},
    {'u': [3.0] + list(range(4, 30))}, {'u': list(range(3, 30)) + [29]},
    {'u': list(range(3, 30)) + [-1]}, {'actions': []}, {'clear': []}, {'extra': 1},
])
def test_compact_screen_rejects_malformed_or_conflicting_output(rules, frames, changes):
    payload = compact_screen()
    payload.update(changes)
    with pytest.raises(ValueError):
        parse_screen(json.dumps(payload), frames, rules)


@pytest.mark.parametrize('indices', [[False], ['0'], [0.0], [-1], [3], [0, 0]])
def test_unused_compact_evidence_indices_are_still_validated(rules, frames, indices):
    payload = {'e': '', 'f': indices, 'c': [], 'n': [], 'u': list(range(31))}
    with pytest.raises(ValueError):
        parse_screen(json.dumps(payload), frames, rules)


def test_empty_compact_screen_preserves_every_rule_as_unknown(rules, frames):
    result = parse_screen(json.dumps({'e': '', 'f': [], 'c': [], 'n': [], 'u': []}), frames, rules)
    assert result == {'actions': [], 'clear': [], 'uncertain': list(range(31)),
                      'omitted_rule_indices': list(range(31))}


def test_partial_compact_candidates_preserve_source_indices_and_unknown_omissions(rules, frames):
    payload = {'e': 'Left person transfers visible notes.', 'f': [0, 2],
               'c': [1, 7, 30], 'n': [], 'u': []}
    result = parse_screen(json.dumps(payload), frames, rules)
    missing = [index for index in range(31) if index not in payload['c']]

    assert result['actions'][0]['rules'] == [1, 7, 30]
    assert [list(rules)[index] for index in result['actions'][0]['rules']] == [
        'synthetic_1', 'synthetic_7', 'synthetic_30']
    assert result['actions'][0]['supporting_timestamps_sec'] == [1.25, 9.75]
    assert result['clear'] == []
    assert result['uncertain'] == result['omitted_rule_indices'] == missing


def test_compact_completion_audits_only_unlisted_indices(rules, frames):
    payload = {'e': '', 'f': [], 'c': [], 'n': [1], 'u': [3]}
    result = parse_screen(json.dumps(payload), frames, rules)
    assert result['uncertain'] == [index for index in range(31) if index != 1]
    assert result['omitted_rule_indices'] == [index for index in range(31) if index not in (1, 3)]
    assert result['clear'] == [1]


def test_legacy_long_screen_still_rejects_omitted_rules(rules, frames):
    payload = valid_screen()
    payload['uncertain'] = []
    with pytest.raises(ValueError, match='Incomplete'):
        parse_screen(json.dumps(payload), frames, rules)
