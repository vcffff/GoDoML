"""CPU-only compact protocol fixtures; no dataset, weights, or model output."""

import json
import time

import pytest
from PIL import Image

from cashml.config import Settings
from cashml.pipeline import AnalysisError, Analyzer
from cashml.video import Frame


@pytest.fixture
def analyzer():
    instance = Analyzer(Settings(mode='mock'))
    instance.rules = {
        key: {'criteria': 'Synthetic visual condition.',
              'applicability_conditions': {condition: 'Synthetic context condition.'}}
        for key, condition in [('fixture_first', 'test_area'),
                               ('fixture_second', 'test_role')]
    }
    return instance


def record(index=0, **changes):
    item = {'c': index, 's': 'object_a', 'o': True, 'm': True,
            'f': [1], 'a': [True], 'e': 'Synthetic supporting observation.'}
    item.update(changes)
    return item


def window(analyzer, items, *, root_array=False):
    wire = items if root_array else {'assessments': items}
    analyzer.infer = lambda *args: json.dumps(wire)
    frames = [Frame(t, Image.new('RGB', (16, 16))) for t in (0.0, 0.5, 1.0)]
    return analyzer._window(frames, time.monotonic() + 10)


def complete(first):
    return [first, record(1, o=False, m=False, s='none')]


@pytest.mark.parametrize('root_array', [False, True])
def test_compact_positive_expands_indices_and_condition_order(analyzer, root_array):
    items = window(analyzer, [record(), record(1)], root_array=root_array)
    assert [item.type_id for item in items] == list(analyzer.rules)
    assert items[0].applicability == {'test_area': True}
    assert items[1].applicability == {'test_role': True}
    assert all(item.supporting_timestamps_sec == [0.5] for item in items)
    assert all(analyzer._positive(item) for item in items)


@pytest.mark.parametrize('states', [
    {'o': False, 'm': False, 'e': 'There is no synthetic bag present.'},
    {'o': True, 'm': False, 'e': 'The synthetic bag does not satisfy the criterion.'},
    {'o': True, 'm': True, 'a': [False]},
])
def test_compact_negative_or_inapplicable_never_creates_positive(analyzer, states):
    parsed = window(analyzer, complete(record(**states)))
    assert not any(analyzer._positive(item) for item in parsed)
    assert not any(analyzer._unknown(item) for item in parsed)
    assert parsed[0].supporting_timestamps_sec == [0.5]


@pytest.mark.parametrize('states', [
    {'o': 'unknown', 'm': 'unknown', 'f': []},
    {'m': 'unknown'},
    {'a': ['unknown']},
    {'a': ['true']},
    {'a': ['false']},
    {'a': ['insufficient context']},
    {'a': [None]},
    {'a': [1]},
])
def test_compact_unknown_is_preserved_without_negative_coercion(analyzer, states):
    parsed = window(analyzer, complete(record(**states)))
    assert analyzer._unknown(parsed[0])
    assert not analyzer._positive(parsed[0])
    if 'a' in states:
        assert parsed[0].applicability == {'test_area': 'unknown'}


@pytest.mark.parametrize('missing', ['c', 's', 'o', 'm', 'f', 'a', 'e'])
def test_compact_requires_every_field(analyzer, missing):
    item = record()
    del item[missing]
    with pytest.raises(AnalysisError):
        window(analyzer, complete(item))


@pytest.mark.parametrize('changes', [
    {'c': True}, {'c': False}, {'c': -1}, {'c': 2}, {'c': 0.0}, {'c': '0'},
    {'a': []}, {'a': [True, True]}, {'a': {'test_area': True}},
    {'unexpected': True}, {'type_id': 'fixture_first'}, {'observed': True},
    {'supporting_timestamps_sec': [0.5]},
    {'o': 'true'}, {'o': 'false'}, {'o': 1},
    {'m': 'true'}, {'m': 'false'}, {'m': 0},
    {'f': [-1]}, {'f': [3]}, {'f': [True]}, {'f': [1.0]}, {'f': ['1']},
    {'f': []},
])
def test_compact_invalid_or_mixed_records_rejected(analyzer, changes):
    with pytest.raises(AnalysisError):
        window(analyzer, complete(record(**changes)))


def test_compact_requires_full_criterion_coverage(analyzer):
    with pytest.raises(AnalysisError):
        window(analyzer, [record()])


def test_canonical_wire_still_accepted(analyzer):
    items = [
        {'type_id': key, 'subject_id': 'object_a', 'observed': True,
         'criterion_met': True, 'supporting_frame_indices': [1],
         'applicability': {condition: True}, 'explanation': 'Synthetic fixture.'}
        for key, condition in [('fixture_first', 'test_area'),
                               ('fixture_second', 'test_role')]
    ]
    parsed = window(analyzer, items)
    assert all(analyzer._positive(item) for item in parsed)
    assert all(item.supporting_timestamps_sec == [0.5] for item in parsed)
