"""Catalog safety tests; synthetic rules are not predictions or new regulations."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from cashml.rules import RuleCatalogError, load_rules


@pytest.fixture
def catalog():
    return {
        'interpretation_constraints': ['Do not infer authority from appearance.'],
        'documents': [{
            'document_id': 'SYNTHETIC_DOC', 'title': 'Synthetic source, not a regulation',
            'official_url': 'urn:test:document',
            'sections': [{'section_id': 'SYNTHETIC_SECTION', 'point': '1',
                          'source_url': 'urn:test:section',
                          'normative_text': 'Exact synthetic normative text.'}]}],
        'video_rules': [{
            'rule_id': 'SOURCE_ID', 'event_type': 'Original event description', 'category': 'synthetic',
            'legal_basis_section_ids': ['SYNTHETIC_SECTION'],
            'applicability_conditions': ['Known operation', 'Complete visible interval'],
            'required_visual_observations': ['Observable transfer sequence'],
            'do_not_flag_when': ['Permitted exception'],
            'non_visual_or_unobservable_limits': ['Authority is not visible'],
            'assessment_mode': 'requires_external_confirmation',
            'evidence_requirements': {'timestamps': 'Seconds from original video',
                                      'sequence_required': True,
                                      'absence_requires_complete_relevant_interval': True,
                                      'description': 'Only observed actions'},
            'additional_source_field': 'Preserve unknown source metadata'}],
        'transitional_provisions': [{
            'official_name': 'Synthetic transition', 'source_url': 'urn:test:transition',
            'effective_until_exclusive': '2027-01-01',
            'affected_sections': ['SYNTHETIC_SECTION'],
            'official_transition_text': 'Original temporary text.',
            'interpretation': 'Use the dated temporary text.'}],
    }


def write_catalog(tmp_path, content):
    path = tmp_path / 'rules.json'
    path.write_text(json.dumps(content, ensure_ascii=False), encoding='utf-8')
    return path


def test_regulatory_rule_preserves_source_semantics_and_order(tmp_path, catalog):
    rules = load_rules(write_catalog(tmp_path, catalog))
    assert list(rules) == ['SOURCE_ID']
    rule = rules['SOURCE_ID']
    source = catalog['video_rules'][0]
    assert rule['source_rule'] == source
    assert rule['type_id'] == rule['rule_id'] == source['rule_id']
    assert rule['event_title'] == rule['event_type'] == source['event_type']
    assert rule['criteria'] == 'Original event description\nObservable transfer sequence'
    assert rule['applicability_conditions'] == {
        'condition_0': 'Known operation', 'condition_1': 'Complete visible interval'}
    for field in ('do_not_flag_when', 'non_visual_or_unobservable_limits', 'assessment_mode',
                  'evidence_requirements', 'additional_source_field'):
        assert rule[field] == source[field]
    assert rule['legal_basis_sections'] == catalog['documents'][0]['sections']
    assert rule['source_documents'][0]['document_id'] == 'SYNTHETIC_DOC'
    assert rule['title'] == catalog['documents'][0]['title']
    assert rule['url'] == 'urn:test:section'
    assert 'SYNTHETIC_SECTION' in rule['section']
    assert rule['transitional_provisions'] == catalog['transitional_provisions']
    assert rule['interpretation_constraints'] == catalog['interpretation_constraints']
    rule['applicability_conditions']['condition_0'] = 'Changed normalized value'
    rule['do_not_flag_when'].append('Changed normalized exception')
    assert rule['source_rule'] == source


def test_new_category_is_loaded_without_hardcoded_allowlist(tmp_path, catalog):
    second = deepcopy(catalog['video_rules'][0])
    second.update(rule_id='ARBITRARY_NEW_ID', category='previously_unseen', assessment_mode='visual_candidate')
    catalog['video_rules'].append(second)
    assert list(load_rules(write_catalog(tmp_path, catalog))) == ['SOURCE_ID', 'ARBITRARY_NEW_ID']


def test_legacy_verified_rules_keep_ids_and_metadata(tmp_path):
    source = {'type_id': 'legacy_id', 'event_title': 'Legacy event', 'verified': True,
              'category_confirmed': False, 'criteria': 'Observable sequence',
              'applicability_conditions': {'area': 'Confirmed area'},
              'title': 'Synthetic source', 'section': '1', 'url': 'urn:test:legacy'}
    disabled = dict(source, type_id='disabled', verified=False)
    rules = load_rules(write_catalog(tmp_path, [source, disabled]))
    assert list(rules) == ['legacy_id']
    assert rules['legacy_id']['source_rule'] == source
    assert rules['legacy_id']['applicability_conditions'] == source['applicability_conditions']
    assert rules['legacy_id']['category_confirmed'] is False


@pytest.mark.parametrize('field,bad', [
    ('rule_id', ''), ('event_type', None), ('category', 4),
    ('legal_basis_section_ids', ['missing']), ('legal_basis_section_ids', []),
    ('legal_basis_section_ids', ['SYNTHETIC_SECTION', 'SYNTHETIC_SECTION']),
    ('applicability_conditions', []), ('applicability_conditions', {'wrong': 'shape'}),
    ('required_visual_observations', ['']), ('do_not_flag_when', 'not a list'),
    ('non_visual_or_unobservable_limits', None), ('assessment_mode', 'unsupported'),
    ('evidence_requirements', {}),
])
def test_invalid_rule_schema_or_references_fail_closed(tmp_path, catalog, field, bad):
    catalog['video_rules'][0][field] = bad
    with pytest.raises(RuleCatalogError):
        load_rules(write_catalog(tmp_path, catalog))


@pytest.mark.parametrize('target', ['rule', 'document', 'section'])
def test_duplicate_source_ids_fail_closed(tmp_path, catalog, target):
    if target == 'rule':
        catalog['video_rules'].append(deepcopy(catalog['video_rules'][0]))
    elif target == 'document':
        catalog['documents'].append(deepcopy(catalog['documents'][0]))
    else:
        catalog['documents'][0]['sections'].append(deepcopy(catalog['documents'][0]['sections'][0]))
    with pytest.raises(RuleCatalogError, match='Duplicate'):
        load_rules(write_catalog(tmp_path, catalog))


@pytest.mark.parametrize('field', ['section_id', 'normative_text', 'source_url'])
def test_missing_normative_source_fields_fail_closed(tmp_path, catalog, field):
    del catalog['documents'][0]['sections'][0][field]
    with pytest.raises(RuleCatalogError):
        load_rules(write_catalog(tmp_path, catalog))


def test_bad_transition_reference_fails_closed(tmp_path, catalog):
    catalog['transitional_provisions'][0]['affected_sections'] = ['missing']
    with pytest.raises(RuleCatalogError, match='transition section'):
        load_rules(write_catalog(tmp_path, catalog))


def test_unreferenced_sections_are_also_validated(tmp_path, catalog):
    catalog['documents'][0]['sections'].append({'section_id': 'UNUSED_BUT_INVALID'})
    with pytest.raises(RuleCatalogError):
        load_rules(write_catalog(tmp_path, catalog))


@pytest.mark.parametrize('field', ['sequence_required', 'absence_requires_complete_relevant_interval'])
def test_string_boolean_does_not_disable_evidence_requirements(tmp_path, catalog, field):
    catalog['video_rules'][0]['evidence_requirements'][field] = 'false'
    with pytest.raises(RuleCatalogError, match='boolean'):
        load_rules(write_catalog(tmp_path, catalog))


@pytest.mark.parametrize('raw', ['[]', '{}', 'null', '{"video_rules":[],"video_rules":[{}]}', 'invalid'])
def test_empty_or_ambiguous_json_fails_closed(tmp_path, raw):
    path = tmp_path / 'invalid.json'
    path.write_text(raw, encoding='utf-8')
    with pytest.raises(RuleCatalogError):
        load_rules(path)


def test_full_local_catalog_preserves_all_rules_when_available():
    path = Path(__file__).resolve().parents[1] / 'data' / 'forte_kz_regulatory_rules.json'
    if not path.is_file():
        pytest.skip('Local regulatory source is not bundled in this checkout')
    source = json.loads(path.read_text(encoding='utf-8'))
    rules = load_rules(path)
    assert list(rules) == [rule['rule_id'] for rule in source['video_rules']]
    assert len(rules) == 31
    assert sum(rule['assessment_mode'] == 'visual_candidate' for rule in rules.values()) == 20
    for original in source['video_rules']:
        rule = rules[original['rule_id']]
        assert rule['source_rule'] == original
        assert list(rule['applicability_conditions'].values()) == original['applicability_conditions']
        assert [section['section_id'] for section in rule['legal_basis_sections']] == original['legal_basis_section_ids']
