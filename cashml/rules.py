"""Load source rules without deriving new prohibitions or discarding exceptions."""

import json
from copy import deepcopy
from pathlib import Path


class RuleCatalogError(ValueError):
    """The rule source cannot be safely interpreted."""


def _text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise RuleCatalogError(f'{field} must be a nonempty string')
    return value


def _object(value, field):
    if not isinstance(value, dict):
        raise RuleCatalogError(f'{field} must be an object')
    return value


def _list(value, field, *, nonempty=False):
    if not isinstance(value, list) or (nonempty and not value):
        raise RuleCatalogError(f'{field} must be a list' + (' with entries' if nonempty else ''))
    return value


def _texts(value, field, *, nonempty=False):
    return [_text(item, field) for item in _list(value, field, nonempty=nonempty)]


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise RuleCatalogError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def _legacy_rules(entries):
    rules = {}
    seen = set()
    for entry in entries:
        source = _object(entry, 'legacy rule')
        rule_id = _text(source.get('type_id'), 'type_id')
        if rule_id in seen:
            raise RuleCatalogError(f'Duplicate rule ID: {rule_id}')
        seen.add(rule_id)
        if type(source.get('verified', False)) is not bool:
            raise RuleCatalogError('verified must be boolean')
        if not source.get('verified', False):
            continue
        for field in ('event_title', 'criteria', 'title', 'section', 'url'):
            _text(source.get(field), field)
        conditions = _object(source.get('applicability_conditions'), 'applicability_conditions')
        if not conditions:
            raise RuleCatalogError('Rules require explicit applicability conditions')
        for key, value in conditions.items():
            _text(key, 'condition key')
            _text(value, 'condition description')
        rule = deepcopy(source)
        rule['source_rule'] = deepcopy(source)
        rule.setdefault('rule_id', rule_id)
        rule.setdefault('event_type', source['event_title'])
        rules[rule_id] = rule
    return rules


def _regulatory_rules(catalog):
    documents = {}
    sections = {}
    section_documents = {}
    for document in _list(catalog.get('documents'), 'documents', nonempty=True):
        _object(document, 'document')
        document_id = _text(document.get('document_id'), 'document_id')
        if document_id in documents:
            raise RuleCatalogError(f'Duplicate document ID: {document_id}')
        for field in ('title', 'official_url'):
            _text(document.get(field), field)
        documents[document_id] = document
        for section in _list(document.get('sections'), 'sections', nonempty=True):
            _object(section, 'section')
            section_id = _text(section.get('section_id'), 'section_id')
            if section_id in sections:
                raise RuleCatalogError(f'Duplicate section ID: {section_id}')
            for field in ('point', 'source_url', 'normative_text'):
                _text(section.get(field), field)
            sections[section_id] = section
            section_documents[section_id] = document_id

    constraints = _texts(catalog.get('interpretation_constraints', []), 'interpretation_constraints')
    incorrect = _list(catalog.get('incorrect_inferences', []), 'incorrect_inferences')
    for item in incorrect:
        _object(item, 'incorrect inference')
        _text(item.get('claim'), 'incorrect inference claim')
        _text(item.get('reason'), 'incorrect inference reason')
    transitions = _list(catalog.get('transitional_provisions', []), 'transitional_provisions')
    for transition in transitions:
        _object(transition, 'transitional provision')
        for field in ('official_name', 'source_url', 'effective_until_exclusive',
                      'official_transition_text', 'interpretation'):
            _text(transition.get(field), field)
        for section_id in _texts(transition.get('affected_sections'), 'affected_sections', nonempty=True):
            if section_id not in sections:
                raise RuleCatalogError(f'Unknown transition section ID: {section_id}')

    rules = {}
    for source in _list(catalog.get('video_rules'), 'video_rules', nonempty=True):
        _object(source, 'video rule')
        rule_id = _text(source.get('rule_id'), 'rule_id')
        if rule_id in rules:
            raise RuleCatalogError(f'Duplicate rule ID: {rule_id}')
        event = _text(source.get('event_type'), 'event_type')
        _text(source.get('category'), 'category')
        mode = source.get('assessment_mode')
        if mode not in ('visual_candidate', 'requires_external_confirmation'):
            raise RuleCatalogError(f'Unknown assessment_mode for {rule_id}')
        conditions = _texts(source.get('applicability_conditions'), 'applicability_conditions', nonempty=True)
        observations = _texts(source.get('required_visual_observations'),
                              'required_visual_observations', nonempty=True)
        for field in ('do_not_flag_when', 'non_visual_or_unobservable_limits'):
            _texts(source.get(field), field)
        evidence = _object(source.get('evidence_requirements'), 'evidence_requirements')
        for field in ('timestamps', 'description'):
            _text(evidence.get(field), field)
        for field in ('sequence_required', 'absence_requires_complete_relevant_interval'):
            if type(evidence.get(field)) is not bool:
                raise RuleCatalogError(f'{field} must be boolean')
        references = _texts(source.get('legal_basis_section_ids'), 'legal_basis_section_ids', nonempty=True)
        if len(references) != len(set(references)):
            raise RuleCatalogError(f'Duplicate legal basis reference for {rule_id}')
        if any(section_id not in sections for section_id in references):
            raise RuleCatalogError(f'Unknown legal basis section for {rule_id}')

        basis = [sections[section_id] for section_id in references]
        document_ids = list(dict.fromkeys(section_documents[section_id] for section_id in references))
        rule = deepcopy(source)
        rule.update({
            'type_id': rule_id,
            'event_title': event,
            'criteria': '\n'.join([event, *observations]),
            'applicability_conditions': {f'condition_{i}': text for i, text in enumerate(conditions)},
            'source_rule': deepcopy(source),
            'legal_basis_sections': deepcopy(basis),
            'source_documents': [{key: deepcopy(value) for key, value in documents[document_id].items()
                                  if key != 'sections'} for document_id in document_ids],
            'title': '; '.join(documents[document_id]['title'] for document_id in document_ids),
            'section': '; '.join(f"{section['section_id']} (пункт {section['point']})" for section in basis),
            'url': basis[0]['source_url'],
            'interpretation_constraints': deepcopy(constraints),
            'incorrect_inferences': deepcopy(incorrect),
            'transitional_provisions': deepcopy([
                item for item in transitions if set(item['affected_sections']) & set(references)]),
        })
        if 'normative_basis_checked_on' in catalog:
            rule['normative_basis_checked_on'] = _text(
                catalog['normative_basis_checked_on'], 'normative_basis_checked_on')
        rules[rule_id] = rule
    return rules


def load_rules(path):
    """Return rules keyed by unchanged source IDs, or fail closed on malformed input.

    Supports the legacy verified-rule list and the regulatory documents/video_rules
    object. Conditions retain source order; normative text and exceptions are never
    inferred from category names. Only the supplied JSON is read.
    """
    try:
        catalog = json.loads(Path(path).read_text(encoding='utf-8-sig'), object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuleCatalogError(f'Cannot read rule catalog: {exc}') from exc
    if isinstance(catalog, list):
        rules = _legacy_rules(catalog)
    elif isinstance(catalog, dict):
        rules = _regulatory_rules(catalog)
    else:
        raise RuleCatalogError('Rule catalog must be a list or regulatory object')
    if not rules:
        raise RuleCatalogError('Rule catalog contains no active rules')
    return rules
