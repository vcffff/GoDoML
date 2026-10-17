"""Print protocol shape diagnostics only, never model-derived descriptions."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('audit', type=Path)
args = parser.parse_args()
audit = json.loads(args.audit.read_text(encoding='utf-8'))
rules_path = Path(__file__).resolve().parents[1] / 'rules.json'
rules = {rule['type_id']: rule for rule in json.loads(rules_path.read_text(encoding='utf-8'))}
shapes = []
expected = {'type_id', 'subject_id', 'observed', 'criterion_met',
            'supporting_timestamps_sec', 'explanation', 'applicability'}
for failure in audit.get('unparsed_windows', []):
    shape = {'validation_error_types': failure['validation_error_types']}
    try:
        value = json.loads(failure['raw_response'])
        shape['root_type'] = type(value).__name__
        records = value if isinstance(value, list) else value.get('assessments', []) if isinstance(value, dict) else []
        shape['record_types'] = [type(record).__name__ for record in records] if isinstance(records, list) else [type(records).__name__]
        if isinstance(records, list):
            shape['records'] = [{'known_field_types': {key: type(record[key]).__name__
                                                      for key in expected if key in record},
                                 'missing_fields': sorted(expected - set(record)),
                                 'extra_field_count': len(set(record) - expected)}
                                for record in records if isinstance(record, dict)]
            shape['negative_assessments_citing_frames'] = sum(
                record.get('observed') is False and record.get('criterion_met') is False
                and bool(record.get('supporting_timestamps_sec'))
                for record in records if isinstance(record, dict))
            shape['applicability_values_copied_from_criteria'] = sum(
                value == rules.get(record.get('type_id'), {}).get('applicability_conditions', {}).get(key)
                for record in records if isinstance(record, dict)
                for key, value in record.get('applicability', {}).items())
            shape['applicability_exact_boolean_strings'] = sum(
                isinstance(value, str) and value in ('true', 'false')
                for record in records if isinstance(record, dict)
                for value in record.get('applicability', {}).values())
            shape['applicability_value_types'] = [type(value).__name__
                for record in records if isinstance(record, dict)
                for value in record.get('applicability', {}).values()]
    except (ValueError, TypeError):
        shape['json_valid'] = False
    shapes.append(shape)
print(json.dumps({'coverage_complete': audit['coverage_complete'],
                  'parsed_windows': len(audit['windows']), 'unparsed_shapes': shapes}))
