"""Technical format diagnostics only; never prints video-derived text."""
import json
from pathlib import Path

from cashml.schema import WindowResult

path = Path('results/smoke/window_outputs.json')
results = []
for raw in json.loads(path.read_text(encoding='utf-8')):
    item = {'characters': len(raw), 'fenced': raw.startswith('```'),
            'ends_brace': raw.rstrip().endswith('}')}
    if raw.startswith('```'):
        raw = raw.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    try:
        parsed = WindowResult.model_validate_json(raw)
        item['valid_schema'] = True
        item['assessment_count'] = len(parsed.assessments)
        allowed = {'camera_phone_in_cash_area', 'personal_bag_in_cash_area'}
        item['unknown_ids'] = sum(e.type_id not in allowed for e in parsed.assessments)
    except ValueError:
        item['valid_schema'] = False
        try:
            obj = json.loads(raw)
            item['json_type'] = type(obj).__name__
            item['expected_keys'] = isinstance(obj, dict) and set(obj) == {'assessments'}
        except json.JSONDecodeError as exc:
            item['parse_error'] = exc.msg
            item['error_position'] = exc.pos
    results.append(item)
print(json.dumps(results))
