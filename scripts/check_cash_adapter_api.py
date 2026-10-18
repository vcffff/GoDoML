"""Exercise real adapter loading and the public upload route entirely locally."""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    out = ROOT / 'outputs/cash-lora'
    examples = json.loads((out / 'examples.json').read_text(encoding='utf-8'))
    example = next(e for e in examples if e['split'] == 'validation'
                   and e['target']['assessments'][0]['m'] is True)
    for name, value in {
        'ML_MODE': 'local', 'ML_MODEL_DIR': str(ROOT / 'models/Qwen3-VL-2B-Instruct'),
        'ML_ADAPTER_DIR': str(out / 'best-adapter'), 'ML_RULES_FILE': str(ROOT / 'data/rules_v2.json'),
        'ML_LOAD_4BIT': '1', 'ML_CONSTRAINED_JSON': '1', 'ML_STAGED_CONTEXT': '1', 'ML_FRAMES_PER_WINDOW': '16',
        'ML_MAX_SIDE': '224', 'ML_WINDOW_SEC': '150', 'ML_STRIDE_SEC': '150',
        'ML_RULE_IDS': example['audit']['rule_id'], 'ML_MAX_NEW_TOKENS': '768',
        'ML_TIMEOUT_SEC': '300'}.items():
        os.environ[name] = value
    from fastapi.testclient import TestClient

    from app import create_app
    from cashml.config import Settings
    from cashml.pipeline import Analyzer
    from cashml.schema import AnalysisResult
    analyzer = Analyzer(Settings.from_env())
    app = create_app(settings=analyzer.settings, analyzer=analyzer)
    with TestClient(app) as client:
        health = client.get('/health').json()
        assert health['ready'] and health['mode'] == 'local' and not health['mock']
        started = time.perf_counter()
        with (ROOT / 'data' / example['audit']['video']).open('rb') as source:
            response = client.post('/analyze', files={'file': ('input.mov', source, 'video/quicktime')})
        if response.status_code == 200:
            AnalysisResult.model_validate(response.json())
        result = {'adapter_loaded': True, 'health': health, 'status_code': response.status_code,
                  'elapsed_sec': time.perf_counter() - started, 'response': response.json(),
                  'transport': 'local ASGI TestClient; real model and upload route, no external network'}
        (out / 'api_constrained_wire_audit.json').write_text(json.dumps(analyzer.wire_audit,
            ensure_ascii=False, indent=2), encoding='utf-8')
        (out / 'api_constrained_check.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({k: v for k, v in result.items() if k != 'response'}), flush=True)


if __name__ == '__main__':
    main()
