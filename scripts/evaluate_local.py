"""Run frozen development inputs offline; print technical metadata, never video text.

Labels are joined only by the separate evaluator. Timings exclude model loading,
HTTP upload and response serialization and must not be reported as API SLA proof.
"""

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import sys
import time
import types
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cashml import pipeline
from cashml.config import ROOT, Settings


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def select_cases(manifest, selected):
    cases = manifest['cases']
    ids = [case['case_id'] for case in cases]
    if len(ids) != len(set(ids)) or len(selected) != len(set(selected)):
        raise ValueError('Duplicate case ids')
    if not selected or not set(selected) <= set(ids):
        raise ValueError('Select cases from the frozen inference manifest')
    for case_id in ids:
        if Path(case_id).name != case_id or case_id in ('.', '..') or '\\' in case_id:
            raise ValueError('Invalid case id')
    by_id = {case['case_id']: case for case in cases}
    return [by_id[case_id] for case_id in selected]


def state_counts(audit):
    counts = {key: Counter() for key in ('observed', 'criterion_met', 'applicability')}
    for window in audit:
        for item in window['assessments']:
            for key in ('observed', 'criterion_met'):
                counts[key][str(item[key]).lower()] += 1
            counts['applicability'].update(str(value).lower() for value in item['applicability'].values())
    return {key: dict(value) for key, value in counts.items()}


def run(manifest_path, selected, destination, pipeline_source=None, phase='candidate'):
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if selected == ['all']:
        selected = [case['case_id'] for case in manifest['cases']]
    cases = select_cases(manifest, selected)
    implementation = pipeline
    if pipeline_source is not None:
        source_dir = pipeline_source.resolve().parent
        if not all((source_dir / name).is_file() for name in ('schema.py', 'video.py')):
            raise ValueError('Frozen baseline requires sibling schema.py and video.py snapshots')
        package = types.ModuleType('_cashml_frozen_baseline')
        package.__path__ = [str(source_dir)]
        sys.modules[package.__name__] = package
        spec = importlib.util.spec_from_file_location(package.__name__ + '.pipeline', pipeline_source)
        implementation = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = implementation
        spec.loader.exec_module(implementation)
    settings = Settings.from_env()
    if settings.mode != 'local':
        raise ValueError('ML_MODE=local is required; mock results are not evaluation')
    destination.mkdir(parents=True, exist_ok=False)
    source_dir = pipeline_source.resolve().parent if pipeline_source else ROOT / 'cashml'
    sources = [source_dir / 'pipeline.py', ROOT / 'cashml/config.py',
               source_dir / 'schema.py', source_dir / 'video.py', settings.rules_file]
    if pipeline_source is None:
        sources += [path for name in ('rules.py', 'actions.py') if (path := source_dir / name).exists()]
    snapshot = destination / 'source'
    snapshot.mkdir()
    for path in sources:
        (snapshot / path.name).write_bytes(path.read_bytes())
    write(destination / 'config_snapshot.json', {
        'settings': {key: str(value) if isinstance(value, Path) else value
                     for key, value in dataclasses.asdict(settings).items() if key != 'api_key'},
        'source_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'case_ids': selected, 'labels_read_by_inference': False,
        'evaluation_phase': 'baseline_all42' if phase == 'baseline' else 'candidate',
        'candidate_frozen_before_holdout': phase == 'candidate',
        'timing_scope': 'local pipeline only; excludes load/upload/HTTP serialization',
    })
    import torch
    import transformers

    analyzer = implementation.Analyzer(settings)
    started = time.perf_counter()
    analyzer.load()
    write(destination / 'runtime.json', {
        'model_load_sec': time.perf_counter() - started, 'torch': torch.__version__,
        'transformers': transformers.__version__, 'gpu': torch.cuda.get_device_name(),
    })
    summaries = []
    for case in cases:
        folder = destination / case['case_id']
        folder.mkdir()
        record = {'case_id': case['case_id'], 'outcome': 'in_flight_unknown', 'mode': 'local',
                  'timing_scope': 'local_pipeline_only'}
        write(folder / 'request.json', record)
        print(json.dumps({'case_id': case['case_id'], 'status': 'started'}), flush=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            response = analyzer.analyze_video(case['video_path'])
            write(folder / 'response.json', response)
            record['outcome'] = 'success'
            record['prediction'] = response['verdict']
        except implementation.AnalysisInconclusive:
            record['outcome'] = 'inconclusive'
        except implementation.AnalysisTimeout:
            record['outcome'] = 'timeout'
        except Exception as exc:  # noqa: BLE001 -- never print private error contents
            record.update(outcome='processing_error', error_type=type(exc).__name__)
        finally:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            record.update(elapsed_sec=round(elapsed, 3),
                          pipeline_budget_pass=elapsed <= settings.timeout_sec,
                          peak_vram_mib=round(torch.cuda.max_memory_allocated() / 1024**2, 3),
                          coverage_complete=analyzer.coverage_complete,
                          completed_windows=len(analyzer.assessment_audit),
                          state_counts=state_counts(analyzer.assessment_audit))
            write(folder / 'assessments.json', {
                'coverage_complete': analyzer.coverage_complete,
                'windows': analyzer.assessment_audit, 'unparsed_windows': analyzer.unparsed_audit,
                'protocol_normalizations': analyzer.protocol_audit,
                'model_wire_outputs': analyzer.wire_audit,
                'action_screenings': getattr(analyzer, 'action_audit', []),
                'stage_metrics': getattr(analyzer, 'stage_metrics', {}),
            })
            write(folder / 'request.json', record)
            summaries.append(record)
            write(destination / 'summary.json', {'cases': summaries,
                  'classification_validated': False, 'timing_scope': 'local_pipeline_only'})
            print(json.dumps(record), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--case-id', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pipeline-source', type=Path)
    parser.add_argument('--phase', choices=('baseline', 'candidate'), default='candidate')
    args = parser.parse_args()
    run(args.manifest, args.case_id, args.output, args.pipeline_source, args.phase)
