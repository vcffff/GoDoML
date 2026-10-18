"""Read complete local v2 sources, split original videos, then extract images."""
import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def sha(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def clean(text):
    # Cross-video names and source labels belong exclusively in the audit metadata.
    text = re.sub(r'[^.!?]*(?:[ХX]-\d+|[Ии]сходн\w* метк)[^.!?]*[.!?]?', '', text)
    return text.strip()


def prepare(args):
    from cashml.training_protocol import load_rules, prompt_for_rule
    from cashml.video import Video

    data = ROOT / 'data'
    out = ROOT / args.output
    annotations = read(data / 'annotations_v2.json')
    raw_rules = read(data / 'rules_v2.json')  # Parse every document and rule, not JSON text SFT.
    if sha(data / 'rules_v2.json') != annotations['rules_sha256']:
        raise ValueError('Rules checksum differs from annotations')
    records = annotations['records']
    missing = [(data / r['video']).resolve().as_posix() for r in records
               if not (data / r['video']).is_file()]
    if missing:
        write(out / 'missing_paths.json', missing)
        raise FileNotFoundError('\n'.join(missing))
    hashes = {r['video']: sha(data / r['video']) for r in records}
    # Union split_group, byte-identical videos, and declared derivatives.
    parent = {r['video']: r['video'] for r in records}

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    def union(a, b):
        parent[find(b)] = find(a)

    seen = {}
    for r in records:
        for key in ('group:' + r['split_group'], 'sha:' + hashes[r['video']]):
            if key in seen:
                union(r['video'], seen[key])
            seen[key] = r['video']
        if r.get('duplicate_of'):
            union(r['video'], r['duplicate_of'])
    groups = defaultdict(list)
    for r in records:
        groups[find(r['video'])].append(r)
    manifest_path = out / 'split_manifest.json'
    fingerprint = {'annotations': sha(data / 'annotations_v2.json'),
                   'rules': sha(data / 'rules_v2.json'), 'videos': hashes}
    if manifest_path.exists():
        manifest = read(manifest_path)
        if manifest['fingerprint'] != fingerprint:
            raise ValueError('Sources changed; use a new output directory, preserve the old split')
        assignments = manifest['assignments']
    else:
        previous = Path(args.previous_split) if args.previous_split else (
            ROOT / 'results/fixed_subset_20261017/split_manifest.json')
        old = {}
        if previous.exists():
            prior = read(previous)
            for key, target in [('development', 'train'), ('train', 'train'),
                                ('holdout', 'validation'), ('validation', 'validation')]:
                for item in prior.get(key, []):
                    name = item if isinstance(item, str) else item.get('video_path', item.get('video'))
                    if name:
                        old[Path(name.replace('\\', '/')).name] = target
            if not old:
                raise ValueError('Unrecognized previous split schema; do not silently replace it')
        assignments = {}
        strata = defaultdict(list)
        for key, members in groups.items():
            pinned = {old[Path(r['video']).name] for r in members if Path(r['video']).name in old}
            if len(pinned) > 1:
                raise ValueError('Previous split leaks duplicate groups; resolve explicitly')
            if pinned:
                for r in members:
                    assignments[r['video']] = next(iter(pinned))
            else:
                strata[members[0]['legal_match']].append(key)
        rng = random.Random(args.seed)
        for keys in strata.values():
            rng.shuffle(keys)
            count = max(1, round(len(keys) * .2)) if len(keys) > 1 else 0
            for i, key in enumerate(keys):
                for r in groups[key]:
                    assignments[r['video']] = 'validation' if i < count else 'train'
        manifest = {'fingerprint': fingerprint, 'seed': args.seed, 'assignments': assignments,
                    'previous_split': str(previous) if previous.exists() else None,
                    'limitation': None if previous.exists() else
                    'Historical manifest absent. This is a new split, not a pristine historical holdout.'}
        write(manifest_path, manifest)  # MUST precede extraction.
    for members in groups.values():
        assert len({assignments[r['video']] for r in members}) == 1
    rules = load_rules(data / 'rules_v2.json')
    examples = []
    excluded = []
    for members in groups.values():
        r = next((x for x in members if not x.get('duplicate_of')), members[0])
        ids = [r['rule_id']] if r.get('rule_id') else r.get('comparison_rule_ids', [])
        # No direct rule: supervise observations against an explicitly unconfirmed candidate.
        if not ids:
            section_ids = r.get('related_sections_not_proving_violation', [])
            ids = [key for key, rule in rules.items()
                   if set(section_ids) & set(rule['training_rule']['legal_basis_section_ids'])][:1]
        if not ids:
            excluded.append({'video': r['video'], 'reason': 'No annotated applicable/comparison rule'})
            continue
        ident = hashes[r['video']][:16]
        with Video(data / r['video']) as video:
            # Uniform sampling is independent of verdict and the annotated event timecode.
            frames = video.sample(0, max(0, video.duration - 1 / video.fps), args.frames, args.side)
        paths = []
        for i, frame in enumerate(frames):
            path = out / 'frames' / ident / f'{i:02d}.jpg'
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.image.save(path, quality=95)
            paths.append(str(path.relative_to(ROOT)))
        observed = [o for o in r['observations'] if any(
            o['start_sec'] <= f.timestamp <= o['end_sec'] for f in frames)]
        # Do not assert an entire annotated before/after sequence when a phase was missed.
        complete = len(observed) == len(r['observations'])
        for rule_id in ids:
            rule = rules[rule_id]
            match = r['legal_match']
            supported = match == 'supported_in_staged_context' and complete
            negative = match == 'source_negative'
            state = True if supported else False if negative else 'unknown'
            explanation = ' '.join(clean(o['action']) for o in observed)
            if supported:
                explanation += f' В заданном учебном контексте связь с {rule_id} подтверждена. '
                explanation += clean(r['reason'])
            elif negative:
                explanation += f' В выбранных кадрах нарушение {rule_id} не подтверждено; другие критерии не оценены.'
            else:
                explanation += f' Нарушение {rule_id} не установлено. ' + ' '.join(r.get('missing_context', []))
            if match == 'user_confirmed_scenario':
                # User clarification is retained in audit, not taught as visually inferred information.
                explanation += ' Недостача по изображениям не определяется: суммы неразличимы.'
            indices = [i for i, f in enumerate(frames) if any(
                o['start_sec'] <= f.timestamp <= o['end_sec'] for o in observed)]
            target = {'assessments': [{'c': 0, 's': 'scene', 'o': True if observed else 'unknown',
                       'm': state, 'f': indices, 'a': [True if supported else 'unknown'] *
                       len(rule['applicability_conditions']), 'e': explanation[:1490]}]}
            examples.append({'id': ident + '-' + rule_id, 'split': assignments[r['video']],
                             'images': paths, 'timestamps': [f.timestamp for f in frames],
                             'prompt': prompt_for_rule(rule, staged=True), 'target': target,
                             'audit': {'video': r['video'], 'split_group': r['split_group'],
                                       'legal_match': match, 'rule_id': rule_id,
                                       'original_verdict': r['verdict'], 'timecode_sec': r['timecode_sec'],
                                       'scenario_clarification': r.get('scenario_clarification'),
                                       'all_observation_phases_sampled': complete}})
    write(out / 'examples.json', examples)
    summary = {'source_records': len(records), 'unique_video_groups': len(groups),
               'examples': dict(Counter(e['split'] for e in examples)), 'excluded': excluded,
               'frames_per_example': args.frames, 'max_side': args.side,
               'documents_read': len(raw_rules['documents']), 'rules_read': len(rules),
               'limitations': ['Sparse frames cannot establish absence between frames.',
                              'Targets follow supplied annotations; frame semantics need human review.']}
    write(out / 'data_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='outputs/cash-lora')
    parser.add_argument('--previous-split')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--frames', type=int, default=16)
    parser.add_argument('--side', type=int, default=224)
    prepare(parser.parse_args())
