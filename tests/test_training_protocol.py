import json
import time

import pytest
from PIL import Image

from cashml.config import Settings
from cashml.pipeline import AnalysisError, Analyzer
from cashml.training_protocol import compact_schema, load_rules, prompt_for_rule
from cashml.video import Frame


@pytest.fixture
def v2(tmp_path):
    path = tmp_path / 'rules.json'
    path.write_text(json.dumps({'documents': [{'title': 'Fixture', 'official_url': 'urn:fixture',
        'sections': [{'section_id': 'S1', 'normative_text': 'Synthetic rule.'}]}],
        'video_rules': [{'rule_id': 'R1', 'event_type': 'Synthetic event',
        'legal_basis_section_ids': ['S1'], 'applicability_conditions': ['Known scene'],
        'required_visual_observations': ['Visible sequence']}]}), encoding='utf-8')
    return path


def test_v2_uses_same_rule_id_reference_and_prompt(v2):
    rule = load_rules(v2)['R1']
    assert rule['section'] == 'S1' and rule['url'] == 'urn:fixture'
    assert 'Synthetic rule.' in prompt_for_rule(rule)
    instance = Analyzer(Settings(mode='mock', rules_file=v2))
    instance.load()
    prompts = []

    def infer(frames, prompt, deadline):
        prompts.append(prompt)
        return json.dumps({'assessments': [{'c': 0, 's': 'scene', 'o': True,
            'm': 'unknown', 'f': [0], 'a': ['unknown'], 'e': 'Cannot establish context.'}]})
    instance.infer = infer
    result = instance._window([Frame(1., Image.new('RGB', (32, 32)))], time.monotonic() + 10)
    assert result[0].type_id == 'R1'
    assert instance._unknown(result[0]) and not instance._positive(result[0])
    assert prompts == [prompt_for_rule(rule, staged=False)]


def test_v2_retains_evidence_rejection(v2):
    instance = Analyzer(Settings(mode='mock', rules_file=v2))
    instance.load()
    instance.infer = lambda *args: json.dumps({'assessments': [{'c': 0, 's': 'scene',
        'o': True, 'm': True, 'f': [99], 'a': [True], 'e': 'Invalid frame.'}]})
    with pytest.raises(AnalysisError, match='outside'):
        instance._window([Frame(1., Image.new('RGB', (32, 32)))], time.monotonic() + 10)


def test_portable_adapter_environment(monkeypatch):
    monkeypatch.setenv('ML_ADAPTER_DIR', './outputs/cash-lora/best-adapter')
    monkeypatch.setenv('ML_LOAD_4BIT', '1')
    settings = Settings.from_env()
    assert settings.adapter_dir == './outputs/cash-lora/best-adapter'
    assert settings.load_4bit == 1
    assert 'Magzhan' not in str(settings.model_dir)


def test_failed_tail_seek_recovers_exact_frame_and_pts():
    import cv2
    import numpy as np

    from cashml.video import Video

    class Capture:
        position = 0
        failed_seek = False

        def set(self, key, value):
            self.position = value
            self.failed_seek = value == 9

        def read(self):
            if self.failed_seek:
                return False, None
            index = self.position
            self.position += 1
            return True, np.full((32, 32, 3), index, dtype=np.uint8)

        def get(self, key):
            assert key == cv2.CAP_PROP_POS_MSEC
            return (self.position - 1) * 100

    video = Video.__new__(Video)
    video.cap, video.count, video.fps, video.duration = Capture(), 10, 10, 1
    frame = video.frame(.9)
    assert frame.timestamp == .9
    assert frame.image.getpixel((0, 0)) == (9, 9, 9)


def test_constrained_schema_preserves_unknown_and_cyrillic():
    lmfe = pytest.importorskip('lmformatenforcer')
    from lmformatenforcer.consts import COMPLETE_ALPHABET

    payload = {'assessments': [{'c': 0, 's': 'scene', 'o': True, 'm': 'unknown',
        'f': [0, 2], 'a': ['unknown', False], 'e': 'Суммы неразличимы.'}]}

    def accepted(value):
        parser = lmfe.JsonSchemaParser(compact_schema(2, 3), config=lmfe.CharacterLevelParserConfig(
            alphabet=COMPLETE_ALPHABET + ''.join(chr(i) for i in range(0x400, 0x500))))
        for char in json.dumps(value, ensure_ascii=False):
            if char not in parser.get_allowed_characters():
                return False
            parser = parser.add_character(char)
        return parser.can_end()

    assert accepted(payload)
    payload['assessments'][0]['a'].append(True)
    assert not accepted(payload)
    payload['assessments'][0]['a'].pop()
    payload['assessments'][0]['f'] = [99]
    assert not accepted(payload)
