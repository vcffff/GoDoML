"""Inference deadline/telemetry tests without loading weights or private frames."""

import json
import sys
import time
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from cashml.config import Settings
from cashml.pipeline import AnalysisTimeout, Analyzer, InputBudgetExceeded


class Batch(dict):
    def to(self, device):
        assert device == 'cuda'
        return self


class Output:
    shape = (1, 8)

    def __getitem__(self, key):
        return 'generated token slice'


@pytest.mark.parametrize('fail', [False, True])
def test_generate_uses_greedy_inference_and_records_failed_attempts(monkeypatch, fail):
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(
        cuda=SimpleNamespace(synchronize=lambda: None), inference_mode=nullcontext))
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        StoppingCriteria=object, StoppingCriteriaList=list))
    analyzer = Analyzer(Settings())
    analyzer.processor = SimpleNamespace(
        apply_chat_template=lambda *a, **k: Batch(input_ids=SimpleNamespace(shape=(1, 5)),
                                                  token_type_ids='unsupported'),
        decode=lambda *a, **k: 'synthetic output')

    def generate(**kwargs):
        assert 'token_type_ids' not in kwargs
        assert kwargs['do_sample'] is False
        assert kwargs['max_new_tokens'] == analyzer.settings.max_new_tokens
        if fail:
            raise RuntimeError('synthetic generation failure')
        return Output()

    analyzer.model = SimpleNamespace(generate=generate)
    if fail:
        with pytest.raises(RuntimeError):
            analyzer.infer([], 'test', time.monotonic() + 10)
    else:
        assert analyzer.infer([], 'test', time.monotonic() + 10) == 'synthetic output'
    call = analyzer.stage_metrics['calls'][0]
    assert call['input_tokens'] == 5
    assert call['completed'] is not fail
    assert call['generation_sec'] >= 0
    if not fail:
        assert call['generated_tokens'] == 3


def test_preprocessing_exhausting_deadline_never_starts_generation(monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(
        cuda=SimpleNamespace(synchronize=lambda: None), inference_mode=nullcontext))
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        StoppingCriteria=object, StoppingCriteriaList=list))
    analyzer = Analyzer(Settings())
    ticks = iter([0.0, 2.0])
    monkeypatch.setattr(time, 'monotonic', lambda: next(ticks))
    analyzer.processor = SimpleNamespace(apply_chat_template=lambda *a, **k:
                                         Batch(input_ids=SimpleNamespace(shape=(1, 5))))
    analyzer.model = SimpleNamespace(generate=lambda **kwargs: pytest.fail('Deadline already expired'))
    with pytest.raises(AnalysisTimeout):
        analyzer.infer([], 'test', 1.0)


def test_large_input_is_rejected_before_cuda_allocation(monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        StoppingCriteria=object, StoppingCriteriaList=list))

    class OversizedBatch(dict):
        def to(self, device):
            pytest.fail('Oversized input must stay off CUDA')

    analyzer = Analyzer(Settings(max_input_tokens=100))
    analyzer.processor = SimpleNamespace(apply_chat_template=lambda *a, **k:
                                         OversizedBatch(input_ids=SimpleNamespace(shape=(1, 101))))
    analyzer.model = SimpleNamespace(generate=lambda **kwargs: pytest.fail('Input already over budget'))
    with pytest.raises(InputBudgetExceeded):
        analyzer.infer([], 'test', time.monotonic() + 10)


@pytest.mark.parametrize(('checks', 'bits', 'basis'), [
    ('YNU', '010', 'A'), ('NNN', '000', 'O'), ('UUU', '000', 'U')])
def test_prefix_constraint_allows_all_verdict_codes_and_exact_json_then_eos(
        monkeypatch, checks, bits, basis):
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(
        cuda=SimpleNamespace(synchronize=lambda: None), inference_mode=nullcontext))
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        StoppingCriteria=object, StoppingCriteriaList=list))
    analyzer = Analyzer(Settings())
    tokenizer = SimpleNamespace(eos_token_id=0,
                                encode=lambda text, **kwargs: [ord(char) for char in text])
    analyzer.processor = SimpleNamespace(
        tokenizer=tokenizer,
        apply_chat_template=lambda *args, **kwargs: Batch(input_ids=SimpleNamespace(shape=(1, 5))),
        decode=lambda tokens, **kwargs: ''.join(chr(token) for token in tokens if token != 0))
    expected = json.dumps({'checks': checks, 'frames': bits, 'basis': basis}, separators=(',', ':'))

    class Generated:
        def __init__(self, tokens):
            self.tokens = tokens
            self.shape = (1, 5 + len(tokens))

        def __getitem__(self, key):
            assert key == (0, slice(5, None))
            return self.tokens

    def generate(**kwargs):
        assert kwargs['do_sample'] is False
        assert kwargs['num_beams'] == 1
        assert kwargs['max_new_tokens'] == len(expected) + 1
        callback = kwargs['prefix_allowed_tokens_fn']
        check_start = expected.index(checks)
        bit_start = expected.index('"frames":"') + len('"frames":"')
        basis_start = expected.index('"basis":"') + len('"basis":"')
        tokens = []
        for position, char in enumerate(expected):
            allowed = callback(0, SimpleNamespace(shape=(5 + position,)))
            if check_start <= position < check_start + len(checks):
                assert set(allowed) == {ord(value) for value in 'YNU'}
            elif bit_start <= position < bit_start + len(bits):
                assert set(allowed) == {ord('0'), ord('1')}
            elif position == basis_start:
                assert set(allowed) == {ord(value) for value in 'AOU'}
            else:
                assert allowed == [ord(char)]
            assert 0 not in allowed
            assert ord(char) in allowed
            tokens.append(ord(char))
        assert callback(0, SimpleNamespace(shape=(5 + len(expected),))) == [0]
        assert callback(0, SimpleNamespace(shape=(6 + len(expected),))) == [0]
        return Generated([*tokens, 0])

    analyzer.model = SimpleNamespace(generate=generate)
    result = analyzer.infer([], 'Synthetic grammar fixture', time.monotonic() + 10,
                            response_shape=(3, 3))
    assert json.loads(result) == {'checks': checks, 'frames': bits, 'basis': basis}
    assert analyzer.stage_metrics['calls'][0]['completed'] is True
