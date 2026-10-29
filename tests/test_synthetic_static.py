from __future__ import annotations

import json
from dataclasses import asdict
from importlib.metadata import version

import pytest
import torch
from packaging.version import Version

from dart import cli
from dart.models.replay import CudaGraphTarget
from dart.verification import state_sequences, synthetic
from dart.verification.synthetic_replay import load_state_replay

pytestmark = pytest.mark.skipif(Version(version('transformers')) < Version('5.16.1'),
                                reason='static execution requires transformers >= 5.16.1')


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)

    def unexpected(*args, **kwargs):
        pytest.fail('synthetic static check used CUDA or loaded a checkpoint')

    for name in ('current_stream', 'Stream', 'CUDAGraph', 'synchronize'):
        monkeypatch.setattr(torch.cuda, name, unexpected)
    monkeypatch.setattr(synthetic.LlamaForCausalLM, 'from_pretrained', unexpected)
    yield
    torch.set_num_threads(threads)


@pytest.mark.parametrize('seed', [17, 29])
def test_static_output_and_state_checks_use_real_backend_and_close_it(seed, monkeypatch):
    calls, closed = [], []
    predictions, close = CudaGraphTarget.forward_predictions, CudaGraphTarget.close

    def record(self, state, tokens):
        calls.append(len(tokens))
        return predictions(self, state, tokens)

    def release(self):
        close(self)
        closed.append(self)

    monkeypatch.setattr(CudaGraphTarget, 'forward_predictions', record)
    monkeypatch.setattr(CudaGraphTarget, 'close', release)
    output = synthetic.run_synthetic_check(seed, execution='static')
    assert output.passed and len(output.results) == 12
    assert {row.backend for row in output.results} == {'greedy', 'block'}
    assert calls and max(calls) > 1
    state = synthetic.run_synthetic_state_check(seed, execution='static')
    assert state.passed, state
    assert len(state.steps) == 28
    trees = [step for step in state.steps if step.operation.action == 'tree']
    assert len(trees) == 2 and all(step.operation.expect_error == 'ValueError' for step in trees)
    assert len(closed) == 2 and all(target._closed and target._workspace is None for target in closed)


@pytest.mark.parametrize('state', [False, True])
def test_static_cli_preserves_rng_threads_and_writes_only_on_request(tmp_path, monkeypatch, capsys, state):
    monkeypatch.chdir(tmp_path)
    threads, rng = torch.get_num_threads(), torch.random.get_rng_state().clone()
    args = ['check', '--synthetic', '--execution', 'static', *(['--state'] if state else [])]
    assert cli.main(args) == 0
    assert 'static' in capsys.readouterr().out
    assert torch.get_num_threads() == threads and torch.equal(rng, torch.random.get_rng_state())
    assert list(tmp_path.iterdir()) == []
    path, again = tmp_path / 'check.json', tmp_path / 'again.json'
    assert cli.main([*args, '--seed', '29', '--out', str(path)]) == 0
    report = json.loads(path.read_text(encoding='utf-8'))
    assert report['schema'] == ('dart-synthetic-state-check-v2' if state else 'dart-synthetic-check-v2')
    assert report['execution'] == {'backend': 'static', 'capacity': 64, 'max_block_tokens': 5,
                                   'capture_predictions': True}
    assert '--execution' in report['reproduce'] and 'static' in report['scope']
    assert cli.main([*report['reproduce'][1:], '--out', str(again)]) == 0
    assert json.loads(again.read_text(encoding='utf-8')) == report
    if state:
        replay = load_state_replay(path)
        assert replay.execution == 'static'
        assert cli.main(['check', '--synthetic', '--state', '--replay', str(path)]) == 0


def test_static_failure_reduction_and_saved_replay_keep_the_backend(tmp_path, monkeypatch):
    path = tmp_path / 'reduced.json'
    closed, close = [], CudaGraphTarget.close

    def release(self):
        close(self)
        closed.append(self)

    monkeypatch.setattr(CudaGraphTarget, 'close', release)
    with monkeypatch.context() as fault:
        fault.setattr(CudaGraphTarget, 'crop', lambda *a: None)
        assert cli.main(['check', '--synthetic', '--state', '--execution', 'static',
                         '--reduce', '16', '--out', str(path)]) == 1
        data = json.loads(path.read_text(encoding='utf-8'))
        reduction = data['reduction']
        assert reduction['reproduced'] and reduction['accepted'] > 0
        assert reduction['failure'] == reduction['final_failure']
        assert len(closed) == reduction['evaluations'] <= 16
        assert all(target._closed and target._workspace is None for target in closed)
        replay = load_state_replay(path)
        assert replay.execution == 'static' and replay.selection == 'reduced'
        assert len(replay.operations) < 28
        assert cli.main(data['reproduce'][1:]) == 1
    assert cli.main(data['reproduce'][1:]) == 0


@pytest.mark.parametrize('kind', ['prediction', 'exception'])
def test_static_output_check_detects_backend_faults_and_closes_target(monkeypatch, kind):
    closed, close, predictions = [], CudaGraphTarget.close, CudaGraphTarget.forward_predictions

    def release(self):
        close(self)
        closed.append(self)

    def faulty(self, state, tokens):
        if kind == 'exception':
            raise RuntimeError('static prediction failed')
        result = predictions(self, state, tokens)
        return [(token + 1) % synthetic.VOCAB_SIZE for token in result]

    monkeypatch.setattr(CudaGraphTarget, 'close', release)
    with monkeypatch.context() as fault:
        fault.setattr(CudaGraphTarget, 'forward_predictions', faulty)
        report = synthetic.run_synthetic_check(execution='static')
        assert not report.passed and report.failures
        expected = {'exception'} if kind == 'exception' else {'tokens', 'eos'}
        assert {row.failure.kind for row in report.failures} <= expected
    assert len(closed) == 1 and closed[0]._closed and closed[0]._workspace is None
    assert synthetic.run_synthetic_check(execution='static').passed


@pytest.mark.parametrize('state', [False, True])
def test_reference_error_closes_static_backend_and_writes_nothing(tmp_path, monkeypatch, state):
    closed, close = [], CudaGraphTarget.close

    def release(self):
        close(self)
        closed.append(self)

    def fail(*args, **kwargs):
        raise RuntimeError('reference failed')

    monkeypatch.setattr(CudaGraphTarget, 'close', release)
    if state:
        monkeypatch.setattr(state_sequences, '_reference_forward', fail)
    else:
        monkeypatch.setattr(synthetic, 'hf_decoder', lambda model: fail)
    threads = torch.get_num_threads()
    with pytest.raises(RuntimeError, match='reference failed'):
        cli.main(['check', '--synthetic', *(['--state'] if state else []), '--execution', 'static',
                  '--out', str(tmp_path / 'failed.json')])
    assert torch.get_num_threads() == threads and list(tmp_path.iterdir()) == []
    assert len(closed) == 1 and closed[0]._closed and closed[0]._workspace is None


@pytest.mark.parametrize('key,value', [('capacity', 65), ('capacity', 64.0), ('max_block_tokens', 5.0),
                                      ('capture_predictions', 1), ('unknown', True)])
def test_replay_requires_exact_execution_settings(tmp_path, key, value):
    path = tmp_path / 'check.json'
    assert cli.main(['check', '--synthetic', '--state', '--execution', 'static', '--out', str(path)]) == 0
    data = json.loads(path.read_text(encoding='utf-8'))
    data['execution'][key] = value
    path.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(ValueError, match='execution'):
        load_state_replay(path)


@pytest.mark.parametrize('execution', [None, True, 'cuda_graph', {}])
def test_invalid_library_execution_fails_before_model_creation(execution, monkeypatch):
    monkeypatch.setattr(synthetic, '_synthetic_models', lambda *a: pytest.fail('built a model'))
    for check in (synthetic.run_synthetic_check, synthetic.run_synthetic_state_check):
        with pytest.raises(ValueError, match='execution'):
            check(execution=execution)


def test_static_execution_cannot_be_combined_with_a_custom_adapter(monkeypatch):
    monkeypatch.setattr(synthetic, '_synthetic_models', lambda *a: pytest.fail('built a model'))
    with pytest.raises(ValueError, match='adapter_factory'):
        synthetic.run_synthetic_check(execution='static', adapter_factory=lambda target: {})


@pytest.mark.parametrize('execution', ['static', 'eager'])
def test_execution_option_is_only_for_synthetic_checks(execution, monkeypatch):
    monkeypatch.setattr(cli, '_run', lambda *a: pytest.fail('loaded a local model'))
    with pytest.raises(SystemExit) as error:
        cli.main(['check', '--model', 'unused', '--execution', execution])
    assert error.value.code == 2


@pytest.mark.parametrize('change', [
    {'execution': {'backend': 'cuda_graph'}}, {'execution': {'backend': 'static', 'capacity': 65}},
    {'execution': None}, {'execution': {}}, {'schema': 'dart-synthetic-state-check-v1'},
])
def test_replay_rejects_mislabelled_or_unbounded_execution(tmp_path, monkeypatch, change):
    data = {'schema': 'dart-synthetic-state-check-v2', 'seed': 17,
            'model': {'architecture': 'LlamaForCausalLM', 'config': synthetic.synthetic_model_config(),
                      'device': 'cpu', 'dtype': 'float32', 'threads': 1},
            'execution': {'backend': 'static', 'capacity': 64, 'max_block_tokens': 5, 'capture_predictions': True},
            'steps': [{'operation': asdict(synthetic.synthetic_state_operations()[0])}], **change}
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps(data), encoding='utf-8')
    monkeypatch.setattr(synthetic, '_synthetic_models', lambda *a: pytest.fail('built a model'))
    with pytest.raises(ValueError):
        load_state_replay(path)


def test_replay_backend_cannot_be_overridden(tmp_path, monkeypatch):
    path = tmp_path / 'static.json'
    assert cli.main(['check', '--synthetic', '--state', '--execution', 'static', '--out', str(path)]) == 0
    monkeypatch.setattr(synthetic, '_synthetic_models', lambda *a: pytest.fail('built a model'))
    with pytest.raises(SystemExit) as error:
        cli.main(['check', '--synthetic', '--state', '--replay', str(path), '--execution', 'eager'])
    assert error.value.code == 2
