"""Exercise native CPU/CUDA readers in child processes: bad readers can corrupt memory."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.nep_to_gpumd import element_table


def write_model(path, symbols, zbl=None):
    # Zero descriptor coefficients make q=0. Each type has a distinct,
    # independently calculable two-neuron network, including the final type.
    count = len(symbols)
    lines = [f"nep5{'_zbl' if zbl else ''} {count} {' '.join(symbols)}"]
    if zbl:
        lines.append('zbl ' + zbl)
    lines += ['cutoff 2.5 2.5 100 100', 'n_max 0 0', 'basis_size 0 0',
              'l_max 4 2 1', 'ANN 2 0']
    values = []
    for t in range(count):
        values += [0.] * 14 + [.1 * (t + 1), .2] + [1. + .01 * t, 2., -.01 * t]
    values += [0.] + [0.] * (2 * count * count) + [1.] * 7
    path.write_text('\n'.join(lines + [str(v) for v in values]) + '\n')
    return path


@pytest.fixture(params=['cpu', 'cuda'])
def backend(request):
    if request.param == 'cuda':
        import torch
        if not torch.cuda.is_available():
            pytest.skip('requires an allocated GPU')
    return request.param


def run_native(backend, path, case='normal'):
    env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get('PYTHONPATH', ''))
    proc = subprocess.run([sys.executable, __file__, '--worker', backend, str(path), case],
                          env=env, text=True, capture_output=True, timeout=60)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    payload = [line for line in proc.stdout.splitlines() if line.startswith('RESULT ')]
    assert payload, proc.stdout[-2000:] + proc.stderr[-2000:]
    return json.loads(payload[-1][7:])


def test_all_118_element_networks_keep_their_parameters(tmp_path, backend):
    path = write_model(tmp_path/'all.txt', element_table[1:])
    result = run_native(backend, path)
    assert 'error' not in result, result
    t = np.arange(118)
    expected = -(1 + .01*t) * np.tanh(.1*(t+1)) - 2*np.tanh(.2) + .01*t
    np.testing.assert_allclose(result['energy'], expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(result['force'], 0., atol=1e-8)


@pytest.mark.parametrize('header,match', [
    ('nep5 0', 'element count'),
    ('nep5 -1 H', 'element count'),
    ('nep5 119 ' + ' '.join(element_table[1:] + ['H']), 'element count'),
    ('nep5 1junk H', 'element count'),
    ('nep5 2 H', 'atom symbols'),
    ('nep5 1 Xx', 'unknown element'),
    ('nep5 2 H H', 'duplicate element'),
])
def test_reject_invalid_model_elements(tmp_path, backend, header, match):
    path = write_model(tmp_path/'bad.txt', ['H'])
    path.write_text(header + '\n' + path.read_text().split('\n', 1)[1])
    result = run_native(backend, path)
    assert result.get('error_type') == 'ValueError', result
    assert match in result['error'], result


@pytest.mark.parametrize('case,match', [
    ('negative_type', 'type index'), ('type_past_end', 'type index'),
    ('short_box', 'box'), ('short_position', 'position'), ('empty', 'atom'),
])
def test_reject_invalid_structure_indices_and_lengths(tmp_path, backend, case, match):
    result = run_native(backend, write_model(tmp_path/'one.txt', ['H']), case)
    assert result.get('error_type') == 'ValueError', result
    assert match in result['error'], result


def test_high_atomic_number_universal_zbl_is_finite(tmp_path, backend):
    result = run_native(backend, write_model(tmp_path/'og.txt', ['Og'], '1 2'), 'pair')
    assert 'error' not in result, result
    assert np.isfinite(result['energy']).all()
    assert sum(result['energy']) > 0  # Repulsive ZBL for two Og atoms 1.2 A apart.
    assert np.isfinite(result['force']).all()
    assert np.max(np.abs(result['force'])) > 1


def test_typewise_zbl_checks_active_atomic_number(tmp_path, backend):
    result = run_native(backend, write_model(tmp_path/'og.txt', ['Og'], '1 2 .7'), 'pair')
    assert result.get('error_type') == 'ValueError', result
    assert 'typewise ZBL' in result['error'], result


def test_flexible_zbl_rejects_parameter_table_overflow(tmp_path, backend):
    result = run_native(backend, write_model(tmp_path/'flex.txt', element_table[1:12], '0 0'))
    assert result.get('error_type') == 'ValueError', result
    assert 'flexible ZBL' in result['error'], result


def test_inference_requires_a_loaded_model(tmp_path, backend):
    result = run_native(backend, write_model(tmp_path/'one.txt', ['H']), 'unloaded')
    assert result.get('error_type') == 'ValueError', result
    assert 'loaded' in result['error'], result


def worker(backend, path, case):
    try:
        if backend == 'cuda':
            import torch
            import importlib
            build = 'hip' if torch.version.hip else 'cuda'
            calc = importlib.import_module(f'src.feature.NEP_GPU.build.{build}.nep_gpu').NEP()
            if case != 'unloaded':
                calc.init_from_file(path, False, 0)
        else:
            from src.feature.nep_find_neigh.findneigh import FindNeigh
            calc = FindNeigh()
            if case != 'unloaded':
                calc.init_model(path)
        count = int(Path(path).read_text().splitlines()[0].split()[1])
        types = np.arange(count, dtype=np.int32)
        positions = np.zeros((3, count))
        positions[0] = np.arange(count)*3.
        box = (np.eye(3)*400.).reshape(-1)
        if case == 'negative_type': types[0] = -1
        if case == 'type_past_end': types[0] = count
        if case == 'short_box': box = box[:8]
        if case == 'short_position': positions = positions.reshape(-1)[:-1]
        if case == 'empty': types,positions = types[:0],positions[:,:0]
        if case == 'pair':
            types=np.array([0,0],dtype=np.int32)
            positions=np.array([[0.,1.2],[0.,0.],[0.,0.]])
        out = calc.inference(types.tolist(),box.tolist(),positions.reshape(-1).tolist())
        payload = {'energy':np.asarray(out[0]).tolist(),'force':np.asarray(out[1]).tolist()}
    except ValueError as exc:
        payload = {'error_type':type(exc).__name__,'error':str(exc)}
    print('RESULT ' + json.dumps(payload), flush=True)


if __name__ == '__main__':
    worker(*sys.argv[2:])
