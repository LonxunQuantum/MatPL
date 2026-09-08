"""Regression tests through real NEP descriptor, force and BEC operators."""
import copy
import io
import unittest
from unittest.mock import patch

import numpy as np
import torch


def make_small_model(charge=False, device='cuda', hidden=33, gpumd=None):
    from src.user.input_param import InputParam
    from src.model.nep_net import NEP
    config = {
        'model_type': 'NEP', 'atom_type': [1, 8, 29], 'seed': 123,
        'precision': 'float64', 'recover_train': False,
        'model': {'descriptor': {'cutoff': [5.0, 4.0], 'n_max': [2, 1],
                  'basis_size': [3, 3], 'l_max': [4, 2, 1], 'zbl': 1.4,
                  'charge_mode': 2 if charge else False,
                  'gpumd_nep4': bool(charge) if gpumd is None else gpumd},
                  'fitting_net': {'network_size': ([hidden] if isinstance(hidden, int) else list(hidden)) + [1]}},
        'optimizer': {'optimizer': 'ADAM', 'train_energy': True, 'train_force': True},
    }
    params = InputParam(config, 'TRAIN')
    model = NEP(params, [0.1, -0.2, 0.3], q_scaler=np.linspace(0.3, 0.8, 15),
                dtype=torch.float64, device=torch.device(device)).double().to(device)
    return model


def make_small_sample(device='cuda'):
    from src.pre_data.nep_lmdb_dataset import NepLmdbDataset
    from src.pre_data.nep_data_loader import variable_length_collate_fn
    dataset = NepLmdbDataset([], [1, 8, 29], 5.0, 4.0)
    frames = [
        {'numbers': [8, 1, 8], 'positions': [[1, 1, 1], [2.2, 1.3, 1], [1.4, 2.7, 1.5]]},
        {'numbers': [1, 8], 'positions': [[1.1, 1, 1], [2.5, 1.2, 1.4]]},
    ]
    samples = []
    for frame in frames:
        frame.update(cell=np.eye(3)*8.0, pbc=[True]*3, energy=-1.0,
                     forces=np.zeros((len(frame['numbers']), 3)), stress=np.zeros(6))
        samples.append(dataset._convert_frame(frame))
    return {k: v.to(device) for k, v in variable_length_collate_fn(samples).items()}


def neighbor_inputs(model, sample):
    from src.utils.op_loader import load_calc_ops
    ops = load_calc_ops()
    nr, na = ops.calculate_maxneigh(sample['num_atom'], sample['box'],
        sample['box_original'], sample['num_cell'], sample['position'],
        model.cutoff_radial, model.cutoff_angular, len(model.atom_type),
        sample['atom_type_map'], False)
    nr, na = max(10, int(nr.max())), max(10, int(na.max()))
    nnr, nna, nlr, nla, rr, ra = ops.calculate_neighbor(
        sample['num_atom'], sample['atom_type_map'], model.atom_type_device - 1,
        sample['box'], sample['box_original'], sample['num_cell'], sample['position'],
        model.cutoff_radial, model.cutoff_angular, nr, na, True)
    return nnr, nlr, rr, nna, nla, ra, sample['num_atom'], sample['atom_type_map']


def evaluate(model, sample, neighbors, fused=True):
    kwargs = dict(charge_label=sample['charge'], position=sample['position'],
                  box_original=sample['box_original'], volume=sample['volume'],
                  need_force=True, need_bec=bool(model.charge_mode),
                  need_charge_virial=True, need_charge_energy=True)
    if fused:
        kwargs['fitting_groups'] = sample['fitting_groups']
    return model(*neighbors, **kwargs)


def training_loss(outputs):
    # Include energy, force, virial and (when present) charge/BEC adjoints.
    return sum((i + 1) * value.square().mean() for i, value in enumerate(outputs)
               if value is not None and value.numel())


class TestFusedFallback(unittest.TestCase):
    def test_cpu_metadata_keeps_original_fitting_and_gradients(self):
        from src.model.nep_fused_fitting import build_fitting_groups
        model = make_small_model(device='cpu')
        types = torch.tensor([1, 0, 1])
        features = torch.randn(3, 15, dtype=torch.float64, requires_grad=True)
        baseline = model.calculate_Ei_with_grad(types, features, features.device)
        try:
            actual = model.calculate_Ei_with_grad(types, features, features.device,
                        fitting_groups=build_fitting_groups(types))
        except TypeError as exc:
            self.fail('NEP fitting does not yet accept CPU grouping metadata: ' + str(exc))
        for expected, observed in zip(baseline, actual):
            if expected is not None:
                torch.testing.assert_close(observed, expected)
        actual[0].sum().backward()
        self.assertIsNotNone(features.grad)


@unittest.skipUnless(torch.cuda.is_available(), 'requires a Slurm GPU allocation')
class TestFusedTraining(unittest.TestCase):
    def compare_step(self, charge):
        from src.model import nep_net
        from src.model.nep_fused_fitting import fused_fitting, fused_charge_fitting
        old = make_small_model(charge)
        new = copy.deepcopy(old)
        sample = make_small_sample()
        neighbors = neighbor_inputs(old, sample)
        old_optim = torch.optim.Adam(old.parameters(), lr=1e-4)
        new_optim = torch.optim.Adam(new.parameters(), lr=1e-4)
        baseline = evaluate(old, sample, neighbors, False)
        fitting_name = 'fused_charge_fitting' if charge else 'fused_fitting'
        fitting_function = fused_charge_fitting if charge else fused_fitting
        with patch.object(nep_net, fitting_name, wraps=fitting_function) as called:
            actual = evaluate(new, sample, neighbors, True)
            self.assertEqual(called.call_count, 1)
        for expected, observed in zip(baseline, actual):
            if expected is None:
                self.assertIsNone(observed)
            else:
                torch.testing.assert_close(observed, expected, rtol=2e-9, atol=2e-10)
        training_loss(baseline).backward()
        training_loss(actual).backward()
        for (name, param), (new_name, new_param) in zip(old.named_parameters(), new.named_parameters()):
            self.assertEqual(name, new_name)
            with self.subTest(parameter=name, charge=charge):
                if param.grad is None:
                    self.assertIsNone(new_param.grad)
                else:
                    torch.testing.assert_close(new_param.grad, param.grad, rtol=2e-8, atol=2e-9)
        old_optim.step()
        new_optim.step()
        for p, q in zip(old.parameters(), new.parameters()):
            torch.testing.assert_close(p, q, rtol=2e-9, atol=2e-10)
        self.assertIsNone(new.fitting_net[2].layers[0].weight.grad)
        # Real serialization and exporter consume the original parameter layout.
        buffer = io.BytesIO()
        torch.save(new.state_dict(), buffer)
        buffer.seek(0)
        restored = make_small_model(charge)
        restored.load_state_dict(torch.load(buffer))
        self.assertEqual(new.get_nn_params(), restored.get_nn_params())
        return new, sample, neighbors

    def test_energy_force_virial_and_adam_update_match_original(self):
        self.compare_step(False)

    def test_charge_bec_and_shared_bias_gradients_match_original(self):
        self.compare_step(True)

    def test_charge_only_step_preserves_unused_energy_head_and_adam_moments(self):
        from src.model.nep_fused_fitting import build_fitting_groups
        model = make_small_model(charge=True, gpumd=False)
        baseline = copy.deepcopy(model)
        types = torch.tensor([1, 0, 1], device='cuda')
        groups = build_fitting_groups(types.cpu()).to('cuda')
        features = torch.randn(3, 15, dtype=torch.float64, device='cuda')
        optimizers = [torch.optim.Adam(m.parameters(), lr=1e-3) for m in (baseline, model)]
        for selected in ((0, 1, 2, 3), (1,), (3,)):
            for m, opt, metadata in zip((baseline, model), optimizers, (None, groups)):
                opt.zero_grad(set_to_none=True)
                out = m.calculate_Ei_with_grad(types, features, features.device, fitting_groups=metadata)
                sum(out[i].square().sum() for i in selected).backward()
                if 0 not in selected and 2 not in selected:
                    self.assertIsNone(m.fitting_net[0].energy_head.weight.grad)
                    self.assertIsNone(m.fitting_net[0].energy_head.bias.grad)
                opt.step()
            for actual, expected in zip(model.parameters(), baseline.parameters()):
                torch.testing.assert_close(actual, expected, rtol=2e-9, atol=2e-10)

    def test_autograd_force_and_deeper_network_keep_original_path(self):
        from src.model import nep_net
        from src.model.nep_fused_fitting import fused_fitting
        sample = make_small_sample()
        for analytical, hidden in ((False, 33), (True, [33, 33])):
            with self.subTest(analytical=analytical, hidden=hidden):
                model = make_small_model(hidden=hidden)
                model.use_analytical_nep_grad = analytical
                neighbors = neighbor_inputs(model, sample)
                baseline = evaluate(model, sample, neighbors, False)
                with patch.object(nep_net, 'fused_fitting', wraps=fused_fitting) as called:
                    actual = evaluate(model, sample, neighbors, True)
                    training_loss(actual).backward()
                    self.assertEqual(called.call_count, 0)
                for got, want in zip(actual, baseline):
                    if want is not None:
                        torch.testing.assert_close(got, want)
                self.assertIsNotNone(model.fitting_net[0].layers[0].weight.grad)


if __name__ == '__main__':
    unittest.main()
