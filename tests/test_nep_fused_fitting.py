import importlib
import pickle
import unittest

import torch
import numpy as np


def _adapter():
    return importlib.import_module("src.model.nep_fused_fitting")


def _reference(X, W, b, V, c, groups):
    ys = []
    gs = []
    for packed_type, count in enumerate(groups.counts):
        start = groups.offsets[packed_type]
        ids = groups.atom_ids.narrow(0, int(start), count)
        x = X.index_select(0, ids)
        y = torch.tanh(x @ W[packed_type] + b[packed_type]) @ V[packed_type] + c[packed_type]
        channel_grads = [
            torch.autograd.grad(y[:, q].sum(), x, create_graph=True, retain_graph=True)[0]
            for q in range(y.shape[1])
        ]
        ys.append(y)
        gs.append(torch.stack(channel_grads))
    if ys:
        ordered_y = torch.cat(ys).transpose(0, 1)
        ordered_g = torch.cat(gs, dim=1)
        return (
            torch.zeros(V.shape[2], X.shape[0], dtype=X.dtype, device=X.device)
            .index_copy(1, groups.atom_ids, ordered_y),
            torch.zeros(V.shape[2], X.shape[0], X.shape[1], dtype=X.dtype, device=X.device)
            .index_copy(1, groups.atom_ids, ordered_g),
        )
    return X.new_empty((V.shape[2], 0)), X.new_empty((V.shape[2], 0, X.shape[1]))


class TestFittingGroups(unittest.TestCase):
    def test_first_appearance_order_preserves_original_row_mapping(self):
        groups = _adapter().build_fitting_groups(torch.tensor([2, 0, 2], dtype=torch.int64))
        self.assertEqual(groups.type_ids, (2, 0))
        self.assertEqual(groups.counts, (2, 1))
        self.assertEqual(groups.atom_ids.tolist(), [0, 2, 1])
        self.assertEqual(groups.offsets.tolist(), [0, 2, 3])

    def test_empty_input_has_well_formed_offsets(self):
        groups = _adapter().build_fitting_groups(torch.empty(0, dtype=torch.int32))
        self.assertEqual(groups.type_ids, ())
        self.assertEqual(groups.counts, ())
        self.assertEqual(groups.atom_ids.dtype, torch.int64)
        self.assertEqual(groups.atom_ids.tolist(), [])
        self.assertEqual(groups.offsets.tolist(), [0])

    def test_metadata_is_picklable_and_to_moves_only_indices(self):
        groups = pickle.loads(pickle.dumps(
            _adapter().build_fitting_groups(torch.tensor([3, 1, 3]))
        ))
        moved = groups.to("cpu")
        self.assertEqual(moved.type_ids, (3, 1))
        self.assertEqual(moved.counts, (2, 1))
        self.assertEqual(moved.atom_ids.tolist(), [0, 2, 1])
        self.assertEqual(moved.offsets.tolist(), [0, 2, 3])

    def test_rejects_non_cpu_and_invalid_type_maps(self):
        build = _adapter().build_fitting_groups
        for bad in (
            torch.tensor([0.0, 1.0]),
            torch.tensor([True, False]),
            torch.tensor([[0, 1]]),
            torch.tensor([0, -1]),
        ):
            with self.subTest(dtype=bad.dtype, shape=tuple(bad.shape)):
                with self.assertRaises((TypeError, ValueError)):
                    build(bad)
        if torch.cuda.is_available():
            with self.assertRaises(ValueError):
                build(torch.tensor([0, 1], device="cuda"))

    def test_collate_keeps_labels_and_adds_groups_after_concatenation(self):
        from src.pre_data.nep_data_loader import variable_length_collate_fn

        batch = [
            {"atom_type_map": torch.tensor([2, 0]), "ei": torch.tensor([1.0, 2.0]), "energy": torch.tensor([3.0])},
            {"atom_type_map": torch.tensor([2]), "ei": torch.tensor([4.0]), "energy": torch.tensor([5.0])},
        ]
        result = variable_length_collate_fn(batch)
        self.assertEqual(result["atom_type_map"].tolist(), [2, 0, 2])
        self.assertEqual(result["ei"].tolist(), [1.0, 2.0, 4.0])
        self.assertEqual(result["energy"].tolist(), [[3.0], [5.0]])
        self.assertEqual(result["fitting_groups"].type_ids, (2, 0))
        self.assertEqual(result["fitting_groups"].atom_ids.tolist(), [0, 2, 1])


class TestPackFittingParameters(unittest.TestCase):
    def _fitting(self, bias=True, last_bias=True):
        from src.model.nep_fitting import FittingNet
        return FittingNet([4, 1], bias, False, "tanh", 3, 0.25, last_bias=last_bias).double()

    def _qfitting(self, bias=True, last_bias=True):
        from src.model.nep_fitting import QNEPFittingNet
        return QNEPFittingNet([4], bias, False, "tanh", 3, 0.25, 1, last_bias=last_bias).double()

    def test_fitting_pack_shapes_values_and_gradients(self):
        nets = torch.nn.ModuleList([self._fitting(), self._fitting(), self._fitting()])
        W, b, V, c = _adapter().pack_fitting_parameters(nets, (2, 0))
        self.assertEqual((W.shape, b.shape, V.shape, c.shape),
                         ((2, 3, 4), (2, 4), (2, 4, 1), (2, 1)))
        self.assertTrue(torch.equal(W[0], nets[2].layers[0].weight))
        self.assertTrue(torch.equal(b[1], nets[0].layers[0].bias[0]))
        self.assertTrue(torch.equal(V[0], nets[2].layers[1].weight))
        self.assertTrue(torch.equal(c[1], nets[0].layers[1].bias[0]))
        (W.sum() + b.sum() + V.sum() + c.sum()).backward()
        self.assertIsNone(nets[1].layers[0].weight.grad)
        self.assertIsNone(nets[1].layers[1].bias.grad)
        self.assertIsNotNone(nets[0].layers[0].weight.grad)
        self.assertIsNotNone(nets[2].layers[1].bias.grad)

    def test_absent_biases_are_zero_constants_and_parameters_stay_inactive(self):
        nets = torch.nn.ModuleList([self._fitting(False, False), self._fitting(False, False)])
        W, b, V, c = _adapter().pack_fitting_parameters(nets, (1,))
        self.assertEqual(b.tolist(), [[0.0] * 4])
        self.assertEqual(c.tolist(), [[0.0]])
        (W.sum() + b.sum() + V.sum() + c.sum()).backward()
        self.assertIsNone(nets[0].layers[0].weight.grad)
        self.assertIsNotNone(nets[1].layers[0].weight.grad)

    def test_qnep_combines_heads_and_preserves_optional_biases(self):
        nets = torch.nn.ModuleList([self._qfitting(False, False), self._qfitting(True, True)])
        W, b, V, c = _adapter().pack_fitting_parameters(nets, (0, 1))
        self.assertEqual((W.shape, b.shape, V.shape, c.shape),
                         ((2, 3, 4), (2, 4), (2, 4, 2), (2, 2)))
        self.assertEqual(b[0].tolist(), [0.0] * 4)
        self.assertEqual(c[0].tolist(), [0.0, 0.0])
        self.assertEqual(c[1, 1].item(), 0.0)
        self.assertTrue(torch.equal(V[1, :, 0:1], nets[1].energy_head.weight))
        self.assertTrue(torch.equal(V[1, :, 1:2], nets[1].charge_head.weight))
        (W.sum() + b.sum() + V.sum() + c.sum()).backward()
        self.assertIsNotNone(nets[0].charge_head.weight.grad)
        self.assertIsNone(nets[0].energy_head.bias)
        self.assertIsNone(nets[1].charge_head.bias)

    def test_empty_active_type_set_returns_shaped_empty_packs(self):
        net = self._fitting()
        W, b, V, c = _adapter().pack_fitting_parameters(torch.nn.ModuleList([net]), ())
        self.assertEqual((W.shape, b.shape, V.shape, c.shape),
                         ((0, 3, 4), (0, 4), (0, 4, 1), (0, 1)))
        self.assertIsNone(net.layers[0].weight.grad)

    def test_nep_ignores_imported_hidden_bias_when_bias_flag_is_false(self):
        from src.model.nep_fitting import FittingNet

        params = [
            np.arange(12, dtype=np.float64).reshape(3, 4) / 10,
            np.array([[9.0, 8.0, 7.0, 6.0]], dtype=np.float64),
            np.arange(4, dtype=np.float64).reshape(4, 1) / 7,
            np.array(0.3, dtype=np.float64),
        ]
        net = FittingNet([4, 1], False, False, "tanh", 3, 0.0,
                         nep_txt_param=params, last_bias=True).double()
        x = torch.tensor([[0.2, -0.1, 0.4]], dtype=torch.float64)
        W, b, V, c = _adapter().pack_fitting_parameters([net], (0,))
        packed_result = torch.tanh(x @ W[0] + b[0]) @ V[0] + c[0]
        torch.testing.assert_close(packed_result, net(x))
        packed_result.sum().backward()
        self.assertIsNone(net.layers[0].bias.grad)

    def test_qnep_ignores_imported_hidden_bias_when_bias_flag_is_false(self):
        from src.model.nep_fitting import QNEPFittingNet

        params = [
            np.arange(12, dtype=np.float64).reshape(3, 4) / 10,
            np.array([[9.0, 8.0, 7.0, 6.0]], dtype=np.float64),
            np.arange(8, dtype=np.float64).reshape(4, 2) / 7,
            np.array(0.3, dtype=np.float64),
        ]
        net = QNEPFittingNet([4], False, False, "tanh", 3, 0.0, 1,
                             nep_txt_param=params, last_bias=True).double()
        x = torch.tensor([[0.2, -0.1, 0.4]], dtype=torch.float64)
        W, b, V, c = _adapter().pack_fitting_parameters([net], (0,))
        packed_result = torch.tanh(x @ W[0] + b[0]) @ V[0] + c[0]
        energy, charge = net(x)
        torch.testing.assert_close(packed_result, torch.cat((energy, charge), dim=1))
        packed_result.sum().backward()
        self.assertIsNone(net.layers[0].bias.grad)


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA/HIP GPU")
class TestFusedFittingGPU(unittest.TestCase):
    CASES = ((35, 40, 1), (35, 60, 2), (31, 33, 1), (65, 99, 2), (96, 100, 2))

    def _inputs(self, D, H, Q, atom_types=(2, 0, 2, 3, 0)):
        torch.manual_seed(D * 10000 + H * 10 + Q)
        groups = _adapter().build_fitting_groups(
            torch.tensor(atom_types, dtype=torch.int64)
        ).to("cuda")
        active_types = len(groups.type_ids)
        shapes = (
            (len(atom_types), D),
            (active_types, D, H),
            (active_types, H),
            (active_types, H, Q),
            (active_types, Q),
        )
        return groups, [torch.randn(s, device="cuda", dtype=torch.float64, requires_grad=True) * 0.2 for s in shapes]

    def _assert_close(self, got, want, *, gradcheck=False):
        atol = 2e-5 if gradcheck else 2e-9
        rtol = 2e-5 if gradcheck else 2e-8
        torch.testing.assert_close(got, want, atol=atol, rtol=rtol)

    def test_outputs_and_arbitrary_vjp_match_independent_autograd_reference(self):
        for D, H, Q in self.CASES:
            with self.subTest(D=D, H=H, Q=Q):
                groups, values = self._inputs(D, H, Q)
                X, W, b, V, c = values
                actual = _adapter().fused_fitting(X, W, b, V, c, groups)
                expected = _reference(X, W, b, V, c, groups)
                self._assert_close(actual[0], expected[0])
                self._assert_close(actual[1], expected[1])
                a = torch.randn_like(actual[0])
                u = torch.randn_like(actual[1])
                actual_grad = torch.autograd.grad((actual[0] * a).sum() + (actual[1] * u).sum(), values, retain_graph=True)
                expected_grad = torch.autograd.grad((expected[0] * a).sum() + (expected[1] * u).sum(), values)
                for got, want in zip(actual_grad, expected_grad):
                    self._assert_close(got, want)

    def test_only_y_and_only_g_backward_match_reference(self):
        for output_index in (0, 1):
            with self.subTest(output="YG"[output_index]):
                groups, values = self._inputs(31, 33, 2)
                actual = _adapter().fused_fitting(*values, groups)
                expected = _reference(*values, groups)
                seed = torch.randn_like(actual[output_index])
                actual_grad = torch.autograd.grad(
                    (actual[output_index] * seed).sum(), values,
                    retain_graph=True, allow_unused=True,
                )
                expected_grad = torch.autograd.grad(
                    (expected[output_index] * seed).sum(), values, allow_unused=True,
                )
                for got, want in zip(actual_grad, expected_grad):
                    if want is None:
                        self.assertIsNone(got)
                    else:
                        self._assert_close(got, want)

    def test_ragged_missing_types_permuted_rows_and_empty_atoms(self):
        for atom_types in ((4, 1, 4, 1, 4, 0), ()):
            with self.subTest(atom_types=atom_types):
                groups, values = self._inputs(35, 40, 2, atom_types)
                actual = _adapter().fused_fitting(*values, groups)
                expected = _reference(*values, groups)
                self._assert_close(actual[0], expected[0])
                self._assert_close(actual[1], expected[1])

    def test_small_gradcheck(self):
        groups, values = self._inputs(3, 4, 2, (1, 0, 1))
        self.assertTrue(torch.autograd.gradcheck(
            lambda *xs: _adapter().fused_fitting(*xs, groups),
            tuple(values), eps=1e-6, atol=2e-4, rtol=2e-3, check_undefined_grad=False,
        ))

    def test_autograd_context_does_not_save_n_by_h_cache(self):
        groups, values = self._inputs(35, 40, 2)
        outputs = _adapter().fused_fitting(*values, groups)
        saved = outputs[0].grad_fn.saved_tensors
        allowed_ids = {id(value) for value in values} | {id(groups.atom_ids), id(groups.offsets)}
        self.assertTrue(all(id(tensor) in allowed_ids for tensor in saved))
        self.assertFalse(any(tuple(tensor.shape) == (5, 40) for tensor in saved))

    def test_dataloader_pin_memory_preserves_immutable_host_metadata(self):
        from torch.utils.data._utils.pin_memory import pin_memory

        groups = _adapter().build_fitting_groups(torch.tensor([2, 0, 2]))
        pinned = pin_memory(groups)
        self.assertIsInstance(pinned.type_ids, tuple)
        self.assertIsInstance(pinned.counts, tuple)
        self.assertEqual(pinned.type_ids, (2, 0))
        self.assertEqual(pinned.counts, (2, 1))
        self.assertTrue(pinned.atom_ids.is_pinned())
        self.assertTrue(pinned.offsets.is_pinned())


if __name__ == "__main__":
    unittest.main()
