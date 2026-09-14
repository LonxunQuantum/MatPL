"""HIP AOT coverage for the fused NEP fitting operator."""

from pathlib import Path
import unittest

import torch


REPO = Path(__file__).resolve().parents[1]
IS_HIP = torch.cuda.is_available() and torch.version.hip is not None


def fitting_reference(x, w, b, v, c, atom_ids, offsets):
    n = x.shape[0]
    atom_types = torch.empty(n, dtype=torch.long, device=x.device)
    for type_index in range(offsets.numel() - 1):
        first, end = int(offsets[type_index]), int(offsets[type_index + 1])
        atom_types[atom_ids[first:end]] = type_index
    selected_w = w.index_select(0, atom_types)
    hidden = torch.tanh(
        torch.einsum("nd,ndh->nh", x, selected_w)
        + b.index_select(0, atom_types)
    )
    selected_v = v.index_select(0, atom_types)
    y = torch.einsum("nh,nhq->qn", hidden, selected_v)
    y = y + c.index_select(0, atom_types).transpose(0, 1)
    g = torch.einsum(
        "nhq,ndh->qnd",
        (1.0 - hidden.square()).unsqueeze(-1) * selected_v,
        selected_w,
    )
    return y, g


class HipAotFittingTests(unittest.TestCase):
    def make_inputs(self, n=11, d=35, h=60, q=2, types=3):
        from src.model.nep_fused_fitting import build_fitting_groups

        atom_types = torch.arange(n, dtype=torch.int64) % types
        groups = build_fitting_groups(atom_types).to("cuda")
        values = [
            torch.randn(shape, dtype=torch.float64, device="cuda")
            .mul_(0.1)
            .requires_grad_()
            for shape in (
                (n, d),
                (types, d, h),
                (types, h),
                (types, h, q),
                (types, q),
            )
        ]
        return groups, values

    def test_hip_build_wires_aot_fitting_without_hip_jit(self):
        cmake = (REPO / "src/op/cmake/hip/CMakeLists.txt").read_text()
        launcher = (REPO / "src/op/src/nep_fitting_launcher.cpp").read_text()
        binding = (REPO / "src/op/src/CalcOps_bind.cpp").read_text()
        adapter = (REPO / "src/model/nep_fused_fitting.py").read_text()
        network = (REPO / "src/model/nep_net.py").read_text()

        self.assertTrue((REPO / "src/op/kernel_hip/nep_fitting.hip").is_file())
        self.assertIn("MATPL_FUSED_FITTING_HIP", cmake)
        self.assertIn("nep_fitting_jit.cpp", cmake)
        self.assertIn("MATPL_FUSED_FITTING_HIP", launcher)
        self.assertIn("MATPL_FUSED_FITTING_HIP", binding)
        self.assertNotIn("hiprtc", cmake.lower())
        self.assertIn("torch.version.hip is not None", adapter)
        self.assertNotIn("feats_scaled.is_cuda and not torch.version.hip", network)

    @unittest.skipUnless(IS_HIP, "requires a Slurm DCU allocation")
    def test_hip_aot_forward_backward_matches_reference(self):
        from src.utils.op_loader import load_calc_ops

        ops = load_calc_ops()
        for d, h, q in ((35, 60, 1), (31, 33, 2), (96, 100, 2)):
            with self.subTest(d=d, h=h, q=q):
                torch.manual_seed(d * 1000 + h * 10 + q)
                counts = [5, 2, 4]
                n = sum(counts)
                values = [
                    torch.randn(n, d, dtype=torch.float64, device="cuda") * 0.1,
                    torch.randn(3, d, h, dtype=torch.float64, device="cuda") * 0.1,
                    torch.randn(3, h, dtype=torch.float64, device="cuda") * 0.1,
                    torch.randn(3, h, q, dtype=torch.float64, device="cuda") * 0.1,
                    torch.randn(3, q, dtype=torch.float64, device="cuda") * 0.1,
                ]
                x, w, b, v, c = [
                    value.detach().requires_grad_(True) for value in values
                ]
                atom_ids = torch.tensor(
                    [9, 2, 7, 0, 5, 10, 1, 8, 3, 6, 4],
                    dtype=torch.long,
                    device="cuda",
                )
                offsets = torch.tensor(
                    [0, 5, 7, 11], dtype=torch.long, device="cuda"
                )
                grad_y = torch.randn(q, n, dtype=torch.float64, device="cuda")
                grad_g = torch.randn(q, n, d, dtype=torch.float64, device="cuda")

                expected_y, expected_g = fitting_reference(
                    x, w, b, v, c, atom_ids, offsets
                )
                expected = torch.autograd.grad(
                    (expected_y * grad_y).sum() + (expected_g * grad_g).sum(),
                    (x, w, b, v, c),
                )
                actual_y, actual_g = ops.nep_fitting_forward(
                    x, w, b, v, c, atom_ids, offsets, counts
                )
                actual = ops.nep_fitting_backward(
                    x, w, b, v, c, atom_ids, offsets, counts, grad_y, grad_g
                )
                torch.cuda.synchronize()

                torch.testing.assert_close(
                    actual_y, expected_y, rtol=2e-11, atol=2e-11
                )
                torch.testing.assert_close(
                    actual_g, expected_g, rtol=2e-11, atol=2e-11
                )
                for got, want in zip(actual, expected):
                    torch.testing.assert_close(got, want, rtol=4e-11, atol=4e-11)

    @unittest.skipUnless(IS_HIP, "requires a Slurm DCU allocation")
    def test_hip_aot_uses_current_stream_and_is_reentrant(self):
        from src.model.nep_fused_fitting import fused_fitting

        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            groups, values = self.make_inputs()
            for _ in range(4):
                actual = fused_fitting(*values, groups)
                expected = fitting_reference(
                    *values, groups.atom_ids, groups.offsets
                )
                actual_grad = torch.autograd.grad(
                    sum(t.square().sum() for t in actual), values,
                    retain_graph=True,
                )
                expected_grad = torch.autograd.grad(
                    sum(t.square().sum() for t in expected), values,
                    retain_graph=True,
                )
                for got, want in zip(actual, expected):
                    torch.testing.assert_close(got, want, rtol=2e-11, atol=2e-11)
                for got, want in zip(actual_grad, expected_grad):
                    torch.testing.assert_close(got, want, rtol=4e-11, atol=4e-11)
        torch.cuda.current_stream().wait_stream(stream)

    @unittest.skipUnless(IS_HIP, "requires a Slurm DCU allocation")
    def test_hip_aot_backward_has_no_atom_sized_workspace(self):
        from src.model.nep_fused_fitting import fused_fitting

        groups, values = self.make_inputs(n=4096, d=35, h=80, q=1, types=89)
        y, g = fused_fitting(*values, groups)
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        grads = torch.autograd.grad((y, g), values, (torch.randn_like(y), torch.randn_like(g)))
        torch.cuda.synchronize()
        increase = torch.cuda.max_memory_allocated() - before
        gradient_bytes = sum(t.numel() * t.element_size() for t in grads)
        self.assertLessEqual(increase, gradient_bytes + 2 * 1024**2)
        self.assertTrue(all(torch.isfinite(t).all() for t in grads))


if __name__ == "__main__":
    unittest.main()
