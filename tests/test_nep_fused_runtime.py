import unittest

import torch


@unittest.skipUnless(torch.cuda.is_available() and not torch.version.hip,
                     'requires a CUDA allocation')
class TestFusedRuntime(unittest.TestCase):
    def inputs(self, n=5, d=35, h=60, q=2, types=3):
        from src.model.nep_fused_fitting import build_fitting_groups
        groups = build_fitting_groups(torch.arange(n, dtype=torch.int64) % types).to('cuda')
        values = [torch.randn(shape, device='cuda', dtype=torch.float64).mul_(0.1).requires_grad_()
                  for shape in ((n, d), (types, d, h), (types, h), (types, h, q), (types, q))]
        return groups, values

    def test_nondefault_stream_matches_pytorch_without_host_synchronization(self):
        from src.model.nep_fused_fitting import fused_fitting
        from test_nep_fused_fitting import _reference
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            groups, values = self.inputs()
            actual = fused_fitting(*values, groups)
            expected = _reference(*values, groups)
            actual_grad = torch.autograd.grad(sum(t.square().sum() for t in actual), values)
            expected_grad = torch.autograd.grad(sum(t.square().sum() for t in expected), values)
        torch.cuda.current_stream().wait_stream(stream)
        for got, want in zip(actual_grad, expected_grad):
            torch.testing.assert_close(got, want, rtol=2e-8, atol=2e-9)

    def test_backward_workspace_stays_bounded_for_large_atom_count(self):
        from src.model.nep_fused_fitting import fused_fitting
        groups, values = self.inputs(n=20000, d=96, h=100, types=89)
        y, g = fused_fitting(*values, groups)
        a, u = torch.randn_like(y), torch.randn_like(g)
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        grads = torch.autograd.grad((y, g), values, (a, u))
        torch.cuda.synchronize()
        increase = torch.cuda.max_memory_allocated() - allocated
        gradient_bytes = sum(t.numel()*t.element_size() for t in grads)
        # Allocator rounding allowance; all atom-sized result gradients are
        # accounted for separately from the bounded reduction workspace.
        self.assertLessEqual(increase, gradient_bytes + 16*1024**2 + 2*1024**2)
        self.assertTrue(all(torch.isfinite(t).all() for t in grads))


if __name__ == '__main__':
    unittest.main()
