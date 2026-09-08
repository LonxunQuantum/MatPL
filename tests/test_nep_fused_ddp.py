"""Two real GPUs exercise rank-dependent active fitting parameter sets."""
import copy
from datetime import timedelta
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


class FittingEnsemble(torch.nn.Module):
    def __init__(self, charge=False):
        super().__init__()
        from src.model.nep_fitting import FittingNet, QNEPFittingNet
        self.charge = charge
        if charge:
            self.nets = torch.nn.ModuleList([
                QNEPFittingNet([60], True, False, 'tanh', 35, 0.0, 2).double()
                for _ in range(4)])
        else:
            self.nets = torch.nn.ModuleList([
                FittingNet([60, 1], True, False, 'tanh', 35, 0.0).double()
                for _ in range(4)])

    def forward(self, x, groups):
        from src.model.nep_fused_fitting import pack_fitting_parameters, fused_fitting, fused_charge_fitting
        if self.charge:
            return fused_charge_fitting(x, self.nets, groups)
        return fused_fitting(x, *pack_fitting_parameters(self.nets, groups.type_ids), groups)


def distributed_worker(rank, init_file, charge=False):
    from src.model.nep_fused_fitting import build_fitting_groups
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', init_method='file://' + init_file,
                            rank=rank, world_size=2, timeout=timedelta(seconds=90))
    try:
        torch.manual_seed(17)
        module = FittingEnsemble(charge).cuda(rank)
        original = copy.deepcopy(module)
        wrapped = torch.nn.parallel.DistributedDataParallel(module, device_ids=[rank],
                    find_unused_parameters=True)
        types = torch.tensor([0, 1, 0, 1, 0, 0, 1] if rank == 0 else [2, 1, 2, 2, 1])
        groups = build_fitting_groups(types).to(rank)
        x = torch.randn(len(types), 35, dtype=torch.float64, device=rank)
        output = wrapped(x, groups)
        y, g = (output[rank], output[rank+2]) if charge else output
        (y.square().sum() + g.square().sum()).backward()
        ref_loss = x.new_zeros(())
        for atom_type in groups.type_ids:
            rows = x[types.to(rank) == atom_type].detach().requires_grad_()
            output = original.nets[atom_type](rows)
            ei = output[rank] if charge else output
            de = torch.autograd.grad(ei.sum(), rows, create_graph=True)[0]
            ref_loss = ref_loss + ei.square().sum() + de.square().sum()
        ref_loss.backward()
        for (name, actual), (_, expected) in zip(module.named_parameters(), original.named_parameters()):
            used = torch.tensor(int(expected.grad is not None), device=rank)
            dist.all_reduce(used)
            grad = expected.grad if expected.grad is not None else torch.zeros_like(expected)
            dist.all_reduce(grad)
            grad /= 2
            if not used:
                assert actual.grad is None
            else:
                torch.testing.assert_close(actual.grad, grad, rtol=2e-9, atol=2e-10)
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(torch.cuda.device_count() >= 2, 'requires two allocated GPUs')
class TestFusedDDP(unittest.TestCase):
    def test_different_active_types_match_reference_allreduce(self):
        with tempfile.TemporaryDirectory(prefix='nep-fitting-ddp-') as directory:
            mp.spawn(distributed_worker, args=(directory + '/init',), nprocs=2, join=True)

    def test_different_heads_and_types_match_reference_allreduce(self):
        with tempfile.TemporaryDirectory(prefix='nep-fitting-head-ddp-') as directory:
            mp.spawn(distributed_worker, args=(directory + '/init', True), nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main()
