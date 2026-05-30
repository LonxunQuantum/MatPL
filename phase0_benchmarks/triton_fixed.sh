#!/bin/bash
#SBATCH --job-name=triton_fixed
#SBATCH --partition=4090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --output=/data/home/pfsuo/pfsuo/software/build/PWMLFF_test/MatPL_Agent_dev/MatPL/phase0_benchmarks/triton_fixed.log

module load gcc/8.3.1 cuda/11.8-share openmpi/4.1.6
source /data/home/pfsuo/pfsuo/software/build/PWMLFF_test/libtorch_version/2026.3/MatPL-2026.3/matpl-2026.3/bin/activate

CODE_DIR="/data/home/pfsuo/pfsuo/software/build/PWMLFF_test/MatPL_Agent_dev/MatPL"
export PYTHONPATH="$CODE_DIR"
cd "$CODE_DIR"

echo "=== Triton detection after fix ==="
python -c "
import torch
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'torch.cuda.is_available() = {torch.cuda.is_available()}')

from src.optimizer.hybrid_muon import TRITON_AVAILABLE, HybridMuonOptimizer
print(f'TRITON_AVAILABLE = {TRITON_AVAILABLE}')

# Build minimal model and test
class MiniNEP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 20, bias=False)
        self.fc2 = torch.nn.Linear(20, 5, bias=False)
        self.fc3 = torch.nn.Linear(5, 1, bias=False)
    def forward(self, x):
        return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x)))))

model = MiniNEP().cuda()
opt = HybridMuonOptimizer(
    model.parameters(),
    lr=5e-4, momentum=0.95, weight_decay=1e-3,
    lr_adjust=0.0, lr_adjust_coeff=0.18, muon_mode='slice',
    named_parameters=list(model.named_parameters()),
    enable_gram=True, flash_muon=True, magma_muon=True,
)
print(f'opt._use_flash = {opt._use_flash}')
print(f'opt._gram_orthogonalizer = {opt._gram_orthogonalizer}')

# Run one step
x = torch.randn(4, 10).cuda()
loss = model(x).sum()
loss.backward()
opt.step()
opt.zero_grad()
print('One step with flash/magma paths OK')
"
RC=$?
echo "exit code: $RC"
[ $RC -eq 0 ] && echo "PASS: Triton detection fixed" || echo "FAIL"
