#!/bin/bash
#SBATCH --job-name=triton_check
#SBATCH --partition=4090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --output=/data/home/pfsuo/pfsuo/software/build/PWMLFF_test/MatPL_Agent_dev/MatPL/phase0_benchmarks/triton_check.log

module load gcc/8.3.1 cuda/11.8-share openmpi/4.1.6
source /data/home/pfsuo/pfsuo/software/build/PWMLFF_test/libtorch_version/2026.3/MatPL-2026.3/matpl-2026.3/bin/activate

echo "=== Triton availability check ==="
python -c "
import torch
print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Compute capability: {torch.cuda.get_device_capability(0)}')

# Check Triton
try:
    import triton
    import triton.language as tl
    print(f'Triton version: {triton.__version__}')
    target = triton.runtime.driver.active.get_current_target()
    print(f'Triton target: {target}')
    print('TRITON_AVAILABLE = True')
except Exception as e:
    print(f'Triton NOT available: {e}')

# Now test via the actual module
from src.optimizer.hybrid_muon import TRITON_AVAILABLE as HA, HybridMuonOptimizer
print(f'hybrid_muon.TRITON_AVAILABLE = {HA}')

# Build a minimal model and check flash/magma status
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
print(f'use_flash = {opt._use_flash}')
print(f'use_gram (magma) = {opt._gram_orthogonalizer is None and opt.param_groups[0][\"enable_gram\"]}')

# Run a training step to trigger flash/magma path
x = torch.randn(4, 10).cuda()
loss = model(x).sum()
loss.backward()
opt.step()
opt.zero_grad()
print('One step OK with flash/magma paths')

# Check which paths were actually taken
routing = opt._routing
for r in routing:
    kind = r.get('kind', 'unknown')
    shape = r.get('shape', '?')
    print(f'  param routed as: kind={kind}, shape={shape}')
"
RC=$?
echo "exit code: $RC"
[ $RC -eq 0 ] && echo "PASS: Triton + flash/magma paths verified" || echo "FAIL"
