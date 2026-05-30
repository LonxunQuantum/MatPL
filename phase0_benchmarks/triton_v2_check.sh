#!/bin/bash
#SBATCH --job-name=triton_v2_check
#SBATCH --partition=4090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --output=/data/home/pfsuo/pfsuo/software/build/PWMLFF_test/MatPL_Agent_dev/MatPL/phase0_benchmarks/triton_v2_check.log

module load gcc/8.3.1 cuda/11.8-share openmpi/4.1.6
source /data/home/pfsuo/pfsuo/software/build/PWMLFF_test/libtorch_version/2026.3/MatPL-2026.3/matpl-2026.3/bin/activate

echo "=== Triton 2.2.0 deep check ==="
python -c "
import torch, triton, triton.language as tl
print(f'Triton {triton.__version__}, PyTorch {torch.__version__}')

# Probe driver API surface
import triton.runtime.driver as drv
print(dir(drv)[:20])

# Try the actual active driver API
try:
    a = drv.active
    print(f'driver.active = {a}')
except AttributeError as e:
    print(f'driver.active missing: {e}')

# Try get_current_target directly from runtime
try:
    target = triton.runtime.driver.utils.get_current_target()
    print(f'get_current_target = {target}')
except Exception as e:
    print(f'get_current_target failed: {e}')

# Try the simplest Triton kernel to verify the compiler works
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

N = 1024
x = torch.randn(N, device='cuda')
y = torch.randn(N, device='cuda')
out = torch.empty(N, device='cuda')
grid = lambda META: (triton.cdiv(N, META['BLOCK']),)
add_kernel[grid](x, y, out, N, BLOCK=256)
torch.cuda.synchronize()
assert torch.allclose(out, x + y), 'Kernel output mismatch!'
print('Triton kernel compiles and runs correctly!')
"
RC=$?
echo "exit code: $RC"
