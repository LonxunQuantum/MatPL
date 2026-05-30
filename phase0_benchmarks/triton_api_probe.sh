#!/bin/bash
#SBATCH --job-name=triton_probe
#SBATCH --partition=4090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=00:05:00
#SBATCH --output=/data/home/pfsuo/pfsuo/software/build/PWMLFF_test/MatPL_Agent_dev/MatPL/phase0_benchmarks/triton_api_probe.log

module load gcc/8.3.1 cuda/11.8-share openmpi/4.1.6
source /data/home/pfsuo/pfsuo/software/build/PWMLFF_test/libtorch_version/2026.3/MatPL-2026.3/matpl-2026.3/bin/activate

python -c "
import triton, torch
print(f'Triton {triton.__version__}')

# Probe the driver module structure
import triton.runtime.driver as drv
import triton.runtime as rt

# What backends are registered?
if hasattr(rt, 'backends'):
    print(f'backends: {rt.backends}')
if hasattr(drv, 'backends'):
    print(f'driver.backends: {drv.backends}')

# Check what utils has
if hasattr(drv, 'utils'):
    print(f'driver.utils attrs: {[x for x in dir(drv.utils) if not x.startswith(\"_\")]}')

# Check if there's a way to get active CUDA driver
for attr in ['active', 'cuda', 'driver', 'get_current_device', 'get_active_driver', '_active']:
    if hasattr(drv, attr):
        print(f'driver.{attr} exists')

# Try to access the CUDA driver object directly
try:
    cuda_drv = drv.CudaDriver()
    print(f'CudaDriver attrs: {[x for x in dir(cuda_drv) if not x.startswith(\"_\")]}')
except Exception as e:
    print(f'CudaDriver() failed: {e}')

# Check if we can detect via triton's own internal checks
try:
    from triton.common.backend import get_backend
    print(f'get_backend exists')
except:
    print('no get_backend')

# Fallback: simply check if torch has CUDA and try compiling a trivial kernel to a file
# The real test is whether @triton.jit can generate PTX/cubin
print(f'torch.cuda.is_available() = {torch.cuda.is_available()}')
print(f'torch.cuda.get_device_capability() = {torch.cuda.get_device_capability()}')

# Check if driver module is a package with submodules
import os, inspect
drv_file = inspect.getfile(drv)
print(f'driver module file: {drv_file}')
print(f'driver module dir: {os.path.dirname(drv_file)}')

# List files in the driver directory
drv_dir = os.path.dirname(drv_file)
if os.path.isdir(drv_dir):
    print(f'Files in driver dir: {os.listdir(drv_dir)}')
" 2>&1
