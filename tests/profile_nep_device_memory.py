#!/usr/bin/env python3
"""Sample process GPU memory, including native cudaMalloc buffers, outside timings."""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import threading

import torch

from benchmark_nep_fused_fitting import cpu_batch, full_trial, make_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--backend', choices=('original', 'fused'), required=True)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--pool-size', type=int, default=8192)
    parser.add_argument('--seed', type=int, default=2023)
    args = parser.parse_args()
    if args.batch_size < 1 or args.pool_size < args.batch_size:
        parser.error('require 1 <= batch-size <= pool-size')
    if not torch.cuda.is_available():
        parser.error('requires a Slurm CUDA allocation')
    model, params, _ = make_model(args.case, torch.device('cpu'))
    _, size = cpu_batch(params, [0])
    pool = random.Random(args.seed).sample(range(size), min(size, args.pool_size))
    batch, _ = cpu_batch(params, pool[:args.batch_size])
    samples = []
    monitor = subprocess.Popen(
        ['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
         '--format=csv,noheader,nounits', '--loop-ms=20'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def read_samples():
        for line in monitor.stdout:
            fields = [value.strip() for value in line.split(',')]
            if len(fields) == 2 and fields[0] == str(os.getpid()) and fields[1].isdigit():
                samples.append(int(fields[1]))

    reader = threading.Thread(target=read_samples, daemon=True)
    reader.start()
    try:
        measurement = full_trial(model, batch, args.backend == 'fused', 2, 3)
    finally:
        monitor.terminate()
        monitor.wait(timeout=10)
        reader.join(timeout=10)
    if not samples:
        raise RuntimeError('No per-process GPU memory samples: ' + monitor.stderr.read())
    result = {
        'backend': args.backend, 'structures': min(args.batch_size, len(pool)),
        'atoms': int(batch['atom_type_map'].numel()), 'seed': args.seed,
        'sampled_process_peak_mib': max(samples), 'sample_interval_ms': 20,
        'sample_count': len(samples), 'torch_peak_allocated': measurement['peak_allocated'],
        'torch_peak_reserved': measurement['peak_reserved'],
        'scope': 'fresh process; includes CUDA context/native allocations; sampling may miss brief peaks; not a speed benchmark',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
