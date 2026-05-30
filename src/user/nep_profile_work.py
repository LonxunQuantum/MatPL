"""
CLI entry for `MatPL nep_profile <input.json> [--steps N] [--memsnap] [--out DIR]`.

Runs NEP training under torch.profiler for a limited number of steps,
emitting trace.json + memory snapshot + summary.json.
"""
import os
import json
import argparse

def nep_profile(input_json: dict, extra_args: list = None):
    """
    Profile NEP training. Called from main.py when cmd == 'nep_profile'.
    """
    parser = argparse.ArgumentParser(description="Profile NEP training")
    parser.add_argument("--steps", type=int, default=50,
                        help="Number of training steps to profile (default: 50)")
    parser.add_argument("--memsnap", action="store_true", default=True,
                        help="Record CUDA memory snapshot (default: True)")
    parser.add_argument("--no-memsnap", action="store_false", dest="memsnap",
                        help="Disable memory snapshot recording")
    parser.add_argument("--out", type=str, default=None,
                        help="Output directory (default: ./phase0_profile/)")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Profiler warmup steps (default: 5)")
    parser.add_argument("--active", type=int, default=10,
                        help="Profiler active recording steps (default: 10)")

    args = parser.parse_args(extra_args or [])

    out_dir = args.out or os.path.join(os.getcwd(), "phase0_profile")

    from src.user.input_param import InputParam
    nep_param = InputParam(input_json, "TRAIN")

    # Force single-GPU for profiling (avoid DDP complexity)
    nep_param.multi_gpus = False
    nep_param.multi_nodes = False
    nep_param.world_size = 1
    nep_param.rank = 0
    nep_param.local_rank = 0

    from src.PWMLFF.nep_mods.nep_profiler import run_profile

    print(f"[nep_profile] Running {args.steps} steps, output → {out_dir}")
    print(f"[nep_profile] Memory snapshot: {'ON' if args.memsnap else 'OFF'}")

    summary = run_profile(
        nep_param,
        out_dir=out_dir,
        max_steps=args.steps,
        profiler_warmup=args.warmup,
        profiler_active=args.active,
        record_memory=args.memsnap,
    )

    print(f"\n[nep_profile] Done.")
    print(f"  Steps recorded : {summary['steps_recorded']}")
    print(f"  Wall time      : {summary['wall_seconds']:.2f} s")
    print(f"  Steps/sec      : {summary['steps_per_sec']:.2f}")
    print(f"  Peak CUDA alloc: {summary['peak_cuda_alloc_mb']:.0f} MB")
    print(f"  Output dir     : {summary['out_dir']}")
    print(f"\n  View trace: tensorboard --logdir {out_dir}")
    print(f"  View memory: python -m torch.utils.viz._memory_viz trace_plot {out_dir}/mem_snapshot.pickle -o mem.html")
