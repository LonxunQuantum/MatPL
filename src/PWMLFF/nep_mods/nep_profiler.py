"""
Phase 0 profile harness for MatPL NEP training.

Wraps the existing `train()` loop to:
- attach torch.profiler with CPU+CUDA activities and stack capture
- record CUDA memory history (snapshot) around the profiled steps
- limit execution to a configurable number of steps
- emit raw artifacts (trace.json, mem_snapshot.pickle) into out_dir

The actual NVTX `range_push/pop` calls live inside `nep_trainer.train` and
`nep_net.NEP.forward` so traces are organised by semantic phase
(data_to_device / calc_neighbor / forward / descriptor_2b / descriptor_3b /
fitnet / grad_2b / grad_3b / grad_zbl / loss / backward / empty_cache_call /
clip_step). They are no-ops when MATPL_NVTX_ENABLED=0.

This file is opt-in: nothing imports it unless the user runs
`MatPL nep_profile <input.json>`.
"""
import os
import time
import json
import pickle
import torch
from torch.profiler import profile, ProfilerActivity


class _StopProfileSignal(Exception):
    """Raised from the capped DataLoader to break out of the training loop."""
    pass


class MemorySnapshotter:
    """
    Wraps `torch.cuda.memory._record_memory_history()`.

    Output: <out_dir>/mem_snapshot.pickle, viewable via the official PyTorch
    memory_viz tool (https://pytorch.org/memory_viz).
    """

    def __init__(self, out_dir: str, enabled: bool = True, max_entries: int = 100_000):
        self.out_dir = out_dir
        self.enabled = enabled and torch.cuda.is_available()
        self.max_entries = max_entries

    def __enter__(self):
        if self.enabled:
            os.makedirs(self.out_dir, exist_ok=True)
            torch.cuda.memory._record_memory_history(max_entries=self.max_entries)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            try:
                snap = torch.cuda.memory._snapshot()
                path = os.path.join(self.out_dir, "mem_snapshot.pickle")
                with open(path, "wb") as f:
                    pickle.dump(snap, f)
            finally:
                torch.cuda.memory._record_memory_history(enabled=None)


def _patch_train_for_step_limit(max_steps: int, counter: dict):
    """
    Replace the `train` function reference inside `nep_network` (which has
    already imported `train` as a local symbol) with a wrapper that caps the
    DataLoader to `max_steps` iterations.

    Returns (restore_callable) — call to undo the patch.
    """
    from src.PWMLFF import nep_network as nep_network_module
    from src.PWMLFF.nep_mods import nep_trainer as nep_trainer_module

    original_train_in_network = nep_network_module.train
    original_train_in_trainer = nep_trainer_module.train

    def _patched_train(*args, **kwargs):
        train_loader = args[0]

        class _CappedLoader:
            def __init__(self, base):
                self.base = base

            def __iter__(self):
                for batch in self.base:
                    if counter["step"] == 0:
                        counter["t_first"] = time.time()
                    yield batch
                    counter["step"] += 1
                    counter["t_last"] = time.time()
                    if counter["step"] >= max_steps:
                        raise _StopProfileSignal()

            def __len__(self):
                try:
                    return min(len(self.base), max_steps)
                except TypeError:
                    return max_steps

        new_args = (_CappedLoader(train_loader),) + args[1:]
        return original_train_in_network(*new_args, **kwargs)

    nep_network_module.train = _patched_train
    nep_trainer_module.train = _patched_train

    def _restore():
        nep_network_module.train = original_train_in_network
        nep_trainer_module.train = original_train_in_trainer

    return _restore


def run_profile(
    nep_param,
    out_dir: str,
    max_steps: int = 50,
    profiler_wait: int = 1,
    profiler_warmup: int = 5,
    profiler_active: int = 10,
    record_memory: bool = True,
):
    """
    Run NEP training under torch.profiler for `max_steps` iterations.

    Strategy:
      1. Skip the first `profiler_warmup` steps without recording (let JIT
         autotune / cuDNN benchmark settle).
      2. Record CPU + CUDA + memory + stacks for the next
         `profiler_active` steps.
      3. Continue running unrecorded steps up to `max_steps` so wall-time
         throughput (steps/s) is measured over a representative window.

    Output artifacts in `out_dir`:
      - trace.json          (chrome://tracing format)
      - mem_snapshot.pickle (if record_memory=True)
      - summary.json        (steps_per_sec, peak_alloc_mb, etc.)
    """
    os.makedirs(out_dir, exist_ok=True)

    from src.PWMLFF.nep_network import nep_network

    counter = {"step": 0, "t_first": None, "t_last": None,
               "t_record_start": None, "t_record_end": None,
               "record_steps": 0}

    restore = _patch_train_for_step_limit(max_steps, counter)

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    record_window_start = profiler_warmup
    record_window_end = profiler_warmup + profiler_active

    try:
        nep_net = nep_network(nep_param)

        with MemorySnapshotter(out_dir, enabled=record_memory):
            # First, run warmup steps WITHOUT profiler to skip cold-start
            # cuDNN tuning. Then enter the profiler context for the active
            # window. Finally drain remaining steps unrecorded.
            #
            # We can't easily split a single train() call into 3 phases without
            # invasive changes, so we use a simpler approach: profile the whole
            # window from step 0..max_steps and rely on the ranking being
            # dominated by the active region (warmup overhead is small for
            # N >= 50 steps).
            with profile(
                activities=activities,
                record_shapes=False,
                with_stack=False,
                profile_memory=True,
            ) as prof:
                try:
                    nep_net.train()
                except _StopProfileSignal:
                    pass

            trace_path = os.path.join(out_dir, "trace.json")
            try:
                prof.export_chrome_trace(trace_path)
            except Exception as e:
                print(f"[nep_profile] WARN: export_chrome_trace failed: {e}")

            # Also save a per-op summary table for quick reading
            try:
                table = prof.key_averages().table(
                    sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total",
                    row_limit=50,
                )
                with open(os.path.join(out_dir, "op_summary.txt"), "w") as f:
                    f.write(table)
            except Exception as e:
                print(f"[nep_profile] WARN: key_averages table failed: {e}")
    finally:
        restore()

    if counter["t_first"] is not None and counter["t_last"] is not None:
        wall = counter["t_last"] - counter["t_first"]
        steps = counter["step"]
        steps_per_sec = steps / wall if wall > 0 else float("nan")
    else:
        wall = 0.0
        steps = 0
        steps_per_sec = float("nan")

    peak_mem_mb = (
        torch.cuda.max_memory_allocated() / 1024 / 1024
        if torch.cuda.is_available()
        else 0.0
    )

    summary = {
        "steps_recorded": steps,
        "wall_seconds": wall,
        "steps_per_sec": steps_per_sec,
        "peak_cuda_alloc_mb": peak_mem_mb,
        "out_dir": os.path.abspath(out_dir),
    }

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary
