"""
Parse the profiler artifacts emitted by `MatPL nep_profile` and render a
human-readable Phase 0 baseline report.

Inputs (all under <out_dir>):
- summary.json — wall_seconds, steps_per_sec, peak_cuda_alloc_mb
- *.pt.trace.json — torch.profiler chrome trace (tensorboard format)
- mem_snapshot.pickle — CUDA memory snapshot (optional)

Output:
- phase0_report.md — markdown report with steps/s, time breakdown by NVTX
  range (forward / descriptor_2b / descriptor_3b / fitnet / grad_2b /
  grad_3b / grad_zbl / loss / backward / empty_cache_call / clip_step),
  peak memory, and a ranked bottleneck table.
"""
import os
import json
import gzip
import glob
from collections import defaultdict


PHASE_NAMES = [
    "data_to_device",
    "calc_neighbor",
    "forward",
    "descriptor_2b",
    "descriptor_3b",
    "fitnet_per_type_loop",
    "zbl",
    "grad_2b",
    "grad_3b",
    "grad_zbl",
    "force_virial_gpu",
    "loss",
    "backward",
    "empty_cache_call",
    "clip_step",
]


def _load_trace(trace_path: str) -> dict:
    if trace_path.endswith(".gz"):
        opener = gzip.open
    else:
        opener = open
    with opener(trace_path, "rt") as f:
        return json.load(f)


def _accumulate_durations(trace: dict) -> dict:
    """
    Sum duration (us) per event name, restricted to user-annotated NVTX ranges
    (which torch.profiler records as 'cat': 'user_annotation' or with name
    matching our PHASE_NAMES).
    """
    totals = defaultdict(lambda: {"us": 0.0, "count": 0})
    events = trace.get("traceEvents", [])
    for ev in events:
        name = ev.get("name", "")
        if name not in PHASE_NAMES and not name.startswith("step_"):
            continue
        dur = ev.get("dur")
        if dur is None:
            continue
        totals[name]["us"] += dur
        totals[name]["count"] += 1
    return totals


def render_report(out_dir: str) -> str:
    """
    Read <out_dir>/{summary.json, *.trace.json} and write
    <out_dir>/phase0_report.md. Returns the report path.
    """
    summary_path = os.path.join(out_dir, "summary.json")
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)

    trace_files = sorted(
        glob.glob(os.path.join(out_dir, "trace.json"))
        + glob.glob(os.path.join(out_dir, "*.pt.trace.json"))
        + glob.glob(os.path.join(out_dir, "*.pt.trace.json.gz"))
    )

    breakdown = defaultdict(lambda: {"us": 0.0, "count": 0})
    for tf in trace_files:
        try:
            trace = _load_trace(tf)
        except Exception as e:
            print(f"[profile_report] Skip {tf}: {e}")
            continue
        for name, d in _accumulate_durations(trace).items():
            breakdown[name]["us"] += d["us"]
            breakdown[name]["count"] += d["count"]

    step_total_us = sum(
        v["us"] for k, v in breakdown.items() if k.startswith("step_")
    )

    lines = []
    lines.append("# MatPL NEP Phase 0 Baseline Report\n")
    lines.append("## Run summary\n")
    if summary:
        lines.append(f"- **Steps recorded**: {summary.get('steps_recorded', 'n/a')}")
        lines.append(f"- **Wall time (s)**: {summary.get('wall_seconds', 0):.2f}")
        lines.append(f"- **Steps / sec**: {summary.get('steps_per_sec', 0):.3f}")
        lines.append(f"- **Peak CUDA alloc (MB)**: {summary.get('peak_cuda_alloc_mb', 0):.0f}")
        lines.append(f"- **Output dir**: `{summary.get('out_dir', out_dir)}`")
    else:
        lines.append("(summary.json not found)")
    lines.append("")

    lines.append(f"## Profiler trace files\n")
    if trace_files:
        for tf in trace_files:
            lines.append(f"- `{os.path.basename(tf)}`")
    else:
        lines.append("(no trace.json found — re-run with `MatPL nep_profile <input.json>`)")
    lines.append("")

    lines.append("## Per-phase time breakdown\n")
    lines.append("Sums of NVTX range durations across the recorded active window.")
    lines.append("Columns: phase, total ms, count, ms / call, % of all phases.\n")

    phase_rows = [(k, v["us"] / 1000.0, v["count"]) for k, v in breakdown.items()
                  if k in PHASE_NAMES]
    grand_total_ms = sum(r[1] for r in phase_rows) or 1.0
    phase_rows.sort(key=lambda r: -r[1])

    lines.append("| phase | total (ms) | count | ms/call | % |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, total_ms, count in phase_rows:
        per_call = total_ms / count if count else 0.0
        pct = 100.0 * total_ms / grand_total_ms
        lines.append(f"| `{name}` | {total_ms:.1f} | {count} | {per_call:.2f} | {pct:.1f}% |")
    lines.append("")

    if step_total_us > 0:
        lines.append(f"_(Sum of `step_i` ranges: {step_total_us/1000:.1f} ms; "
                     "use as denominator for per-step % if you prefer.)_\n")

    lines.append("## Memory snapshot\n")
    mem_path = os.path.join(out_dir, "mem_snapshot.pickle")
    if os.path.exists(mem_path):
        size_mb = os.path.getsize(mem_path) / 1024 / 1024
        lines.append(f"- File: `mem_snapshot.pickle` ({size_mb:.1f} MB)")
        lines.append("- Visualize with:")
        lines.append("  ```")
        lines.append(f"  python -m torch.utils.viz._memory_viz trace_plot {mem_path} -o mem.html")
        lines.append("  ```")
    else:
        lines.append("(no mem_snapshot.pickle — run with `--memsnap`)")
    lines.append("")

    op_summary_path = os.path.join(out_dir, "op_summary.txt")
    if os.path.exists(op_summary_path):
        lines.append("## Top-N CUDA / CPU operators\n")
        lines.append("```")
        with open(op_summary_path) as f:
            content = f.read()
        max_chars = 5000
        if len(content) > max_chars:
            content = content[:max_chars] + "\n... (truncated, see op_summary.txt for full)"
        lines.append(content)
        lines.append("```\n")

    lines.append("## Bottleneck shortlist (highest cost first)\n")
    for name, total_ms, count in phase_rows[:5]:
        pct = 100.0 * total_ms / grand_total_ms
        lines.append(f"1. **{name}** — {total_ms:.1f} ms total ({pct:.1f}%)")
    lines.append("")

    lines.append("## Notes for Phase 1 planning\n")
    lines.append("- `empty_cache_call` cost ≈ wasted time per step; check root cause "
                 "is the 3× `autograd.grad(retain_graph=True)` pattern (Phase 1.4 will merge them).")
    lines.append("- If `descriptor_3b` dominates, Phase 3.2 EdgeFeatureCache pays off.")
    lines.append("- If `fitnet_per_type_loop` is large, Phase 2.1 ntypes batchify is worthwhile.")
    lines.append("- If `grad_*` sum > 30%, Phase 1.4 (merge grads) is the priority.")
    lines.append("")

    report_path = os.path.join(out_dir, "phase0_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    return report_path


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: profile_report.py <out_dir>")
        sys.exit(1)
    path = render_report(sys.argv[1])
    print(f"Wrote {path}")
