#!/usr/bin/env python3
"""Compute OMat24 mix:N batch-size distributions using MatPL's epoch-0 order."""

import argparse
import ctypes
import json
import math
import os
import random
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import lmdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MIX_VALUES = np.asarray(
    [1024, 2048, 4096, 6144, 8192, 10240, 12288, 16384, 20480, 40960],
    dtype=np.int32,
)
BLOCK_SIZE = 65536
SEED = 2023
EPOCH = 0


C_SOURCE = r"""
#include <stdint.h>
#include <stddef.h>
#include <omp.h>

void pack_all(
    const int32_t *natoms, int64_t frame_count,
    const int32_t *budgets, int budget_count,
    int64_t histogram_width, int64_t *histograms,
    int64_t *oversized_frames)
{
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < budget_count; ++b) {
        const int64_t budget = budgets[b];
        int64_t *hist = histograms + (int64_t)b * histogram_width;
        int64_t batch_frames = 0;
        int64_t batch_atoms = 0;
        int64_t oversized = 0;
        for (int64_t i = 0; i < frame_count; ++i) {
            const int64_t frame_atoms = natoms[i];
            if (batch_frames > 0 && batch_atoms + frame_atoms > budget) {
                hist[batch_frames] += 1;
                batch_frames = 0;
                batch_atoms = 0;
            }
            if (frame_atoms > budget) {
                hist[1] += 1;
                oversized += 1;
            } else {
                batch_frames += 1;
                batch_atoms += frame_atoms;
            }
        }
        if (batch_frames > 0) hist[batch_frames] += 1;
        oversized_frames[b] = oversized;
    }
}
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="/data/public/wuxingxing/metadata/decompress/Omat24/train",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent),
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epoch", type=int, default=EPOCH)
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    return parser.parse_args()


def decode_json(raw):
    import zlib
    return json.loads(zlib.decompress(raw).decode("utf-8"))


def read_shard_metadata(path):
    env = lmdb.open(
        str(path), subdir=False, readonly=True, lock=False,
        readahead=False, meminit=False,
    )
    try:
        with env.begin(buffers=False) as txn:
            nextid_raw = txn.get(b"nextid")
            if nextid_raw is None:
                raise ValueError("missing nextid")
            nextid = decode_json(nextid_raw)
            deleted_raw = txn.get(b"deleted_ids")
            deleted = [] if deleted_raw is None else decode_json(deleted_raw)
    finally:
        env.close()
    deleted = sorted(set(int(value) for value in deleted))
    return int(nextid), deleted


def valid_shards(source):
    shards = []
    skipped = []
    for path in sorted(source.rglob("*.aselmdb")):
        try:
            nextid, deleted = read_shard_metadata(path)
            logical_length = nextid - 1 - len(deleted)
            if logical_length < 1:
                raise ValueError("empty shard")
        except Exception as exc:
            skipped.append({"path": str(path), "reason": repr(exc)})
            continue
        shards.append((path, nextid, deleted, logical_length))
    if not shards:
        raise RuntimeError("no valid ASE-LMDB shards found")
    return shards, skipped


def metadata_path_for(shard):
    return shard.with_name("metadata_" + shard.stem + ".npz")


def write_canonical_natoms(shards, destination):
    total_frames = sum(item[3] for item in shards)
    result = np.memmap(destination, dtype=np.int32, mode="w+", shape=(total_frames,))
    cursor = 0
    missing = []
    for shard_index, (shard, nextid, deleted, logical_length) in enumerate(shards):
        metadata_path = metadata_path_for(shard)
        if not metadata_path.is_file():
            missing.append(str(metadata_path))
            continue
        with np.load(metadata_path, allow_pickle=False) as archive:
            if "natoms" not in archive.files:
                raise ValueError(f"{metadata_path}: missing natoms")
            values = np.asarray(archive["natoms"])
        if values.ndim != 1:
            raise ValueError(f"{metadata_path}: natoms is not one-dimensional")
        if values.size == nextid - 1:
            if deleted:
                keep = np.ones(values.size, dtype=bool)
                keep[np.asarray(deleted, dtype=np.int64) - 1] = False
                values = values[keep]
        elif values.size != logical_length:
            raise ValueError(
                f"{metadata_path}: natoms length {values.size} does not match "
                f"physical {nextid - 1} or logical {logical_length} length"
            )
        if values.size != logical_length:
            raise ValueError(f"{metadata_path}: deleted-frame filtering failed")
        if not np.issubdtype(values.dtype, np.integer) or np.any(values < 1):
            raise ValueError(f"{metadata_path}: natoms contains invalid values")
        result[cursor:cursor + logical_length] = values.astype(np.int32, copy=False)
        cursor += logical_length
        if (shard_index + 1) % 250 == 0:
            print(
                f"metadata: {shard_index + 1}/{len(shards)} shards, "
                f"{cursor:,}/{total_frames:,} frames",
                flush=True,
            )
    if missing:
        raise FileNotFoundError(
            f"missing {len(missing)} metadata files; first entries: {missing[:10]}"
        )
    if cursor != total_frames:
        raise RuntimeError(f"wrote {cursor} of {total_frames} frames")
    result.flush()
    return result


def write_training_order(canonical, destination, seed, epoch, block_size):
    size = canonical.size
    shuffled = np.memmap(destination, dtype=np.int32, mode="w+", shape=(size,))
    rng = random.Random(seed + epoch)
    block_count = (size + block_size - 1) // block_size
    block_ids = list(range(block_count))
    rng.shuffle(block_ids)
    cursor = 0
    for sequence, block_id in enumerate(block_ids):
        start = block_id * block_size
        stop = min(start + block_size, size)
        indices = list(range(start, stop))
        rng.shuffle(indices)
        count = stop - start
        shuffled[cursor:cursor + count] = canonical[np.asarray(indices, dtype=np.int64)]
        cursor += count
        if (sequence + 1) % 250 == 0 or sequence + 1 == block_count:
            print(
                f"shuffle: {sequence + 1}/{block_count} blocks, "
                f"{cursor:,}/{size:,} frames",
                flush=True,
            )
    shuffled.flush()
    return shuffled


def compile_packer(temp_dir):
    source = temp_dir / "pack_mix.c"
    library = temp_dir / "libpack_mix.so"
    source.write_text(C_SOURCE, encoding="utf-8")
    subprocess.run(
        ["gcc", "-O3", "-fopenmp", "-shared", "-fPIC", str(source), "-o", str(library)],
        check=True,
    )
    loaded = ctypes.CDLL(str(library))
    loaded.pack_all.argtypes = [
        ctypes.POINTER(ctypes.c_int32), ctypes.c_int64,
        ctypes.POINTER(ctypes.c_int32), ctypes.c_int,
        ctypes.c_int64, ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
    ]
    loaded.pack_all.restype = None
    return loaded


def validate_packer(packer):
    natoms = np.asarray([2, 3, 9, 1, 4, 6, 2], dtype=np.int32)
    budgets = np.asarray([5, 8, 12], dtype=np.int32)
    width = 14
    actual = np.zeros((budgets.size, width), dtype=np.int64)
    oversized = np.zeros(budgets.size, dtype=np.int64)
    packer.pack_all(
        natoms.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int64(natoms.size),
        budgets.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int(budgets.size),
        ctypes.c_int64(width), actual.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        oversized.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    expected = np.zeros_like(actual)
    expected_oversized = np.zeros_like(oversized)
    for row, budget in enumerate(budgets.tolist()):
        frames = atoms = 0
        for frame_atoms in natoms.tolist():
            if frames and atoms + frame_atoms > budget:
                expected[row, frames] += 1
                frames = atoms = 0
            if frame_atoms > budget:
                expected[row, 1] += 1
                expected_oversized[row] += 1
            else:
                frames += 1
                atoms += frame_atoms
        if frames:
            expected[row, frames] += 1
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(oversized, expected_oversized)


def weighted_quantile(sizes, counts, fraction):
    target = int(math.ceil(int(counts.sum()) * fraction))
    return int(sizes[np.searchsorted(np.cumsum(counts), target, side="left")])


def save_outputs(output, natoms, histograms, oversized, source_info):
    output.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    plot_payload = []
    for row, budget in enumerate(MIX_VALUES.tolist()):
        counts_all = histograms[row]
        sizes = np.flatnonzero(counts_all)
        counts = counts_all[sizes]
        total_batches = int(counts.sum())
        frequencies = counts / total_batches
        csv_data = np.column_stack((sizes, counts, frequencies, frequencies * 100.0))
        csv_path = output / f"batchsize_frequency_mix_{budget}.csv"
        np.savetxt(
            csv_path,
            csv_data,
            delimiter=",",
            header="batch_size_frames,count,frequency,frequency_percent",
            comments="",
            fmt=["%d", "%d", "%.12g", "%.9g"],
        )
        mean_batch = float(np.dot(sizes.astype(np.float64), counts) / total_batches)
        mode_index = int(np.argmax(counts))
        stats = {
            "mix": budget,
            "total_batches": total_batches,
            "min_batch_size": int(sizes[0]),
            "max_batch_size": int(sizes[-1]),
            "mean_batch_size": mean_batch,
            "median_batch_size": weighted_quantile(sizes, counts, 0.5),
            "p05_batch_size": weighted_quantile(sizes, counts, 0.05),
            "p95_batch_size": weighted_quantile(sizes, counts, 0.95),
            "mode_batch_size": int(sizes[mode_index]),
            "mode_frequency_percent": float(frequencies[mode_index] * 100.0),
            "singleton_frequency_percent": float(counts_all[1] / total_batches * 100.0),
            "oversized_frames": int(oversized[row]),
        }
        summary_rows.append(stats)
        plot_payload.append((budget, sizes, frequencies, stats))

        fig, ax = plt.subplots(figsize=(9.2, 5.4), constrained_layout=True)
        ax.bar(sizes, frequencies * 100.0, width=0.86, color="#2563EB", edgecolor="none")
        ax.set_xlabel("Batch size (frames)")
        ax.set_ylabel("Frequency (%)")
        ax.set_title(f"OMat24 train batch-size distribution — mix:{budget}")
        ax.grid(axis="y", alpha=0.25, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.text(
            0.985, 0.965,
            f"batches: {total_batches:,}\nmean: {mean_batch:.2f}\n"
            f"median: {stats['median_batch_size']}\np95: {stats['p95_batch_size']}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.88, "edgecolor": "#CBD5E1"},
        )
        fig.savefig(output / f"batchsize_distribution_mix_{budget}.png", dpi=200)
        plt.close(fig)

    summary_columns = list(summary_rows[0])
    with (output / "batchsize_distribution_summary.csv").open("w", encoding="utf-8") as stream:
        stream.write(",".join(summary_columns) + "\n")
        for row in summary_rows:
            stream.write(",".join(str(row[key]) for key in summary_columns) + "\n")

    result = dict(source_info)
    result["mix_values"] = MIX_VALUES.tolist()
    result["distributions"] = summary_rows
    (output / "batchsize_distribution_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(2, 5, figsize=(22, 8.5), constrained_layout=True)
    for ax, (budget, sizes, frequencies, stats) in zip(axes.flat, plot_payload):
        ax.bar(sizes, frequencies * 100.0, width=0.86, color="#2563EB", edgecolor="none")
        ax.set_title(f"mix:{budget}  (mean={stats['mean_batch_size']:.1f})")
        ax.set_xlabel("Batch size (frames)")
        ax.set_ylabel("Frequency (%)")
        ax.grid(axis="y", alpha=0.22, linewidth=0.6)
        ax.set_axisbelow(True)
    fig.suptitle("OMat24 train batch-size distributions", fontsize=17)
    fig.savefig(output / "batchsize_distributions_all_mix.png", dpi=180)
    plt.close(fig)

    readme = f"""# OMat24 mix batch-size distribution

- Source: `{source_info['source']}`
- Valid ASE-LMDB shards: {source_info['valid_shards']}
- Skipped invalid shards: {source_info['skipped_shards']}
- Frames: {source_info['frame_count']:,}
- Atoms: {source_info['atom_count']:,}
- Average atoms/frame: {source_info['average_atoms_per_frame']:.6f}
- Minimum/maximum atoms/frame: {source_info['min_atoms_per_frame']} / {source_info['max_atoms_per_frame']}
- Ordering: MatPL `BlockShuffleIndices`, seed={source_info['seed']}, epoch={source_info['epoch']}, block_size={source_info['block_size']}
- Packing: identical greedy rule to `DistributedAtomBatchSampler._global_batches`
- Frequency: `count / total batches` for each mix value

Each `batchsize_frequency_mix_N.csv` stores the plotting data. Each mix value has an independent PNG, and `batchsize_distributions_all_mix.png` is the 2×5 overview.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")


def main():
    args = parse_args()
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    temp_root = Path(os.environ.get("SLURM_TMPDIR", tempfile.gettempdir()))
    work = Path(tempfile.mkdtemp(prefix="omat-mix-", dir=temp_root))
    started = time.time()
    try:
        print(f"source={source}", flush=True)
        print(f"temporary_work={work}", flush=True)
        packer = compile_packer(work)
        validate_packer(packer)
        print("C/OpenMP packer self-test passed", flush=True)
        shards, skipped = valid_shards(source)
        print(f"valid_shards={len(shards)} skipped_shards={len(skipped)}", flush=True)
        canonical = write_canonical_natoms(shards, work / "natoms-canonical.i32")
        frame_count = int(canonical.size)
        atom_count = int(np.sum(canonical, dtype=np.int64))
        source_info = {
            "source": str(source),
            "valid_shards": len(shards),
            "skipped_shards": len(skipped),
            "skipped_details": skipped,
            "frame_count": frame_count,
            "atom_count": atom_count,
            "average_atoms_per_frame": atom_count / frame_count,
            "min_atoms_per_frame": int(canonical.min()),
            "max_atoms_per_frame": int(canonical.max()),
            "seed": args.seed,
            "epoch": args.epoch,
            "block_size": args.block_size,
        }
        print(json.dumps(source_info, indent=2), flush=True)
        ordered = write_training_order(
            canonical, work / "natoms-epoch-order.i32",
            args.seed, args.epoch, args.block_size,
        )
        width = int(MIX_VALUES.max()) + 2
        histograms = np.zeros((MIX_VALUES.size, width), dtype=np.int64)
        oversized = np.zeros(MIX_VALUES.size, dtype=np.int64)
        print("packing all mix values with OpenMP", flush=True)
        packer.pack_all(
            ordered.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int64(frame_count),
            MIX_VALUES.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int(MIX_VALUES.size),
            ctypes.c_int64(width),
            histograms.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            oversized.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        )
        save_outputs(output, canonical, histograms, oversized, source_info)
        print(f"completed_seconds={time.time() - started:.3f}", flush=True)
        print(f"output={output}", flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
