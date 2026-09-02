# NEP OMat24 LMDB Training Design

## Context

MatPL's existing NEP `UniDataset` eagerly asks `pwdata.Config` to materialize every
frame as a Python image object.  It then walks the full image list to derive atom
counts and energy shifts.  In distributed training every rank repeats that state,
so OMat24-scale inputs are not practical even though the optimizer only consumes
one batch at a time.

OMat24 `.aselmdb` files use the ASE LMDB layout: integer frame keys start at `1`,
the `nextid` and optional `deleted_ids` records describe the logical index domain,
and every frame is a zlib-compressed JSON object.  This is not the msgpack schema
used by DPA4C, so MatPL needs an ASE-LMDB decoder while adopting DPA4C's lazy read,
bounded prefetch, deterministic shuffle, and rank-level batch partitioning.

## Goals

- Select the large-data path only when the top-level JSON contains
  `"format": "lmdb"`.
- Accept one `.aselmdb` file, a list of files, a directory, or a mixture; directory
  inputs are searched recursively for all `.aselmdb` files.
- Keep structure data out of resident memory and decode only frames selected for
  the current rank.
- Preserve deterministic epoch shuffle and equal optimizer-step counts across
  single-node and multi-node DDP.
- Support integer frame-count batches and DPA4C-style `"mix:N"` atom-budget
  batches.
- Estimate NEP initialization statistics from a bounded deterministic global
  sample distributed across ranks.
- Leave the existing `UniDataset` and all non-LMDB formats unchanged.

## Non-goals

- Reading DPA4C's msgpack LMDB schema.
- Changing NEP model, loss, or optimizer semantics.
- Converting OMat24 into another on-disk dataset format.
- Installing or upgrading LMDB; the existing `lmdb==1.5.1` is sufficient.

## User Interface

The LMDB path is selected by the existing top-level field:

```json
{
  "format": "lmdb",
  "train_data": [
    "/data/public/wuxingxing/metadata/decompress/Omat24/valid"
  ],
  "lmdb_stat_frames": 32768,
  "optimizer": {
    "batch_size": 32
  }
}
```

`train_data`, `valid_data`, and `test_data` use the same discovery rules.  Inputs
are expanded to resolved absolute file paths, sorted, and deduplicated.  A path
that does not exist, a non-`.aselmdb` file, or a directory with no `.aselmdb`
descendants fails before DDP workers are launched.

`optimizer.batch_size` has two LMDB modes:

- positive integer: number of frames per rank per optimizer step;
- string `"mix:N"`: positive per-rank atom budget.

String batch sizes remain invalid for non-LMDB formats.

`lmdb_stat_frames` is the desired global statistics sample count and defaults to
32768.  The actual sample size is:

```text
min(dataset_frames, lmdb_stat_frames, 32768 * world_size)
```

Thus no rank handles more than `4096 * 8 == 32768` statistics frames.

## Components

### Path discovery and configuration

`WorkFileStructure` retains the original path behavior for every format except
`lmdb`.  LMDB paths are resolved through a focused discovery helper.  `InputParam`
parses `lmdb_stat_frames`, validates it as a positive integer, and serializes it
into checkpoint JSON.

### Lazy ASE-LMDB dataset

`src/pre_data/nep_lmdb_dataset.py` owns the new path.  On construction it opens
each shard only long enough to read `nextid` and optional `deleted_ids`, then
closes it.  Resident indexing state consists of shard paths, counts, cumulative
offsets, and deleted IDs; it does not construct `range(nextid)` as a list.

`__getitem__` maps a global logical index to `(shard, ASE row id)` by binary search.
Each DataLoader process lazily opens read-only environments with:

```python
lmdb.open(
    path,
    subdir=False,
    readonly=True,
    lock=False,
    readahead=False,
    meminit=False,
)
```

An eight-entry per-process LRU bounds open environments.  Dataset pickling and
process forking never carry live LMDB handles.  Environments are closed on LRU
eviction and explicit dataset cleanup.

Frame values are decompressed with `zlib` and decoded with `orjson` when available,
falling back to the standard `json` module.  No pickle deserialization is used.

### Frame conversion

The decoder emits the same mapping consumed by the existing
`variable_length_collate_fn`:

- `numbers` maps to configured NEP type indices;
- `positions`, `cell`, `energy`, and `forces` become typed tensors;
- six-component ASE stress `[xx, yy, zz, yz, xz, xy]` becomes a symmetric 3x3
  virial using `virial = -stress * abs(det(cell))`;
- absent virial uses the existing `-1e6` mask;
- absent atomic energy uses a finite placeholder when `train_ei` is false, while
  `train_ei=true` fails with a label error;
- charge, fragment, and BEC fields use the same defaults as `UniDataset`.

Required-label failures, corrupt JSON/zlib records, non-periodic cells, invalid
shapes, and atom types missing from `atom_type` include the shard path and row key.

### Bounded-memory distributed sampler

The integer sampler generates a deterministic global permutation without a full
`torch.randperm`.  It shuffles block IDs and then shuffles indices inside one
bounded block.  Seed and epoch determine the order.

For a per-rank frame batch `B`, the sampler consumes `B * world_size` global
indices, gives each rank its disjoint `B`-frame slice, and drops the final
incomplete global super-batch.  Every rank therefore has the same number of
optimizer steps with no duplicated training frame.  Rank data is neither a
contiguous source interval nor shared with another rank.

Validation uses ascending global indices and the same global-super-batch split.

### Atom-budget batches

ASE LMDB has no per-frame atom-count metadata.  Strict `mix:N` packing therefore
uses a compact `int32` atom-count sidecar.  On first use, ranks divide LMDB shards,
scan their assigned shards sequentially with readahead enabled, and atomically
write one cache file per shard under `<json_dir>/.matpl_lmdb_cache`.  A fingerprint
of resolved path, size, modification time, `nextid`, and deleted IDs invalidates
stale cache entries.  A distributed barrier makes the completed cache visible
before sampling.  Later runs memory-map the cache.

The mix sampler walks the deterministic global index stream, closes a batch before
adding a frame that would exceed `N`, and lets a frame larger than `N` form a
one-frame batch.  Completed batches are buffered in groups of `world_size`; rank
`r` receives item `r` and a final incomplete rank group is dropped.  Sampler memory
is bounded by one shuffle block and one group of rank batches.

### Global sampled statistics

All ranks independently construct the same deterministic global sample and take
`sample_indices[rank::world_size]`.  Local indices are sorted before reading to
improve storage locality.

Each rank accumulates:

- atom-count sum, maximum, and frame count;
- per-frame composition normal equations for the NEP energy shift;
- descriptor component minima and maxima;
- maximum radial and angular neighbor counts.

Distributed reductions produce identical energy shifts, average/max atom counts,
q-scalers, and neighbor capacities on every rank.  The full dataset is never
scanned for these initialization statistics.

### Integration

`nep_network.load_data()` branches on `format == "lmdb"`.  It constructs the new
datasets, samplers, DataLoaders, and sampled-statistics loader while preserving the
existing return contract.  Epoch updates target `loader.batch_sampler.set_epoch`
for LMDB and retain `DistributedSampler.set_epoch` for legacy datasets.

DataLoader prefetch remains bounded.  `persistent_workers` and `prefetch_factor`
are enabled only when `workers > 0`, so `workers: 0` is also valid.

## Testing and Acceptance

Tests use standard-library `unittest`, because `matpl-2026.3` does not include
pytest.

- Temporary ASE-LMDB fixtures verify discovery, deleted rows, lazy handles,
  conversion, stress sign/order, label errors, and corrupt rows.
- Pure sampler tests verify reproducibility, epoch variation, complete coverage,
  disjoint ranks, equal step counts, bounded blocks, and mix budgets.
- Statistics tests verify global sample caps and simulated multi-rank aggregation.
- A real-data read-only test verifies 11 shards, 1,077,382 frames, random reads,
  and non-linear RSS behavior.
- A small smoke LMDB derived by copying trusted compressed rows is used for one
  short single-GPU run and one single-node four-GPU Slurm run on `q4` or `3090`.
- Integer and `mix:N` paths must both reach training, keep loss/gradients finite,
  and write a checkpoint.

No source or test result is pushed to a GitHub remote.
