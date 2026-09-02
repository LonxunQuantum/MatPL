# NEP OMat24 LMDB Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train MatPL NEP directly from OMat24 ASE-LMDB files with lazy frame reads, bounded-memory shuffle, distributed statistics, and integer or atom-budget batches.

**Architecture:** A dedicated `NepLmdbDataset` path is selected only for `format == "lmdb"`; legacy `UniDataset` remains unchanged.  The dataset decodes one selected zlib-JSON frame at a time, custom samplers build deterministic global rank batches, and bounded global samples provide NEP initialization statistics.

**Tech Stack:** Python 3.11, PyTorch 2.2 distributed/DataLoader APIs, python-lmdb 1.5.1, NumPy, zlib, JSON/orjson, standard-library unittest, Slurm/NCCL.

**Spec:** `docs/superpowers/specs/2026-09-02-nep-omat24-lmdb-design.md`

## Global Constraints

- Activate the large-data path only for top-level `"format": "lmdb"`.
- Do not change `UniDataset` behavior for any existing format.
- Do not install or upgrade LMDB; use the existing `lmdb==1.5.1`.
- `train_data`, `valid_data`, and `test_data` accept recursive directories and `.aselmdb` file lists.
- Default global statistics sample: 32768 frames; hard maximum per rank: 32768 frames.
- Integer `batch_size` means frames per rank; `"mix:N"` means atoms per rank.
- Distributed ranks must have equal optimizer-step counts and disjoint non-padding frames.
- Slurm integration tests may use only `q4` or `3090` partitions.
- Never push changes to a GitHub remote.

---

### Task 1: LMDB path and configuration contract

**Files:**
- Create: `src/pre_data/nep_lmdb_dataset.py`
- Modify: `src/user/work_file_param.py:161-205`
- Modify: `src/user/input_param.py:186-214,258-340`
- Test: `src/test/test_nep_lmdb_config.py`

**Interfaces:**
- Produces: `discover_aselmdb_files(paths: Sequence[str]) -> list[str]`.
- Produces: `InputParam.lmdb_stat_frames: int` with default `32768`.
- Produces: normalized LMDB paths in `WorkFileStructure.*_data_path`.

- [ ] **Step 1: Write failing configuration tests**

Create standard-library tests that build nested temporary directories and assert:

```python
files = discover_aselmdb_files([root, explicit_file, root])
self.assertEqual(files, sorted({str(nested_file.resolve()), str(explicit.resolve())}))
```

Also assert empty directories, wrong file suffixes, and non-existent paths raise
`ValueError`, and `lmdb_stat_frames` rejects zero, booleans, strings, and negative
integers.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
PYTHONPATH=. python -m unittest src.test.test_nep_lmdb_config -v
```

Expected: import failure for `discover_aselmdb_files`.

- [ ] **Step 3: Implement minimal discovery and parsing**

Implement a side-effect-free helper:

```python
def discover_aselmdb_files(paths):
    discovered = set()
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if path.is_dir():
            matches = list(path.rglob("*.aselmdb"))
            if not matches:
                raise ValueError(f"No .aselmdb files found under {path}")
            discovered.update(item.resolve() for item in matches if item.is_file())
        elif path.is_file() and path.suffix == ".aselmdb":
            discovered.add(path)
        else:
            raise ValueError(f"Expected an .aselmdb file or directory: {path}")
    return sorted(str(path) for path in discovered)
```

Branch inside `set_train_valid_file()` only when `self.format == "lmdb"`. Parse
and serialize `lmdb_stat_frames` in `InputParam`.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the unittest command from Step 2. Expected: all tests pass.

- [ ] **Step 5: Commit locally**

```bash
git add -f src/test/test_nep_lmdb_config.py
git add src/pre_data/nep_lmdb_dataset.py src/user/work_file_param.py src/user/input_param.py
git commit -m "feat(nep): add LMDB input configuration"
```

### Task 2: Lazy ASE-LMDB metadata and frame decoder

**Files:**
- Modify: `src/pre_data/nep_lmdb_dataset.py`
- Test: `src/test/test_nep_lmdb_dataset.py`

**Interfaces:**
- Produces: `AseLmdbShard(path: str)` with logical length and row-ID mapping.
- Produces: `NepLmdbDataset(...).__getitem__(index) -> dict[str, torch.Tensor]`.
- Produces: `NepLmdbDataset.close() -> None`.
- Consumes: normalized paths from Task 1.

- [ ] **Step 1: Write a reusable safe LMDB fixture in the test module**

The fixture writes rows as zlib-compressed JSON and metadata with keys `nextid`
and optional `deleted_ids`. Include two periodic structures with different atom
counts and known stress values. Do not use pickle.

- [ ] **Step 2: Write failing metadata/laziness tests**

Assert that construction reads counts without retaining environments or a frame
list, deleted row IDs map correctly, negative/out-of-range indices follow Dataset
semantics, and `len(dataset)` equals the sum of logical shard lengths.

- [ ] **Step 3: Run decoder tests and verify RED**

```bash
PYTHONPATH=. python -m unittest src.test.test_nep_lmdb_dataset -v
```

Expected: missing `AseLmdbShard` and `NepLmdbDataset`.

- [ ] **Step 4: Implement shard metadata and LRU environment ownership**

Keep live environments in an `OrderedDict` capped at eight. Ensure
`__getstate__()` returns state with an empty cache, and `close()` closes all
environments. Map deleted IDs without constructing a full integer ID list.

- [ ] **Step 5: Implement frame conversion**

Decode one row and emit keys matching `variable_length_collate_fn`. Convert ASE
Voigt stress with:

```python
scaled = -np.asarray(stress, dtype=float) * abs(np.linalg.det(cell))
virial = np.array([
    [scaled[0], scaled[5], scaled[4]],
    [scaled[5], scaled[1], scaled[3]],
    [scaled[4], scaled[3], scaled[2]],
])
```

Validate shapes, PBC, required labels, atom types, and finite values. Prefix
exceptions with `<path>: frame key <id>`.

- [ ] **Step 6: Expand tests for tensors and errors, then verify GREEN**

Assert every output tensor shape/dtype, cumulative collate behavior, stress sign
and order, missing virial mask, missing atomic-energy policy, unknown elements,
and corrupt zlib/JSON errors. Run the Step 3 command until green.

- [ ] **Step 7: Commit locally**

```bash
git add -f src/test/test_nep_lmdb_dataset.py
git add src/pre_data/nep_lmdb_dataset.py
git commit -m "feat(nep): lazily decode ASE LMDB frames"
```

### Task 3: Bounded-memory distributed frame sampler

**Files:**
- Modify: `src/pre_data/nep_lmdb_dataset.py`
- Test: `src/test/test_nep_lmdb_sampler.py`

**Interfaces:**
- Produces: `BlockShuffleIndices(size, block_size, seed, epoch, shuffle)` iterable.
- Produces: `DistributedFrameBatchSampler(dataset_size, batch_size, rank, world_size, seed, shuffle)`.
- Produces: `set_epoch(epoch: int)`, `__iter__()`, and exact `__len__()`.

- [ ] **Step 1: Write failing single-rank sampler tests**

Assert a fixed seed/epoch reproduces order, a new epoch changes order, every full
index appears exactly once before the dropped tail, and internal buffers never
exceed the configured block plus one super-batch.

- [ ] **Step 2: Write failing multi-rank sampler tests**

For world sizes 2 and 4, materialize small test outputs and assert equal lengths,
per-rank batch size, pairwise-disjoint frame sets, union coverage of every retained
frame, and ascending validation order.

- [ ] **Step 3: Run tests and verify RED**

```bash
PYTHONPATH=. python -m unittest src.test.test_nep_lmdb_sampler -v
```

Expected: sampler imports fail.

- [ ] **Step 4: Implement block shuffle and global super-batches**

Shuffle the list of block IDs, allocate only one block's indices, shuffle that
block, and stream indices. Group `batch_size * world_size` indices and yield the
slice `[rank * batch_size:(rank + 1) * batch_size]`. Drop a short final group.

- [ ] **Step 5: Run tests and verify GREEN**

Run the Step 3 command. Expected: all sampler tests pass.

- [ ] **Step 6: Commit locally**

```bash
git add -f src/test/test_nep_lmdb_sampler.py
git add src/pre_data/nep_lmdb_dataset.py
git commit -m "feat(nep): add bounded distributed LMDB sampler"
```

### Task 4: Global sampled NEP statistics

**Files:**
- Modify: `src/pre_data/nep_lmdb_dataset.py`
- Modify: `src/PWMLFF/nep_network.py:434-553,774-839`
- Test: `src/test/test_nep_lmdb_stats.py`

**Interfaces:**
- Produces: `select_stat_indices(size, requested, rank, world_size, seed, per_rank_cap=32768) -> list[int]`.
- Produces: `LmdbEnergyStatistics` with normal-equation, atom-count, and merge methods.
- Consumes: `calculate_neighbor_scaler` for local descriptor/neighbor extrema.

- [ ] **Step 1: Write failing sample-selection tests**

Assert deterministic global selection, disjoint rank slices, global default 32768,
dataset-size clipping, requested-size clipping, and the per-rank 32768 hard cap.

- [ ] **Step 2: Write failing aggregate-statistics tests**

Use synthetic frames with known composition vectors and energies. Merge simulated
rank accumulators and compare `energy_shift`, average atoms, and maximum atoms to
direct NumPy calculations.

- [ ] **Step 3: Run tests and verify RED**

```bash
PYTHONPATH=. python -m unittest src.test.test_nep_lmdb_stats -v
```

Expected: selection/statistics APIs are absent.

- [ ] **Step 4: Implement bounded sample selection and CPU accumulators**

Use `random.Random(seed).sample(range(size), count)`, take the rank stride, and
sort local indices before I/O. Accumulate `A.T @ A` and `A.T @ energy` without
retaining frame objects.

- [ ] **Step 5: Integrate distributed reductions**

In the LMDB branch, reduce the normal equations, count/sum/max atoms, descriptor
min/max, and radial/angular neighbor maxima. Solve the energy shift independently
on every rank from the identical reduced normal equations. Handle empty local
sample slices with reduction identity values.

- [ ] **Step 6: Run statistics and existing config tests**

```bash
PYTHONPATH=. python -m unittest \
  src.test.test_nep_lmdb_stats \
  src.test.test_nep_lmdb_config \
  src.test.test_nep_lmdb_dataset -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit locally**

```bash
git add -f src/test/test_nep_lmdb_stats.py
git add src/pre_data/nep_lmdb_dataset.py src/PWMLFF/nep_network.py
git commit -m "feat(nep): distribute sampled LMDB statistics"
```

### Task 5: Atom-count cache and `mix:N` sampler

**Files:**
- Modify: `src/pre_data/nep_lmdb_dataset.py`
- Modify: `src/PWMLFF/nep_network.py:434-553,912-930`
- Test: `src/test/test_nep_lmdb_mix_sampler.py`

**Interfaces:**
- Produces: `LmdbNatomsCache(dataset, cache_dir)` with `build_assigned(rank, world_size)` and random access.
- Produces: `DistributedAtomBatchSampler(natoms, atom_budget, rank, world_size, seed, shuffle)`.
- Consumes: Task 3's block-shuffled index stream.

- [ ] **Step 1: Write failing cache tests**

Build a fixture with known atom counts. Assert first build writes one `int32` file
per shard through an atomic rename, second load reuses it, and file size/mtime or
`nextid` changes invalidate it. Assert cache memory maps rather than loading arrays.

- [ ] **Step 2: Write failing mix sampler tests**

Assert every yielded batch has total atoms `<= N`, except one oversized frame alone;
ranks have equal step counts and disjoint frame sets; epoch changes are deterministic;
and fewer than `world_size` final batches are dropped.

- [ ] **Step 3: Run tests and verify RED**

```bash
PYTHONPATH=. python -m unittest src.test.test_nep_lmdb_mix_sampler -v
```

Expected: cache and mix sampler imports fail.

- [ ] **Step 4: Implement per-shard cache**

Open shards sequentially with `readahead=True`, decode only enough JSON to count
`numbers`, write temporary arrays, `fsync`, and atomically replace final cache files.
Use a JSON manifest carrying the shard fingerprint. In DDP, assign shard index
`i` to rank `i % world_size`, then barrier before memory mapping all shards.

- [ ] **Step 5: Implement online mix packing**

Accumulate frames until the next frame would exceed the budget. Buffer completed
batches in a list of length `world_size`, yield the item for this rank, then clear
the group. Implement and cache exact length for the active epoch using the same
packing rules.

- [ ] **Step 6: Run tests and verify GREEN**

Run the Step 3 command. Expected: all cache and mix tests pass.

- [ ] **Step 7: Commit locally**

```bash
git add -f src/test/test_nep_lmdb_mix_sampler.py
git add src/pre_data/nep_lmdb_dataset.py src/PWMLFF/nep_network.py
git commit -m "feat(nep): support atom-budget LMDB batches"
```

### Task 6: End-to-end NEP loader integration

**Files:**
- Modify: `src/PWMLFF/nep_network.py:434-553,774-930`
- Modify: `src/user/optimizer_param.py:1-35,150-175`
- Test: `src/test/test_nep_lmdb_integration.py`

**Interfaces:**
- Consumes: `NepLmdbDataset`, both distributed batch samplers, sampled statistics, and `variable_length_collate_fn`.
- Produces: the existing `load_data()` tuple and existing trainer batch mapping.

- [ ] **Step 1: Write a failing DataLoader integration test**

Construct a tiny two-shard LMDB and a lightweight `InputParam` stub. Assert the
LMDB branch returns train/validation loaders whose batches contain the exact tensor
keys and ragged atom concatenation expected by `nep_trainer.train`.

- [ ] **Step 2: Run integration test and verify RED**

Run on a GPU node because importing `nep_network` loads CalcOps:

```bash
PYTHONPATH=. python -m unittest src.test.test_nep_lmdb_integration -v
```

Expected: `load_data()` still constructs `UniDataset` for LMDB.

- [ ] **Step 3: Add the isolated `format == "lmdb"` branch**

Construct LMDB datasets and DataLoaders with `batch_sampler`. Set
`persistent_workers` and `prefetch_factor` only for positive worker counts. Keep
the legacy branch textually separate and unchanged.

- [ ] **Step 4: Forward epochs to the correct sampler**

Add a helper that calls `loader.batch_sampler.set_epoch(epoch)` when present and
falls back to the existing `DistributedSampler` logic. Validate integer and
`mix:N` batch-size syntax with contextual errors.

- [ ] **Step 5: Run all pure tests and GPU integration test**

```bash
PYTHONPATH=. python -m unittest \
  src.test.test_nep_lmdb_config \
  src.test.test_nep_lmdb_dataset \
  src.test.test_nep_lmdb_sampler \
  src.test.test_nep_lmdb_stats \
  src.test.test_nep_lmdb_mix_sampler \
  src.test.test_nep_lmdb_integration -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit locally**

```bash
git add -f src/test/test_nep_lmdb_integration.py
git add src/PWMLFF/nep_network.py src/user/optimizer_param.py
git commit -m "feat(nep): train directly from OMat24 LMDB"
```

### Task 7: Real OMat24 and four-GPU acceptance

**Files:**
- Modify outside Git: `/data/home/wuxingxing/datas/training_test/mgpus/omat24/nep.json`
- Modify outside Git: `/data/home/wuxingxing/datas/training_test/mgpus/omat24/train.job`
- Create outside Git: `/data/home/wuxingxing/datas/training_test/mgpus/omat24/smoke.aselmdb`

**Interfaces:**
- Consumes: completed LMDB training path from Tasks 1-6.
- Produces: logs, checkpoint, and reproducible Slurm smoke configuration.

- [ ] **Step 1: Run real-data metadata and memory checks**

Instantiate the directory dataset and assert 11 shards and 1,077,382 frames. Read
fixed and random indices, record RSS before/after construction, and verify no
frame-count-proportional Python list exists.

- [ ] **Step 2: Create a small trusted smoke LMDB**

Copy a bounded set of compressed frame values from the real read-only LMDB into a
new test LMDB and write matching compressed `nextid`. Do not deserialize with
pickle or alter the public dataset.

- [ ] **Step 3: Run one short single-GPU integer-batch job**

Set one epoch, a small statistics count, and integer batch size. Confirm model
initialization, finite first/last loss, and checkpoint creation.

- [ ] **Step 4: Run one short single-node four-GPU integer-batch job**

Use only `#SBATCH --partition=q4,3090`. Confirm all four ranks initialize, take the
same number of steps, and finish without collective mismatch.

- [ ] **Step 5: Run a four-GPU `mix:N` smoke job**

Change only `optimizer.batch_size` to a small `mix:N`. Confirm atom-count cache
creation/reuse, budget-respecting batches, equal rank steps, and checkpoint output.

- [ ] **Step 6: Run regression and syntax checks**

```bash
python -m compileall -q src/pre_data/nep_lmdb_dataset.py \
  src/user/work_file_param.py src/user/input_param.py \
  src/user/optimizer_param.py src/PWMLFF/nep_network.py
git diff --check
git status --short
```

Run the directly executable existing NEP configuration test and all new unittest
modules once more. Expected: zero failures and no unexpected tracked artifacts.

- [ ] **Step 7: Document results without pushing**

Record exact commands, Slurm job IDs, pass/fail results, peak RSS, and known limits
in the final handoff. Leave the local branch and worktree intact. Do not run
`git push`.
