import bisect
import hashlib
import json
import math
import operator
import os
import random
import tempfile
import zlib
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, List, Union

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset


PathLike = Union[str, os.PathLike]


def discover_aselmdb_files(paths: Iterable[PathLike]) -> List[str]:
    """Expand files/directories into a sorted, de-duplicated LMDB shard list."""
    discovered = set()
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if not path.exists():
            raise ValueError("LMDB data path does not exist: {}".format(path))
        if path.is_file():
            if path.suffix != ".aselmdb":
                raise ValueError("Expected an .aselmdb file, got: {}".format(path))
            discovered.add(str(path.resolve()))
            continue
        if not path.is_dir():
            raise ValueError("LMDB data path is neither a file nor directory: {}".format(path))

        directory_shards = [
            str(candidate.resolve())
            for candidate in path.rglob("*.aselmdb")
            if candidate.is_file()
        ]
        if not directory_shards:
            raise ValueError("No .aselmdb files found under directory: {}".format(path))
        discovered.update(directory_shards)

    return sorted(discovered)


def _decode_compressed_json(value: bytes, context: str):
    try:
        return json.loads(zlib.decompress(value).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, zlib.error) as exc:
        raise ValueError("{}: invalid zlib-compressed JSON".format(context)) from exc


class AseLmdbShard:
    """Immutable ASE-LMDB metadata with logical-to-physical row mapping."""

    def __init__(self, path: str):
        self.path = str(Path(path).expanduser().resolve())
        env = lmdb.open(
            self.path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        try:
            with env.begin(buffers=False) as txn:
                nextid_value = txn.get(b"nextid")
                if nextid_value is None:
                    raise ValueError("{}: missing nextid metadata".format(self.path))
                nextid = _decode_compressed_json(nextid_value, self.path + ": nextid")
                deleted_value = txn.get(b"deleted_ids")
                deleted_ids = (
                    []
                    if deleted_value is None
                    else _decode_compressed_json(
                        deleted_value, self.path + ": deleted_ids"
                    )
                )
        finally:
            env.close()

        if isinstance(nextid, bool) or not isinstance(nextid, int) or nextid < 1:
            raise ValueError("{}: nextid must be a positive integer".format(self.path))
        if not isinstance(deleted_ids, list) or any(
            isinstance(row_id, bool) or not isinstance(row_id, int)
            for row_id in deleted_ids
        ):
            raise ValueError("{}: deleted_ids must be an integer list".format(self.path))

        self.nextid = nextid
        self.deleted_ids = tuple(sorted(set(deleted_ids)))
        if any(row_id < 1 or row_id >= nextid for row_id in self.deleted_ids):
            raise ValueError(
                "{}: deleted_ids contains an out-of-range row ID".format(self.path)
            )
        self._length = nextid - 1 - len(self.deleted_ids)

    def __len__(self):
        return self._length

    def row_id(self, logical_index: int) -> int:
        if logical_index < 0 or logical_index >= self._length:
            raise IndexError("LMDB shard index out of range")

        row_id = logical_index + 1
        while True:
            skipped = bisect.bisect_right(self.deleted_ids, row_id)
            mapped = logical_index + 1 + skipped
            if mapped == row_id:
                return row_id
            row_id = mapped


class BlockShuffleIndices:
    """Stream a global permutation while materializing only one index block."""

    def __init__(self, size, block_size, seed, epoch=0, shuffle=True):
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("size must be a non-negative integer")
        if (
            isinstance(block_size, bool)
            or not isinstance(block_size, int)
            or block_size < 1
        ):
            raise ValueError("block_size must be a positive integer")
        self.size = size
        self.block_size = block_size
        self.seed = seed
        self.epoch = epoch
        self.shuffle = shuffle
        self.current_buffered = 0
        self.peak_buffered = 0

    def __len__(self):
        return self.size

    def __iter__(self):
        self.current_buffered = 0
        self.peak_buffered = 0
        if not self.shuffle:
            yield from range(self.size)
            return

        random_generator = random.Random(self.seed + self.epoch)
        block_count = (self.size + self.block_size - 1) // self.block_size
        block_ids = list(range(block_count))
        random_generator.shuffle(block_ids)
        for block_id in block_ids:
            start = block_id * self.block_size
            stop = min(start + self.block_size, self.size)
            block = list(range(start, stop))
            random_generator.shuffle(block)
            self.current_buffered = len(block)
            self.peak_buffered = max(self.peak_buffered, self.current_buffered)
            yield from block
            self.current_buffered = 0


class DistributedFrameBatchSampler:
    """Build global frame batches, then give each rank a disjoint slice."""

    def __init__(
        self,
        dataset_size,
        batch_size,
        rank,
        world_size,
        seed=2023,
        shuffle=True,
        block_size=65536,
    ):
        integer_arguments = {
            "dataset_size": dataset_size,
            "batch_size": batch_size,
            "rank": rank,
            "world_size": world_size,
            "block_size": block_size,
        }
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in integer_arguments.values()
        ):
            raise ValueError("sampler sizes and rank must be integers")
        if dataset_size < 0:
            raise ValueError("dataset_size must be non-negative")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if world_size < 1:
            raise ValueError("world_size must be positive")
        if rank < 0 or rank >= world_size:
            raise ValueError("rank must be in [0, world_size)")
        if block_size < 1:
            raise ValueError("block_size must be positive")

        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.shuffle = shuffle
        self.block_size = block_size
        self.epoch = 0
        self.peak_buffered = 0

    @property
    def super_batch_size(self):
        return self.batch_size * self.world_size

    def __len__(self):
        return self.dataset_size // self.super_batch_size

    def set_epoch(self, epoch):
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        self.epoch = epoch

    def __iter__(self):
        indices = BlockShuffleIndices(
            self.dataset_size,
            self.block_size,
            self.seed,
            epoch=self.epoch,
            shuffle=self.shuffle,
        )
        super_batch = []
        self.peak_buffered = 0
        rank_start = self.rank * self.batch_size
        rank_stop = rank_start + self.batch_size
        for index in indices:
            super_batch.append(index)
            self.peak_buffered = max(
                self.peak_buffered,
                indices.current_buffered + len(super_batch),
            )
            if len(super_batch) == self.super_batch_size:
                rank_batch = super_batch[rank_start:rank_stop]
                super_batch = []
                yield rank_batch


def select_stat_indices(
    size,
    requested,
    rank,
    world_size,
    seed,
    per_rank_cap=32768,
):
    """Select one deterministic global sample and return this rank's sorted slice."""
    arguments = {
        "size": size,
        "requested": requested,
        "rank": rank,
        "world_size": world_size,
        "per_rank_cap": per_rank_cap,
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in arguments.values()
    ):
        raise ValueError("statistics sizes and rank must be integers")
    if size < 0:
        raise ValueError("size must be non-negative")
    if requested < 1:
        raise ValueError("requested must be positive")
    if world_size < 1:
        raise ValueError("world_size must be positive")
    if rank < 0 or rank >= world_size:
        raise ValueError("rank must be in [0, world_size)")
    if per_rank_cap < 1:
        raise ValueError("per_rank_cap must be positive")

    global_count = min(size, requested, per_rank_cap * world_size)
    global_indices = random.Random(seed).sample(range(size), global_count)
    return sorted(global_indices[rank::world_size])


class LmdbEnergyStatistics:
    """Mergeable normal equations and atom-count summaries for LMDB frames."""

    def __init__(self, atom_type_count):
        if (
            isinstance(atom_type_count, bool)
            or not isinstance(atom_type_count, int)
            or atom_type_count < 1
        ):
            raise ValueError("atom_type_count must be a positive integer")
        self.atom_type_count = atom_type_count
        self.ata = np.zeros((atom_type_count, atom_type_count), dtype=np.float64)
        self.ate = np.zeros(atom_type_count, dtype=np.float64)
        self.frame_count = 0
        self.atom_count_sum = 0
        self.max_atoms = 0

    def update(self, composition, energy):
        composition = np.asarray(composition, dtype=np.float64)
        if composition.shape != (self.atom_type_count,):
            raise ValueError(
                "composition has shape {}, expected ({},)".format(
                    composition.shape, self.atom_type_count
                )
            )
        if (
            not np.isfinite(composition).all()
            or (composition < 0).any()
            or not np.equal(composition, np.floor(composition)).all()
        ):
            raise ValueError("composition must contain finite non-negative counts")
        try:
            energy = float(energy)
        except (TypeError, ValueError) as exc:
            raise ValueError("energy must be a finite scalar") from exc
        if not np.isfinite(energy):
            raise ValueError("energy must be a finite scalar")

        atom_count = int(composition.sum())
        self.ata += np.outer(composition, composition)
        self.ate += composition * energy
        self.frame_count += 1
        self.atom_count_sum += atom_count
        self.max_atoms = max(self.max_atoms, atom_count)

    def update_from_sample(self, sample):
        atom_type_map = sample["atom_type_map"]
        if isinstance(atom_type_map, torch.Tensor):
            atom_type_map = atom_type_map.detach().cpu().numpy()
        atom_type_map = np.asarray(atom_type_map, dtype=np.int64).reshape(-1)
        if (
            (atom_type_map < 0).any()
            or (atom_type_map >= self.atom_type_count).any()
        ):
            raise ValueError("atom_type_map contains an out-of-range type index")
        composition = np.bincount(
            atom_type_map, minlength=self.atom_type_count
        ).astype(np.float64)
        energy = sample["energy"]
        if isinstance(energy, torch.Tensor):
            energy = energy.detach().cpu().numpy()
        energy = np.asarray(energy)
        if energy.size != 1:
            raise ValueError("sample energy must contain one scalar")
        self.update(composition, energy.reshape(-1)[0])

    def merge(self, other):
        if not isinstance(other, LmdbEnergyStatistics):
            raise TypeError("can only merge LmdbEnergyStatistics")
        if other.atom_type_count != self.atom_type_count:
            raise ValueError("statistics atom type counts do not match")
        self.ata += other.ata
        self.ate += other.ate
        self.frame_count += other.frame_count
        self.atom_count_sum += other.atom_count_sum
        self.max_atoms = max(self.max_atoms, other.max_atoms)
        return self

    @property
    def average_atoms(self):
        if self.frame_count == 0:
            return 0.0
        return self.atom_count_sum / self.frame_count

    def energy_shift(self):
        if self.frame_count == 0:
            raise ValueError("cannot solve energy shift with no frames")
        shift, _, _, _ = np.linalg.lstsq(self.ata, self.ate, rcond=None)
        return shift.tolist()


class LmdbNatomsCache:
    """Persistent per-shard atom counts, memory-mapped after distributed build."""

    CACHE_VERSION = 1

    def __init__(self, dataset, cache_dir):
        if not isinstance(dataset, NepLmdbDataset):
            raise TypeError("dataset must be a NepLmdbDataset")
        self.dataset = dataset
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self._arrays = []

    def __len__(self):
        return len(self.dataset)

    def _paths(self, shard):
        digest = hashlib.sha256(shard.path.encode("utf-8")).hexdigest()[:24]
        return (
            self.cache_dir / (digest + ".natoms.i32"),
            self.cache_dir / (digest + ".natoms.json"),
        )

    def _fingerprint(self, shard):
        source = Path(shard.path).stat()
        return {
            "version": self.CACHE_VERSION,
            "source_path": shard.path,
            "source_size": source.st_size,
            "source_mtime_ns": source.st_mtime_ns,
            "nextid": shard.nextid,
            "deleted_ids": list(shard.deleted_ids),
            "logical_length": len(shard),
        }

    def _is_valid(self, shard):
        data_path, manifest_path = self._paths(shard)
        if not data_path.is_file() or not manifest_path.is_file():
            return False
        if data_path.stat().st_size != len(shard) * np.dtype(np.int32).itemsize:
            return False
        try:
            with manifest_path.open("r", encoding="utf-8") as stream:
                manifest = json.load(stream)
        except (OSError, ValueError):
            return False
        return manifest == self._fingerprint(shard)

    def build_assigned(self, rank, world_size):
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or isinstance(world_size, bool)
            or not isinstance(world_size, int)
            or world_size < 1
            or rank < 0
            or rank >= world_size
        ):
            raise ValueError("rank must be in [0, world_size)")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        for shard_index, shard in enumerate(self.dataset.shards):
            if shard_index % world_size == rank and not self._is_valid(shard):
                self._build_shard(shard)

    def _build_shard(self, shard):
        data_path, manifest_path = self._paths(shard)
        data_handle = tempfile.NamedTemporaryFile(
            prefix=data_path.name + ".tmp-",
            dir=self.cache_dir,
            delete=False,
        )
        temporary_data = Path(data_handle.name)
        data_handle.close()
        temporary_manifest = None
        env = None
        counts = None
        try:
            if len(shard) > 0:
                counts = np.memmap(
                    temporary_data,
                    dtype=np.int32,
                    mode="w+",
                    shape=(len(shard),),
                )
                env = lmdb.open(
                    shard.path,
                    subdir=False,
                    readonly=True,
                    lock=False,
                    readahead=True,
                    meminit=False,
                )
                with env.begin(buffers=False) as txn:
                    for logical_index in range(len(shard)):
                        row_id = shard.row_id(logical_index)
                        context = "{}: frame key {}".format(shard.path, row_id)
                        value = txn.get(str(row_id).encode("ascii"))
                        if value is None:
                            raise ValueError("{}: row is missing".format(context))
                        frame = _decode_compressed_json(value, context)
                        if not isinstance(frame, dict) or not isinstance(
                            frame.get("numbers"), list
                        ):
                            raise ValueError(
                                "{}: numbers must be a JSON array".format(context)
                            )
                        natoms = len(frame["numbers"])
                        if natoms < 1 or natoms > np.iinfo(np.int32).max:
                            raise ValueError(
                                "{}: invalid atom count {}".format(context, natoms)
                            )
                        counts[logical_index] = natoms
                counts.flush()
                del counts
                counts = None
            with temporary_data.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary_data, data_path)

            manifest_handle = tempfile.NamedTemporaryFile(
                prefix=manifest_path.name + ".tmp-",
                dir=self.cache_dir,
                mode="w",
                encoding="utf-8",
                delete=False,
            )
            temporary_manifest = Path(manifest_handle.name)
            with manifest_handle:
                json.dump(self._fingerprint(shard), manifest_handle, sort_keys=True)
                manifest_handle.flush()
                os.fsync(manifest_handle.fileno())
            os.replace(temporary_manifest, manifest_path)
        finally:
            if counts is not None:
                del counts
            if env is not None:
                env.close()
            if temporary_data.exists():
                temporary_data.unlink()
            if temporary_manifest is not None and temporary_manifest.exists():
                temporary_manifest.unlink()

    def load(self):
        self.close()
        arrays = []
        for shard in self.dataset.shards:
            if not self._is_valid(shard):
                raise ValueError(
                    "atom-count cache is missing or stale for {}".format(shard.path)
                )
            data_path, _ = self._paths(shard)
            if len(shard) == 0:
                arrays.append(np.empty(0, dtype=np.int32))
            else:
                arrays.append(
                    np.memmap(
                        data_path,
                        dtype=np.int32,
                        mode="r",
                        shape=(len(shard),),
                    )
                )
        self._arrays = arrays
        return self

    def close(self):
        for array in self._arrays:
            mmap_object = getattr(array, "_mmap", None)
            if mmap_object is not None:
                mmap_object.close()
        self._arrays = []

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __getitem__(self, index):
        if not self._arrays:
            raise RuntimeError("atom-count cache must be loaded before use")
        shard, shard_index = self.dataset._locate(index)
        cache_shard_index = bisect.bisect_right(
            self.dataset._shard_ends,
            (operator.index(index) % len(self.dataset)),
        )
        return int(self._arrays[cache_shard_index][shard_index])


class DistributedAtomBatchSampler:
    """Pack the global shuffled frame stream into equal-step rank atom batches."""

    def __init__(
        self,
        natoms,
        atom_budget,
        rank,
        world_size,
        seed=2023,
        shuffle=True,
        block_size=65536,
    ):
        integer_arguments = {
            "atom_budget": atom_budget,
            "rank": rank,
            "world_size": world_size,
            "block_size": block_size,
        }
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in integer_arguments.values()
        ):
            raise ValueError("atom budget, rank, and sampler sizes must be integers")
        if atom_budget < 1:
            raise ValueError("atom_budget must be positive")
        if world_size < 1:
            raise ValueError("world_size must be positive")
        if rank < 0 or rank >= world_size:
            raise ValueError("rank must be in [0, world_size)")
        if block_size < 1:
            raise ValueError("block_size must be positive")

        self.natoms = natoms
        self.atom_budget = atom_budget
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.shuffle = shuffle
        self.block_size = block_size
        self.epoch = 0
        self.peak_completed_batches = 0
        self._length_cache = {}

    def set_epoch(self, epoch):
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        self.epoch = epoch

    def _atom_count(self, index):
        try:
            count = operator.index(self.natoms[index])
        except (TypeError, ValueError) as exc:
            raise ValueError("natoms values must be positive integers") from exc
        if isinstance(count, bool) or count < 1:
            raise ValueError("natoms values must be positive integers")
        return count

    def _global_batches(self):
        indices = BlockShuffleIndices(
            len(self.natoms),
            self.block_size,
            self.seed,
            epoch=self.epoch,
            shuffle=self.shuffle,
        )
        batch = []
        batch_atoms = 0
        for index in indices:
            frame_atoms = self._atom_count(index)
            if batch and batch_atoms + frame_atoms > self.atom_budget:
                yield batch
                batch = []
                batch_atoms = 0
            if frame_atoms > self.atom_budget:
                yield [index]
            else:
                batch.append(index)
                batch_atoms += frame_atoms
        if batch:
            yield batch

    def __len__(self):
        cache_key = (self.epoch, self.shuffle)
        if cache_key not in self._length_cache:
            global_batch_count = sum(1 for _ in self._global_batches())
            self._length_cache[cache_key] = global_batch_count // self.world_size
        return self._length_cache[cache_key]

    def __iter__(self):
        completed_batches = []
        self.peak_completed_batches = 0
        for batch in self._global_batches():
            completed_batches.append(batch)
            self.peak_completed_batches = max(
                self.peak_completed_batches, len(completed_batches)
            )
            if len(completed_batches) == self.world_size:
                rank_batch = completed_batches[self.rank]
                completed_batches = []
                yield rank_batch


def _get_det(box: np.ndarray) -> float:
    return float(np.linalg.det(box.reshape((3, 3))))


def _get_area(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.linalg.norm(np.cross(first, second)))


def _expand_box(lattice, cutoff_radial, num_cell, box):
    a = lattice[0::3]
    b = lattice[1::3]
    c = lattice[2::3]
    determinant = _get_det(lattice)
    volume = abs(determinant)
    if not np.isfinite(volume) or volume <= 0:
        raise ValueError("cell must have a finite, non-zero volume")

    num_cell[0] = int(math.ceil(2.0 * cutoff_radial / (volume / _get_area(b, c))))
    num_cell[1] = int(math.ceil(2.0 * cutoff_radial / (volume / _get_area(c, a))))
    num_cell[2] = int(math.ceil(2.0 * cutoff_radial / (volume / _get_area(a, b))))

    box[0:9:3] = lattice[0::3] * num_cell[0]
    box[1:9:3] = lattice[1::3] * num_cell[1]
    box[2:9:3] = lattice[2::3] * num_cell[2]

    box[9] = box[4] * box[8] - box[5] * box[7]
    box[10] = box[2] * box[7] - box[1] * box[8]
    box[11] = box[1] * box[5] - box[2] * box[4]
    box[12] = box[5] * box[6] - box[3] * box[8]
    box[13] = box[0] * box[8] - box[2] * box[6]
    box[14] = box[2] * box[3] - box[0] * box[5]
    box[15] = box[3] * box[7] - box[4] * box[6]
    box[16] = box[1] * box[6] - box[0] * box[7]
    box[17] = box[0] * box[4] - box[1] * box[3]

    expanded_determinant = determinant * int(num_cell[0]) * int(num_cell[1]) * int(num_cell[2])
    if expanded_determinant == 0:
        raise ValueError("cutoff_radial must produce a non-zero expanded cell")
    box[9:18] /= expanded_determinant
    return volume


def _default_bec(numbers: np.ndarray, fill_metal_bec: bool) -> np.ndarray:
    bec = np.full((numbers.size, 9), -1e6, dtype=float)
    if not fill_metal_bec:
        return bec
    identity = np.eye(3, dtype=float).reshape(9)
    bec[np.isin(numbers, (3, 11, 19))] = identity
    bec[np.isin(numbers, (12, 20))] = 2.0 * identity
    return bec


class NepLmdbDataset(Dataset):
    """Lazy, worker-safe NEP dataset for zlib-JSON ASE-LMDB shards."""

    def __init__(
        self,
        data_paths,
        atom_types,
        cutoff_radial=0,
        cutoff_angular=0,
        cal_energy=False,
        batch_max_types=-1,
        dtype=torch.float64,
        index_type=torch.int64,
        use_cartesian=True,
        fill_metal_bec=False,
        train_ei=False,
        max_open_shards=8,
    ):
        super().__init__()
        if not use_cartesian:
            raise ValueError("ASE-LMDB NEP training currently requires Cartesian positions")
        if isinstance(max_open_shards, bool) or not isinstance(max_open_shards, int) or max_open_shards < 1:
            raise ValueError("max_open_shards must be a positive integer")

        self.dtype = dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
        self.index_type = (
            index_type
            if isinstance(index_type, torch.dtype)
            else getattr(torch, index_type)
        )
        self.dirs = discover_aselmdb_files(data_paths)
        self.shards = [AseLmdbShard(path) for path in self.dirs]
        self._shard_ends = []
        total = 0
        for shard in self.shards:
            total += len(shard)
            self._shard_ends.append(total)
        self.total_images = total

        self.atom_types = np.asarray(atom_types, dtype=np.int64)
        if self.atom_types.ndim != 1 or len(set(self.atom_types.tolist())) != len(self.atom_types):
            raise ValueError("atom_types must be a one-dimensional list of unique elements")
        self._type_to_index = {
            int(atomic_number): index
            for index, atomic_number in enumerate(self.atom_types.tolist())
        }
        self.cutoff_radial = cutoff_radial
        self.cutoff_angular = cutoff_angular
        self.cal_energy = cal_energy
        self.batch_max_types = batch_max_types
        self.fill_metal_bec = fill_metal_bec
        self.train_ei = train_ei
        self.max_open_shards = max_open_shards
        self._env_cache = OrderedDict()
        self._env_pid = os.getpid()

        # These are filled from the bounded global statistics pass before training.
        self.max_NN_radial = 100
        self.max_NN_angular = 100
        self.max_atom_nums = 1
        self.avg_image_atom = None
        self.energy_shift = [0.0 for _ in self.atom_types]

    def __len__(self):
        return self.total_images

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_env_cache"] = OrderedDict()
        state["_env_pid"] = None
        return state

    def close(self):
        while self._env_cache:
            _, env = self._env_cache.popitem(last=False)
            env.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def get_energy_shift(self):
        return self.energy_shift

    def _environment(self, path: str):
        current_pid = os.getpid()
        if self._env_pid != current_pid:
            self.close()
            self._env_pid = current_pid
        env = self._env_cache.pop(path, None)
        if env is None:
            env = lmdb.open(
                path,
                subdir=False,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
        self._env_cache[path] = env
        while len(self._env_cache) > self.max_open_shards:
            _, oldest = self._env_cache.popitem(last=False)
            oldest.close()
        return env

    def _locate(self, index: int):
        try:
            index = operator.index(index)
        except TypeError as exc:
            raise TypeError("LMDB dataset indices must be integers") from exc
        if index < 0:
            index += self.total_images
        if index < 0 or index >= self.total_images:
            raise IndexError("LMDB dataset index out of range")
        shard_index = bisect.bisect_right(self._shard_ends, index)
        shard_start = 0 if shard_index == 0 else self._shard_ends[shard_index - 1]
        return self.shards[shard_index], index - shard_start

    def __getitem__(self, index):
        shard, shard_index = self._locate(index)
        row_id = shard.row_id(shard_index)
        context = "{}: frame key {}".format(shard.path, row_id)
        env = self._environment(shard.path)
        with env.begin(buffers=False) as txn:
            value = txn.get(str(row_id).encode("ascii"))
        if value is None:
            raise ValueError("{}: row is missing".format(context))
        frame = _decode_compressed_json(value, context)
        if not isinstance(frame, dict):
            raise ValueError("{}: decoded frame must be an object".format(context))
        try:
            return self._convert_frame(frame)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("{}: {}".format(context, exc)) from exc

    @staticmethod
    def _finite_array(frame, key, shape=None, dtype=float):
        if key not in frame:
            raise ValueError("missing required field '{}'".format(key))
        value = np.asarray(frame[key], dtype=dtype)
        if shape is not None and value.shape != shape:
            raise ValueError(
                "{} has shape {}, expected {}".format(key, value.shape, shape)
            )
        if not np.isfinite(value).all():
            raise ValueError("{} contains non-finite values".format(key))
        return value

    def _convert_frame(self, frame):
        raw_numbers = self._finite_array(frame, "numbers")
        if raw_numbers.ndim != 1 or raw_numbers.size == 0:
            raise ValueError("numbers must be a non-empty one-dimensional array")
        if not np.equal(raw_numbers, np.floor(raw_numbers)).all():
            raise ValueError("numbers must contain integer atomic numbers")
        numbers = raw_numbers.astype(np.int64)
        natoms = numbers.size

        unknown = sorted(set(numbers.tolist()) - set(self._type_to_index))
        if unknown:
            raise ValueError("atom types {} are absent from configured atom_type".format(unknown))
        atom_type_map = np.asarray(
            [self._type_to_index[int(number)] for number in numbers], dtype=np.int64
        )

        positions = self._finite_array(frame, "positions", (natoms, 3))
        forces = self._finite_array(frame, "forces", (natoms, 3))
        cell = self._finite_array(frame, "cell", (3, 3))
        pbc = np.asarray(frame.get("pbc"))
        if pbc.shape != (3,) or not np.asarray(pbc, dtype=bool).all():
            raise ValueError("pbc must mark all three cell directions periodic")

        energy_array = self._finite_array(frame, "energy")
        if energy_array.size != 1:
            raise ValueError("energy must be a scalar")
        energy = float(energy_array.reshape(-1)[0])

        volume = abs(float(np.linalg.det(cell)))
        if not np.isfinite(volume) or volume <= 0:
            raise ValueError("cell must have a finite, non-zero volume")
        stress = frame.get("stress")
        if stress is None:
            virial = np.full(9, -1e6, dtype=float)
        else:
            stress = np.asarray(stress, dtype=float)
            if stress.shape != (6,):
                raise ValueError("stress has shape {}, expected (6,)".format(stress.shape))
            if not np.isfinite(stress).all():
                raise ValueError("stress contains non-finite values")
            scaled = -stress * volume
            virial = np.array(
                [
                    [scaled[0], scaled[5], scaled[4]],
                    [scaled[5], scaled[1], scaled[3]],
                    [scaled[4], scaled[3], scaled[2]],
                ]
            ).reshape(-1)

        atomic_energy = None
        for key in ("atomic_energy", "atomic_energies", "energies"):
            if key in frame:
                atomic_energy = np.asarray(frame[key], dtype=float)
                break
        if atomic_energy is None:
            if self.train_ei:
                raise ValueError("atomic energies are required when train_ei is true")
            atomic_energy = np.zeros(natoms, dtype=float)
        elif atomic_energy.shape != (natoms,):
            raise ValueError(
                "atomic energies have shape {}, expected ({},)".format(
                    atomic_energy.shape, natoms
                )
            )
        elif not np.isfinite(atomic_energy).all():
            raise ValueError("atomic energies contain non-finite values")

        num_cell = np.zeros(3, dtype=int)
        box = np.zeros(18, dtype=float)
        lattice = cell.T.flatten()
        expanded_volume = _expand_box(
            lattice, self.cutoff_radial, num_cell, box
        )

        fragment = np.full(natoms, -1, dtype=np.int64)
        fragment_charge = np.full(natoms, np.nan, dtype=float)
        total_charge = float(frame.get("charge", frame.get("total_charge", 0.0)))
        if not np.isfinite(total_charge):
            raise ValueError("charge must be finite")

        return {
            "max_allow_atom_type": torch.tensor(
                [self.batch_max_types], dtype=self.index_type
            ),
            "box": torch.as_tensor(box, dtype=self.dtype),
            "box_original": torch.as_tensor(lattice, dtype=self.dtype),
            "num_cell": torch.as_tensor(num_cell, dtype=self.index_type),
            "volume": torch.tensor([expanded_volume], dtype=self.dtype),
            "atom_type_map": torch.as_tensor(atom_type_map, dtype=self.index_type),
            "num_atom": torch.tensor([natoms], dtype=self.index_type),
            "atom_type_image": torch.as_tensor(
                np.unique(numbers), dtype=self.index_type
            ),
            "force": torch.as_tensor(forces, dtype=self.dtype),
            "ei": torch.as_tensor(atomic_energy, dtype=self.dtype),
            "energy": torch.tensor([energy], dtype=self.dtype),
            "fragment": torch.as_tensor(fragment, dtype=self.index_type),
            "fragment_charge": torch.as_tensor(fragment_charge, dtype=self.dtype),
            "charge": torch.tensor([total_charge], dtype=self.dtype),
            "position": torch.as_tensor(positions, dtype=self.dtype),
            "virial": torch.as_tensor(virial, dtype=self.dtype),
            "bec": torch.as_tensor(
                _default_bec(numbers, self.fill_metal_bec), dtype=self.dtype
            ),
        }
