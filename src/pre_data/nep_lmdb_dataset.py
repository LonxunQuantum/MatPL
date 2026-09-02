import bisect
import json
import math
import operator
import os
import random
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
