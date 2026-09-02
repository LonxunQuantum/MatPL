import os
from pathlib import Path
from typing import Iterable, List, Union


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
