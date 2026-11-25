"""Input identity and execution helpers used by the controller."""

import functools
import hashlib
import subprocess
from pathlib import Path

from cases import BATIO_DATA, load_config, training_config


@functools.lru_cache(maxsize=128)
def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_identity(code):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(code), *args], text=True).strip()

    diff = subprocess.check_output(["git", "-C", str(code), "diff", "--binary", "HEAD"])
    return {"commit": git("rev-parse", "HEAD"), "status": git("status", "--porcelain"),
            "diff_sha256": hashlib.sha256(diff).hexdigest()}


def input_identity(case, source):
    if case.suite == "regression":
        config = training_config(source, case.system, {"batch_size": case.options["batch_size"]})
        names = [source / "nep.json"]
        for key in ("train_data", "valid_data", "datasets_path", "raw_files"):
            names.extend(Path(path) for path in config.get(key, []))
    elif case.suite == "test":
        template = source / ("gputest.json" if case.gpus else "cputest.json")
        load_config(template)
        names = [source / BATIO_DATA, source / "nep.txt", template]
    elif case.suite == "ase":
        names = [source / BATIO_DATA, source / "nep.txt"]
    else:
        names = [source / BATIO_DATA, source / "nep.json"]
        if case.options.get("resume"):
            names.append(source / "gputest.json")
    if len(set(names)) != len({path.name for path in names}):
        raise ValueError("input files have conflicting basenames")
    return {path.name: file_digest(path) for path in names}
