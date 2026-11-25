"""The supported quick-test matrix and its configuration templates."""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


CASES = ("16_metal", "MPtraj", "AuAg", "LiSiC", "BaTiO", "Si-SiO2-La2O3-HfO2-TiN")
SUITES = ("regression", "train", "optimizer", "test", "ase", "lmdb")
BATIO_DATA = "reference-structures-BaTiO-R2SCAN.xyz"
DEFAULT_CODE = Path(__file__).resolve().parents[2]
REFERENCE = "941d15ac53a25343ba38d40c804efdc8aa3107dc"


@dataclass(frozen=True)
class Case:
    suite: str
    variant: str
    system: str = "BaTiO"
    gpus: int = 1
    cpus: int = 1
    smoke: bool = False
    options: dict = field(default_factory=dict)

    @property
    def key(self):
        return f"{self.suite}/{self.system}/{self.variant}"


def collect_cases():
    matrix = [
        Case("regression", f"batch{batch}", system, options={"batch_size": batch})
        for system in CASES for batch in (1, 32)
    ]
    matrix += [
        Case("train", "gpu1", smoke=True),
        Case("train", "gpu2", gpus=2, cpus=4),
        Case("train", "cpu1", gpus=0),
        Case("train", "autograd", options={"analytical": False}),
        Case("train", "resume", options={"resume": True}),
    ]
    for charge in (False, True):
        for bec in (False, True):
            if not (charge and bec):
                matrix.append(Case("train", f"charge{int(charge)}-bec{int(bec)}",
                                   options={"train_charge": charge, "train_bec": bec}))
    for restart in (False, True):
        for scale in (False, True):
            matrix.append(Case("optimizer", f"restart{int(restart)}-scale{int(scale)}",
                               options={"restart": restart, "scale_lr": scale}))
    for device, gpus, cpus in (("cpu1", 0, 1), ("cpu4", 0, 4), ("gpu1", 1, 1), ("gpu2", 2, 2)):
        for method in ("ewald", "pppm"):
            matrix.append(Case("test", f"{device}-{method}", gpus=gpus, cpus=cpus,
                               smoke=device in ("cpu1", "gpu1") and method == "ewald",
                               options={"kspace": method}))
    for method in ("default", "invalid"):
        matrix.append(Case("test", f"cpu1-{method}", gpus=0, options={"kspace": method}))
    for device, gpus in (("cpu", 0), ("gpu", 1)):
        matrix.append(Case("ase", f"{device}-calculator", gpus=gpus, smoke=True))
        for method in ("ewald", "pppm"):
            matrix.append(Case("ase", f"{device}-{method}", gpus=gpus,
                               options={"kspace": method, "total_charge": 0.0}))
        matrix.append(Case("ase", f"{device}-charged", gpus=gpus,
                           options={"kspace": "pppm", "total_charge": 1.0}))
    for gpus in (1, 2):
        for kind, batch in (("frames", 4), ("mix", "mix:64")):
            matrix.append(Case("lmdb", f"gpu{gpus}-{kind}", gpus=gpus, cpus=2 * gpus,
                               options={"batch_size": batch}))
    return matrix


def select_cases(matrix, profile, suites, systems, variants):
    return [
        case for case in matrix
        if (profile == "full" or case.smoke)
        and (not suites or case.suite in suites)
        and (not systems or case.system in systems)
        and (not variants or case.variant in variants or case.key in variants)
    ]


def load_config(path):
    return json.loads(path.read_text(encoding="utf-8"))


def training_config(source, system, overrides):
    config = load_config(source / "nep.json")
    for field_name in ("train_data", "valid_data", "datasets_path", "raw_files"):
        if field_name in config:
            paths = config[field_name]
            if isinstance(paths, str):
                paths = [paths]
            config[field_name] = [str((source / path).resolve(strict=True)) for path in paths]
    config.pop("model_load_file", None)
    config.update(recover_train=False, model_store_dir="model_record", model_name="nep_model.ckpt")
    if system == "BaTiO":
        config["model_type"] = "NEP_CHARGE2"
        for key in ("charge_mode", "gpumd_nep4"):
            config["model"]["descriptor"].pop(key, None)
    optimizer = config["optimizer"]
    optimizer.pop("_t_0", None)
    optimizer.pop("_t_mult", None)
    optimizer.update(overrides)
    return config


def read_frames(path):
    with path.open(encoding="utf-8") as source:
        frame_id = 0
        for line in source:
            if not line.strip():
                continue
            natoms = int(line)
            header = next(source)
            atoms = [next(source) for _ in range(natoms)]
            label = "bec" if re.search(r"(?:^|:)bec:", header.lower()) else "other"
            if label == "other" and re.search(r"\b(?:stress|virial)=", header.lower()):
                label = "stress"
            yield frame_id, "".join((line, header, *atoms)), label
            frame_id += 1


def split_frames(frames, train_size=48, valid_size=8):
    groups = {label: [frame for frame in frames if frame[2] == label]
              for label in ("bec", "stress")}
    count = (train_size + valid_size) // 2
    if any(len(group) < count for group in groups.values()):
        raise ValueError("BaTiO requires both BEC-labeled and stress-labeled frames")
    balanced = [row for pair in zip(groups["bec"][:count], groups["stress"][:count]) for row in pair]
    return balanced[:train_size], balanced[train_size:train_size + valid_size]
