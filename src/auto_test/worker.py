"""Run one case on an allocated node with the MatPL environment loaded."""

import json
import math
import os
import re
import shutil
import subprocess
import sys
import traceback
import zlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from cases import BATIO_DATA, Case, load_config, read_frames, split_frames, training_config
from results import compare_predictions, read_epochs, read_predictions, write_json


def verify(condition, message):
    if not condition:
        raise ValueError(message)


def matpl(work, command, config, *, cpu_workers=None):
    work.mkdir(parents=True, exist_ok=True)
    input_path = work / f"{command}.json"
    write_json(input_path, config)
    executable = shutil.which("MATPL")
    expected = Path(os.environ["MATPL_QA_CODE"]) / "main.py"
    verify(executable and Path(executable).resolve() == expected.resolve(),
           f"MATPL resolved to {executable}; expected {expected}")
    if cpu_workers is None:
        args = [executable, command, str(input_path)]
    else:
        launcher = ("import multiprocessing, runpy, sys\n"
                    "count = int(sys.argv.pop(1))\n"
                    "sys.argv.pop(0)\n"
                    "multiprocessing.cpu_count = lambda: count\n"
                    "runpy.run_path(sys.argv[0], run_name='__main__')")
        args = [sys.executable, "-c", launcher, str(cpu_workers), executable, command, str(input_path)]
    logfile = work / f"{command}.log"
    with logfile.open("w", encoding="utf-8") as stream:
        process = subprocess.run(args, cwd=work, stdout=stream, stderr=subprocess.STDOUT)
    if process.returncode:
        raise RuntimeError(f"MATPL {command} exited {process.returncode}; see {logfile}")
    return logfile.read_text(encoding="utf-8", errors="replace")


def dataset_paths(work, frames, prefix, use_lmdb):
    xyz = work / f"{prefix}.xyz"
    xyz.write_text("".join(frame[1] for frame in frames), encoding="utf-8")
    if not use_lmdb:
        return [str(xyz)]
    from ase.io import read
    import lmdb

    path = work / f"{prefix}.aselmdb"
    env = lmdb.open(str(path), subdir=False, map_size=32 * 1024 * 1024)
    try:
        with env.begin(write=True) as txn:
            for index, atoms in enumerate(read(xyz, index=":"), start=1):
                batch = {
                    "numbers": atoms.numbers.tolist(), "positions": atoms.positions.tolist(),
                    "cell": atoms.cell.array.tolist(), "pbc": atoms.pbc.tolist(),
                    "energy": float(atoms.get_potential_energy()), "charge": 0.0,
                    "forces": atoms.get_forces().tolist(),
                }
                if "stress" in atoms.calc.results:
                    batch["stress"] = atoms.get_stress().tolist()
                txn.put(str(index).encode("ascii"), zlib.compress(json.dumps(batch).encode()))
            txn.put(b"nextid", zlib.compress(json.dumps(index + 1).encode()))
    finally:
        env.close()
    return [str(path)]


def train_once(case, work, config, expected_peak=None):
    log = matpl(work, "train", config)
    if expected_peak is not None:
        match = re.search(r"\bpeak_lr=([0-9.eE+-]+)", log)
        verify(match and math.isclose(float(match.group(1)), expected_peak, rel_tol=1e-8),
               f"training did not use peak LR {expected_peak:.9g}")
    if case.gpus > 1:
        for rank in range(case.gpus):
            verify(f"Rank {rank}: LocalRank {rank}, device cuda:{rank}" in log,
                   f"missing CUDA/HIP rank {rank}")
    else:
        device = "cuda:0" if case.gpus else "cpu"
        verify(f"Using device: {device}" in log, f"training did not use {device}")
    target = work / "model_record"
    metrics = read_epochs(target / "epoch_train.dat")
    opt = config["optimizer"]
    required = {"loss", "last_lr_used"}
    for flag, metric in (
        ("train_energy", "RMSE_Etot(eV/atom)"), ("train_force", "RMSE_F(eV/Å)"),
        ("train_virial", "RMSE_virial(eV/atom)"), ("train_charge", "RMSE_charge"),
        ("train_bec", "RMSE_BEC"),
    ):
        if opt.get(flag):
            required.add(metric)
        elif flag in opt:
            verify(all(metric not in row for row in metrics.values()), f"unexpected {metric}")
    epochs = opt["epochs"]
    verify(all(required <= row.keys() for row in metrics.values()), "required train metrics missing")
    ckpt_path = target / "nep_model.ckpt"
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    verify(state["epoch"] == epochs and max(metrics) == epochs, f"expected training through epoch {epochs}")
    verify(state["optimizer_updates"] > 0, "checkpoint has no optimizer updates")
    verify(bool(state.get("optimizer")) and bool(state.get("state_dict")), "checkpoint is incomplete")
    verify(all(t.isfinite().all().item() for t in state["state_dict"].values()
               if isinstance(t, torch.Tensor)), "checkpoint has nonfinite weights")
    if case.system == "BaTiO":
        with (target / "nep5.txt").open(encoding="utf-8") as model:
            verify("_charge2 " in model.readline(), "expected exported qNEP model")
    validation = {}
    if config.get("valid_data"):
        validation = read_epochs(target / "epoch_valid.dat")
        verify(validation.keys() == metrics.keys(), "training and validation epochs differ")
    return metrics, validation, state, ckpt_path


def cli_prediction(case, work, source, frames, model, method):
    work.mkdir(parents=True, exist_ok=True)
    data = dataset_paths(work, frames, "test", False)
    if model.suffix == ".ckpt":
        model = Path(shutil.copy2(model, work / model.name))
    config = load_config(source / ("gputest.json" if case.gpus else "cputest.json"))
    config.update(model_type="NEP", format="extxyz", test_data=data, model_load_file=str(model),
                  test_dir_name="test_result")
    config.pop("kspace", None)
    if method != "default":
        config["kspace"] = method
    output = matpl(work, "test", config, cpu_workers=None if case.gpus else case.cpus)
    device_log = f"The GPUs: {case.gpus}" if case.gpus else f"The CPUs: {case.cpus}"
    verify(re.search(rf"^{re.escape(device_log)}\s*$", output, re.MULTILINE),
           f"case did not use {device_log}")
    if method == "invalid":
        verify("is invalid" in output and "Use default 'ewald'" in output,
               "invalid kspace did not emit fallback warning")
    else:
        verify("Detected kspace option" not in output, "valid kspace emitted a fallback warning")
    return read_predictions(work / "test_result", frames)


def train(case, work, source):
    options = case.options
    regression = case.suite == "regression"
    overrides = {"epochs": 1, "batch_size": options.get("batch_size", 4)}
    if not regression:
        overrides.update(train_charge=options.get("train_charge", True),
                         train_bec=options.get("train_bec", True))
    config = training_config(source, case.system, overrides)
    config.update(device="cuda" if case.gpus else "cpu", seed=config.get("seed", 2023))
    opt = config["optimizer"]
    peak = None
    if not regression:
        opt.update(print_freq=10, scale_lr=options.get("scale_lr", False), warm_epochs=0)
        for key in ("t_0", "t_mult"):
            opt.pop(key, None)
        if options.get("restart"):
            opt.update(t_0=1, t_mult=2, epochs=4)
        config["model"]["fitting_net"]["use_analytical_nep_grad"] = options.get("analytical", True)
        train_frames, valid_frames = split_frames(list(read_frames(source / BATIO_DATA)))
        write_json(work / "frames.json", {"train": [f[0] for f in train_frames],
                                          "valid": [f[0] for f in valid_frames]})
        use_lmdb = case.suite == "lmdb"
        config["train_data"] = dataset_paths(work, train_frames, "train", use_lmdb)
        config["valid_data"] = dataset_paths(work, valid_frames, "valid", use_lmdb)
        if use_lmdb:
            config.update(format="lmdb", precision="float64", workers=0,
                          lmdb_stat_frames=len(train_frames))
            opt.update(train_bec=False, train_bec_ion=False)
    if case.suite == "optimizer":
        opt.pop("scaling_method", None)
        opt.update(scale_method="sqrt", decay_step=5000)
        peak = opt["learning_rate"] * (math.sqrt(opt["batch_size"]) if opt["scale_lr"] else 1)
    initial = work / "initial" if options.get("resume") else work
    metrics, validation, state, model = train_once(case, initial, config, peak)
    if case.suite == "optimizer":
        scheduler = state.get("scheduler")
        if opt.get("t_0"):
            verify(scheduler and scheduler["T_mult"] == 2 and scheduler["T_i"] == 4 * scheduler["T_0"],
                   "checkpoint did not cross the restart boundary")
            verify(math.isclose(scheduler["base_lrs"][0], peak, rel_tol=1e-10),
                   "scheduler peak learning rate is wrong")
        else:
            verify(scheduler is None, "scheduler was active with restart disabled")
            verify(math.isclose(metrics[1]["last_lr_used"], peak, rel_tol=1e-8),
                   "optimizer used the wrong LR with restart disabled")
            verify(math.isclose(state["optimizer"]["param_groups"][0]["lr"], peak, rel_tol=1e-8),
                   "checkpoint stored the wrong LR with restart disabled")
    if options.get("resume"):
        opt.update(epochs=2, reset_epoch=False)
        config.update(recover_train=True, model_load_file=str(model))
        last, last_validation, after, resumed = train_once(case, work / "resume", config)
        verify(after["optimizer_updates"] > state["optimizer_updates"],
               "resume did not advance optimizer updates")
        verify(after["lr_progress_epochs"] > state["lr_progress_epochs"],
               "resume did not advance learning-rate progress")
        ckpt_prediction = cli_prediction(case, work / "ckpt-test", source, valid_frames, resumed, "ewald")
        text_prediction = cli_prediction(case, work / "text-test", source, valid_frames,
                                         resumed.with_name("nep5.txt"), "ewald")
        verify(not compare_predictions(ckpt_prediction, text_prediction, 1e-6, 1e-7),
               "checkpoint and its exported text predictions differ")
        metrics.update(last)
        validation.update(last_validation)
        state = after
    return {"epochs": {str(index): values for index, values in metrics.items()},
            "validation_epochs": {str(index): values for index, values in validation.items()},
            "optimizer_updates": state["optimizer_updates"]}


def ase_predictions(case, work, source, frames):
    from ase.io import read
    from src.ase.calculate import MatPL_calculator
    from src.mods.infer import Inference

    xyz, = dataset_paths(work, frames, "ase", False)
    model = str(source / "nep.txt")
    method = case.options.get("kspace")
    images = read(xyz, index=":")
    result = {"frame_ids": [frame[0] for frame in frames], "natoms": [len(atoms) for atoms in images],
              "energy": [], "atomic_energy": [], "force": [], "virial": []}
    if method:
        result.update(charge=[], bec=[])
        device = torch.device("cuda" if case.gpus else "cpu")
        calc = Inference(model, device)
        total_charge = case.options["total_charge"]
        screened_charge = 0.0
        if total_charge:
            from src.user.nep_param import NepParam

            params = NepParam()
            params.set_nep_nn_c_param_from_nep_txt(model, calc.model_atom_type)
            screened_charge = total_charge / params.sqrt_epsilon_inf
    else:
        calc = MatPL_calculator(model_file=model)
        verify(calc.device.type == ("cuda" if case.gpus else "cpu"), "unexpected ASE device")
    for atoms in images:
        if method:
            energy, atomic, forces, virial, charge, bec = calc.ase_nep_infer(
                np.asarray(atoms.cell), atoms.positions, atoms.get_chemical_symbols(),
                kspace_method=method, total_charge=total_charge)
            verify(np.asarray(charge).shape == (len(atoms),) and np.asarray(bec).shape == (len(atoms), 9),
                   "ASE inference returned wrong charge or BEC shape")
            verify(abs(float(np.sum(charge)) - screened_charge) < 1e-4,
                   f"ASE screened charge sum differs from {screened_charge:.9g}")
            result["charge"].extend(np.asarray(charge).reshape(-1, 1).tolist())
            result["bec"].extend(np.asarray(bec).tolist())
            virial = np.asarray(virial).reshape(-1)[[0, 1, 2, 4, 5, 8]]
        else:
            atoms.calc = calc
            energy = atoms.get_potential_energy()
            atomic = atoms.get_potential_energies()
            forces = atoms.get_forces()
            stress = atoms.get_stress()
            verify(stress.shape == (6,), "ASE returned wrong stress shape")
            virial = -atoms.get_volume() * stress[[0, 5, 4, 1, 3, 2]]
        verify(np.asarray(atomic).shape == (len(atoms),) and np.asarray(forces).shape == (len(atoms), 3),
               "ASE returned wrong atomic energy or force shape")
        verify(math.isclose(float(energy), float(np.sum(atomic)), rel_tol=1e-7, abs_tol=1e-5),
               "ASE atomic energies do not sum to total energy")
        result["energy"].append([float(energy)])
        result["atomic_energy"].extend(np.asarray(atomic).reshape(-1, 1).tolist())
        result["force"].extend(np.asarray(forces).tolist())
        result["virial"].append(virial.tolist())
    if not method:
        atoms = images[0]
        original = atoms.get_forces().copy()
        atoms.positions[0, 0] += 0.05
        updated = atoms.get_forces()
        verify(np.isfinite(updated).all() and not np.allclose(original, updated),
               "ASE forces did not update after moving an atom")
    return result


def verify_kspace_difference(pppm, ewald):
    delta = float(np.max(np.abs(np.asarray(pppm["virial"]) - ewald["virial"])))
    verify(delta > 1e-5, f"BaTiO PPPM and Ewald virials were indistinguishable (delta={delta:.3g})")


def main(work):
    request = load_config(work / "case.json")
    case = Case(**request["case"])
    source = Path(request["data_root"]) / case.system
    try:
        from runtime import file_digest, input_identity
        from src.utils.op_loader import get_library_path

        verify(input_identity(case, source) == request["inputs"], "inputs changed after submitting the job")
        actual = torch.cuda.device_count()
        verify(actual == case.gpus, f"requested {case.gpus} visible GPUs/DCUs, found {actual}")
        backend = ("hip" if torch.version.hip else "cuda") if actual else "cpu"
        extension = "hip" if torch.version.hip else "cuda"
        root = Path(os.environ["MATPL_QA_CODE"])
        library_paths = (
            root / "src/feature/nep_find_neigh/findneigh.so",
            root / f"src/feature/NEP_GPU/build/{extension}/nep_gpu.so",
            get_library_path(backend, root / "src"),
        )
        libraries = {str(path.relative_to(root)): file_digest(path)
                     for path in library_paths if path.is_file()}
        runtime = {"backend": backend, "torch": str(torch.__version__), "binary_hashes": libraries,
                   "device_names": [torch.cuda.get_device_name(i) for i in range(actual)],
                   "node": os.uname().nodename}
        if case.suite in ("regression", "train", "optimizer", "lmdb"):
            values = train(case, work, source)
        else:
            _, frames = split_frames(list(read_frames(source / BATIO_DATA)))
            write_json(work / "frames.json", {"test": [frame[0] for frame in frames]})
            if case.suite == "test":
                method = case.options["kspace"]
                model = source / "nep.txt"
                values = cli_prediction(case, work, source, frames, model, method)
                if method != "ewald":
                    ewald = cli_prediction(case, work / "ewald-check", source, frames, model, "ewald")
                    if method == "pppm":
                        verify_kspace_difference(values, ewald)
                    else:
                        verify(not compare_predictions(values, ewald, 1e-12, 1e-12),
                               f"kspace={method} did not produce the Ewald result")
            else:
                values = ase_predictions(case, work, source, frames)
                if case.options.get("kspace") == "pppm":
                    ewald_case = replace(case, options={**case.options, "kspace": "ewald"})
                    ewald_work = work / "ewald-check"
                    ewald_work.mkdir()
                    ewald = ase_predictions(ewald_case, ewald_work, source, frames)
                    verify_kspace_difference(values, ewald)
        file_digest.cache_clear()
        verify(input_identity(case, source) == request["inputs"], "inputs changed while the job was running")
        verify(libraries == {str(path.relative_to(root)): file_digest(path)
                             for path in library_paths if path.is_file()},
               "compiled libraries changed while the job was running")
        write_json(work / "result.json", {"status": "PASS", "runtime": runtime, "values": values})
        return 0
    except Exception as exc:
        traceback.print_exc()
        write_json(work / "result.json", {"status": "FAIL", "message": str(exc)})
        return 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]).resolve()))
