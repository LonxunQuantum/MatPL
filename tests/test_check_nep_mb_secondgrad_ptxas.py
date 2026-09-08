"""Exercise the resource gate with complete and deliberately broken build logs."""
import subprocess
import sys
from pathlib import Path

import pytest

CHECKER = Path(__file__).with_name("check_nep_mb_secondgrad_ptxas.py")


def record(sm, threads, registers=252, stack=0, stores=0, loads=0):
    name = f"_Z23nep_mb_secondgrad_fusedILi5ELi9ELi4ELb1ELb1ELi4ELi{threads}EEv19NepMbSecondGradArgs"
    return (
        f"ptxas info    : Compiling entry function '{name}' for 'sm_{sm}'\n"
        f"ptxas info    : Function properties for {name}\n"
        f"    {stack} bytes stack frame, {stores} bytes spill stores, {loads} bytes spill loads\n"
        f"ptxas info    : Used {registers} registers, 544 bytes cmem[0]\n"
        "ptxas info    : Function properties for __internal_trig_reduction_slowpathd\n"
        "    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n"
    )


def complete_log():
    return "build noise\n" + "".join(record(sm, cta) for sm in (60, 70, 86) for cta in (32, 64))


def run_checker(tmp_path, log):
    path = tmp_path / "build.log"
    path.write_text(log)
    return subprocess.run([sys.executable, str(CHECKER), str(path)], capture_output=True, text=True)


def test_complete_resource_matrix_passes(tmp_path):
    result = run_checker(tmp_path, complete_log())
    assert result.returncode == 0, result.stderr
    assert "PASS" in result.stdout
    assert "sm_86" in result.stdout and "CTA=64" in result.stdout


@pytest.mark.parametrize("log,reason", [
    ("", "missing"),
    (complete_log().replace(record(86, 64), ""), "missing"),
    (complete_log().replace(record(60, 32), record(60, 32, registers=253)), "register"),
    (complete_log().replace(record(70, 64), record(70, 64, stores=8)), "spill"),
    (complete_log().replace(record(86, 32), record(86, 32, loads=16)), "spill"),
    (complete_log() + record(89, 32, registers=255), "register"),
    (complete_log().replace("Used 252 registers", "unavailable", 1), "incomplete"),
    (complete_log().replace("0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads", "", 1), "incomplete"),
], ids=["empty", "missing-cta", "register-limit", "spill-store", "spill-load", "extra-arch", "missing-registers", "missing-memory"])
def test_invalid_resources_fail(tmp_path, log, reason):
    result = run_checker(tmp_path, log)
    assert result.returncode != 0
    assert reason in result.stderr.lower()


def test_stack_is_reported_without_being_treated_as_spill(tmp_path):
    result = run_checker(tmp_path, complete_log().replace(record(60, 32), record(60, 32, stack=64)))
    assert result.returncode == 0, result.stderr
    assert "stack=64" in result.stdout


@pytest.mark.parametrize("stores,loads", [(8, 0), (0, 16)])
def test_helper_only_spills_fail(tmp_path, stores, loads):
    helper = ("ptxas info    : Function properties for _Z19accumulate_neighbor_helper\n"
              f"    0 bytes stack frame, {stores} bytes spill stores, {loads} bytes spill loads\n")
    log = complete_log().replace(record(70, 64), record(70, 64) + helper)
    result = run_checker(tmp_path, log)
    assert result.returncode != 0
    assert "spill" in result.stderr.lower()
    assert "accumulate_neighbor_helper" in result.stderr
    assert "sm_70" in result.stderr and "CTA=64" in result.stderr


def test_helper_stack_alone_passes(tmp_path):
    helper = ("ptxas info    : Function properties for _Z19accumulate_neighbor_helper\n"
              "    96 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n")
    result = run_checker(tmp_path, complete_log() + helper)
    assert result.returncode == 0, result.stderr
    assert "stack=96" in result.stdout


def test_incomplete_helper_fails(tmp_path):
    helper = "ptxas info    : Function properties for _Z19accumulate_neighbor_helper\n"
    result = run_checker(tmp_path, complete_log() + helper)
    assert result.returncode != 0
    assert "incomplete" in result.stderr.lower()


def test_unrelated_entry_helper_spills_are_ignored(tmp_path):
    unrelated = ("ptxas info    : Compiling entry function '_Z6legacyv' for 'sm_70'\n"
                 "ptxas info    : Function properties for _Z19accumulate_neighbor_helper\n"
                 "    96 bytes stack frame, 8 bytes spill stores, 16 bytes spill loads\n")
    result = run_checker(tmp_path, complete_log() + unrelated)
    assert result.returncode == 0, result.stderr
