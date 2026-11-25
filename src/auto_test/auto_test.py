#!/usr/bin/env python3
"""Run the BaTiO functional matrix or compare six-system training against a base."""

import argparse
import concurrent.futures
import datetime
import math
import re
import shlex
import subprocess
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path

# Running the CLI directly must leave the source checkout free of bytecode.
sys.dont_write_bytecode = True

from cases import CASES, DEFAULT_CODE, REFERENCE, SUITES, collect_cases, load_config, select_cases
from results import compare_metrics, compare_predictions, write_json
from runtime import code_identity, input_identity


SOURCE_DIR = Path(__file__).resolve().parent
BUNDLED_MANIFEST = SOURCE_DIR / "reference" / "manifest.json"
DATA = SOURCE_DIR / "quick_test"
MATRIX = collect_cases()
VERSIONS = 1
SLURM_RETRY_DELAYS = (5, 10, 20)
SITE_FIELDS = ("partition", "cpu_partition", "gres", "time", "jobs",
               "conda_sh", "conda_env", "cuda_module", "code")


def csv(values):
    return [item for value in values for item in value.split(",") if item]


def selection_arguments(parser):
    parser.add_argument("--profile", choices=["smoke", "full"])
    parser.add_argument("--suite", action="append", default=[], help="comma-separated test groups")
    parser.add_argument("--case", action="append", default=[], help="comma-separated system names")
    parser.add_argument("--variant", action="append", default=[], help="names or full IDs from list")


def workspace_argument(parser):
    parser.add_argument("--work-dir", "--work_dir", type=Path, required=True, metavar="PATH",
                        help="required path outside Git checkouts for runs, candidates and baselines")


def require_outside_git(path):
    if any((ancestor / ".git").exists() for ancestor in (path, *path.parents)):
        raise ValueError(f"work and artifact paths must be outside Git checkouts: {path}")


def resolve_work_dir(path):
    path = path.expanduser().resolve()
    require_outside_git(path)
    if path.exists() and not path.is_dir():
        raise ValueError(f"--work-dir is not a directory: {path}")
    return path


def workspace_path(work_dir, *parts):
    path = work_dir.joinpath(*parts).resolve()
    try:
        path.relative_to(work_dir)
    except ValueError:
        raise ValueError(f"artifact path resolves outside the work directory: {path}") from None
    require_outside_git(path)
    return path


def read_site_config(path):
    path = path.expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"site config is not a file: {path}")
    names = [f"MATPL_QA_{field.upper()}" for field in SITE_FIELDS]
    script = r"""
config=$1
shift
unset "$@"
source "$config" >&2
for name in "$@"; do
    if [[ -v "$name" ]]; then
        printf '%s\0%s\0' "$name" "${!name}"
    fi
done
"""
    process = subprocess.run(["bash", "-e", "-c", script, "matpl-qa-site", str(path), *names],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode:
        detail = process.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"could not load site config {path}: {detail or process.returncode}")
    values = process.stdout.decode("utf-8").split("\0")[:-1]
    return {key[len("MATPL_QA_"):].lower(): value for key, value in zip(values[::2], values[1::2])}


def execution_arguments(parser, site):
    workspace_argument(parser)
    parser.add_argument("--site-config", type=Path, help="Bash file with MATPL_QA_* site defaults")
    parser.add_argument("--data-root", type=Path, default=DATA,
                        help="test input directory; defaults to the bundled quick_test")
    parser.add_argument("--code", type=Path, default=site.get("code", DEFAULT_CODE),
                        help="MatPL checkout containing env.sh")
    parser.add_argument("--executor", choices=["slurm", "local"], default="slurm")
    parser.add_argument("--partition", default=site.get("partition", "3090,q4"))
    parser.add_argument("--cpu-partition", default=site.get("cpu_partition"), help="defaults to --partition")
    parser.add_argument("--gres", default=site.get("gres", "gpu"), help="Slurm GPU resource name, e.g. gpu or dcu")
    parser.add_argument("--time", default=site.get("time", "00:30:00"), help="Slurm time limit per case")
    parser.add_argument("--jobs", type=int, default=site.get("jobs", 4), help="maximum active Slurm submissions")
    parser.add_argument("--conda-sh", default=site.get("conda_sh",
                        "/data/home/wuxingxing/anaconda3/etc/profile.d/conda.sh"))
    parser.add_argument("--conda-env", default=site.get("conda_env", "matpl-2026.3"))
    parser.add_argument("--cuda-module", default=site.get("cuda_module", "cuda/11.8-share"),
                        help="set empty for preloaded/HIP environment")
    parser.add_argument("--dry-run", action="store_true", help="print job commands without creating files")


def comparison_arguments(parser):
    parser.add_argument("--base", help="reference ID; defaults to the workspace base or the bundled base")
    parser.add_argument("--atol", type=float, default=1e-9)
    parser.add_argument("--rtol", type=float, default=1e-10)


def make_parser(site=None):
    site = {} if site is None else site
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("list", help="show the selected cases")
    selection_arguments(listing)
    run = sub.add_parser("run", help="run functional tests and compare selected regression cases")
    selection_arguments(run)
    execution_arguments(run, site)
    comparison_arguments(run)
    compare = sub.add_parser("compare", help="compare a finished run's training artifacts without new jobs")
    compare.add_argument("run_id")
    workspace_argument(compare)
    comparison_arguments(compare)
    base = sub.add_parser("base", help="create and publish reference results")
    actions = base.add_subparsers(dest="action", required=True)
    build = actions.add_parser("build", help="run selected cases to create a candidate base")
    selection_arguments(build)
    execution_arguments(build, site)
    build.add_argument("--reference", default=REFERENCE, help="clean reference checkout commit SHA")
    promote = actions.add_parser("promote", help="publish a successful candidate")
    promote.add_argument("candidate", help="ID printed by base build")
    workspace_argument(promote)
    return parser


def choose(args):
    suites, systems, variants = csv(args.suite), csv(args.case), csv(args.variant)
    if set(suites) - set(SUITES) or set(systems) - set(CASES):
        raise ValueError(f"unknown suite or system: {suites}, {systems}")
    if args.command == "base" and not suites and not variants:
        suites = ["regression"]
    profile = args.profile or (
        "full" if args.command in ("list", "base") or suites or systems or variants else "smoke"
    )
    selected = select_cases(MATRIX, profile, suites, systems, variants)
    matches = {case.variant for case in selected} | {case.key for case in selected}
    if not selected or set(variants) - matches:
        raise ValueError(f"no matching cases or unknown variants: {set(variants) - matches}")
    return selected


def job_command(args, case, work):
    script = SOURCE_DIR / "run_job.sh"
    job_args = [str(work), str(args.code), str(case.gpus), str(case.cpus),
                args.conda_sh, args.conda_env, args.cuda_module, str(SOURCE_DIR)]
    if args.executor == "local":
        return ["bash", "-l", str(script), *job_args]
    partition = args.partition if case.gpus else args.cpu_partition or args.partition
    resources = [f"--gres={args.gres}:{case.gpus}"] if case.gpus else []
    return [
        "sbatch", "--parsable", "--wait", "--nodes=1", "--ntasks=1",
        f"--partition={partition}", f"--cpus-per-task={case.cpus}", f"--time={args.time}",
        f"--output={work / 'slurm-%j.log'}", f"--job-name=nepqa-{case.variant}",
        *resources, str(script), *job_args,
    ]


def run_job(args, case, work, inputs):
    work.mkdir(parents=True)
    write_json(work / "case.json", {"case": asdict(case), "data_root": str(args.data_root),
                                   "inputs": inputs})
    path = work / "result.json"
    submit_log = work / "submit.log"
    command = job_command(args, case, work)
    delays = (*SLURM_RETRY_DELAYS, None) if args.executor == "slurm" else (None,)
    with submit_log.open("w+", encoding="utf-8", errors="replace") as log:
        for attempt, delay in enumerate(delays, 1):
            start = log.tell()
            process = subprocess.run(command, cwd=work, stdout=log, stderr=subprocess.STDOUT)
            log.seek(start)
            output = log.read()
            job_id = re.search(r"^(\d+)(?:;\S+)?[ \t]*$", output, re.MULTILINE)
            errors = [line for line in output.splitlines() if line.startswith("sbatch: error:")]
            rejected = (
                args.executor == "slurm" and process.returncode and errors and not job_id
                and not path.exists() and not any(work.glob("slurm-*.log"))
            )
            balance_lookup_failed = (
                rejected
                and "sbatch: error: Failed to query the balance." in output
                and "sbatch: error: Batch job submission failed:" in output
            )
            if not balance_lookup_failed or delay is None:
                break
            log.write(f"\nRetrying Slurm submission in {delay}s (attempt {attempt + 1}/{len(delays)})\n")
            log.flush()
            time.sleep(delay)

    if path.is_file():
        result = load_config(path)
    elif rejected:
        result = {"status": "FAIL", "message": f"Slurm submission failed after {attempt} attempt(s): "
                  f"{errors[0]} (see {submit_log})"}
    else:
        job_log = work / f"slurm-{job_id[1]}.log" if job_id else submit_log
        result = {"status": "FAIL", "message": f"job did not write result.json "
                  f"(exit={process.returncode}; see {job_log})"}
    if process.returncode and result["status"] == "PASS":
        result = {"status": "FAIL", "message": f"job exited {process.returncode}"}
    if attempt > 1:
        result["submission_attempts"] = attempt
    return result


def current_base_id(work_dir):
    pointer = workspace_path(work_dir, "baselines", "current.json")
    if pointer.is_file():
        return load_config(pointer)["id"]
    return load_config(BUNDLED_MANIFEST)["id"] if BUNDLED_MANIFEST.is_file() else None


def base_index(work_dir, base_id=None):
    if base_id is None:
        base_id = current_base_id(work_dir)
        if base_id is None:
            return {}
    if Path(base_id).name != base_id or base_id in (".", ".."):
        raise ValueError(f"invalid base ID: {base_id}")
    if BUNDLED_MANIFEST.is_file():
        bundled = load_config(BUNDLED_MANIFEST)
        if base_id == bundled["id"]:
            return bundled["entries"]
    return load_config(workspace_path(work_dir, "baselines", base_id, "manifest.json"))["entries"]


def promote_base(work_dir, candidate):
    import fcntl

    base = workspace_path(work_dir, "baselines")
    base.mkdir(parents=True, exist_ok=True)
    with workspace_path(work_dir, "baselines", ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return publish_base(work_dir, candidate)


def compare_values(actual, expected, atol, rtol):
    if "epochs" not in actual:
        return compare_predictions(actual, expected, atol, rtol)
    differences = compare_metrics(actual["epochs"], expected["epochs"], atol, rtol)
    differences.extend(f"validation {difference}" for difference in compare_metrics(
        actual.get("validation_epochs", {}), expected.get("validation_epochs", {}), atol, rtol))
    if actual["optimizer_updates"] != expected["optimizer_updates"]:
        differences.append(f"optimizer updates differ: {actual['optimizer_updates']} "
                           f"vs {expected['optimizer_updates']}")
    return differences


def assess_difference(item, result, base, atol, rtol):
    if item["inputs"] != base["inputs"]:
        item.update(status="BLOCKED", message="inputs differ from the base")
        return
    actual, expected = result["runtime"], base["runtime"]
    fields = ("backend", "torch", "device_names")
    environment = {field: {"baseline": expected[field], "actual": actual[field]}
                   for field in fields if actual[field] != expected[field]}
    if environment:
        item["environment_differences"] = environment
    differences = compare_values(result["values"], base["values"], atol, rtol)
    if differences:
        item.update(status="FAIL", differences=differences)


def report_environment(key, item):
    fields = item.get("environment_differences")
    if fields:
        print(f"NOTE {key}: environment differs from baseline ({', '.join(fields)})", file=sys.stderr)


def check_tolerance(args):
    if not all(math.isfinite(value) and value >= 0 for value in (args.atol, args.rtol)):
        raise ValueError("tolerances must be finite and nonnegative")


def compare_saved_run(args):
    check_tolerance(args)
    if Path(args.run_id).name != args.run_id or args.run_id in (".", ".."):
        raise ValueError(f"invalid run ID: {args.run_id}")
    run = workspace_path(args.work_dir, "runs", args.run_id)
    summary = load_config(run / "summary.json")
    if not summary["code_unchanged"]:
        raise ValueError("the MatPL checkout changed in the saved run")
    base = args.base or current_base_id(args.work_dir)
    entries = base_index(args.work_dir, base) if base else {}
    cases = {}
    for key, item in summary["cases"].items():
        if not key.startswith("regression/"):
            continue
        if key not in entries:
            cases[key] = {**item, "status": "NO_BASE", "message": "no published base entry"}
            continue
        result_file = run / key / "result.json"
        if not result_file.is_file():
            cases[key] = {**item, "status": "BLOCKED", "message": "missing saved result.json"}
            continue
        result = load_config(result_file)
        cases[key] = {**{k: v for k, v in result.items() if k != "values"}, "inputs": item["inputs"]}
        if result["status"] == "PASS":
            assess_difference(cases[key], result, entries[key], args.atol, args.rtol)
    if not cases:
        raise ValueError("run has no six-system regression cases")
    passed = all(item["status"] == "PASS" for item in cases.values())
    report = run / f"comparison-{uuid.uuid4().hex[:8]}.json"
    write_json(report, {"base_id": base, "atol": args.atol, "rtol": args.rtol,
                        "passed": passed, "cases": cases})
    for key, item in cases.items():
        print(f"{item['status']} {key} {item.get('message', '')}")
        report_environment(key, item)
        for difference in item.get("differences", []):
            print(f"  {difference}")
    print(f"{sum(item['status'] == 'PASS' for item in cases.values())}/{len(cases)} passed  {report}")
    return 0 if passed else 1


def publish_base(work_dir, candidate):
    if Path(candidate).name != candidate or candidate in (".", ".."):
        raise ValueError(f"invalid candidate ID: {candidate}")
    run = workspace_path(work_dir, "candidates", candidate)
    summary = load_config(run / "summary.json")
    if summary["version"] != VERSIONS or not summary["passed"] or not summary["cases"]:
        raise ValueError("candidate is incomplete or has failed cases")
    current = current_base_id(work_dir)
    if current == candidate:
        print(f"Base {candidate} is already current")
        return 0
    dest = workspace_path(work_dir, "baselines", candidate)
    parent = summary.get("base_parent")
    entries = base_index(work_dir, current)
    if current != parent:
        parent_entries = base_index(work_dir, parent) if parent else {}
        overlapping = [key for key in summary["cases"]
                       if entries.get(key) != parent_entries.get(key)]
        if overlapping:
            raise ValueError(f"base changed after this candidate started: {', '.join(overlapping)}")
    for key, item in summary["cases"].items():
        result = load_config(run / key / "result.json")
        if result["status"] != "PASS":
            raise ValueError(f"candidate case did not pass: {key}")
        previous = entries.get(key)
        if previous and previous["inputs"] == item["inputs"]:
            differences = compare_values(result["values"], previous["values"], 1e-9, 0.0)
            for difference in differences:
                print(f"{key} {difference}")
        entries[key] = {"candidate": candidate, "code": summary["code"], "inputs": item["inputs"],
                        "runtime": result["runtime"], "values": result["values"]}
    dest.mkdir(parents=True, exist_ok=True)
    manifest = {"version": VERSIONS, "id": candidate, "parent": current, "entries": entries}
    if (dest / "manifest.json").is_file():
        if load_config(dest / "manifest.json") != manifest:
            raise ValueError(f"reference {candidate} is already published against a different base")
    else:
        write_json(dest / "manifest.json", manifest)
    write_json(workspace_path(work_dir, "baselines", "current.json"), {"id": candidate})
    print(f"Published base {candidate} with {len(entries)} cases")
    return 0


def compare_functional(results, summary):
    def check(candidate, reference, properties=None):
        if candidate not in results or reference not in results:
            return
        actual, expected = results[candidate], results[reference]
        if actual["status"] != "PASS":
            return
        if expected["status"] != "PASS":
            summary[candidate].update(status="FAIL", message=f"comparison case failed: {reference}")
            return
        actual, expected = actual["values"], expected["values"]
        if properties:
            actual = {key: actual[key] for key in properties}
            expected = {key: expected[key] for key in properties}
        differences = compare_predictions(actual, expected, 5e-5, 1e-6)
        if differences:
            summary[candidate].update(status="FAIL", differences=differences,
                                      message=f"comparison against {reference} failed")

    for method in ("ewald", "pppm"):
        for other, reference in (("cpu4", "cpu1"), ("gpu2", "gpu1")):
            check(f"test/BaTiO/{other}-{method}", f"test/BaTiO/{reference}-{method}")
    for method in ("default", "invalid"):
        check(f"test/BaTiO/cpu1-{method}", "test/BaTiO/cpu1-ewald")
    props = ("frame_ids", "natoms", "energy", "atomic_energy", "force", "virial")
    for device in ("cpu", "gpu"):
        check(f"ase/BaTiO/{device}-calculator", f"test/BaTiO/{device}1-ewald", props)
        for method in ("ewald", "pppm"):
            check(f"ase/BaTiO/{device}-{method}", f"test/BaTiO/{device}1-{method}")


def execute(args, selected):
    building = args.command == "base"
    args.code = args.code.resolve(strict=True)
    args.data_root = args.data_root.resolve(strict=True)
    if args.jobs <= 0 or not (args.code / "env.sh").is_file():
        raise ValueError("--jobs must be positive and the checkout must contain env.sh")
    if args.executor == "local" and args.jobs != 1:
        raise ValueError("use --jobs 1 inside an allocated node when using --executor local")
    if not building:
        check_tolerance(args)
    identity = code_identity(args.code)
    if building and (not re.fullmatch(r"[0-9a-fA-F]{7,40}", args.reference)
                     or not identity["commit"].startswith(args.reference.lower()) or identity["status"]):
        raise ValueError(f"base requires a clean checkout at {args.reference}; found {identity}")
    base_parent = current_base_id(args.work_dir)
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    root = workspace_path(args.work_dir, "candidates" if building else "runs", run_id)
    if args.dry_run:
        for case in selected:
            print(case.key, shlex.join(job_command(args, case, root / case.key)))
        return 0
    base_id = None if building else args.base or base_parent
    base_entries = base_index(args.work_dir, base_id) if base_id else {}
    summary, results, pending = {}, {}, {}
    root.mkdir(parents=True)
    print(f"{'Candidate' if building else 'Run'} {run_id}  {root}", flush=True)
    if base_id:
        print(f"Base {base_id}", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for case in selected:
            try:
                inputs = input_identity(case, args.data_root / case.system)
                summary[case.key] = {"inputs": inputs}
                base_entry = base_entries.get(case.key)
                if not building:
                    if case.suite == "regression" and base_entry is None:
                        summary[case.key].update(status="NO_BASE", message="no published base entry")
                        continue
                    if base_entry is not None and base_entry["inputs"] != inputs:
                        summary[case.key].update(status="BLOCKED", message="inputs differ from the base")
                        continue
                pending[pool.submit(run_job, args, case, root / case.key, inputs)] = case
            except (OSError, ValueError) as exc:
                summary[case.key] = {"status": "BLOCKED", "message": str(exc)}
        for future in concurrent.futures.as_completed(pending):
            case = pending[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"status": "FAIL", "message": str(exc)}
            results[case.key] = result
            item = summary[case.key]
            item.update({key: value for key, value in result.items() if key != "values"})
            print(f"{result['status']} {case.key} {result.get('message', '')}", flush=True)
    if not building:
        for key, result in results.items():
            if key not in base_entries or result["status"] != "PASS":
                continue
            assess_difference(summary[key], result, base_entries[key], args.atol, args.rtol)
    compare_functional(results, summary)
    unchanged = code_identity(args.code) == identity
    if not unchanged:
        print("FAIL MatPL checkout changed while tests were running", file=sys.stderr)
    passed = unchanged and all(item["status"] == "PASS" for item in summary.values())
    write_json(root / "summary.json", {"version": VERSIONS, "id": run_id, "code": identity,
                                      "base_parent": base_parent, "base_id": base_id,
                                      "code_unchanged": unchanged,
                                      "passed": passed, "cases": summary})
    for key, item in summary.items():
        report_environment(key, item)
        if item["status"] != "PASS":
            print(f"{item['status']} {key} {item.get('message', '')}", file=sys.stderr)
            for difference in item.get("differences", []):
                print(f"  {difference}", file=sys.stderr)
    print(f"{sum(item['status'] == 'PASS' for item in summary.values())}/{len(summary)} passed"
          f"  {root / 'summary.json'}")
    return 0 if passed else 1


def main():
    args = make_parser().parse_args()
    try:
        if getattr(args, "site_config", None) is not None:
            args = make_parser(read_site_config(args.site_config)).parse_args()
        if args.command != "list":
            args.work_dir = resolve_work_dir(args.work_dir)
        if args.command == "compare":
            return compare_saved_run(args)
        if args.command == "base" and args.action == "promote":
            return promote_base(args.work_dir, args.candidate)
        selected = choose(args)
        if args.command == "list":
            for case in selected:
                print(f"{case.key:<60} cpus={case.cpus} gpus={case.gpus}")
            return 0
        return execute(args, selected)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
