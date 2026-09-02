#!/bin/bash

set -euo pipefail

readonly CASES=(
    "16_metal"
    "AuAg"
    "LiSiC"
    "MPtraj"
    "Si-SiO2-La2O3-HfO2-TiN"
)
readonly BATCH_SIZES=(1 32)

die() {
    echo "ERROR: $*" >&2
    exit 1
}

canonical_path() {
    python3 - "$1" <<'PY'
import os
import sys

print(os.path.realpath(os.path.expanduser(sys.argv[1])))
PY
}

set_batch_size() {
    local json_file=$1
    local batch_size=$2
    python3 - "$json_file" "$batch_size" <<'PY'
import json
import os
from pathlib import Path
import sys

path = Path(sys.argv[1])
batch_size = int(sys.argv[2])
with path.open("r", encoding="utf-8") as stream:
    config = json.load(stream)

optimizer = config.get("optimizer")
if not isinstance(optimizer, dict) or "batch_size" not in optimizer:
    raise SystemExit(f"optimizer.batch_size is missing in {path}")
optimizer["batch_size"] = batch_size

temporary = path.with_name(f".{path.name}.quick-train.tmp")
with temporary.open("w", encoding="utf-8") as stream:
    json.dump(config, stream, ensure_ascii=False, indent=4)
    stream.write("\n")
os.replace(temporary, path)
PY
}

command -v python3 >/dev/null 2>&1 || die "python3 was not found"
command -v sbatch >/dev/null 2>&1 || die "sbatch was not found"

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
output_dir=$(canonical_path "${1:-${PWD}/quick_train}")
home_dir=$(canonical_path "${HOME:-/}")

[[ "$output_dir" != "/" ]] || die "refusing to use / as the output directory"
[[ "$output_dir" != "$home_dir" ]] || die "refusing to use the home directory as output"
[[ "$output_dir" != "$script_dir" ]] || die "refusing to delete the quick_test source directory"
[[ "$script_dir" != "$output_dir/"* ]] || die "refusing to delete an ancestor of quick_test"

for case in "${CASES[@]}"; do
    case_dir="$script_dir/$case"
    [[ "$output_dir" != "$case_dir" ]] || die "refusing to use source case $case as output"
    [[ "$output_dir" != "$case_dir/"* ]] || die "refusing to use a path inside source case $case as output"
    for required in nep.json run.sh batch1_epoch_train.dat batch32_epoch_train.dat; do
        [[ -f "$case_dir/$required" ]] || die "missing required file: $case_dir/$required"
    done
done

if [[ -e "$output_dir" ]]; then
    echo "Removing existing output directory: $output_dir"
    rm -rf -- "$output_dir"
fi
mkdir -p -- "$output_dir"

declare -a job_cases=()
declare -a job_batches=()
declare -a job_dirs=()

for case in "${CASES[@]}"; do
    for batch_size in "${BATCH_SIZES[@]}"; do
        job_dir="$output_dir/$case/batch$batch_size"
        mkdir -p -- "$job_dir"
        cp -a -- "$script_dir/$case/." "$job_dir/"
        rm -rf -- "$job_dir/model_record" "$job_dir/test_result"
        find "$job_dir" -maxdepth 1 -type f -name 'slurm-*.out' -delete
        set_batch_size "$job_dir/nep.json" "$batch_size"
        job_cases+=("$case")
        job_batches+=("$batch_size")
        job_dirs+=("$job_dir")
    done
done

jobs_file="$output_dir/jobs.tsv"
errors_file="$output_dir/submission_errors.log"
printf 'case\tbatch_size\tjob_id\tstatus\twork_dir\n' >"$jobs_file"
: >"$errors_file"

submitted=0
failed=0
for index in "${!job_dirs[@]}"; do
    case=${job_cases[$index]}
    batch_size=${job_batches[$index]}
    job_dir=${job_dirs[$index]}

    if submit_output=$(cd -- "$job_dir" && sbatch --parsable run.sh 2>&1); then
        job_id=$(printf '%s\n' "$submit_output" | tail -n 1)
        job_id=${job_id%%;*}
        if [[ "$job_id" =~ ^[0-9]+$ ]]; then
            printf '%s\t%s\t%s\tSUBMITTED\t%s\n' \
                "$case" "$batch_size" "$job_id" "$job_dir" >>"$jobs_file"
            echo "Submitted $case batch_size=$batch_size as job $job_id"
            ((submitted += 1))
            continue
        fi
        submit_output="unexpected sbatch output: $submit_output"
    fi

    printf '%s\t%s\t-\tSUBMIT_FAILED\t%s\n' \
        "$case" "$batch_size" "$job_dir" >>"$jobs_file"
    printf '[%s batch_size=%s]\n%s\n' \
        "$case" "$batch_size" "$submit_output" >>"$errors_file"
    echo "Failed to submit $case batch_size=$batch_size" >&2
    ((failed += 1))
done

echo
echo "Submission summary: $submitted submitted, $failed failed"
echo "Job list: $jobs_file"
echo "After all jobs finish, compare with:"
echo "python3 $script_dir/compare_quick_train.py $output_dir"

((failed == 0))
