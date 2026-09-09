#!/usr/bin/env python3
"""Report and validate CUDA resources for NEP fitting JIT CUBIN files."""

import argparse
import json
from pathlib import Path
import re
import subprocess


REQUIRED = {
    "fitting_atoms_forward",
    "fitting_atoms_backward",
    "fitting_parameter_partials",
    "fitting_parameter_reduce",
}
FUNCTION = re.compile(r"^ Function (\S+):$")
RESOURCE = re.compile(
    r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)"
)


def parse_resource_usage(text):
    records = {}
    name = None
    for line in text.splitlines():
        match = FUNCTION.match(line)
        if match:
            name = match.group(1)
            continue
        if name is not None and (match := RESOURCE.search(line)):
            registers, stack, shared, local = map(int, match.groups())
            records[name] = {
                "registers": registers,
                "stack_bytes": stack,
                "shared_bytes": shared,
                "local_bytes": local,
                "spill_bytes": local,
            }
            name = None
    return records


def inspect(path):
    completed = subprocess.run(
        ["cuobjdump", "--dump-resource-usage", str(path)],
        check=True, text=True, capture_output=True,
    )
    records = parse_resource_usage(completed.stdout)
    missing = REQUIRED - records.keys()
    if missing:
        raise SystemExit(f"{path}: missing JIT kernels: {sorted(missing)}")
    spilling = {name: record for name, record in records.items()
                if name in REQUIRED and record["local_bytes"] != 0}
    if spilling:
        raise SystemExit(f"{path}: local/spill memory detected: {spilling}")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cubins", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {str(path): inspect(path) for path in args.cubins}
    payload = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
