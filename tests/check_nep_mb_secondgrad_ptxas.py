#!/usr/bin/env python3
"""Gate verbose ptxas build logs for the fused FP64 NEP second gradient.

Build with NVCC_APPEND_FLAGS=--ptxas-options=-v, then pass the complete log.
Missing/truncated fused entries fail closed. Other kernels are ignored.
"""
import argparse
import dataclasses
from pathlib import Path
import re

KERNEL = "nep_mb_secondgrad_fused"


@dataclasses.dataclass(frozen=True)
class ResourceRecord:
    name: str
    sm: int
    cta_threads: int
    registers: int
    stack_bytes: int
    spill_load_bytes: int
    spill_store_bytes: int


ENTRY = re.compile(r"Compiling entry function '([^']+)' for 'sm_(\d+)'")
PROPERTIES = re.compile(r"Function properties for (\S+)")
MEMORY = re.compile(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads")
REGISTERS = re.compile(r"Used (\d+) registers")
# The final integer template argument is CTA_THREADS; earlier integers include
# NMAX, NBASIS, LMAX3, and TYPE_TILE and must not be confused with it.
CTA = re.compile(r"nep_mb_secondgrad_fusedI.*Li(\d+)EEv")


def parse(log):
    records = []
    pending = None
    properties_seen = False
    memory = None
    for line in log.splitlines():
        entry = ENTRY.search(line)
        properties = PROPERTIES.search(line)
        if entry:
            if pending is not None:
                raise SystemExit(f"incomplete fused resource record: {pending[0]}")
            name, sm = entry.groups()
            if KERNEL in name:
                cta = CTA.search(name)
                if not cta:
                    raise SystemExit(f"cannot parse fused CTA template argument: {name}")
                pending = (name, int(sm), int(cta[1]))
                properties_seen, memory = False, None
        elif pending is not None:
            if properties:
                if properties_seen or properties[1] != pending[0]:
                    raise SystemExit(f"incomplete fused resource record: {pending[0]}")
                properties_seen = True
            elif match := MEMORY.search(line):
                if not properties_seen or memory is not None:
                    raise SystemExit(f"incomplete fused resource record: {pending[0]}")
                memory = tuple(map(int, match.groups()))
            elif match := REGISTERS.search(line):
                if not properties_seen or memory is None:
                    raise SystemExit(f"incomplete fused resource record: {pending[0]}")
                stack, stores, loads = memory
                records.append(ResourceRecord(*pending, int(match[1]), stack, loads, stores))
                pending = None
    if pending is not None:
        raise SystemExit(f"incomplete fused resource record: {pending[0]}")
    return records


def validate(records):
    fused = [record for record in records if KERNEL in record.name]
    required = {(sm, threads) for sm in (60, 70, 86) for threads in (32, 64)}
    present = {(record.sm, record.cta_threads) for record in fused}
    if missing := required - present:
        raise SystemExit(f"missing fused resource records: {sorted(missing)}")
    for record in fused:
        if record.spill_load_bytes or record.spill_store_bytes:
            raise SystemExit(f"{record.name}: spills detected: {record}")
        if record.registers >= 253:
            raise SystemExit(f"{record.name}: register count did not improve: {record.registers}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_log", type=Path)
    args = parser.parse_args()
    records = parse(args.build_log.read_text())
    for record in records:
        print(f"sm_{record.sm} CTA={record.cta_threads} registers={record.registers} "
              f"stack={record.stack_bytes} spill_loads={record.spill_load_bytes} "
              f"spill_stores={record.spill_store_bytes}")
    validate(records)
    print(f"PASS: {len(records)} fused resource records; no spills and registers < 253")


if __name__ == "__main__":
    main()
