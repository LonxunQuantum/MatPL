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
class CalleeRecord:
    name: str
    stack_bytes: int
    spill_load_bytes: int
    spill_store_bytes: int


@dataclasses.dataclass(frozen=True)
class ResourceRecord:
    name: str
    sm: int
    cta_threads: int
    registers: int
    stack_bytes: int
    spill_load_bytes: int
    spill_store_bytes: int
    callees: tuple[CalleeRecord, ...] = ()


ENTRY = re.compile(r"Compiling entry function '([^']+)' for 'sm_(\d+)'")
PROPERTIES = re.compile(r"Function properties for (\S+)")
MEMORY = re.compile(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads")
REGISTERS = re.compile(r"Used (\d+) registers")
# The final integer template argument is CTA_THREADS; earlier integers include
# NMAX, NBASIS, LMAX3, and TYPE_TILE and must not be confused with it.
CTA = re.compile(r"nep_mb_secondgrad_fusedI.*Li(\d+)EEv")

# CUDA 11.8 emits these out-of-line callees for the supported 5/9/4,
# four-/five-body specialization, for both CTA widths. Check the set rather
# than a following entry or a particular callee order: a complete log may end
# here, and a truncated log may omit whole property blocks.
REQUIRED_CALLEES = frozenset(
    [f"_ZN21nep_mb_secondgrad_opt22accumulate_cross_basisILi{i}ELi5ELi9ELi4EEEvRKNS_19AngularContributionEii" for i in range(1, 5)] +
    [f"_ZN21nep_mb_secondgrad_opt23accumulate_direct_basisILi{i}ELi5ELi9ELi4EEEvRKNS_19AngularContributionEii" for i in range(1, 5)] +
    [f"_ZN21nep_mb_secondgrad_opt24accumulate_angular_orderILi{i}ELi5ELi9ELi4ELb1ELb1ELi4EEEvRKNS_15NeighborContextIXT1_EEE" for i in range(1, 5)] +
    [f"_ZN21nep_mb_secondgrad_opt26accumulate_{body}_body_basisILb{i}ELi5ELi9ELi4EEEvRKNS_19AngularContributionEii" for body in ("four", "five") for i in (0, 1)] +
    ["__internal_trig_reduction_slowpathd"]
)


def parse(log):
    records = []
    pending = None
    properties_seen = False
    memory = None
    active_index = None
    callee = None
    for line in log.splitlines():
        entry = ENTRY.search(line)
        properties = PROPERTIES.search(line)
        if entry:
            if callee is not None:
                raise SystemExit(f"incomplete fused callee resource record: {callee}")
            active_index = None
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
                active_index = len(records) - 1
                pending = None
        elif active_index is not None:
            # ptxas emits callee properties after the entry's register line.
            # They belong to that fused entry until the next compiling entry.
            if properties:
                if callee is not None:
                    raise SystemExit(f"incomplete fused callee resource record: {callee}")
                callee = properties[1]
            elif callee is not None and (match := MEMORY.search(line)):
                stack, stores, loads = map(int, match.groups())
                record = records[active_index]
                helper = CalleeRecord(callee, stack, loads, stores)
                records[active_index] = dataclasses.replace(record, callees=record.callees + (helper,))
                callee = None
    if callee is not None:
        raise SystemExit(f"incomplete fused callee resource record: {callee}")
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
        missing_callees = REQUIRED_CALLEES - {callee.name for callee in record.callees}
        if missing_callees:
            raise SystemExit(
                f"incomplete fused callee coverage: sm_{record.sm} "
                f"CTA={record.cta_threads} {record.name}: missing {sorted(missing_callees)}")
        for callee in record.callees:
            if callee.spill_load_bytes or callee.spill_store_bytes:
                raise SystemExit(f"sm_{record.sm} CTA={record.cta_threads} {record.name}: "
                                 f"callee spills detected: {callee}")
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
        for callee in record.callees:
            print(f"  callee={callee.name} stack={callee.stack_bytes} "
                  f"spill_loads={callee.spill_load_bytes} spill_stores={callee.spill_store_bytes}")
    validate(records)
    print(f"PASS: {len(records)} fused resource records; no spills and registers < 253")


if __name__ == "__main__":
    main()
