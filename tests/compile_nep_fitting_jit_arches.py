#!/usr/bin/env python3
"""Compile the embedded NEP fitting source with NVRTC for selected SM targets."""

import argparse
import ctypes
import ctypes.util
from pathlib import Path
import re


def embedded_source(header):
    text = header.read_text()
    match = re.search(r'R"MATPL_JIT\((.*)\)MATPL_JIT";', text, re.S)
    if not match:
        raise SystemExit(f"cannot extract embedded CUDA source from {header}")
    return match.group(1).encode()


def check(nvrtc, result, operation, program=None):
    if result == 0:
        return
    log = ""
    if program:
        size = ctypes.c_size_t()
        nvrtc.nvrtcGetProgramLogSize(program, ctypes.byref(size))
        if size.value:
            buffer = ctypes.create_string_buffer(size.value)
            nvrtc.nvrtcGetProgramLog(program, buffer)
            log = buffer.value.decode(errors="replace")
    nvrtc.nvrtcGetErrorString.restype = ctypes.c_char_p
    message = nvrtc.nvrtcGetErrorString(result).decode()
    raise SystemExit(f"{operation} failed: {message}\n{log}")


def compile_one(nvrtc, source, sm, d, h, q):
    program = ctypes.c_void_p()
    check(nvrtc, nvrtc.nvrtcCreateProgram(
        ctypes.byref(program), source, b"nep_fitting_jit.cu",
        0, None, None), "nvrtcCreateProgram")
    options = [b"--std=c++14", f"--gpu-architecture=sm_{sm}".encode(),
               f"-DJIT_D={d}".encode(), f"-DJIT_H={h}".encode(),
               f"-DJIT_Q={q}".encode()]
    option_array = (ctypes.c_char_p * len(options))(*options)
    try:
        check(nvrtc, nvrtc.nvrtcCompileProgram(
            program, len(options), option_array), "nvrtcCompileProgram", program)
        size = ctypes.c_size_t()
        check(nvrtc, nvrtc.nvrtcGetCUBINSize(program, ctypes.byref(size)),
              "nvrtcGetCUBINSize", program)
        cubin = ctypes.create_string_buffer(size.value)
        check(nvrtc, nvrtc.nvrtcGetCUBIN(program, cubin),
              "nvrtcGetCUBIN", program)
        return cubin.raw
    finally:
        nvrtc.nvrtcDestroyProgram(ctypes.byref(program))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--header", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sm", nargs="+", type=int, default=[60, 70, 86, 89])
    parser.add_argument("--d", type=int, default=96)
    parser.add_argument("--h", type=int, default=100)
    parser.add_argument("--q", type=int, default=2)
    args = parser.parse_args()
    library = ctypes.util.find_library("nvrtc") or "libnvrtc.so"
    nvrtc = ctypes.CDLL(library)
    nvrtc.nvrtcCreateProgram.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_char_p,
        ctypes.c_int, ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_char_p),
    ]
    nvrtc.nvrtcCompileProgram.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    nvrtc.nvrtcGetProgramLogSize.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    nvrtc.nvrtcGetProgramLog.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    nvrtc.nvrtcGetCUBINSize.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    nvrtc.nvrtcGetCUBIN.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    nvrtc.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    nvrtc.nvrtcGetErrorString.restype = ctypes.c_char_p
    source = embedded_source(args.header)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for sm in args.sm:
        binary = compile_one(nvrtc, source, sm, args.d, args.h, args.q)
        output = args.output_dir / f"nepfit-nvrtc-sm{sm}-d{args.d}-h{args.h}-q{args.q}.cubin"
        output.write_bytes(binary)
        print(f"{output} {len(binary)} bytes")


if __name__ == "__main__":
    main()
