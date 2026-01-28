#!/bin/bash
cd NEP_GPU/build
cmake -Dpybind11_DIR=$(python -m pybind11 --cmakedir) .. && make
mv nep3_module*.so nep_gpu.so
cd ../../

