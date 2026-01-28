#!/bin/bash
cd nep_find_neigh/build
cmake -Dpybind11_DIR=$(python -m pybind11 --cmakedir) .. && make
cp findneigh.* ../findneigh.so
cd ../..

