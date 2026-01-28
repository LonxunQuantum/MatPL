#!/bin/bash
rm -f bin/*.r
rm -f bin/*.x
rm -f bin/*.py
rm -rf op/build

#nep cpu
rm -rf feature/nep_find_neigh/build/*
rm feature/nep_find_neigh/findneigh.so
#nep gpu
rm -rf feature/NEP_GPU/build/*

make clean -C pre_data/gen_feature
make clean -C pre_data/fit
make clean -C pre_data/fortran_code  # for tf, will delete it
make clean -C md/fortran_code

