#!/bin/bash
TMP_EXTRACT_DIR=$1
A_ROOT=$2
B_ROOT=$3
VERSION=$4
CHECK_DIFF_SCRIPT="${TMP_EXTRACT_DIR}/check_diff.sh"
bash "${CHECK_DIFF_SCRIPT}" "${TMP_EXTRACT_DIR}" "${A_ROOT}" "${B_ROOT}" "${VERSION}" 
exit_code=$?
py_change=$((exit_code & 1))
nep_cpu_change=$(( (exit_code >> 1) & 1 ))
nep_change=$(( (exit_code >> 2) & 1 ))
op_change=$(( (exit_code >> 3) & 1 ))
fortran_change=$(( (exit_code >> 4) & 1 ))
lmp_change=$(( (exit_code >> 5) & 1 ))
lmpfortran_change=$(( (exit_code >> 6) & 1 ))

echo "Check completed! Summary of results:"
echo "py_change=$py_change"
echo "nep_cpu_change=$nep_cpu_change"
echo "nep_change=$nep_change"
echo "op_change=$op_change"
echo "fortran_change=$fortran_change"
echo "lmp_change=$lmp_change"
echo "lmpfortran_change=$lmpfortran_change"
echo "----------------------------------------"
