#!/bin/bash
TMP_EXTRACT_DIR=$1
A_ROOT=$2
B_ROOT=$3
VERSION=$4
CPU_ONLY=$5

# Check necessary directories and files
if [ ! -d "$A_ROOT" ]; then
    echo "Error: Patch directory ${A_ROOT} does not exist"
    rm -rf "${TMP_EXTRACT_DIR}"
    exit 1
fi
if [ ! -d "$B_ROOT" ]; then
    echo "Error: Target directory ${B_ROOT} does not exist"
    rm -rf "${TMP_EXTRACT_DIR}"
    exit 1
fi

py_change=0
nep_cpu_change=0
nep_change=0
op_change=0
lmp_change=0
fortran_change=0

echo "Starting directory comparison..."
echo "A Directory: $A_ROOT"
echo "B Directory: $B_ROOT"
echo "----------------------------------------"

# 1. Check all .py files, excluding example and .git directories
echo "1. Checking all .py files (excluding example and .git directories)..."
echo "Python file differences:"
tmp_a=$(mktemp)
tmp_b=$(mktemp)
find "$A_ROOT" -type f -name "*.py" -not -path "*/example/*" -not -path "*/.git/*" > "$tmp_a"
find "$B_ROOT" -type f -name "*.py" -not -path "*/example/*" -not -path "*/.git/*" > "$tmp_b"

while IFS= read -r file_a; do
    rel_path=${file_a#$A_ROOT/}
    file_b="$B_ROOT/$rel_path"
    if [ -f "$file_b" ]; then
        if ! diff -q "$file_a" "$file_b" > /dev/null 2>&1; then
            echo " Modified: $rel_path"
            py_change=1
        fi
    else
        echo " Added: $rel_path"
        py_change=1
    fi
done < "$tmp_a"

while IFS= read -r file_b; do
    rel_path=${file_b#$B_ROOT/}
    file_a="$A_ROOT/$rel_path"
    if [ ! -f "$file_a" ]; then
        echo " Deleted: $rel_path"
        py_change=1
    fi
done < "$tmp_b"
rm -f "$tmp_a" "$tmp_b"

if [ $py_change -eq 0 ]; then
    echo " No changes in Python files"
else
    echo " Changes detected in Python files"
fi
echo "----------------------------------------"

# 2. Check src/feature/nep_find_neigh directory (excluding build)
echo "2. Checking src/feature/nep_find_neigh directory..."
nep_cpu_dir_A="$A_ROOT/src/feature/nep_find_neigh"
nep_cpu_dir_B="$B_ROOT/src/feature/nep_find_neigh"
if [ -d "$nep_cpu_dir_A" ] && [ -d "$nep_cpu_dir_B" ]; then
    if diff -rq --exclude="build" --exclude="*.so" "$nep_cpu_dir_A" "$nep_cpu_dir_B" > /dev/null 2>&1; then
        echo " No changes in nep_find_neigh directory"
    else
        nep_cpu_change=1
        echo " Changes detected in nep_find_neigh directory"
    fi
else
    if [ ! -d "$nep_cpu_dir_A" ] && [ -d "$nep_cpu_dir_B" ]; then
        nep_cpu_change=1
        echo " nep_find_neigh directory deleted"
    elif [ -d "$nep_cpu_dir_A" ] && [ ! -d "$nep_cpu_dir_B" ]; then
        nep_cpu_change=1
        echo " nep_find_neigh directory added"
    else
        echo " nep_find_neigh directory does not exist in A or B"
    fi
fi
echo "----------------------------------------"

# 3. Check src/feature/NEP_GPU directory (excluding build)
echo "3. Checking src/feature/NEP_GPU directory..."
nep_gpu_dir_A="$A_ROOT/src/feature/NEP_GPU"
nep_gpu_dir_B="$B_ROOT/src/feature/NEP_GPU"
if [ -d "$nep_gpu_dir_A" ] && [ -d "$nep_gpu_dir_B" ]; then
    if diff -rq --exclude="build" "$nep_gpu_dir_A" "$nep_gpu_dir_B" > /dev/null 2>&1; then
        echo " No changes in NEP_GPU directory"
    else
        nep_change=1
        echo " Changes detected in NEP_GPU directory"
    fi
else
    if [ ! -d "$nep_gpu_dir_A" ] && [ -d "$nep_gpu_dir_B" ]; then
        nep_change=1
        echo " NEP_GPU directory deleted"
    elif [ -d "$nep_gpu_dir_A" ] && [ ! -d "$nep_gpu_dir_B" ]; then
        nep_change=1
        echo " NEP_GPU directory added"
    else
        echo " NEP_GPU directory does not exist in A or B"
    fi
fi
echo "----------------------------------------"

# 4. Check src/op directory (excluding build)
echo "4. Checking src/op directory..."
op_dir_A="$A_ROOT/src/op"
op_dir_B="$B_ROOT/src/op"
if [ -d "$op_dir_A" ] && [ -d "$op_dir_B" ]; then
    if diff -rq --exclude="build" "$op_dir_A" "$op_dir_B" > /dev/null 2>&1; then
        echo " No changes in op directory"
    else
        op_change=1
        echo " Changes detected in op directory"
    fi
else
    if [ ! -d "$op_dir_A" ] && [ -d "$op_dir_B" ]; then
        op_change=1
        echo " op directory deleted"
    elif [ -d "$op_dir_A" ] && [ ! -d "$op_dir_B" ]; then
        op_change=1
        echo " op directory added"
    else
        echo " op directory does not exist in A or B"
    fi
fi
echo "----------------------------------------"

# 5. Check src/pre_data directory and all .f90 files in it and its subdirectories
echo "5. Checking src/pre_data directory and all .f90 files..."
pre_data_dir_A="$A_ROOT/src/pre_data"
pre_data_dir_B="$B_ROOT/src/pre_data"

if [ -d "$pre_data_dir_A" ] && [ -d "$pre_data_dir_B" ]; then
    tmp_f90_a=$(mktemp)
    tmp_f90_b=$(mktemp)
    find "$pre_data_dir_A" -type f -name "*.f90" > "$tmp_f90_a"
    find "$pre_data_dir_B" -type f -name "*.f90" > "$tmp_f90_b"
    
    echo "Fortran (.f90) file differences in src/pre_data:"
    while IFS= read -r file_a; do
        rel_path=${file_a#$A_ROOT/}
        file_b="$B_ROOT/$rel_path"
        if [ -f "$file_b" ]; then
            if ! diff -q "$file_a" "$file_b" > /dev/null 2>&1; then
                echo " Modified: $rel_path"
                fortran_change=1
            fi
        else
            echo " Added: $rel_path"
            fortran_change=1
        fi
    done < "$tmp_f90_a"
    
    while IFS= read -r file_b; do
        rel_path=${file_b#$B_ROOT/}
        file_a="$A_ROOT/$rel_path"
        if [ ! -f "$file_a" ]; then
            echo " Deleted: $rel_path"
            fortran_change=1
        fi
    done < "$tmp_f90_b"
    
    rm -f "$tmp_f90_a" "$tmp_f90_b"
    
    if [ $fortran_change -eq 0 ]; then
        echo " No changes in Fortran (.f90) files in src/pre_data"
    else
        echo " Changes detected in Fortran (.f90) files in src/pre_data"
    fi
elif [ ! -d "$pre_data_dir_A" ] && [ -d "$pre_data_dir_B" ]; then
    fortran_change=1
    echo " src/pre_data directory deleted"
elif [ -d "$pre_data_dir_A" ] && [ ! -d "$pre_data_dir_B" ]; then
    fortran_change=1
    echo " src/pre_data directory added"
else
    echo " src/pre_data directory does not exist in A or B"
fi
echo "----------------------------------------"

echo "Checking src/md directory and all .f90 files..."
md_dir_A="$A_ROOT/src/md"
md_dir_B="$B_ROOT/src/md"

if [ -d "$md_dir_A" ] && [ -d "$md_dir_B" ]; then
    tmp_f90_md_a=$(mktemp)
    tmp_f90_md_b=$(mktemp)
    find "$md_dir_A" -type f -name "*.f90" > "$tmp_f90_md_a"
    find "$md_dir_B" -type f -name "*.f90" > "$tmp_f90_md_b"
    
    echo "Fortran (.f90) file differences in src/md:"
    while IFS= read -r file_a; do
        rel_path=${file_a#$A_ROOT/}
        file_b="$B_ROOT/$rel_path"
        if [ -f "$file_b" ]; then
            if ! diff -q "$file_a" "$file_b" > /dev/null 2>&1; then
                echo " Modified: $rel_path"
                fortran_change=1
            fi
        else
            echo " Added: $rel_path"
            fortran_change=1
        fi
    done < "$tmp_f90_md_a"
    
    while IFS= read -r file_b; do
        rel_path=${file_b#$B_ROOT/}
        file_a="$A_ROOT/$rel_path"
        if [ ! -f "$file_a" ]; then
            echo " Deleted: $rel_path"
            fortran_change=1
        fi
    done < "$tmp_f90_md_b"
    
    rm -f "$tmp_f90_md_a" "$tmp_f90_md_b"
    
    if [ $fortran_change -eq 0 ]; then
        echo " No changes in Fortran (.f90) files in src/md"
    else
        echo " Changes detected in Fortran (.f90) files in src/md"
    fi
elif [ ! -d "$md_dir_A" ] && [ -d "$md_dir_B" ]; then
    fortran_change=1
    echo " src/md directory deleted"
elif [ -d "$md_dir_A" ] && [ ! -d "$md_dir_B" ]; then
    fortran_change=1
    echo " src/md directory added"
else
    echo " src/md directory does not exist in A or B"
fi
echo "----------------------------------------"

# 6. Check lammps (new structure for 2026.3)
echo "6. Checking lammps (lmp_nepkokkos_cmake) related files and directories..."
LMP_BASE_A="$A_ROOT/src/lmps/lammps-${VERSION}/lmp_nepkokkos_cmake"
LMP_BASE_B="$B_ROOT/src/lmps/lammps-${VERSION}/lmp_nepkokkos_cmake"

if [ -d "$LMP_BASE_A" ] && [ -d "$LMP_BASE_B" ]; then
    # 1. kknep-patch.sh
    echo "  Checking kknep-patch.sh..."
    KKNEP_A="${LMP_BASE_A}/kknep-patch.sh"
    KKNEP_B="${LMP_BASE_B}/kknep-patch.sh"
    if [ -f "$KKNEP_A" ] && [ -f "$KKNEP_B" ]; then
        if ! diff -q "$KKNEP_A" "$KKNEP_B" > /dev/null 2>&1; then
            lmp_change=1
            echo "   Modified: kknep-patch.sh"
        fi
    elif [ -f "$KKNEP_A" ] && [ ! -f "$KKNEP_B" ]; then
        lmp_change=1
        echo "   Added: kknep-patch.sh"
    elif [ ! -f "$KKNEP_A" ] && [ -f "$KKNEP_B" ]; then
        lmp_change=1
        echo "   Deleted: kknep-patch.sh"
    fi

    # 2. KOKKOS/*.cpp or *.h
    echo "  Checking KOKKOS/*.cpp *.h files..."
    tmp_kok_a=$(mktemp)
    tmp_kok_b=$(mktemp)
    find "${LMP_BASE_A}/KOKKOS" -type f \( -name "*.cpp" -o -name "*.h" \) 2>/dev/null > "$tmp_kok_a" || true
    find "${LMP_BASE_B}/KOKKOS" -type f \( -name "*.cpp" -o -name "*.h" \) 2>/dev/null > "$tmp_kok_b" || true

    while IFS= read -r file_a; do
        rel_path=${file_a#$A_ROOT/}
        file_b="$B_ROOT/$rel_path"
        if [ -f "$file_b" ]; then
            if ! diff -q "$file_a" "$file_b" > /dev/null 2>&1; then
                echo "   Modified: $rel_path"
                lmp_change=1
            fi
        else
            echo "   Added: $rel_path"
            lmp_change=1
        fi
    done < "$tmp_kok_a" 2>/dev/null || true

    while IFS= read -r file_b; do
        rel_path=${file_b#$B_ROOT/}
        file_a="$A_ROOT/$rel_path"
        if [ ! -f "$file_a" ]; then
            echo "   Deleted: $rel_path"
            lmp_change=1
        fi
    done < "$tmp_kok_b" 2>/dev/null || true
    rm -f "$tmp_kok_a" "$tmp_kok_b"

    # 3. nep_gpu/utilities and nep_gpu/force下的 .cu/.cuh
    echo "  Checking nep_gpu/utilities and nep_gpu/force *.cu *.cuh files..."
    for sub in "nep_gpu/utilities" "nep_gpu/force"; do
        dir_a="${LMP_BASE_A}/${sub}"
        dir_b="${LMP_BASE_B}/${sub}"
        if [ -d "$dir_a" ] && [ -d "$dir_b" ]; then
            tmp_cu_a=$(mktemp)
            tmp_cu_b=$(mktemp)
            find "$dir_a" -type f \( -name "*.cu" -o -name "*.cuh" \) > "$tmp_cu_a"
            find "$dir_b" -type f \( -name "*.cu" -o -name "*.cuh" \) > "$tmp_cu_b"

            while IFS= read -r file_a; do
                rel_path=${file_a#$A_ROOT/}
                file_b="$B_ROOT/$rel_path"
                if [ -f "$file_b" ]; then
                    if ! diff -q "$file_a" "$file_b" > /dev/null 2>&1; then
                        echo "   Modified: $rel_path"
                        lmp_change=1
                    fi
                else
                    echo "   Added: $rel_path"
                    lmp_change=1
                fi
            done < "$tmp_cu_a"

            while IFS= read -r file_b; do
                rel_path=${file_b#$B_ROOT/}
                file_a="$A_ROOT/$rel_path"
                if [ ! -f "$file_a" ]; then
                    echo "   Deleted: $rel_path"
                    lmp_change=1
                fi
            done < "$tmp_cu_b"

            rm -f "$tmp_cu_a" "$tmp_cu_b"
        fi
    done

    if [ $lmp_change -eq 0 ]; then
        echo " No changes in lmp_nepkokkos_cmake related files"
    fi
elif [ ! -d "$LMP_BASE_A" ] && [ -d "$LMP_BASE_B" ]; then
    lmp_change=1
    echo " lmp_nepkokkos_cmake directory deleted"
elif [ -d "$LMP_BASE_A" ] && [ ! -d "$LMP_BASE_B" ]; then
    lmp_change=1
    echo " lmp_nepkokkos_cmake directory added"
else
    echo " lmp_nepkokkos_cmake directory does not exist in A or B"
fi
echo "----------------------------------------"

# 计算组合状态码
exit_code=$((py_change * 1 + nep_cpu_change * 2 + nep_change * 4 + op_change * 8 + fortran_change * 16 + lmp_change * 32))
exit $exit_code
