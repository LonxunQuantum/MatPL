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
lmpfortran_change=0
fortran_change=0  # 新增的Fortran文件检查标志

echo "Starting directory comparison..."
echo "A Directory: $A_ROOT"
echo "B Directory: $B_ROOT"
echo "----------------------------------------"
# 1. Check all .py files, excluding example and .git directories
echo "1. Checking all .py files (excluding example and .git directories)..."
echo "Python file differences:"
# Create temporary files to store find results
tmp_a=$(mktemp)
tmp_b=$(mktemp)
find "$A_ROOT" -type f -name "*.py" -not -path "*/example/*" -not -path "*/.git/*" > "$tmp_a"
find "$B_ROOT" -type f -name "*.py" -not -path "*/example/*" -not -path "*/.git/*" > "$tmp_b"
# Check .py files in A_ROOT
while IFS= read -r file_a; do
    # Get relative path from A_ROOT
    rel_path=${file_a#$A_ROOT/}
    file_b="$B_ROOT/$rel_path"
    if [ -f "$file_b" ]; then
        # File exists in both A and B, compare contents
        if ! diff -q "$file_a" "$file_b" > /dev/null 2>&1; then
            echo " Modified: $rel_path"
            py_change=1
        fi
    else
        # File only exists in A
        echo " Added: $rel_path"
        py_change=1
    fi
done < "$tmp_a"
# Check .py files unique to B_ROOT
while IFS= read -r file_b; do
    rel_path=${file_b#$B_ROOT/}
    file_a="$A_ROOT/$rel_path"
    if [ ! -f "$file_a" ]; then
        echo " Deleted: $rel_path"
        py_change=1
    fi
done < "$tmp_b"
# Clean up temporary files
rm -f "$tmp_a" "$tmp_b"
# Output Python file check results
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
        diff -rq --exclude="build" --exclude="*.so" "$nep_cpu_dir_A" "$nep_cpu_dir_B" | while read line; do
            echo " $line"
        done
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
        diff -rq --exclude="build" "$nep_gpu_dir_A" "$nep_gpu_dir_B" | while read line; do
            echo " $line"
        done
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
        diff -rq --exclude="build" "$op_dir_A" "$op_dir_B" | while read line; do
            echo " $line"
        done
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

# 检查整个pre_data目录是否存在
if [ -d "$pre_data_dir_A" ] && [ -d "$pre_data_dir_B" ]; then
    # 创建临时文件来存储find结果
    tmp_f90_a=$(mktemp)
    tmp_f90_b=$(mktemp)
    
    # 查找所有.f90文件
    find "$pre_data_dir_A" -type f -name "*.f90" > "$tmp_f90_a"
    find "$pre_data_dir_B" -type f -name "*.f90" > "$tmp_f90_b"
    
    echo "Fortran (.f90) file differences:"
    # 检查A_ROOT中的.f90文件
    while IFS= read -r file_a; do
        # 获取相对路径
        rel_path=${file_a#$A_ROOT/}
        file_b="$B_ROOT/$rel_path"
        if [ -f "$file_b" ]; then
            # 文件在A和B中都存在，比较内容
            if ! diff -q "$file_a" "$file_b" > /dev/null 2>&1; then
                echo " Modified: $rel_path"
                fortran_change=1
            fi
        else
            # 文件只在A中存在
            echo " Added: $rel_path"
            fortran_change=1
        fi
    done < "$tmp_f90_a"
    
    # 检查B_ROOT中独有的.f90文件
    while IFS= read -r file_b; do
        rel_path=${file_b#$B_ROOT/}
        file_a="$A_ROOT/$rel_path"
        if [ ! -f "$file_a" ]; then
            echo " Deleted: $rel_path"
            fortran_change=1
        fi
    done < "$tmp_f90_b"
    
    # 清理临时文件
    rm -f "$tmp_f90_a" "$tmp_f90_b"
    
    # 输出结果
    if [ $fortran_change -eq 0 ]; then
        echo " No changes in Fortran (.f90) files in src/pre_data"
    else
        echo " Changes detected in Fortran (.f90) files in src/pre_data"
    fi
    
elif [ ! -d "$pre_data_dir_A" ] && [ -d "$pre_data_dir_B" ]; then
    fortran_change=1
    echo " src/pre_data directory deleted"
    echo " Changes detected in Fortran files"
elif [ -d "$pre_data_dir_A" ] && [ ! -d "$pre_data_dir_B" ]; then
    fortran_change=1
    echo " src/pre_data directory added"
    echo " Changes detected in Fortran files"
else
    echo " src/pre_data directory does not exist in A or B"
fi
echo "----------------------------------------"
echo "Checking src/md directory and all .f90 files..."
md_dir_A="$A_ROOT/src/md"
md_dir_B="$B_ROOT/src/md"

if [ -d "$md_dir_A" ] && [ -d "$md_dir_B" ]; then
    # 创建临时文件来存储find结果
    tmp_f90_md_a=$(mktemp)
    tmp_f90_md_b=$(mktemp)
    
    # 查找所有.f90文件
    find "$md_dir_A" -type f -name "*.f90" > "$tmp_f90_md_a"
    find "$md_dir_B" -type f -name "*.f90" > "$tmp_f90_md_b"
    
    echo "Fortran (.f90) file differences in src/md:"
    # 检查A_ROOT中的.f90文件
    while IFS= read -r file_a; do
        # 获取相对路径
        rel_path=${file_a#$A_ROOT/}
        file_b="$B_ROOT/$rel_path"
        if [ -f "$file_b" ]; then
            # 文件在A和B中都存在，比较内容
            if ! diff -q "$file_a" "$file_b" > /dev/null 2>&1; then
                echo " Modified: $rel_path"
                fortran_change=1
            fi
        else
            # 文件只在A中存在
            echo " Added: $rel_path"
            fortran_change=1
        fi
    done < "$tmp_f90_md_a"
    
    # 检查B_ROOT中独有的.f90文件
    while IFS= read -r file_b; do
        rel_path=${file_b#$B_ROOT/}
        file_a="$A_ROOT/$rel_path"
        if [ ! -f "$file_a" ]; then
            echo " Deleted: $rel_path"
            fortran_change=1
        fi
    done < "$tmp_f90_md_b"
    
    # 清理临时文件
    rm -f "$tmp_f90_md_a" "$tmp_f90_md_b"
    
    # 输出结果
    if [ $fortran_change -eq 0 ]; then
        echo " No changes in Fortran (.f90) files in src/md"
    else
        echo " Changes detected in Fortran (.f90) files in src/md"
    fi
    
elif [ ! -d "$md_dir_A" ] && [ -d "$md_dir_B" ]; then
    fortran_change=1
    echo " src/md directory deleted"
    echo " Changes detected in Fortran files in src/md"
elif [ -d "$md_dir_A" ] && [ ! -d "$md_dir_B" ]; then
    fortran_change=1
    echo " src/md directory added"
    echo " Changes detected in Fortran files in src/md"
else
    echo " src/md directory does not exist in A or B"
fi
echo "----------------------------------------"

# 6. Check lammps-libtorch related files and directories
echo "6. Checking lammps-libtorch related files and directories..."
lmp_matpl_dir_A="$A_ROOT/src/lmps/lammps-libtorch/MATPL"
lmp_matpl_dir_B="$B_ROOT/src/lmps/lammps-libtorch/MATPL"
lmp_makefile_A="$A_ROOT/src/lmps/lammps-libtorch/Makefile.mpi"
lmp_makefile_B="$B_ROOT/src/lmps/lammps-libtorch/Makefile.mpi"
# Check MATPL directory
if [ -d "$lmp_matpl_dir_A" ] && [ -d "$lmp_matpl_dir_B" ]; then
    if ! diff -rq "$lmp_matpl_dir_A" "$lmp_matpl_dir_B" > /dev/null 2>&1; then
        lmp_change=1
        echo " Changes detected in MATPL directory"
    fi
elif [ ! -d "$lmp_matpl_dir_A" ] && [ -d "$lmp_matpl_dir_B" ]; then
    lmp_change=1
    echo " MATPL directory deleted"
elif [ -d "$lmp_matpl_dir_A" ] && [ ! -d "$lmp_matpl_dir_B" ]; then
    lmp_change=1
    echo " MATPL directory added"
fi
# Check Makefile.mpi file
if [ -f "$lmp_makefile_A" ] && [ -f "$lmp_makefile_B" ]; then
    if ! diff -q "$lmp_makefile_A" "$lmp_makefile_B" > /dev/null 2>&1; then
        lmp_change=1
        echo " Changes detected in Makefile.mpi file"
    fi
elif [ ! -f "$lmp_makefile_A" ] && [ -f "$lmp_makefile_B" ]; then
    lmp_change=1
    echo " Makefile.mpi file deleted"
elif [ -f "$lmp_makefile_A" ] && [ ! -f "$lmp_makefile_B" ]; then
    lmp_change=1
    echo " Makefile.mpi file added"
fi
if [ $lmp_change -eq 0 ]; then
    echo " No changes in lammps-libtorch related files"
fi
echo "----------------------------------------"
# 7. Check lammps-fortran related files and directories
echo "7. Checking lammps-fortran related files and directories..."
lmp_fortran_matpl_dir_A="$A_ROOT/src/lmps/lammps-fortran/MATPL"
lmp_fortran_matpl_dir_B="$B_ROOT/src/lmps/lammps-fortran/MATPL"
lmp_fortran_makefile_A="$A_ROOT/src/lmps/lammps-fortran/Makefile.mpi"
lmp_fortran_makefile_B="$B_ROOT/src/lmps/lammps-fortran/Makefile.mpi"
# Check MATPL directory
if [ -d "$lmp_fortran_matpl_dir_A" ] && [ -d "$lmp_fortran_matpl_dir_B" ]; then
    if ! diff -rq "$lmp_fortran_matpl_dir_A" "$lmp_fortran_matpl_dir_B" > /dev/null 2>&1; then
        lmpfortran_change=1
        echo " Changes detected in MATPL directory"
    fi
elif [ ! -d "$lmp_fortran_matpl_dir_A" ] && [ -d "$lmp_fortran_matpl_dir_B" ]; then
    lmpfortran_change=1
    echo " MATPL directory deleted"
elif [ -d "$lmp_fortran_matpl_dir_A" ] && [ ! -d "$lmp_fortran_matpl_dir_B" ]; then
    lmpfortran_change=1
    echo " MATPL directory added"
fi
# Check Makefile.mpi file
if [ -f "$lmp_fortran_makefile_A" ] && [ -f "$lmp_fortran_makefile_B" ]; then
    if ! diff -q "$lmp_fortran_makefile_A" "$lmp_fortran_makefile_B" > /dev/null 2>&1; then
        lmpfortran_change=1
        echo " Changes detected in Makefile.mpi file"
    fi
elif [ ! -f "$lmp_fortran_makefile_A" ] && [ -f "$lmp_fortran_makefile_B" ]; then
    lmpfortran_change=1
    echo " Makefile.mpi file deleted"
elif [ -f "$lmp_fortran_makefile_A" ] && [ ! -f "$lmp_fortran_makefile_B" ]; then
    lmpfortran_change=1
    echo " Makefile.mpi file added"
fi
if [ $lmpfortran_change -eq 0 ]; then
    echo " No changes in lammps-fortran related files"
fi
echo "----------------------------------------"

# 计算组合状态码 (每个标志占一位)
exit_code=$((py_change * 1 + nep_cpu_change * 2 + nep_change * 4 + op_change * 8 + fortran_change * 16 + lmp_change * 32 + lmpfortran_change * 64))
exit $exit_code

