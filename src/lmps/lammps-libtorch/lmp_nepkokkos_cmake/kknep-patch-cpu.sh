#!/bin/bash

# Check for help option
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "This script is used to copy the MatPL-2026.3 lammps interface files to the corresponding lammps directory:"
    echo "bash kknep-patch.sh lammps/rootdir"
    exit 0
fi

# Check if lammps root directory is provided
if [ $# -ne 1 ]; then
    echo "Error: Please provide the lammps source directory as an argument"
    echo "Usage: $0 <lammpsroot>"
    exit 1
fi

LAMMPSROOT=$1

# Check if the provided directory exists
if [ ! -d "$LAMMPSROOT" ]; then
    echo "Error: Directory '$LAMMPSROOT' does not exist"
    exit 1
fi

echo "Starting patch process for LAMMPS source directory: $LAMMPSROOT"

# Now copy files after successful CMakeLists.txt modification
echo "Starting file copy process..."

# Copy CPU files to src/
echo "Copying CPU files to src/ directory..."
CPU_FILES=("nep_cpu.cpp" "nep_cpu.h" "pair_nep.cpp" "pair_nep.h")
for file in "${CPU_FILES[@]}"; do
    if [ -f "$file" ]; then
        cp -f "$file" "$LAMMPSROOT/src/"
        echo "  - $file copied successfully"
    else
        echo "  Warning: $file not found in current directory"
    fi
done

# Copy nep_gpu directory to src/
echo "Copying nep_gpu directory to src/..."
if [ -d "nep_gpu" ]; then
    cp -rf nep_gpu "$LAMMPSROOT/src/"
    echo "  - nep_gpu directory copied successfully"
else
    echo "  Warning: nep_gpu directory not found in current directory"
fi

# Copy KOKKOS files to src/KOKKOS/
echo "Copying KOKKOS files to src/KOKKOS/ directory..."
if [ -d "KOKKOS" ]; then
    mkdir -p "$LAMMPSROOT/src/KOKKOS"
    if [ -f "KOKKOS/pair_nep_kokkos.cpp" ]; then
        cp -f "KOKKOS/pair_nep_kokkos.cpp" "$LAMMPSROOT/src/KOKKOS/"
        echo "  - pair_nep_kokkos.cpp copied successfully"
    else
        echo "  Warning: KOKKOS/pair_nep_kokkos.cpp not found"
    fi
    
    if [ -f "KOKKOS/pair_nep_kokkos.h" ]; then
        cp -f "KOKKOS/pair_nep_kokkos.h" "$LAMMPSROOT/src/KOKKOS/"
        echo "  - pair_nep_kokkos.h copied successfully"
    else
        echo "  Warning: KOKKOS/pair_nep_kokkos.h not found"
    fi
else
    echo "  Warning: KOKKOS directory not found in current directory"
fi

# Copy MATPLDP and MATPLD3 directories to src/
echo "Copying MATPLDP and MATPLD3 directories to src/..."
if [ -d "MATPLDP" ]; then
    cp -rf MATPLDP "$LAMMPSROOT/src/"
    echo "  - MATPLDP directory copied successfully"
else
    echo "  Warning: MATPLDP directory not found in current directory"
fi

if [ -d "MATPLD3" ]; then
    cp -rf MATPLD3 "$LAMMPSROOT/src/"
    echo "  - MATPLD3 directory copied successfully"
else
    echo "  Warning: MATPLD3 directory not found in current directory"
fi

echo "File copy process completed successfully!"
echo ""
echo "Patch process completed successfully!"
echo ""
echo "Compilation Environment:"
echo "Recommended compilation environment: openmpi4.1.4 cmake/3.31.6 gcc8.n"
echo ""
echo "Compilation Process:"
echo "cd $LAMMPSROOT"
echo "mkdir build & cd build"
echo "cmake -C ../cmake/presets/basic.cmake ../cmake"
echo ""
echo "cmake --build . --parallel 4 #(number of parallel compilation cores)"
echo ""
echo ""
echo "The current pure CPU version does not support the DP model. To use it, please install the MatPL-2025.3 LAMMPS interface(http://doc.lonxun.com/MatPL/install/)."
echo ""

######### 手动复制文件 #########
# # 1. 复制 CPU 相关源文件到 src/
# cp nep_cpu.cpp nep_cpu.h pair_nep.cpp pair_nep.h <LAMMPSROOT>/src/

# # 2. 复制 nep_gpu 目录（包含 GPU 加速的 NEP 代码）到 src/
# cp -r nep_gpu <LAMMPSROOT>/src/

# # 3. 复制 KOKKOS 版本的 NEP 文件到 src/KOKKOS/（若目录不存在则先创建）
# mkdir -p <LAMMPSROOT>/src/KOKKOS
# cp KOKKOS/pair_nep_kokkos.cpp KOKKOS/pair_nep_kokkos.h <LAMMPSROOT>/src/KOKKOS/

# # 4. 复制 MATPLDP 接口目录到 src/
# cp -r MATPLDP <LAMMPSROOT>/src/

# # 5. 复制 MATPLD3 接口目录到 src/
# cp -r MATPLD3 <LAMMPSROOT>/src/
######### 手动复制文件 #########