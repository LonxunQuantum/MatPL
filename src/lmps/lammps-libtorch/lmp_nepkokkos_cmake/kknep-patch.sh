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

# Backup CMakeLists.txt before modification with timestamp
CMAKE_FILE="$LAMMPSROOT/cmake/CMakeLists.txt"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [ -f "$CMAKE_FILE" ]; then
    cp "$CMAKE_FILE" "$CMAKE_FILE.bk.$TIMESTAMP"
    echo "  - CMakeLists.txt backed up as CMakeLists.txt.bk.$TIMESTAMP"
else
    echo "  Error: CMakeLists.txt not found in $LAMMPSROOT/cmake/"
    exit 1
fi

# Modify CMakeLists.txt to add NEP_KK package and update project settings
echo "Modifying CMakeLists.txt to add CUDA support and NEP_KK package..."

# First modification: Update project to include CUDA
PROJECT_LINE='project(lammps CXX)'
NEW_PROJECT_LINE='project(lammps CXX CUDA)'

# Check if project line already has CUDA to avoid duplication
if grep -q "$NEW_PROJECT_LINE" "$CMAKE_FILE"; then
    echo "  - Project already includes CUDA, no change needed"
elif grep -q "$PROJECT_LINE" "$CMAKE_FILE"; then
    sed -i "s|${PROJECT_LINE}|${NEW_PROJECT_LINE}|g" "$CMAKE_FILE"
    echo "  - Project configuration updated to include CUDA"
else
    echo "Error: Unsupported LAMMPS version detected"
    echo "Current LAMMPS version is not supported."
    echo "Supported versions: 2023 and 2024 released versions of LAMMPS"
    echo "Unsupported versions: 2025 released versions, develop or stable branch code"
    echo "Please check and try again with a supported version."
    exit 1
fi

# Second modification: Add NEP_KK package after the packages foreach loop
# Find the position to insert the new content
INSERT_POINT=$(grep -n "foreach(PKG_WITH_INCL CORESHELL" "$CMAKE_FILE" | tail -1 | cut -d: -f1)

if [ -z "$INSERT_POINT" ]; then
    echo "  Error: Could not find the insertion point for NEP_KK package in CMakeLists.txt"
    echo "  Please check the file structure and modify manually"
    exit 1
fi

# Check if NEP_KK package is already added to avoid duplication
if grep -q "PKG_NEP_KK" "$CMAKE_FILE"; then
    echo "  - NEP_KK package already exists in CMakeLists.txt, skipping addition"
else
    # Calculate the line number after the foreach block
    END_LINE=$((INSERT_POINT + 5))

    # Create a temporary file for the modified content
    TEMP_FILE=$(mktemp)

    # Copy lines before the insertion point
    head -n $END_LINE "$CMAKE_FILE" > "$TEMP_FILE"

    # Append the new NEP_KK content
    cat >> "$TEMP_FILE" << 'EOF'

######################################################################
# package of NEP with KOKKOS
######################################################################
if(PKG_NEP_KK)
  # Mandatory dependency check
  if(NOT PKG_KOKKOS)
    message(FATAL_ERROR "NEP_KK requires KOKKOS package. Enable with -DPKG_KOKKOS=yes")
  endif()
  
  if(NOT Kokkos_ENABLE_CUDA)
    message(FATAL_ERROR "NEP_KK requires CUDA support. Enable with -DKokkos_ENABLE_CUDA=yes")
  endif()
  
  message(STATUS "NEP_KK: Building with mandatory KOKKOS and CUDA")
  
  # Collect source files
  file(GLOB NEP_KK_SOURCES
    ${LAMMPS_SOURCE_DIR}/nep_gpu/force/*.cu
    ${LAMMPS_SOURCE_DIR}/nep_gpu/utilities/*.cu
  )
  
  file(GLOB NEP_KOKKOS_SOURCES 
    ${LAMMPS_SOURCE_DIR}/KOKKOS/pair_nep_kokkos.cpp
  )
  # Simply add to compilation - let Kokkos configuration handle these files
  target_sources(lammps PRIVATE ${NEP_KOKKOS_SOURCES} ${NEP_KK_SOURCES})
  # Only set include directories
  target_include_directories(lammps PRIVATE 
    ${LAMMPS_SOURCE_DIR}/KOKKOS
    ${LAMMPS_SOURCE_DIR}/nep_gpu/force
    ${LAMMPS_SOURCE_DIR}/nep_gpu/utilities
  )
endif()

######################################################################
# package of DP with KOKKOS
######################################################################
if(PKG_MATPLDP)
  # 1. 查找 PyTorch
  find_package(Torch REQUIRED)
  # 2. 链接 PyTorch 库到 LAMMPS 库（使用 PUBLIC 以便传递给可执行文件）
  target_link_libraries(lammps PUBLIC ${TORCH_LIBRARIES})
  # 3. 添加包源文件（假设文件位于 src/MATPLDP/ 下）
  file(GLOB MATPLDP_SOURCES ${LAMMPS_SOURCE_DIR}/MATPLDP/*.cpp)
  target_sources(lammps PRIVATE ${MATPLDP_SOURCES})
  # 4. 添加包含目录（如果头文件需要被其他源文件引用）
  target_include_directories(lammps PRIVATE ${LAMMPS_SOURCE_DIR}/MATPLDP)
  RegisterStyles(${LAMMPS_SOURCE_DIR}/MATPLDP)
endif()

######################################################################
# package of D3
######################################################################
if(PKG_MATPLD3)
  # ==================== CUDA 强制检查（和 NEP_KK 一样） ====================
  find_package(CUDAToolkit REQUIRED)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "PKG_MATPLD3 requires CUDA (nvcc). 请确保 CUDA 已安装！")
  endif()
  
  message(STATUS "PKG_MATPLD3: Building with CUDA (toolkit ${CUDAToolkit_VERSION})")

  file(GLOB MATPLD3_SOURCES ${CONFIGURE_DEPENDS}
    ${LAMMPS_SOURCE_DIR}/MATPLD3/*.cu
    ${LAMMPS_SOURCE_DIR}/MATPLD3/*.cpp
  )

  message(STATUS "LAMMPS_SOURCE_DIR = ${LAMMPS_SOURCE_DIR}")
  message(STATUS "Looking for MATPLD3 files in: ${LAMMPS_SOURCE_DIR}/MATPLD3")
  if(MATPLD3_SOURCES)
    message(STATUS "MATPLD3 sources found: ${MATPLD3_SOURCES}")
  else()
    message(FATAL_ERROR "No source files found in ${LAMMPS_SOURCE_DIR}/MATPLD3")
  endif()
  target_sources(lammps PRIVATE ${MATPLD3_SOURCES})
  target_include_directories(lammps PRIVATE 
    ${LAMMPS_SOURCE_DIR}/MATPLD3
    ${CUDAToolkit_INCLUDE_DIRS}
  )
  RegisterStyles(${LAMMPS_SOURCE_DIR}/MATPLD3)
endif()

######################################################################
EOF

    # Append the remaining content
    tail -n +$((END_LINE + 1)) "$CMAKE_FILE" >> "$TEMP_FILE"

    # Replace the original file
    mv "$TEMP_FILE" "$CMAKE_FILE"

    echo "  - CMakeLists.txt modified successfully with NEP_KK package"
fi

echo "CMakeLists.txt modification completed successfully!"

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
echo "Recommended compilation environment: cuda/11.6 (with nvcc compiler) openmpi4.1.4 cmake/3.31.6 gcc8.n"
echo ""
echo "Compilation Process:"
echo "cd $LAMMPSROOT"
echo "mkdir build & cd build"
echo "cmake -C ../cmake/presets/basic.cmake \\"
echo "    -DPKG_MESONT=no \\"
echo "    -DPKG_JPEG=no \\"
echo "    -DPKG_KOKKOS=yes \\"
echo "    -DPKG_NEP_KK=yes \\"
echo "    -DKokkos_ENABLE_CUDA=yes \\"
echo "    -DKokkos_ENABLE_OPENMP=yes \\"
echo "    -DKokkos_ENABLE_CUDA_LAMBDA=yes \\"
echo "    -DFFT_KOKKOS=CUFFT \\"
echo "    -DKokkos_ARCH_AMPERE86=ON \\"
echo "    -DTEST_TIME=ON \\"
echo "    ../cmake"
echo ""
echo "cmake --build . --parallel 4 #(number of parallel compilation cores)"
echo ""
echo ""
echo "If you also need to compile the DP interface, please import the PyTorch path, import the MKL library, and enable the C++ STD17 standard for compilation."
echo "export Torch_DIR=\$(python -c \"import torch; print(torch.utils.cmake_prefix_path)\")/Torch"
echo "Then, add the following option in cmake:"
echo "-DTorch_DIR=\${Torch_DIR} \\"
echo "-DCMAKE_CXX_STANDARD=17 \\"
echo "-DPKG_MATPLDP=yes \\"
echo ""
echo "For the D3 interface, please add the following option in cmake. Note that D3 requires CUDA support and cannot be used in combination with matpl/nep/kk."
echo "-DPKG_MATPLD3=yes \\"
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