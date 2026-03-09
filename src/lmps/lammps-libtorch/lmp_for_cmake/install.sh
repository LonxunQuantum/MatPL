#!/bin/bash

# Check if lammps_root_dir is provided
if [ -z "$1" ]; then
    echo "Error: Please provide the LAMMPS source root directory as the first argument"
    echo "Usage: $0 <lammps_root_dir> [install_dir]"
    exit 1
fi

# Set variables
LAMMPS_ROOT_DIR="$1"
INSTALL_DIR="${2:-$LAMMPS_ROOT_DIR/build}"  # Default install path is lammps_root_dir/build
BUILD_DIR="$LAMMPS_ROOT_DIR/build"
CMAKE_DIR="$LAMMPS_ROOT_DIR/cmake"
PRESETS_DIR="$CMAKE_DIR/presets"
MATPL_DIR="./MATPL"

# Check if LAMMPS root directory exists
if [ ! -d "$LAMMPS_ROOT_DIR" ]; then
    echo "Error: LAMMPS root directory $LAMMPS_ROOT_DIR does not exist"
    exit 1
fi

# Step 1: Copy files, backing up existing CMakeLists.txt and basic.cmake
echo "Copying CMakeLists.txt and basic.cmake..."
# Check and backup CMakeLists.txt
if [ -f "$CMAKE_DIR/CMakeLists.txt" ]; then
    echo "Backing up $CMAKE_DIR/CMakeLists.txt to CMakeLists.txt.bk"
    mv "$CMAKE_DIR/CMakeLists.txt" "$CMAKE_DIR/CMakeLists.txt.bk"
fi
cp ./CMakeLists.txt "$CMAKE_DIR/"

# Check and backup basic.cmake
if [ -f "$PRESETS_DIR/basic.cmake" ]; then
    echo "Backing up $PRESETS_DIR/basic.cmake to basic.cmake.bk"
    mv "$PRESETS_DIR/basic.cmake" "$PRESETS_DIR/basic.cmake.bk"
fi
cp ./basic.cmake "$PRESETS_DIR/"

# Copy other files
echo "Copying pair_d3.* and MATPL files..."
cp ./pair_d3.cu ./pair_d3.h ./pair_d3_pars.h "$LAMMPS_ROOT_DIR/src/"
cp "$MATPL_DIR"/{dftd3para.h,nep_cpu.cpp,nep_cpu.h,pair_matpl.cpp,pair_matpl.h} "$LAMMPS_ROOT_DIR/src/"
# Recursively copy NEP_GPU directory
if [ -d "$MATPL_DIR/NEP_GPU" ]; then
    echo "Copying NEP_GPU directory to $LAMMPS_ROOT_DIR/src/NEP_GPU..."
    cp -r "$MATPL_DIR/NEP_GPU" "$LAMMPS_ROOT_DIR/src/"
else
    echo "Error: NEP_GPU directory not found in $MATPL_DIR"
    exit 1
fi

# Step 2: Create build directory
echo "Creating build directory: $BUILD_DIR"
mkdir -p "$BUILD_DIR"

# Step 3: Run CMake configuration
echo "Running CMake configuration with install prefix: $INSTALL_DIR..."
cd "$BUILD_DIR" || { echo "Error: Cannot enter $BUILD_DIR"; exit 1; }
cmake ../cmake -C ../cmake/presets/basic.cmake -DCMAKE_PREFIX_PATH=$(python3 -c 'import torch;print(torch.utils.cmake_prefix_path)') -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR"

# Step 4: Compile
echo "Starting compilation..."
make -j4
if [ $? -ne 0 ]; then
    echo "Error: Compilation failed"
    exit 1
fi

# Step 5: Install to INSTALL_DIR
echo "Installing to $INSTALL_DIR..."
make install
if [ $? -ne 0 ]; then
    echo "Error: Installation failed"
    exit 1
fi

# Step 6: Generate environment variable script and output
ENV_SCRIPT="$INSTALL_DIR/env.sh"
echo "Generating environment variable script: $ENV_SCRIPT"
cat > "$ENV_SCRIPT" << EOF
export PATH="$INSTALL_DIR/bin:\$PATH"
export LD_LIBRARY_PATH="\$LD_LIBRARY_PATH:$(python3 -c "import torch; print(torch.__path__[0])")/lib:$(dirname $(dirname $(which python3)))/lib"
EOF

# Make env.sh executable
chmod +x "$ENV_SCRIPT"

# Output environment variables to screen
echo "Environment variable configuration:"
cat "$ENV_SCRIPT"

echo "Compilation and installation completed! LAMMPS executable is located at: $INSTALL_DIR/bin"
echo "Environment variable script generated: $ENV_SCRIPT"
