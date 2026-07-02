#!/bin/bash

# Default make command (single core) and NEP types
MAKE_CMD="make"
COMPILE_FORTRAN=0

# Define directory variables
BASE_DIR=$(pwd)  # src directory
BIN_DIR="$BASE_DIR/bin"
LIB_DIR="$BASE_DIR/lib"
NEP_CPU_DIR="$BASE_DIR/feature/nep_find_neigh"
NEP_GPU_DIR="$BASE_DIR/feature/NEP_GPU"
OP_DIR="$BASE_DIR/op"

# Function to display help information
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -h              Show this help message"
    echo "  -jN             Use N parallel jobs for compilation (e.g., -j4)"
    echo "  -m nn           Compile Fortran codes (required for NN and Linear models)"
    echo ""
    echo "Examples:"
    echo "  $0                     # Default compilation without Fortran"
    echo "  $0 -j4                 # Use 4 parallel jobs"
    echo "  $0 -m nn               # Compile Fortran codes"
    echo "  $0 -j4 -m nn      # Use 4 jobs, compile Fortran"
    exit 0
}

# Parse command line arguments
while [ $# -gt 0 ]; do
    case $1 in
        -h)
            show_help
            ;;
        -j*)
            MAKE_CMD="make $1"
            shift
            ;;
        -m)
            if [ "$2" = "nn" ]; then
                COMPILE_FORTRAN=1
                shift 2
            else
                echo "Error: -m option requires 'nn' argument"
                exit 1
            fi
            ;;
        *)
            echo "Error: Unknown option $1"
            show_help
            ;;
    esac
done

echo "Using MAKE_CMD = $MAKE_CMD"
if [ $COMPILE_FORTRAN -eq 1 ]; then
    echo "Compile Fortran codes: Yes"
else
    echo "Compile Fortran codes: No"
fi

mkdir -p "$BIN_DIR"
mkdir -p "$LIB_DIR"

# Compile Fortran codes if requested
if [ $COMPILE_FORTRAN -eq 1 ]; then
    echo "Compiling Fortran codes..."
    
    # List of directories containing Fortran code
    for dir in "pre_data/gen_feature" "pre_data/fit" "pre_data/fortran_code" "md/fortran_code"; do
        echo "Compiling in $dir..."
        if ! make -C "$dir"; then
            echo "Error: Compilation failed in $dir"
            echo "Fortran compilation is required for NN and Linear models."
            exit 1
        fi
    done
    
    # Check for required Fortran compiled files
    missing_files=""
    
    for file in "main_MD.x" "gen_dR.x"; do
        if [ ! -f "$BIN_DIR/$file" ]; then
            missing_files="$missing_files $file"
        fi
    done
    
    if [ -n "$missing_files" ]; then
        echo "Error: Missing required Fortran compiled files:$missing_files"
        exit 1
    fi
    
    if [ ! -f "$LIB_DIR/NeighConst.so" ]; then
        echo "Error: $LIB_DIR/NeighConst.so not found (Fortran compilation product)"
        exit 1
    fi
    
    echo "Fortran compilation completed successfully"
else
    echo "Skipping Fortran compilation (NN and Linear models will not be available)"
fi

# make nep-cpu interface
echo "Building NEP-CPU interface..."
if [ -d "$NEP_CPU_DIR" ]; then
    cd "$NEP_CPU_DIR"
    rm -rf build/*
    rm -f findneigh.so 2>/dev/null
    mkdir -p build
    cd build
    if cmake -Dpybind11_DIR=$(python -m pybind11 --cmakedir) .. && $MAKE_CMD; then
        echo "Compile nep_cpu interface success"
    else
        echo "Warning: Failed to build NEP-CPU interface"
    fi
    cd "$BASE_DIR"  # Return to base directory
else
    echo "Warning: NEP-CPU directory not found: $NEP_CPU_DIR"
fi

# make nep-gpu interface
echo "Building NEP-GPU interface..."

# Check if CUDA is available
if command -v nvcc >/dev/null 2>&1 || [ -n "$CUDA_HOME" ] || [ -n "$CUDA_PATH" ]; then
    if [ -d "$NEP_GPU_DIR" ]; then
        mkdir -p "$NEP_GPU_DIR/build"
        cd "$NEP_GPU_DIR/build"
        if cmake -Dpybind11_DIR=$(python -m pybind11 --cmakedir) .. && $MAKE_CMD; then
            echo "compile nep_gpu interface success"
        else
            echo "Warning: Failed to build NEP-GPU interface"
        fi
        cd "$BASE_DIR"  # Return to base directory
    else
        echo "Warning: NEP-GPU directory not found: $NEP_GPU_DIR"
    fi
else
    echo "Warning: CUDA not detected, skipping NEP-GPU compilation"
    echo "         To compile with GPU support, please install CUDA toolkit"
fi

# Build operators
echo "Building operators..."
if [ -d "$OP_DIR" ]; then
    cd "$OP_DIR"
    rm -rf build
    mkdir -p build
    cd build
    # for bigmodel the types should be 100
    if cmake .. && $MAKE_CMD; then
        echo "Operators built successfully"
    else
        echo "Warning: Failed to build operators"
    fi
    cd "$BASE_DIR"  # Return to base directory
else
    echo "Warning: Operators directory not found: $OP_DIR"
fi

# Create symbolic links in bin directory
echo "Creating symbolic links in bin directory..."
cd "$BIN_DIR"

# Create symbolic link for MD executable only if it exists
if [ -f "../md/fortran_code/main_MD.x" ]; then
    ln -sf ../md/fortran_code/main_MD.x .
    echo "Created symbolic link for main_MD.x"
elif [ $COMPILE_FORTRAN -eq 1 ]; then
    echo "Error: main_MD.x should have been created by Fortran compilation but was not found"
    exit 1
else
    echo "Note: main_MD.x not available (requires Fortran compilation with -m nn)"
fi

# Create symbolic links for Python executables
ln -sf ../../main.py ./MATPL
ln -sf ../../main.py ./matpl
ln -sf ../../main.py ./MatPL
ln -sf ../../main.py ./PWMLFF
ln -sf ../../main.py ./pwmlff
# ln -sf ../../main_mnode.py ./MNEP

cd "$BASE_DIR"  # Return to base directory

# Get parent directory of BASE_DIR (project root)
PARENT_DIR=$(dirname "$BASE_DIR")

# write environment to env.sh
cat <<EOF > "$PARENT_DIR/env.sh"
# Environment for MatPL
export PYTHONPATH=$PARENT_DIR:\$PYTHONPATH
export PATH=$BIN_DIR:\$PATH
EOF

echo ""
echo "================================="
if [ $COMPILE_FORTRAN -eq 0 ]; then
    echo "WARNING: Fortran codes were not compiled."
    echo "NN and Linear models will not be available."
    echo "To enable these models, recompile with the '-m nn' option:"
    echo "  sh build.sh -m nn"
    echo ""
fi

echo "MatPL has been successfully installed."
echo "Please load the MatPL environment variables before use."
echo ""
echo "Recommended method:"
echo "  source $PARENT_DIR/env.sh"
echo ""
echo "Or manually set environment variables:"
echo "  export PYTHONPATH=$PARENT_DIR:\$PYTHONPATH"
echo "  export PATH=$BIN_DIR:\$PATH"
echo "================================="
