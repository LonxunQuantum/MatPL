#!/bin/bash

# Default make command (single core) and NEP types
MAKE_CMD="make"
JOBS=1
COMPILE_FORTRAN=0
DRY_RUN=0

# Define directory variables
BASE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)  # src directory
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
    echo "  --dry-run       Print the selected operator build commands and exit"
    echo ""
    echo "Examples:"
    echo "  $0                     # Default compilation without Fortran"
    echo "  $0 -j4                 # Use 4 parallel jobs"
    echo "  $0 -m nn               # Compile Fortran codes"
    echo "  $0 -j4 -m nn           # Use 4 jobs, compile Fortran"
}

# Parse command line arguments
while [ $# -gt 0 ]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -j*)
            JOBS="${1#-j}"
            if ! [[ "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
                echo "Error: -j requires a positive integer"
                exit 1
            fi
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
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        *)
            echo "Error: Unknown option $1"
            show_help
            exit 1
            ;;
    esac
done

if ! RESOLVED_BACKEND=$(python -c \
    "import torch; print('hip' if getattr(torch.version, 'hip', None) else ('cuda' if torch.version.cuda else 'cpu'))"); then
    echo "Error: Unable to detect the PyTorch accelerator backend"
    exit 1
fi

case "$RESOLVED_BACKEND" in
    cuda|hip|cpu)
        ;;
    *)
        echo "Error: PyTorch backend detection returned '$RESOLVED_BACKEND'"
        exit 1
        ;;
esac

OP_BACKENDS=("$RESOLVED_BACKEND")
if [ "$RESOLVED_BACKEND" != "cpu" ]; then
    OP_BACKENDS+=("cpu")
fi
NEP_GPU_BUILD_DIR="$NEP_GPU_DIR/build/$RESOLVED_BACKEND"
NEP_GPU_CUDACXX=""
NEP_GPU_TOOLKIT_ROOT=""

find_dtk_nvcc() {
    local dtk_root candidate
    for candidate in "${MATPL_DTK_NVCC:-}" "${CUDACXX:-}"; do
        if [ -n "$candidate" ] && [ -x "$candidate" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    for dtk_root in "${MATPL_DTK_ROOT:-}" "${ROCM_PATH:-}"; do
        [ -n "$dtk_root" ] || continue
        for candidate in \
            "$dtk_root"/cuda/cuda-*/bin/nvcc \
            "$dtk_root/cuda/cuda/bin/nvcc"; do
            if [ -x "$candidate" ]; then
                printf '%s\n' "$candidate"
                return 0
            fi
        done
    done
    return 1
}

if [ "$RESOLVED_BACKEND" = "cuda" ]; then
    NEP_GPU_CUDACXX=$(command -v nvcc 2>/dev/null || true)
    NEP_GPU_TOOLKIT_ROOT="${CUDAToolkit_ROOT:-${CUDA_HOME:-${CUDA_PATH:-}}}"
    if [ -z "$NEP_GPU_TOOLKIT_ROOT" ] && [ -n "$NEP_GPU_CUDACXX" ]; then
        NEP_GPU_TOOLKIT_ROOT=$(cd "$(dirname "$NEP_GPU_CUDACXX")/.." && pwd)
    fi
elif [ "$RESOLVED_BACKEND" = "hip" ]; then
    NEP_GPU_CUDACXX=$(find_dtk_nvcc || true)
    if [ -n "${MATPL_DTK_CUDA_ROOT:-}" ]; then
        NEP_GPU_TOOLKIT_ROOT="$MATPL_DTK_CUDA_ROOT"
    elif [ -n "${MATPL_DTK_ROOT:-}" ]; then
        NEP_GPU_TOOLKIT_ROOT="$MATPL_DTK_ROOT/cuda/cuda"
    elif [ -n "${ROCM_PATH:-}" ]; then
        NEP_GPU_TOOLKIT_ROOT="$ROCM_PATH/cuda/cuda"
    fi
fi

echo "Using MAKE_CMD = $MAKE_CMD"
if [ $COMPILE_FORTRAN -eq 1 ]; then
    echo "Compile Fortran codes: Yes"
else
    echo "Compile Fortran codes: No"
fi
echo "Resolved accelerator backend: $RESOLVED_BACKEND"
echo "Operator build backends: ${OP_BACKENDS[*]}"
echo "NEP-GPU backend: $RESOLVED_BACKEND"
if [ -n "$NEP_GPU_CUDACXX" ]; then
    echo "NEP-GPU CUDA compiler: $NEP_GPU_CUDACXX"
    echo "NEP-GPU build directory: $NEP_GPU_BUILD_DIR"
else
    echo "NEP-GPU CUDA compiler: unavailable"
fi

if [ "$DRY_RUN" -eq 1 ]; then
    if [ -n "$NEP_GPU_CUDACXX" ]; then
        echo "PATH=$(dirname "$NEP_GPU_CUDACXX"):\$PATH CUDACXX=$NEP_GPU_CUDACXX cmake -S $NEP_GPU_DIR -B $NEP_GPU_BUILD_DIR -DCUDAToolkit_ROOT=$NEP_GPU_TOOLKIT_ROOT"
        echo "cmake --build $NEP_GPU_BUILD_DIR --parallel $JOBS"
    else
        echo "Skipping NEP-GPU interface for backend $RESOLVED_BACKEND"
    fi
    for OP_BACKEND in "${OP_BACKENDS[@]}"; do
        OP_BUILD_DIR="$OP_DIR/build/$OP_BACKEND"
        OP_BACKEND_UPPER="${OP_BACKEND^^}"
        echo "cmake -S $OP_DIR -B $OP_BUILD_DIR -DMATPL_GPU_BACKEND=$OP_BACKEND_UPPER"
        echo "cmake --build $OP_BUILD_DIR --parallel $JOBS"
    done
    exit 0
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

# CUDA uses the native nvcc. DCU uses DTK's nvcc-compatible wrapper while
# retaining the single NEP_GPU CUDA source tree.
if [ -n "$NEP_GPU_CUDACXX" ]; then
    if [ -d "$NEP_GPU_DIR" ]; then
        mkdir -p "$NEP_GPU_BUILD_DIR"
        if PATH="$(dirname "$NEP_GPU_CUDACXX"):$PATH" \
            CUDACXX="$NEP_GPU_CUDACXX" \
            cmake -S "$NEP_GPU_DIR" -B "$NEP_GPU_BUILD_DIR" \
                -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
                -DCUDAToolkit_ROOT="$NEP_GPU_TOOLKIT_ROOT" && \
            PATH="$(dirname "$NEP_GPU_CUDACXX"):$PATH" \
            cmake --build "$NEP_GPU_BUILD_DIR" --parallel "$JOBS"; then
            echo "compile nep_gpu interface success"
        else
            echo "Warning: Failed to build NEP-GPU interface"
        fi
    else
        echo "Warning: NEP-GPU directory not found: $NEP_GPU_DIR"
    fi
else
    echo "Warning: No CUDA-compatible compiler found for backend $RESOLVED_BACKEND"
    echo "         Skipping NEP-GPU compilation"
fi

# Build operators
echo "Building operators..."
if [ -d "$OP_DIR" ]; then
    for OP_BACKEND in "${OP_BACKENDS[@]}"; do
        OP_BUILD_DIR="$OP_DIR/build/$OP_BACKEND"
        OP_BACKEND_UPPER="${OP_BACKEND^^}"
        case "$OP_BUILD_DIR" in
            "$OP_DIR"/build/cuda|"$OP_DIR"/build/hip|"$OP_DIR"/build/cpu)
                ;;
            *)
                echo "Error: Refusing to clean unexpected build directory: $OP_BUILD_DIR"
                exit 1
                ;;
        esac
        rm -rf "$OP_BUILD_DIR"
        mkdir -p "$OP_BUILD_DIR"
        # for bigmodel the types should be 100
        if cmake -S "$OP_DIR" -B "$OP_BUILD_DIR" \
            -DMATPL_GPU_BACKEND="$OP_BACKEND_UPPER" && \
            cmake --build "$OP_BUILD_DIR" --parallel "$JOBS"; then
            echo "Operators built successfully for backend $OP_BACKEND"
        else
            echo "Error: Failed to build operators for backend $OP_BACKEND"
            exit 1
        fi
    done
    cd "$BASE_DIR"  # Return to base directory
else
    echo "Error: Operators directory not found: $OP_DIR"
    exit 1
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
