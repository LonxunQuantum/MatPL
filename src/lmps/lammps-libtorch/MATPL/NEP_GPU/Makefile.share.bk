# Compiler
CC = nvcc

### NVCC FLAGS
# Kepler
#Kepler = -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_35,code=compute_35

# Maxwell, GTX980
#Maxwell = -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52

# Pascal, GTX1080, Titan X, Titan Xp
Pascal = -gencode arch=compute_60,code=sm_60 -gencode arch=compute_60,code=compute_60

# Volta, Titan V
#Volta = -gencode arch=compute_70,code=sm_70 -gencode arch=compute_70,code=compute_70

# Turing, RTX 2060, RTX 2070, RTX 2080, RTX 2080Ti, Titan RTX
#Turing = -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75

# Ampere, A100
AmpereD = -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80

# Ampere, RTX 3070, RTX 3080, RTX 3090
Ampere = -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86
# Ampere, RTX 4090
Ampere4 = -gencode arch=compute_89,code=sm_89 -gencode arch=compute_89,code=compute_89

# Combine all GPU targets
GPU_TARGETS =  $(Pascal) $(AmpereD) $(Ampere) $(Ampere4)

# Flags
CFLAGS = -std=c++17 -O3 $(GPU_TARGETS) -Xcompiler="-fPIC"
INC = -I./ -I./force -I./utilities
LDFLAGS = -shared $(GPU_TARGETS)
LIBS = -lcublas -lcusolver

# Source files
CUDA_SOURCES = force/nep3.cu utilities/error.cu utilities/gpu_vector.cu utilities/main_common.cu

# Object files
CUDA_OBJECTS = $(CUDA_SOURCES:.cu=.o)

# Shared library
SHARED_LIB = ../../libnep_gpu.so

all: $(SHARED_LIB)

$(SHARED_LIB): $(CUDA_OBJECTS)
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBS)

%.o: %.cu
	$(CC) $(CFLAGS) $(INC) -c $< -o $@

clean:
	rm -f force/*.o utilities/*.o $(SHARED_LIB)

.PHONY: all clean

