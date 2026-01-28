#include <iostream>

template<typename DType>
__global__ void compress_calc(
    DType * G,
    const DType * f2,
    const DType * coefficient,
    const int sij_num,
    const int layer_node,
    const int coe_num
    )
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < sij_num) {
        DType f2_val = f2[tid];
        int coefficient_base = tid * layer_node * coe_num;
        
        for (int i = 0; i < layer_node; i++) {
            DType coeff_0 = coefficient[coefficient_base + i * coe_num];
            DType coeff_1 = coefficient[coefficient_base + i * coe_num + 1];
            DType coeff_2 = coefficient[coefficient_base + i * coe_num + 2];
            DType coeff_3 = coefficient[coefficient_base + i * coe_num + 3];
            
            DType result = 0.0;
            
            if (coe_num == 4) {
                result = f2_val * (f2_val * (f2_val * coeff_0 + coeff_1) + coeff_2) + coeff_3;
            } else {
                DType coeff_4 = coefficient[coefficient_base + i * coe_num + 4];
                DType coeff_5 = coefficient[coefficient_base + i * coe_num + 5];
                
                result = f2_val * (f2_val * (f2_val * (f2_val * (f2_val * coeff_0 + coeff_1) + coeff_2) + coeff_3) + coeff_4) + coeff_5;
            }
            
            G[tid * layer_node + i] = result;
        }
    }
}

template<typename DType>
void launch_calculate_compress(
    const DType * f2,
    const DType * coefficient,
    const int sij_num,
    const int layer_node,
    const int coe_num,
    DType * G,
    const int device_id
) {
    cudaSetDevice(device_id);
    const int blockSize = 256;
    const int gridSize = (sij_num + blockSize -1) / blockSize;
    // Launch the kernel
    compress_calc<<<gridSize, blockSize>>>(G, f2, coefficient, sij_num, layer_node, coe_num);
}

template void launch_calculate_compress(
    const float * f2,
    const float * coefficient,
    const int sij_num,
    const int layer_node,
    const int coe_num,
    float * G,
    const int device_id
    );

template void launch_calculate_compress(
    const double * f2,
    const double * coefficient,
    const int sij_num,
    const int layer_node,
    const int coe_num,
    double * G,
    const int device_id
    );