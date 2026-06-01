#include "./utilities/nep_utilities.cuh"
#include "./utilities/nep_feature.cuh"
#include "./utilities/nep_mbgrad.cuh"
#include "./utilities/error.cuh"
#include "./utilities/gpu_vector.cuh"
#include <iostream>
#include <cuda_runtime.h>

template <typename T>
__global__ void compute_gradsecond_mbgradout(
    const T *grad_second,
    const T *dfeat_drij,
    T *gradsecond_gradout,
    int atom_nums,
    int maxneighs,
    int feat_mb_num)
{
    int atom_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int feat_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (atom_idx < atom_nums && feat_idx < feat_mb_num) {
        T sum = T(0.0);
        for (int neigh_idx = 0; neigh_idx < maxneighs; ++neigh_idx) {
            for (int dim = 0; dim < 4; ++dim) {
                int dfeat_index = ((atom_idx * maxneighs + neigh_idx) * feat_mb_num + feat_idx) * 4 + dim;
                int grad_index = (atom_idx * maxneighs + neigh_idx) * 4 + dim;
                sum += dfeat_drij[dfeat_index] * grad_second[grad_index];
            }
        }
        gradsecond_gradout[atom_idx * feat_mb_num + feat_idx] = sum;
    }
}

template <typename T>
void launch_calculate_nepmbfeat_secondgradout(
    const T * grad_second,
    const T * dfeat_b,
    T * gradsecond_gradout,
    const int atom_nums, 
    const int maxneighs, 
    const int feat_mb_nums, 
    const int device
) {
    cudaSetDevice(device);
    dim3 blockDim(16, 16);
    dim3 gridDim((atom_nums + blockDim.x - 1) / blockDim.x, (feat_mb_nums + blockDim.y - 1) / blockDim.y);
    compute_gradsecond_mbgradout<<<gridDim, blockDim>>>(
        grad_second, 
        dfeat_b, 
        gradsecond_gradout,
        atom_nums, 
        maxneighs, 
        feat_mb_nums
        );

    CUDA_CHECK_KERNEL
}

template <typename T>
void launch_calculate_nepmbfeat_secondgradout_c3_bk(
    const T * grad_second,
    const T * d12,
    const int64_t * NL,
    const T * de_dfeat,
    const T * dsnlm_dc,
    const T * sum_fxyz,
    const int64_t * atom_map,
    const T * coeff3,
    T * gradsecond_c3,
    const T rcut_angular,
    const int atom_nums, 
    const int maxneighs, 
    const int n_max_3b, 
    const int n_base_3b, 
    const int atom_types, 
    const int lmax_3,
    const int lmax_4,
    const int lmax_5,
    const int feat_2b_num,
    const int multi_feat_num,
    const int device
){
    cudaSetDevice(device);
    // 
    T rcinv_angular = 1.0 / rcut_angular;
    int atom_types_sq = atom_types * atom_types;
    int total_elements = atom_nums * maxneighs;
    int threads_per_block = 256;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    const int N = atom_nums;
    GPU_Vector<T> dfeat_c3; //(N * maxneighs * atom_types * n_max_3b * n_base_3b, 0.0);
    GPU_Vector<int> unique_types;     // atom_map 中元素的不重复子表，顺序为它在atom_map中的顺序 例如 对于[0,1,29,4,39,1,1,1,1,1,4,29,...]这个atom_map,这里为[0,1,29,4,39], num_unique=5
    GPU_Vector<int> num_unique;
    GPU_Vector<int> unique_types_map; // 不重复子表在周期表中位置，例如这里为[0,1,-1,-1,3,-1,...,2,-1,...,4,-1,-1,...]，即它的下标[0,1,4,29,39]值分别为[0,1,3,2,4]，周期表其他位置填充-1
    GPU_Vector<T> tmp_dfeat_c3(N * atom_types * n_max_3b * n_base_3b, 0.0);
    
    int cpu_num_value = 0;
    
    unique_types.resize(100, -1);
    num_unique.resize(1, 0);
    unique_types_map.resize(100, -1);
    buildTypeMapKernel<<<(N + 256 - 1) / 256, 256>>>(atom_map, unique_types_map.data(), unique_types.data(), atom_nums, num_unique.data());
    std::vector<int> cpu_num_unique(1);
    num_unique.copy_to_host(cpu_num_unique.data());
    cpu_num_value = cpu_num_unique[0];

    // std::vector<int> cpu_unique_types_map(100);
    // unique_types_map.copy_to_host(cpu_unique_types_map.data());
    // for(int ii = 0; ii < 100; ii++){
    //     printf("I-%d=%d ", ii, cpu_unique_types_map[ii]);
    // }
    // printf("\n");
    // std::vector<int> cpu_unique_types(100);
    // unique_types.copy_to_host(cpu_unique_types.data());
    // for(int ii = 0; ii < 100; ii++){
    //     printf("U-%d=%d ", ii, cpu_unique_types[ii]);
    // }
    // printf("\n");
    // printf("num_unique types is %d\n", cpu_num_value);

    // printf("N %d maxneighs %d cpu_num_value %d n_max_3b %d n_base_3b %d atom_types %d\n",N, maxneighs, cpu_num_value, n_max_3b, n_base_3b, atom_types);
    dfeat_c3.resize(N * maxneighs * n_max_3b * cpu_num_value * n_base_3b, 0.0);

    find_angular_gardc_neigh_bk<<<num_blocks, threads_per_block>>>(
        total_elements,
        grad_second,
        d12, 
        NL,
        de_dfeat-feat_2b_num, 
        dsnlm_dc,
        sum_fxyz,
        atom_map,                // for a list of atom type: [55, 19, 79, 79]
        unique_types_map.data(), //the value of index 55 19 79 is 0 1 2 
        unique_types.data(),     // the value of index 0 1 2 is 55 19 79 
        cpu_num_value,
        coeff3,
        dfeat_c3.data(),
        rcut_angular,
        rcinv_angular,
        atom_nums, 
        maxneighs, 
        n_max_3b,
        n_base_3b,
        atom_types,
        atom_types_sq,
        lmax_3,
        lmax_4,
        lmax_5,
        feat_2b_num,
        multi_feat_num
        );
    CUDA_CHECK_KERNEL
    cudaDeviceSynchronize();

    // std::vector<double> cpu_dfeat_c3(N * maxneighs * n_max_3b * cpu_num_value * n_base_3b);
    // dfeat_c3.copy_to_host(cpu_dfeat_c3.data());
    // double tmptest = 0.0;
    // for(int i=0; i < N;i++) {
    //     if (i > 1) continue;
    //     int id=i * maxneighs * n_max_3b * cpu_num_value * n_base_3b;
    //     for(int j=0;j < maxneighs; j++) {
    //         int jd=id + j * n_max_3b * cpu_num_value * n_base_3b;
    //         for(int k=0; k < cpu_num_value;k++) {
    //             for(int h=0;h < n_max_3b; h++){
    //                 
    //                 int kd = jd + h * cpu_num_value * n_base_3b + k * n_base_3b;
    //                 printf("dfeat_c3[n%d m%d t%d mx%d]=",i,j,k,h);
    //                 for(int p=0;p<n_base_3b;p++){
    //                     printf(" %f ", cpu_dfeat_c3[kd+p]);
    //                     tmptest += cpu_dfeat_c3[kd+p];
    //                 }
    //                 printf("\n");
    //             }
    //         }
    //     }
    // }
    // printf("=====res %.17lf =====\n", tmptest);

    aggregate_dfeat_c3<<<(N - 1) / 64 + 1, 64>>>(
        NL,
        atom_map,
        dfeat_c3.data(),
        tmp_dfeat_c3.data(),
        unique_types.data(),
        cpu_num_value,
        N,
        maxneighs, 
        atom_types,
        n_max_3b,
        n_base_3b
    );
    CUDA_CHECK_KERNEL
    cudaDeviceSynchronize();

    // std::vector<double> cpu_tmp_dfeat_c3(N * atom_types * n_max_3b * n_base_3b);
    // tmp_dfeat_c3.copy_to_host(cpu_tmp_dfeat_c3.data());
    // for(int i=0; i < N;i++) {
    //     if (i > 1) continue;
    //     int id=i * atom_types * n_max_3b * n_base_3b;
    //     for(int j=0;j < atom_types; j++) {
    //         int jd=id + j*n_max_3b * n_base_3b;
    //         for(int k=0; k< n_max_3b;k++) {
    //             int kd=jd + k*n_base_3b;
    //             printf("OPT[N%d T%d NM%d]=",i,j,k);
    //             for(int p=0;p<n_base_3b;p++){
    //                 printf(" %f ", cpu_tmp_dfeat_c3[kd+p]);
    //             }
    //             printf("\n");
    //         }
    //     }
    // }

    total_elements = N * n_max_3b * n_base_3b;
    threads_per_block = 256;
    num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    aggregate_features<T><<<num_blocks, threads_per_block>>>(
    tmp_dfeat_c3.data(), 
    atom_map, 
    gradsecond_c3, 
    N, 
    atom_types, 
    n_max_3b, 
    n_base_3b);   
    CUDA_CHECK_KERNEL
    cudaDeviceSynchronize();
}

template <typename T>
void launch_calculate_nepmbfeat_secondgradout_c3(
    const T * grad_second,
    const T * d12,
    const int64_t * NL,
    const T * de_dfeat,
    const T * dsnlm_dc,
    const T * sum_fxyz,
    const int64_t * atom_map,
    const T * coeff3,
    T * gradsecond_c3,
    const T rcut_angular,
    const int atom_nums, 
    const int maxneighs, 
    const int n_max_3b, 
    const int n_base_3b, 
    const int atom_types, 
    const int lmax_3,
    const int lmax_4,
    const int lmax_5,
    const int feat_2b_num,
    const int multi_feat_num,
    const int device
){
    cudaSetDevice(device);
    T rcinv_angular = 1.0 / rcut_angular;
    int atom_types_sq = atom_types * atom_types;
    // 计算共享内存大小
    const int uniq_map_size = 100;
    const int uniq_type_size = 100;
    const int Fp_size = MAX_DIM_ANGULAR;
    const int sum_fxyz_size = n_max_3b * NUM_OF_ABC;
    
    size_t shared_mem_size = 
        uniq_map_size * sizeof(int) + 
        uniq_type_size * sizeof(int) + 
        Fp_size * sizeof(T) + 
        sum_fxyz_size * sizeof(T);
    // 对齐到256字节边界
    shared_mem_size = (shared_mem_size + 255) & ~255;
    // 检查共享内存是否超过限制
    // cudaDeviceProp prop;
    // cudaGetDeviceProperties(&prop, device);
    // printf("Shared memory required (%zu bytes) exceeds device limit (%zu bytes)\n",\
               shared_mem_size, prop.sharedMemPerBlock);
    // 每个线程块处理一个原子，每个线程处理一个近邻
    int threads_per_block = 256;
    if (maxneighs < 256) {
        threads_per_block = 256;  // 保持warp对齐
    } else if (maxneighs > 1024) {
        threads_per_block = 1024;  // 最大线程数
    } else {
        // 找到最接近maxneighs的32的倍数
        threads_per_block = ((maxneighs + 31) / 32) * 32;
    }
    int num_blocks = atom_nums;  // 每个块处理一个中心原子
    
    int cpu_num_value = 0;

    GPU_Vector<T> tmp_dfeat_c3(atom_nums * atom_types * n_max_3b * n_base_3b, 0.0);
    GPU_Vector<int> unique_types(100, -1);     // atom_map 中元素的不重复子表，顺序为它在atom_map中的顺序 例如 对于[0,1,29,4,39,1,1,1,1,1,4,29,...]这个atom_map,这里为[0,1,29,4,39], num_unique=5
    GPU_Vector<int> num_unique(1, 0);
    GPU_Vector<int> unique_types_map(100, -1); // 不重复子表在周期表中位置，例如这里为[0,1,-1,-1,3,-1,...,2,-1,...,4,-1,-1,...]，即它的下标[0,1,4,29,39]值分别为[0,1,3,2,4]，周期表其他位置填充-1
    buildTypeMapKernel<<<(atom_nums + 256 - 1) / 256, 256>>>(
        atom_map, unique_types_map.data(), unique_types.data(), 
        atom_nums, num_unique.data());
    
    std::vector<int> cpu_num_unique(1);
    num_unique.copy_to_host(cpu_num_unique.data());
    cpu_num_value = cpu_num_unique[0];
    // printf("  cpu_num_value: %d\n", cpu_num_value);

    GPU_Vector<T> dfeat_c3(atom_nums * maxneighs * n_max_3b * cpu_num_value * n_base_3b, 0.0);; //(N * maxneighs * atom_types * n_max_3b * n_base_3b, 0.0);

    find_angular_gardc_neigh<T><<<num_blocks, threads_per_block, shared_mem_size>>>(
        atom_nums * maxneighs,  // 总元素数
        grad_second,
        d12, 
        NL,
        de_dfeat - feat_2b_num, 
        dsnlm_dc,
        sum_fxyz,
        atom_map,
        unique_types_map.data(),
        unique_types.data(),
        cpu_num_value,
        coeff3,
        dfeat_c3.data(),
        rcut_angular,
        rcinv_angular,
        atom_nums, 
        maxneighs, 
        n_max_3b,
        n_base_3b,
        atom_types,
        atom_types_sq,
        lmax_3,
        lmax_4,
        lmax_5,
        feat_2b_num,
        multi_feat_num
    );
    CUDA_CHECK_KERNEL
    cudaDeviceSynchronize();

    // std::vector<double> cpu_dfeat_c3(N * maxneighs * n_max_3b * cpu_num_value * n_base_3b);
    // dfeat_c3.copy_to_host(cpu_dfeat_c3.data());
    // double tmptest = 0.0;
    // for(int i=0; i < N;i++) {
    //     if (i > 1) continue;
    //     int id=i * maxneighs * n_max_3b * cpu_num_value * n_base_3b;
    //     for(int j=0;j < maxneighs; j++) {
    //         int jd=id + j * n_max_3b * cpu_num_value * n_base_3b;
    //         for(int k=0; k < cpu_num_value;k++) {
    //             for(int h=0;h < n_max_3b; h++){
    //                 
    //                 int kd = jd + h * cpu_num_value * n_base_3b + k * n_base_3b;
    //                 printf("dfeat_c3[n%d m%d t%d mx%d]=",i,j,k,h);
    //                 for(int p=0;p<n_base_3b;p++){
    //                     printf(" %f ", cpu_dfeat_c3[kd+p]);
    //                     tmptest += cpu_dfeat_c3[kd+p];
    //                 }
    //                 printf("\n");
    //             }
    //         }
    //     }
    // }
    // printf("=====res %.17lf =====\n", tmptest);

    aggregate_dfeat_c3<T><<<(atom_nums - 1) / 64 + 1, 64>>>(
        NL,
        atom_map,
        dfeat_c3.data(),
        tmp_dfeat_c3.data(),
        unique_types.data(),
        cpu_num_value,
        atom_nums,
        maxneighs, 
        atom_types,
        n_max_3b,
        n_base_3b
    );
    CUDA_CHECK_KERNEL
    cudaDeviceSynchronize();

    int total_elements = atom_nums * n_max_3b * n_base_3b;
    threads_per_block = 256;
    num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    aggregate_features<T><<<num_blocks, threads_per_block>>>(
    tmp_dfeat_c3.data(), 
    atom_map, 
    gradsecond_c3, 
    atom_nums, 
    atom_types, 
    n_max_3b, 
    n_base_3b);   
    CUDA_CHECK_KERNEL
    cudaDeviceSynchronize();
}
// Explicit instantiations
template void launch_calculate_nepmbfeat_secondgradout<double>(
    const double*, const double*, double*, const int, const int, const int, const int);
template void launch_calculate_nepmbfeat_secondgradout<float>(
    const float*, const float*, float*, const int, const int, const int, const int);

template void launch_calculate_nepmbfeat_secondgradout_c3_bk<double>(
    const double*, const double*, const int64_t*, const double*, const double*,
    const double*, const int64_t*, const double*, double*,
    const double, const int, const int, const int, const int, const int,
    const int, const int, const int, const int, const int, const int);
template void launch_calculate_nepmbfeat_secondgradout_c3_bk<float>(
    const float*, const float*, const int64_t*, const float*, const float*,
    const float*, const int64_t*, const float*, float*,
    const float, const int, const int, const int, const int, const int,
    const int, const int, const int, const int, const int, const int);

template void launch_calculate_nepmbfeat_secondgradout_c3<double>(
    const double*, const double*, const int64_t*, const double*, const double*,
    const double*, const int64_t*, const double*, double*,
    const double, const int, const int, const int, const int, const int,
    const int, const int, const int, const int, const int, const int);
template void launch_calculate_nepmbfeat_secondgradout_c3<float>(
    const float*, const float*, const int64_t*, const float*, const float*,
    const float*, const int64_t*, const float*, float*,
    const float, const int, const int, const int, const int, const int,
    const int, const int, const int, const int, const int, const int);
