#include "./utilities/error.cuh"
#include "./utilities/gpu_vector.cuh"
#include "./utilities/nep_utilities.cuh"
#include "./utilities/nep_feature.cuh"
#include <iostream>

template <typename T>
void launch_calculate_nepmbfeat_grad(
            const T * grad_output,
            const T * coeff3, 
            const T * r12,
            const int64_t * NL, 
            const int64_t * atom_map, 
            T * sum_fxyz,
            T * grad_coeff3, 
            T * grad_d12_3b,
            T * dsnlm_dc, // dsnlm/dc_NK_IJ used in second grad mb c
            T * dfeat_drij,
            const T rcut_angular,
            const int atom_nums, 
            const int neigh_num, 
            const int feat_2b_num, 
            const int n_max_3b, 
            const int n_base_3b,
            const int lmax_3,
            const int lmax_4,
            const int lmax_5,
            const int n_types, 
            const int device_id
) {
    cudaSetDevice(device_id);
    const int N = atom_nums; // N = natoms * batch_size
    const int num_types_sq = n_types * n_types;
    T rcinv_angular = 1.0 / rcut_angular;
    
    int feat_3b_num = 0;
    if (lmax_3 > 0) feat_3b_num += n_max_3b * lmax_3;
    if (lmax_4 > 0) feat_3b_num += n_max_3b;
    if (lmax_5 > 0) feat_3b_num += n_max_3b;

    // 计算共享内存大小
    const int Fp_size = MAX_DIM_ANGULAR;
    const int sum_fxyz_size = n_max_3b * NUM_OF_ABC;
    size_t shared_mem_size = Fp_size * sizeof(T) + sum_fxyz_size * sizeof(T);
    // 对齐到256字节边界
    shared_mem_size = (shared_mem_size + 255) & ~255;
    int maxneighs = neigh_num;
    int threads_per_block = 256;
    if (maxneighs < 256) {
        threads_per_block = 256;  // 保持warp对齐，即使近邻数少也充分利用SM
    } else if (maxneighs > 1024) {
        threads_per_block = 1024;  // 最大线程数
    } else {
        // 找到最接近maxneighs的32的倍数
        threads_per_block = ((maxneighs + 31) / 32) * 32;
    }
    // printf("nepmbfeat_grad dfeat_c3 N * n_types * n_max_3b * n_base_3b=%d %d %d %d\n", N, n_types, n_max_3b, n_base_3b);
    GPU_Vector<T> dfeat_c3(N * n_types * n_max_3b * n_base_3b, 0.0);
    // const int BLOCK_SIZE = 64;
    const int grid_size = N; // 每个block处理一个中心原子
    // find_angular_gard<<<grid_size, BLOCK_SIZE>>>(
    find_angular_gard<T><<<grid_size, threads_per_block, shared_mem_size>>>(
        N,
        n_types,
        num_types_sq,
        neigh_num,
        lmax_3,
        lmax_4,
        lmax_5,
        feat_2b_num, 
        feat_3b_num,
        rcut_angular,
        rcinv_angular,
        n_max_3b, 
        n_base_3b,
        NL,
        r12,
        coeff3,
        atom_map,
        grad_output - feat_2b_num,
        sum_fxyz,
        dsnlm_dc,
        dfeat_c3.data(),
        dfeat_drij,//[batch*atom, neighbornum, 3b_feat_num, 4]
        grad_d12_3b
    );
    CUDA_CHECK_KERNEL

    // std::vector<double> cpu_dfeat_c3(N * n_types * n_max_3b * n_base_3b);
    // dfeat_c3.copy_to_host(cpu_dfeat_c3.data());
    // double tmptest = 0.0;
    // for(int i=0; i < N;i++) {
    //     if (i > 1) continue;
    //     int id=i * n_types * n_max_3b * n_base_3b;
    //     for(int j=0;j < n_types; j++) {
    //         int jd=id + j*n_max_3b * n_base_3b;
    //         for(int k=0; k< n_max_3b;k++) {
    //             int kd=jd + k*n_base_3b;
    //             printf("dfeat_c3[n%d t%d mx%d]=",i,j,k);
    //             for(int p=0;p<n_base_3b;p++){
    //                 printf(" %f ", cpu_dfeat_c3[kd+p]);
    //                 tmptest += cpu_dfeat_c3[kd+p];
    //             }
    //             printf("\n");
    //         }
    //     }
    // }
    // printf("======test %.17lf ======\n\n\n", tmptest);

    // std::vector<double> cpu_dfeat_drij(N * neigh_num * 4);
    // cudaMemcpy(cpu_dfeat_drij.data(), dfeat_drij, N * neigh_num * 4 * sizeof(double) , cudaMemcpyDeviceToHost);
    // double tmpdrij = 0.0;
    // for(int i=0; i < N;i++) {
    //     if (i > 1) continue;
    //     int id=i * neigh_num * 4;
    //     for(int j=0;j < neigh_num; j++) {
    //         int jd=id + j*4;
    //         printf("dfeat_drij[n%d j%d]= %f %f %f %f\n",i, j, \
    //         cpu_dfeat_drij[jd], cpu_dfeat_drij[jd+1], cpu_dfeat_drij[jd+2], cpu_dfeat_drij[jd+3]);
    //         tmpdrij += (cpu_dfeat_drij[jd], cpu_dfeat_drij[jd+1], cpu_dfeat_drij[jd+2], cpu_dfeat_drij[jd+3]);
    //     }
    // }
    // printf("======test %.17lf ======\n", tmpdrij);

    // print_dfeat_c3(dfeat_c3.data(), N, n_types, n_max_3b, n_base_3b);

    int total_elements = N * n_max_3b * n_base_3b;
    threads_per_block = 256;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    aggregate_features<T><<<num_blocks, threads_per_block>>>(
    dfeat_c3.data(), 
    atom_map, 
    grad_coeff3, 
    N, 
    n_types, 
    n_max_3b, 
    n_base_3b);
    CUDA_CHECK_KERNEL
    // print_grad_coeff3(grad_coeff3, n_types, n_max_3b, n_base_3b);

}


// int calculate_optimal_threads_per_block(int n_max_3b, int device_id) {
//     cudaDeviceProp props;
//     cudaGetDeviceProperties(&props, device_id);
    
//     size_t per_thread_memory = sizeof(double) * 
//         (MAX_DIM_ANGULAR + NUM_OF_ABC * MAX_NUM_N);
//     // 计算最大线程数（基于共享内存限制）
//     int max_threads_by_mem = props.sharedMemPerBlock / per_thread_memory;
//     // 设置合理的范围
//     const int MIN_THREADS = 1;
//     const int MAX_THREADS = 256;
//     const int WARP_SIZE = 32;
//     // 限制在合理范围内
//     int threads = max(MIN_THREADS, min(MAX_THREADS, max_threads_by_mem));
//     // 如果大于等于32，调整到warp的倍数
//     if (threads >= WARP_SIZE) {
//         threads = (threads / WARP_SIZE) * WARP_SIZE;
//     }
//     // 确保不超过硬件限制
//     threads = min(threads, props.maxThreadsPerBlock);
//     // printf("thread %d\n", threads);
//     return threads;
// }

// void launch_calculate_nepmbfeat_grad(
//             const double * grad_output,
//             const double * coeff3, 
//             const double * r12,
//             const int64_t * NL, 
//             const int64_t * atom_map, 
//             double * sum_fxyz,
//             double * grad_coeff3, 
//             double * grad_d12_3b,
//             double * dsnlm_dc, // dsnlm/dc_NK_IJ used in second grad mb c
//             double * dfeat_drij,
//             const double rcut_angular,
//             const int atom_nums, 
//             const int neigh_num, 
//             const int feat_2b_num, 
//             const int n_max_3b, 
//             const int n_base_3b,
//             const int lmax_3,
//             const int lmax_4,
//             const int lmax_5,
//             const int n_types, 
//             const int device_id
// ) {
//     cudaSetDevice(device_id);
//     const int num_types_sq = n_types * n_types;
//     double rcinv_angular = 1.0 / rcut_angular;
    
//     int feat_3b_num = 0;
//     if (lmax_3 > 0) feat_3b_num += n_max_3b * lmax_3;
//     if (lmax_4 > 0) feat_3b_num += n_max_3b;
//     if (lmax_5 > 0) feat_3b_num += n_max_3b;

//     int THREADS_PER_BLOCK = calculate_optimal_threads_per_block(n_max_3b, device_id);  // 根据共享内存大小调整
//     THREADS_PER_BLOCK = 8;
//     // 计算共享内存大小
//     const int Fp_size = MAX_DIM_ANGULAR;
//     const int sum_fxyz_size = n_max_3b * NUM_OF_ABC;
//     size_t shared_mem_size = THREADS_PER_BLOCK * (Fp_size + sum_fxyz_size) * sizeof(double);
//     // printf("threadnum %d Fp_size %d fxyz %d shared_mem_size %d\n", THREADS_PER_BLOCK, Fp_size, sum_fxyz_size, shared_mem_size);
//     // 对齐到256字节边界
//     shared_mem_size = (shared_mem_size + 255) & ~255;
//     // printf("aline threadnum %d THREADS_PER_BLOCK %d \n", shared_mem_size, THREADS_PER_BLOCK);

//     GPU_Vector<double> dfeat_c3(atom_nums * n_types * n_max_3b * n_base_3b, 0.0);
//     const int grid_size = (atom_nums + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
//     find_angular_gard<<<grid_size, THREADS_PER_BLOCK, shared_mem_size>>>(
//         atom_nums,
//         n_types,
//         num_types_sq,
//         neigh_num,
//         lmax_3,
//         lmax_4,
//         lmax_5,
//         feat_2b_num, 
//         feat_3b_num,
//         rcut_angular,
//         rcinv_angular,
//         n_max_3b, 
//         n_base_3b,
//         NL,
//         r12,
//         coeff3,
//         atom_map,
//         grad_output - feat_2b_num,
//         sum_fxyz,
//         dsnlm_dc,
//         dfeat_c3.data(),
//         dfeat_drij,//[batch*atom, neighbornum, 3b_feat_num, 4]
//         grad_d12_3b
//     );
//     CUDA_CHECK_KERNEL

//     // std::vector<double> cpu_dfeat_c3(atom_nums * n_types * n_max_3b * n_base_3b);
//     // dfeat_c3.copy_to_host(cpu_dfeat_c3.data());
//     // double tmptest = 0.0;
//     // for(int i=0; i < atom_nums;i++) {
//     //     if (i > 1) continue;
//     //     int id=i * n_types * n_max_3b * n_base_3b;
//     //     for(int j=0;j < n_types; j++) {
//     //         int jd=id + j*n_max_3b * n_base_3b;
//     //         for(int k=0; k< n_max_3b;k++) {
//     //             int kd=jd + k*n_base_3b;
//     //             printf("dfeat_c3[n%d t%d mx%d]=",i,j,k);
//     //             for(int p=0;p<n_base_3b;p++){
//     //                 printf(" %f ", cpu_dfeat_c3[kd+p]);
//     //                 tmptest += cpu_dfeat_c3[kd+p];
//     //             }
//     //             printf("\n");
//     //         }
//     //     }
//     // }
//     // printf("======test %.17lf ======\n\n\n", tmptest);

//     // std::vector<double> cpu_dfeat_drij(atom_nums * neigh_num * 4);
//     // cudaMemcpy(cpu_dfeat_drij.data(), dfeat_drij, atom_nums * neigh_num * 4 * sizeof(double) , cudaMemcpyDeviceToHost);
//     // double tmpdrij = 0.0;
//     // for(int i=0; i < atom_nums;i++) {
//     //     if (i > 1) continue;
//     //     int id=i * neigh_num * 4;
//     //     for(int j=0;j < neigh_num; j++) {
//     //         int jd=id + j*4;
//     //         printf("dfeat_drij[n%d j%d]= %f %f %f %f\n",i, j, \
//     //         cpu_dfeat_drij[jd], cpu_dfeat_drij[jd+1], cpu_dfeat_drij[jd+2], cpu_dfeat_drij[jd+3]);
//     //         tmpdrij += (cpu_dfeat_drij[jd], cpu_dfeat_drij[jd+1], cpu_dfeat_drij[jd+2], cpu_dfeat_drij[jd+3]);
//     //     }
//     // }
//     // printf("======test %.17lf ======\n", tmpdrij);

//     int total_elements = atom_nums * n_max_3b * n_base_3b;
//     int threads_per_block = 256;
//     int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
//     aggregate_features<T><<<num_blocks, threads_per_block>>>(
//     dfeat_c3.data(), 
//     atom_map, 
//     grad_coeff3, 
//     atom_nums, 
//     n_types, 
//     n_max_3b, 
//     n_base_3b);
//     CUDA_CHECK_KERNEL

//     // print_grad_coeff3(grad_coeff3, n_types, n_max_3b, n_base_3b);

// }

// 把sum_fxyz Fp 放入共享内存区，线程粒度为中心原子，速度提升有限
// void launch_calculate_nepmbfeat_grad_bk(
//             const double * grad_output,
//             const double * coeff3, 
//             const double * r12,
//             const int64_t * NL, 
//             const int64_t * atom_map, 
//             double * sum_fxyz,
//             double * grad_coeff3, 
//             double * grad_d12_3b,
//             double * dsnlm_dc, // dsnlm/dc_NK_IJ used in second grad mb c
//             double * dfeat_drij,
//             const double rcut_angular,
//             const int atom_nums, 
//             const int neigh_num, 
//             const int feat_2b_num, 
//             const int n_max_3b, 
//             const int n_base_3b,
//             const int lmax_3,
//             const int lmax_4,
//             const int lmax_5,
//             const int n_types, 
//             const int device_id
// ) {
//     cudaSetDevice(device_id);
//     const int N = atom_nums; // N = natoms * batch_size
//     const int num_types_sq = n_types * n_types;
//     double rcinv_angular = 1.0 / rcut_angular;
    
//     int feat_3b_num = 0;
//     if (lmax_3 > 0) feat_3b_num += n_max_3b * lmax_3;
//     if (lmax_4 > 0) feat_3b_num += n_max_3b;
//     if (lmax_5 > 0) feat_3b_num += n_max_3b;

//     // 计算共享内存大小
//     const int Fp_size = MAX_DIM_ANGULAR;
//     const int sum_fxyz_size = n_max_3b * NUM_OF_ABC;
//     size_t shared_mem_size = Fp_size * sizeof(double) + sum_fxyz_size * sizeof(double);
//     // 对齐到256字节边界
//     shared_mem_size = (shared_mem_size + 255) & ~255;
//     int maxneighs = neigh_num;
//     int threads_per_block = 256;
//     if (maxneighs < 256) {
//         threads_per_block = 256;  // 保持warp对齐，即使近邻数少也充分利用SM
//     } else if (maxneighs > 1024) {
//         threads_per_block = 1024;  // 最大线程数
//     } else {
//         // 找到最接近maxneighs的32的倍数
//         threads_per_block = ((maxneighs + 31) / 32) * 32;
//     }
//     // printf("nepmbfeat_grad dfeat_c3 N * n_types * n_max_3b * n_base_3b=%d %d %d %d\n", N, n_types, n_max_3b, n_base_3b);
//     GPU_Vector<double> dfeat_c3(N * n_types * n_max_3b * n_base_3b, 0.0);
//     // const int BLOCK_SIZE = 64;
//     const int grid_size = N; // 每个block处理一个中心原子
//     // find_angular_gard<<<grid_size, BLOCK_SIZE>>>(
//     find_angular_gard_bk<<<grid_size, threads_per_block, shared_mem_size>>>(
//         N,
//         n_types,
//         num_types_sq,
//         neigh_num,
//         lmax_3,
//         lmax_4,
//         lmax_5,
//         feat_2b_num, 
//         feat_3b_num,
//         rcut_angular,
//         rcinv_angular,
//         n_max_3b, 
//         n_base_3b,
//         NL,
//         r12,
//         coeff3,
//         atom_map,
//         grad_output - feat_2b_num,
//         sum_fxyz,
//         dsnlm_dc,
//         dfeat_c3.data(),
//         dfeat_drij,//[batch*atom, neighbornum, 3b_feat_num, 4]
//         grad_d12_3b
//     );
//     CUDA_CHECK_KERNEL

//     // std::vector<double> cpu_dfeat_c3(N * n_types * n_max_3b * n_base_3b);
//     // dfeat_c3.copy_to_host(cpu_dfeat_c3.data());
//     // double tmptest = 0.0;
//     // for(int i=0; i < N;i++) {
//     //     if (i > 1) continue;
//     //     int id=i * n_types * n_max_3b * n_base_3b;
//     //     for(int j=0;j < n_types; j++) {
//     //         int jd=id + j*n_max_3b * n_base_3b;
//     //         for(int k=0; k< n_max_3b;k++) {
//     //             int kd=jd + k*n_base_3b;
//     //             printf("dfeat_c3[n%d t%d mx%d]=",i,j,k);
//     //             for(int p=0;p<n_base_3b;p++){
//     //                 printf(" %f ", cpu_dfeat_c3[kd+p]);
//     //                 tmptest += cpu_dfeat_c3[kd+p];
//     //             }
//     //             printf("\n");
//     //         }
//     //     }
//     // }
//     // printf("======test %.17lf ======\n\n\n", tmptest);

//     // std::vector<double> cpu_dfeat_drij(N * neigh_num * 4);
//     // cudaMemcpy(cpu_dfeat_drij.data(), dfeat_drij, N * neigh_num * 4 * sizeof(double) , cudaMemcpyDeviceToHost);
//     // double tmpdrij = 0.0;
//     // for(int i=0; i < N;i++) {
//     //     if (i > 1) continue;
//     //     int id=i * neigh_num * 4;
//     //     for(int j=0;j < neigh_num; j++) {
//     //         int jd=id + j*4;
//     //         printf("dfeat_drij[n%d j%d]= %f %f %f %f\n",i, j, \
//     //         cpu_dfeat_drij[jd], cpu_dfeat_drij[jd+1], cpu_dfeat_drij[jd+2], cpu_dfeat_drij[jd+3]);
//     //         tmpdrij += (cpu_dfeat_drij[jd], cpu_dfeat_drij[jd+1], cpu_dfeat_drij[jd+2], cpu_dfeat_drij[jd+3]);
//     //     }
//     // }
//     // printf("======test %.17lf ======\n", tmpdrij);

//     // print_dfeat_c3(dfeat_c3.data(), N, n_types, n_max_3b, n_base_3b);

//     int total_elements = N * n_max_3b * n_base_3b;
//     threads_per_block = 256;
//     int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
//     aggregate_features<T><<<num_blocks, threads_per_block>>>(
//     dfeat_c3.data(), 
//     atom_map, 
//     grad_coeff3, 
//     N, 
//     n_types, 
//     n_max_3b, 
//     n_base_3b);
//     CUDA_CHECK_KERNEL
//     // print_grad_coeff3(grad_coeff3, n_types, n_max_3b, n_base_3b);

// }
// Explicit instantiations
template void launch_calculate_nepmbfeat_grad<double>(
    const double*, const double*, const double*, const int64_t*, const int64_t*,
    double*, double*, double*, double*, double*,
    const double, const int, const int, const int, const int, const int,
    const int, const int, const int, const int, const int);
template void launch_calculate_nepmbfeat_grad<float>(
    const float*, const float*, const float*, const int64_t*, const int64_t*,
    float*, float*, float*, float*, float*,
    const float, const int, const int, const int, const int, const int,
    const int, const int, const int, const int, const int);

