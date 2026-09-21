#include "hip/hip_runtime.h"
#include "./utilities/common.cuh"
#include "./utilities/nep_mbgrad_opt.cuh"
#include "./utilities/nep3_small_box.cuh"
#include "./utilities/nep3_small_box_mbgrad.cuh"
#include "./utilities/nep3_c3_wave_opt_legacy.cuh"
#include "./utilities/error.cuh"
#include "./utilities/gpu_vector.cuh"
#include <algorithm>
#include <c10/hip/HIPStream.h>
#include <c10/hip/HIPCachingAllocator.h>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <hip/hip_runtime.h>

static bool nep_mb_secondgrad_omat24_supported(
    int n_max_3b, int n_base_3b, int atom_types, int lmax_3,
    int lmax_4, int lmax_5) {
    return n_max_3b == 6 && n_base_3b == 9 && atom_types >= 1 &&
           atom_types <= NEP_MAX_ELEMENT_TYPES && lmax_3 == 4 &&
           lmax_4 > 0 && lmax_5 > 0;
}

bool should_recompute_nep_mb_dsnlm(
    const int n_max_3b, const int n_base_3b, const int atom_types,
    const int lmax_3, const int lmax_4, const int lmax_5, const int) {
    return nep_mb_secondgrad_omat24_supported(
        n_max_3b, n_base_3b, atom_types, lmax_3, lmax_4, lmax_5);
}

static void launch_nep_mb_secondgrad_omat24(
    const NepMbSecondGradArgs& args, int device) {
    hipSetDevice(device);
    if (args.atom_count == 0) return;
    constexpr size_t shared_bytes = nep_mb_secondgrad_shared_bytes<6, 9, 4>();
    const auto stream = c10::hip::getCurrentHIPStream(device);
    constexpr int angular_values_per_neighbor = 138;
    constexpr int weighted_values_per_center = 6 * 24;
    const size_t angular_values =
        static_cast<size_t>(args.atom_count) * args.max_neighbors *
        angular_values_per_neighbor;
    const size_t workspace_bytes =
        (angular_values + static_cast<size_t>(args.atom_count) *
         weighted_values_per_center) * sizeof(double);
    auto workspace_deleter = [](void* ptr) {
        c10::hip::HIPCachingAllocator::raw_delete(ptr);
    };
    std::unique_ptr<void, decltype(workspace_deleter)> angular_workspace(
        c10::hip::HIPCachingAllocator::raw_alloc_with_stream(
            workspace_bytes, stream.stream()),
        workspace_deleter);
    double* workspace = static_cast<double*>(angular_workspace.get());
    NepMbSecondGradArgs kernel_args = args;
    kernel_args.angular_workspace = workspace;
    kernel_args.weighted_sums = workspace + angular_values;
    const int rows = std::max(
        args.atom_count * args.max_neighbors, args.atom_count * 6);
    constexpr int precompute_threads = 256;
    nep_mb_secondgrad_precompute_angular<precompute_threads><<<
        (rows + precompute_threads - 1) / precompute_threads,
        precompute_threads, 0, stream.stream()>>>(kernel_args);
    CUDA_CHECK_KERNEL
    nep_mb_secondgrad_fused<6, 9, 4, true, true, 4, 128><<<
        args.atom_count, 128, shared_bytes, stream.stream()>>>(kernel_args);
    CUDA_CHECK_KERNEL
}

__global__ void compute_gradsecond_mbgradout(
    const double *grad_second, // Shape: [batch_size, atom_nums, maxneighs, 4]
    const double *dfeat_drij,    // Shape: [batch_size, atom_nums, maxneighs, feat_mb_num, 4]
    double *gradsecond_gradout, // Shape: [batch_size, atom_nums, feat_mb_num]
    int atom_nums,
    int maxneighs,
    int feat_mb_num)
{
    int atom_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int feat_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (atom_idx < atom_nums && feat_idx < feat_mb_num) {
        float sum = 0.0f;
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

void launch_calculate_nepmbfeat_secondgradout(
    const double * grad_second,
    const double * dfeat_b,
    double * gradsecond_gradout,
    const int atom_nums,
    const int maxneighs,
    const int feat_mb_nums,
    const int device
) {
    hipSetDevice(device);
    dim3 blockDim(16, 16);
    dim3 gridDim((atom_nums + blockDim.x - 1) / blockDim.x, (feat_mb_nums + blockDim.y - 1) / blockDim.y);
    const auto stream = c10::hip::getCurrentHIPStream(device);
    compute_gradsecond_mbgradout<<<gridDim, blockDim, 0, stream.stream()>>>(
        grad_second,
        dfeat_b,
        gradsecond_gradout,
        atom_nums,
        maxneighs,
        feat_mb_nums
        );

    CUDA_CHECK_KERNEL
}

void launch_calculate_nepmbfeat_secondgradout_c3(
    const double * grad_second,
    const double * d12,
    const int64_t * NL,
    const double * de_dfeat,
    const double * dsnlm_dc,
    const double * sum_fxyz,
    const int64_t * atom_map,
    const double * coeff3,
    double * gradsecond_c3,
    const double rcut_angular,
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
    const bool recompute_dsnlm,
    const int device
){
    if (recompute_dsnlm) {
        const NepMbSecondGradArgs args{
            grad_second, d12, NL, de_dfeat, sum_fxyz, atom_map, coeff3,
            nullptr, nullptr, gradsecond_c3, rcut_angular,
            1.0 / rcut_angular, atom_nums,
            maxneighs, atom_types, feat_2b_num, multi_feat_num};
        launch_nep_mb_secondgrad_omat24(args, device);
        return;
    }
    hipSetDevice(device);
    // // 每个线程块的线程数 (这里选择 8 * 16 * 2 = 256，保证不会超过 1024)
    double rcinv_angular = 1.0 / rcut_angular;
    int atom_types_sq = atom_types * atom_types;
    // 按照邻居粒度存在一个问题，线程内分配的数组太多了，可能炸显存，但是速度会快，后面可以试试

    // const int BLOCK_SIZE = 64;
    // const int N = batch_size * atom_nums; // N = natoms * batch_size
    // const int grid_size = (N - 1) / BLOCK_SIZE + 1;

    // GPU_Vector<double> dfeat_c3(N * atom_types * n_max_3b * n_base_3b, 0.0);
    // find_angular_gardc_small_box<<<grid_size, BLOCK_SIZE>>>(
    //     N,
    //     grad_second,
    //     d12,
    //     NL,
    //     de_dfeat-feat_2b_num,
    //     dsnlm_dc,
    //     sum_fxyz,
    //     atom_map,
    //     coeff3,
    //     dfeat_c3.data(),
    //     rcut_angular,
    //     rcinv_angular,
    //     batch_size,
    //     atom_nums,
    //     maxneighs,
    //     n_max_3b,
    //     n_base_3b,
    //     atom_types,
    //     atom_types_sq,
    //     lmax_3,
    //     lmax_4,
    //     lmax_5,
    //     feat_2b_num,
    //     multi_feat_num
    //     );
    // hipDeviceSynchronize();
    // printf("==== tmp_c3 =====\n");
    // std::vector<double> tmp_c3(N * atom_types * n_max_3b * n_base_3b);
    // dfeat_c3.copy_to_host(tmp_c3.data());
    // for(int i = 0; i < N; i++) {
    //     for(int j = 0; j < atom_types; j++) {
    //         for (int n = 0; n < n_max_3b; n++) {
    //             printf("tmp_c3[i %d][J %d][n %d][k:] = ", i, j, n);
    //             for(int k = 0; k < n_base_3b; k++) {
    //                 printf("%f ", tmp_c3[i * atom_types * n_max_3b * n_base_3b + j * n_max_3b * n_base_3b + n * n_base_3b + k]);
    //             }
    //             printf("\n");
    //         }
    //     }
    // }

    int total_elements = atom_nums * maxneighs;
    int threads_per_block = 128;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    const int N = atom_nums;
    GPU_Vector<double> tmp_dfeat_c3(
        N * atom_types * n_max_3b * n_base_3b, 0.0);
    // find_angular_gardc_neigh_optimized<<<num_blocks, threads_per_block>>>(
    //     total_elements,
    // The LDS kernel is runtime-generic: only the reduction strategy is a
    // compile-time choice. Descriptor dimensions remain runtime parameters.
    // Keep the total below 48 KiB even when the device reports a larger limit:
    // this preserves sufficient occupancy on the supported DCU targets.
    int device_shared_limit = 0;
    hipDeviceGetAttribute(
        &device_shared_limit, hipDeviceAttributeMaxSharedMemoryPerBlock, device);
    const size_t static_shared_bytes =
        (NUM_OF_ABC * MAX_NUM_N + MAX_DIM_ANGULAR) * sizeof(double);
    const size_t c3_elements = static_cast<size_t>(atom_types) * n_max_3b * n_base_3b;
    const size_t dynamic_shared_bytes = c3_elements * sizeof(double);
    const size_t shared_limit = std::min(
        static_cast<size_t>(std::max(device_shared_limit, 0)),
        static_cast<size_t>(48 * 1024));
    const int fp_elements = n_max_3b *
        (lmax_3 + (lmax_4 > 0 ? 1 : 0) + (lmax_5 > 0 ? 1 : 0));
    const bool use_lds_opt =
        atom_types > 0 &&
        n_max_3b > 0 && n_max_3b <= MAX_NUM_N &&
        n_base_3b > 0 && n_base_3b <= MAX_NUM_N &&
        fp_elements >= 0 && fp_elements <= MAX_DIM_ANGULAR &&
        static_shared_bytes + dynamic_shared_bytes <= shared_limit;
    size_t selected_shared_bytes = static_shared_bytes + dynamic_shared_bytes;

    bool use_aot_specialization = false;
#define MATPL_NEP_C3_SPECIALIZATION(BASE, MAX, TYPES, L3, L4, L5) \
    { \
      constexpr size_t specialization_shared_bytes = \
          (NUM_OF_ABC * MAX + MAX * ((L3) + ((L5) > 0 ? 2 : 1)) + \
              14 * (5 * NUM_OF_ABC - 1) + 14 * BASE) * sizeof(double) + \
          sizeof(int); \
      if (!use_aot_specialization && n_base_3b == BASE && n_max_3b == MAX && \
        atom_types == TYPES && lmax_3 == L3 && lmax_4 == L4 && lmax_5 == L5) { \
        nep3_opt_legacy::find_angular_gardc_neigh_opt<BASE, MAX, TYPES, L3, L5><<< \
            N, threads_per_block>>>( \
            N, grad_second, d12, NL, de_dfeat - feat_2b_num, dsnlm_dc, \
            sum_fxyz, atom_map, coeff3, tmp_dfeat_c3.data(), rcut_angular, \
            rcinv_angular, atom_nums, maxneighs, n_max_3b, n_base_3b, \
            atom_types, atom_types_sq, lmax_3, lmax_4, lmax_5, feat_2b_num, \
            multi_feat_num); \
        selected_shared_bytes = specialization_shared_bytes; \
        use_aot_specialization = true; \
      } \
    }
#include "./utilities/nep3_c3_specialization_whitelist.def"
#undef MATPL_NEP_C3_SPECIALIZATION

    static bool c3_path_logged = false;
    if (!c3_path_logged && std::getenv("MATPL_NEP_DEBUG_OPT_PATH") != nullptr) {
        const char* c3_path = use_aot_specialization ? "aot-wave-specialized-opt" :
            (use_lds_opt ? "runtime-lds-opt" : "base-global-atomic");
        std::cerr << "NEP C3 path: " << c3_path
                  << " (types=" << atom_types
                  << ", n_max_3b=" << n_max_3b
                  << ", n_base_3b=" << n_base_3b
                  << ", lmax=" << lmax_3 << "/" << lmax_4 << "/" << lmax_5
                  << ", shared=" << selected_shared_bytes
                  << "/" << shared_limit << " bytes)" << std::endl;
        c3_path_logged = true;
    }

    if (use_aot_specialization) {
      // The legacy wave-level specialization was launched from the whitelist.
    } else if (use_lds_opt) {
      find_angular_gardc_neigh_optimized_2<true><<<
          N, threads_per_block, dynamic_shared_bytes>>>(
          N, grad_second, d12, NL, de_dfeat - feat_2b_num, dsnlm_dc,
          sum_fxyz, atom_map, coeff3, tmp_dfeat_c3.data(), rcut_angular,
          rcinv_angular, atom_nums, maxneighs, n_max_3b, n_base_3b,
          atom_types, atom_types_sq, lmax_3, lmax_4, lmax_5, feat_2b_num,
          multi_feat_num);
    } else {
      find_angular_gardc_neigh_optimized_2<false><<<N, threads_per_block>>>(
          N, grad_second, d12, NL, de_dfeat - feat_2b_num, dsnlm_dc,
          sum_fxyz, atom_map, coeff3, tmp_dfeat_c3.data(), rcut_angular,
          rcinv_angular, atom_nums, maxneighs, n_max_3b, n_base_3b,
          atom_types, atom_types_sq, lmax_3, lmax_4, lmax_5, feat_2b_num,
          multi_feat_num);
    }
    CUDA_CHECK_KERNEL
    hipDeviceSynchronize();

    total_elements = N * n_max_3b * n_base_3b;
    threads_per_block = 256;
    num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    aggregate_features<<<num_blocks, threads_per_block>>>(
    tmp_dfeat_c3.data(),
    atom_map,
    gradsecond_c3,
    N,
    atom_types,
    n_max_3b,
    n_base_3b);
    CUDA_CHECK_KERNEL
    hipDeviceSynchronize();
}
