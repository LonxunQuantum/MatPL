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
#include <cstring>
#include "./utilities/nep_mb_secondgrad_hiprtc.hpp"
#include <hip/hip_runtime.h>

// 与 CUDA NVRTC 路径对齐的数学参数范围；默认包含 HfO2。
static constexpr size_t nep_mb_secondgrad_runtime_shared_bytes(int nn,int kb) {
 const auto align=[](size_t v){return (v+7)&~size_t(7);};
 size_t bytes=align(4*sizeof(unsigned int));
 bytes=align(bytes+NEP_MAX_ELEMENT_TYPES*sizeof(int));bytes=align(bytes+sizeof(int));
 bytes=align(bytes+static_cast<size_t>(nn)*6*sizeof(double));
 bytes=align(bytes+static_cast<size_t>(nn)*24*sizeof(double));
 bytes=align(bytes+static_cast<size_t>(4)*kb*24*sizeof(double));
 return align(bytes+static_cast<size_t>(4)*nn*kb*sizeof(double));
}
static_assert(nep_mb_secondgrad_runtime_shared_bytes(20,20)==nep_mb_secondgrad_shared_bytes<20,20,4>(),"shared layout mismatch");
static int nep_mb_secondgrad_mode() {
 const char* mode=std::getenv("MATPL_NEP_MB_SECONDGRAD_MODE");
 if(mode) {
  if(std::strcmp(mode,"auto")==0)return 0;
  if(std::strcmp(mode,"optimized")==0)return 1;
  if(std::strcmp(mode,"legacy")==0)return -1;
  TORCH_CHECK(false,"MATPL_NEP_MB_SECONDGRAD_MODE must be auto, optimized, or legacy");
 }
 const char* compat=std::getenv("MATPL_NEP_MB_SECONDGRAD_FUSED");
 return compat && compat[0]=='0' ? -1 : 0;
}
bool should_recompute_nep_mb_dsnlm(const int nn,const int kb,const int types,
 const int l3,const int l4,const int l5,const int device) {
 const int mode=nep_mb_secondgrad_mode();if(mode<0)return false;
 hipDeviceProp_t prop{};matpl_mb_secondgrad_rtc::driver(hipGetDeviceProperties(&prop,device),"hipGetDeviceProperties");
 const bool supported=nn>=5 && nn<=20 && kb>=5 && kb<=20 && types>=1 && types<=NEP_MAX_ELEMENT_TYPES && l3==4 &&
  ((l4==2 && (l5==0 || l5==1)) || (l4==0 && l5==0)) && nep_mb_secondgrad_runtime_shared_bytes(nn,kb)<=prop.sharedMemPerBlock;
 TORCH_CHECK(mode!=1 || supported,"unsupported optimized NEP many-body second-gradient shape: NMAX=",nn," NBASIS=",kb," TYPES=",types," L3=",l3," L4=",l4," L5=",l5);
 return supported;
}
static void launch_nep_mb_secondgrad_parameterized_dispatch(const NepMbSecondGradArgs& args,int nn,int kb,int l4,int l5,int device) {
 hipSetDevice(device);if(args.atom_count==0)return;
 const auto stream=c10::hip::getCurrentHIPStream(device);
 const size_t angular_values=static_cast<size_t>(args.atom_count)*args.max_neighbors*(120+2*kb);
 const size_t bytes=(angular_values+static_cast<size_t>(args.atom_count)*nn*24)*sizeof(double);
 auto deleter=[](void* p){c10::hip::HIPCachingAllocator::raw_delete(p);};
 std::unique_ptr<void,decltype(deleter)> workspace(c10::hip::HIPCachingAllocator::raw_alloc_with_stream(bytes,stream.stream()),deleter);
 NepMbSecondGradArgs a=args;a.angular_workspace=static_cast<double*>(workspace.get());a.weighted_sums=a.angular_workspace+angular_values;
 matpl_mb_secondgrad_rtc::launch(a,nn,kb,l4>0,l5>0,device,nep_mb_secondgrad_runtime_shared_bytes(nn,kb));
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
        launch_nep_mb_secondgrad_parameterized_dispatch(args,n_max_3b,n_base_3b,lmax_4,lmax_5,device);
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
