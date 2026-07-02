#pragma once

#include "../utilities/common.cuh"
#include "../utilities/error.cuh"
#include "../utilities/gpu_vector.cuh"
#include "nepkk.cuh"
#include <cmath>
#include <vector>

__device__ void nepkk_ewald_cross_product(const NEP_FLOAT a[3], const NEP_FLOAT b[3], NEP_FLOAT c[3])
{
  c[0] = a[1] * b[2] - a[2] * b[1];
  c[1] = a[2] * b[0] - a[0] * b[2];
  c[2] = a[0] * b[1] - a[1] * b[0];
}

__device__ NEP_FLOAT nepkk_ewald_get_area(const NEP_FLOAT* a, const NEP_FLOAT* b)
{
  const NEP_FLOAT s1 = a[1] * b[2] - a[2] * b[1];
  const NEP_FLOAT s2 = a[2] * b[0] - a[0] * b[2];
  const NEP_FLOAT s3 = a[0] * b[1] - a[1] * b[0];
  return sqrt(s1 * s1 + s2 * s2 + s3 * s3);
}

inline void nepkk_ewald_cross_product_host(const NEP_FLOAT a[3], const NEP_FLOAT b[3], NEP_FLOAT c[3])
{
  c[0] = a[1] * b[2] - a[2] * b[1];
  c[1] = a[2] * b[0] - a[0] * b[2];
  c[2] = a[0] * b[1] - a[1] * b[0];
}

inline NEP_FLOAT nepkk_ewald_get_area_host(const NEP_FLOAT* a, const NEP_FLOAT* b)
{
  const NEP_FLOAT s1 = a[1] * b[2] - a[2] * b[1];
  const NEP_FLOAT s2 = a[2] * b[0] - a[0] * b[2];
  const NEP_FLOAT s3 = a[0] * b[1] - a[1] * b[0];
  return sqrt(s1 * s1 + s2 * s2 + s3 * s3);
}

inline int nepkk_find_k_and_G_charge2_host(
  const NEP_FLOAT alpha,
  const NEP_FLOAT alpha_factor,
  const NEPKK_Box& box,
  GPU_Vector<int>& num_kpoints,
  GPU_Vector<NEP_FLOAT>& kx,
  GPU_Vector<NEP_FLOAT>& ky,
  GPU_Vector<NEP_FLOAT>& kz,
  GPU_Vector<NEP_FLOAT>& G,
  GPU_Vector<NEP_FLOAT>& S_real,
  GPU_Vector<NEP_FLOAT>& S_imag)
{
  const NEP_FLOAT det = NEP_FLOAT(box.h[0] * (box.h[4] * box.h[8] - box.h[5] * box.h[7]) +
                                  box.h[1] * (box.h[5] * box.h[6] - box.h[3] * box.h[8]) +
                                  box.h[2] * (box.h[3] * box.h[7] - box.h[4] * box.h[6]));
  const NEP_FLOAT a1[3] = {NEP_FLOAT(box.h[0]), NEP_FLOAT(box.h[3]), NEP_FLOAT(box.h[6])};
  const NEP_FLOAT a2[3] = {NEP_FLOAT(box.h[1]), NEP_FLOAT(box.h[4]), NEP_FLOAT(box.h[7])};
  const NEP_FLOAT a3[3] = {NEP_FLOAT(box.h[2]), NEP_FLOAT(box.h[5]), NEP_FLOAT(box.h[8])};
  NEP_FLOAT b1[3] = {FLOAT_LIT(0.0)};
  NEP_FLOAT b2[3] = {FLOAT_LIT(0.0)};
  NEP_FLOAT b3[3] = {FLOAT_LIT(0.0)};
  nepkk_ewald_cross_product_host(a2, a3, b1);
  nepkk_ewald_cross_product_host(a3, a1, b2);
  nepkk_ewald_cross_product_host(a1, a2, b3);

  const NEP_FLOAT two_pi = FLOAT_LIT(6.2831853);
  const NEP_FLOAT two_pi_over_det = two_pi / det;
  for (int d = 0; d < 3; ++d) {
    b1[d] *= two_pi_over_det;
    b2[d] *= two_pi_over_det;
    b3[d] *= two_pi_over_det;
  }

  const NEP_FLOAT volume_k = two_pi * two_pi * two_pi / abs(det);
  const int n1_max = int(alpha * two_pi * nepkk_ewald_get_area_host(b2, b3) / volume_k);
  const int n2_max = int(alpha * two_pi * nepkk_ewald_get_area_host(b3, b1) / volume_k);
  const int n3_max = int(alpha * two_pi * nepkk_ewald_get_area_host(b1, b2) / volume_k);
  const NEP_FLOAT ksq_max = two_pi * two_pi * alpha * alpha;

  std::vector<NEP_FLOAT> cpu_kx;
  std::vector<NEP_FLOAT> cpu_ky;
  std::vector<NEP_FLOAT> cpu_kz;
  std::vector<NEP_FLOAT> cpu_G;

  for (int n1 = 0; n1 <= n1_max; ++n1) {
    for (int n2 = -n2_max; n2 <= n2_max; ++n2) {
      for (int n3 = -n3_max; n3 <= n3_max; ++n3) {
        const int nsq = n1 * n1 + n2 * n2 + n3 * n3;
        if (nsq == 0 || (n1 == 0 && n2 < 0) || (n1 == 0 && n2 == 0 && n3 < 0)) continue;
        const NEP_FLOAT kx_value = n1 * b1[0] + n2 * b2[0] + n3 * b3[0];
        const NEP_FLOAT ky_value = n1 * b1[1] + n2 * b2[1] + n3 * b3[1];
        const NEP_FLOAT kz_value = n1 * b1[2] + n2 * b2[2] + n3 * b3[2];
        const NEP_FLOAT ksq = kx_value * kx_value + ky_value * ky_value + kz_value * kz_value;
        if (ksq < ksq_max) {
          cpu_kx.emplace_back(kx_value);
          cpu_ky.emplace_back(ky_value);
          cpu_kz.emplace_back(kz_value);
          cpu_G.emplace_back(FLOAT_LIT(2.0) * abs(two_pi_over_det) / ksq * exp(-ksq * alpha_factor));
        }
      }
    }
  }

  const int cpu_num_kpoints = int(cpu_kx.size());
  if (cpu_num_kpoints > int(kx.size())) {
    kx.resize(cpu_num_kpoints);
    ky.resize(cpu_num_kpoints);
    kz.resize(cpu_num_kpoints);
    G.resize(cpu_num_kpoints);
    S_real.resize(cpu_num_kpoints);
    S_imag.resize(cpu_num_kpoints);
  }

  num_kpoints.copy_from_host(&cpu_num_kpoints, 1);
  kx.copy_from_host(cpu_kx.data(), cpu_num_kpoints);
  ky.copy_from_host(cpu_ky.data(), cpu_num_kpoints);
  kz.copy_from_host(cpu_kz.data(), cpu_num_kpoints);
  G.copy_from_host(cpu_G.data(), cpu_num_kpoints);
  return cpu_num_kpoints;
}

__global__ void nepkk_find_k_and_G_charge2(
  const int num_kpoints_max,
  const NEP_FLOAT alpha,
  const NEP_FLOAT alpha_factor,
  const NEPKK_Box box,
  int* g_num_kpoints,
  NEP_FLOAT* g_kx,
  NEP_FLOAT* g_ky,
  NEP_FLOAT* g_kz,
  NEP_FLOAT* g_G)
{
  if (threadIdx.x + blockIdx.x * blockDim.x == 0) {
    const NEP_FLOAT det = NEP_FLOAT(box.h[0] * (box.h[4] * box.h[8] - box.h[5] * box.h[7]) +
                                    box.h[1] * (box.h[5] * box.h[6] - box.h[3] * box.h[8]) +
                                    box.h[2] * (box.h[3] * box.h[7] - box.h[4] * box.h[6]));
    const NEP_FLOAT a1[3] = {NEP_FLOAT(box.h[0]), NEP_FLOAT(box.h[3]), NEP_FLOAT(box.h[6])};
    const NEP_FLOAT a2[3] = {NEP_FLOAT(box.h[1]), NEP_FLOAT(box.h[4]), NEP_FLOAT(box.h[7])};
    const NEP_FLOAT a3[3] = {NEP_FLOAT(box.h[2]), NEP_FLOAT(box.h[5]), NEP_FLOAT(box.h[8])};
    NEP_FLOAT b1[3] = {FLOAT_LIT(0.0)};
    NEP_FLOAT b2[3] = {FLOAT_LIT(0.0)};
    NEP_FLOAT b3[3] = {FLOAT_LIT(0.0)};
    nepkk_ewald_cross_product(a2, a3, b1);
    nepkk_ewald_cross_product(a3, a1, b2);
    nepkk_ewald_cross_product(a1, a2, b3);

    const NEP_FLOAT two_pi = FLOAT_LIT(6.2831853);
    const NEP_FLOAT two_pi_over_det = two_pi / det;
    for (int d = 0; d < 3; ++d) {
      b1[d] *= two_pi_over_det;
      b2[d] *= two_pi_over_det;
      b3[d] *= two_pi_over_det;
    }

    const NEP_FLOAT volume_k = two_pi * two_pi * two_pi / abs(det);
    int n1_max = int(alpha * two_pi * nepkk_ewald_get_area(b2, b3) / volume_k);
    int n2_max = int(alpha * two_pi * nepkk_ewald_get_area(b3, b1) / volume_k);
    int n3_max = int(alpha * two_pi * nepkk_ewald_get_area(b1, b2) / volume_k);
    NEP_FLOAT ksq_max = two_pi * two_pi * alpha * alpha;

    int nk = 0;
    for (int n1 = 0; n1 <= n1_max; ++n1) {
      for (int n2 = -n2_max; n2 <= n2_max; ++n2) {
        for (int n3 = -n3_max; n3 <= n3_max; ++n3) {
          const int nsq = n1 * n1 + n2 * n2 + n3 * n3;
          if (nsq == 0 || (n1 == 0 && n2 < 0) || (n1 == 0 && n2 == 0 && n3 < 0)) continue;
          const NEP_FLOAT kx = n1 * b1[0] + n2 * b2[0] + n3 * b3[0];
          const NEP_FLOAT ky = n1 * b1[1] + n2 * b2[1] + n3 * b3[1];
          const NEP_FLOAT kz = n1 * b1[2] + n2 * b2[2] + n3 * b3[2];
          const NEP_FLOAT ksq = kx * kx + ky * ky + kz * kz;
          if (ksq < ksq_max) {
            if (nk < num_kpoints_max) {
              g_kx[nk] = kx;
              g_ky[nk] = ky;
              g_kz[nk] = kz;
              g_G[nk] = FLOAT_LIT(2.0) * abs(two_pi_over_det) / ksq * exp(-ksq * alpha_factor);
            }
            ++nk;
          }
        }
      }
    }
    g_num_kpoints[0] = nk < num_kpoints_max ? nk : num_kpoints_max;
  }
}

__global__ void nepkk_find_structure_factor_charge2(
  const int N,
  const int num_kpoints,
  const NEP_FLOAT* g_charge,
  const NEP_FLOAT* g_pos,
  const NEP_FLOAT* g_kx,
  const NEP_FLOAT* g_ky,
  const NEP_FLOAT* g_kz,
  NEP_FLOAT* g_S_real,
  NEP_FLOAT* g_S_imag)
{
  int nk = blockIdx.x * blockDim.x + threadIdx.x;
  if (nk < num_kpoints) {
    NEP_FLOAT S_real = FLOAT_LIT(0.0);
    NEP_FLOAT S_imag = FLOAT_LIT(0.0);
    for (int n = 0; n < N; ++n) {
      const NEP_FLOAT kr = g_kx[nk] * g_pos[n * 3] + g_ky[nk] * g_pos[n * 3 + 1] + g_kz[nk] * g_pos[n * 3 + 2];
      const NEP_FLOAT q = g_charge[n];
      S_real += q * cos(kr);
      S_imag -= q * sin(kr);
    }
    g_S_real[nk] = S_real;
    g_S_imag[nk] = S_imag;
  }
}

__global__ void nepkk_find_force_charge_reciprocal_space_charge2(
  const int N,
  const int num_kpoints,
  const NEP_FLOAT alpha_factor,
  const NEP_FLOAT* g_charge,
  const NEP_FLOAT* g_pos,
  const NEP_FLOAT* g_kx,
  const NEP_FLOAT* g_ky,
  const NEP_FLOAT* g_kz,
  const NEP_FLOAT* g_G,
  const NEP_FLOAT* g_S_real,
  const NEP_FLOAT* g_S_imag,
  NEP_FLOAT* g_D_real,
  double* g_f,
  double* g_virial,
  const int vflag_either,
  const int cvflag_atom,
  const int vatom_num)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    const NEP_FLOAT q = g_charge[n];
    NEP_FLOAT temp_virial_sum[6] = {FLOAT_LIT(0.0)};
    NEP_FLOAT temp_force_sum[3] = {FLOAT_LIT(0.0)};
    NEP_FLOAT temp_D_real_sum = FLOAT_LIT(0.0);
    for (int nk = 0; nk < num_kpoints; ++nk) {
      const NEP_FLOAT kx = g_kx[nk];
      const NEP_FLOAT ky = g_ky[nk];
      const NEP_FLOAT kz = g_kz[nk];
      const NEP_FLOAT kr = kx * g_pos[n * 3] + ky * g_pos[n * 3 + 1] + kz * g_pos[n * 3 + 2];
      const NEP_FLOAT G = g_G[nk];
      const NEP_FLOAT S_real = g_S_real[nk];
      const NEP_FLOAT S_imag = g_S_imag[nk];
      const NEP_FLOAT sin_kr = sin(kr);
      const NEP_FLOAT cos_kr = cos(kr);
      const NEP_FLOAT imag_term = G * (S_real * sin_kr + S_imag * cos_kr);
      const NEP_FLOAT GSE = G * (S_real * cos_kr - S_imag * sin_kr);
      const NEP_FLOAT qGSE = q * GSE;
      const NEP_FLOAT alpha_k_factor = FLOAT_LIT(2.0) * alpha_factor + FLOAT_LIT(2.0) / (kx * kx + ky * ky + kz * kz);
      temp_virial_sum[0] += qGSE * (FLOAT_LIT(1.0) - alpha_k_factor * kx * kx);
      temp_virial_sum[1] += qGSE * (FLOAT_LIT(1.0) - alpha_k_factor * ky * ky);
      temp_virial_sum[2] += qGSE * (FLOAT_LIT(1.0) - alpha_k_factor * kz * kz);
      temp_virial_sum[3] -= qGSE * (alpha_k_factor * kx * ky);
      temp_virial_sum[4] -= qGSE * (alpha_k_factor * ky * kz);
      temp_virial_sum[5] -= qGSE * (alpha_k_factor * kz * kx);
      temp_D_real_sum += GSE;
      temp_force_sum[0] += kx * imag_term;
      temp_force_sum[1] += ky * imag_term;
      temp_force_sum[2] += kz * imag_term;
    }
    g_D_real[n] = FLOAT_LIT(2.0) * K_C_SP * temp_D_real_sum;
    const NEP_FLOAT charge_factor = FLOAT_LIT(2.0) * K_C_SP * q;
    g_f[n * 3] += double(charge_factor * temp_force_sum[0]);
    g_f[n * 3 + 1] += double(charge_factor * temp_force_sum[1]);
    g_f[n * 3 + 2] += double(charge_factor * temp_force_sum[2]);
    if (vflag_either) {
      const double virial_xx = double(K_C_SP * temp_virial_sum[0]);
      const double virial_yy = double(K_C_SP * temp_virial_sum[1]);
      const double virial_zz = double(K_C_SP * temp_virial_sum[2]);
      const double virial_xy = double(K_C_SP * temp_virial_sum[3]);
      const double virial_yz = double(K_C_SP * temp_virial_sum[4]);
      const double virial_zx = double(K_C_SP * temp_virial_sum[5]);
      g_virial[n * vatom_num + 0] += virial_xx;
      g_virial[n * vatom_num + 1] += virial_yy;
      g_virial[n * vatom_num + 2] += virial_zz;
      g_virial[n * vatom_num + 3] += virial_xy;
      g_virial[n * vatom_num + 4] += virial_zx;
      g_virial[n * vatom_num + 5] += virial_yz;
      if (cvflag_atom) {
        g_virial[n * vatom_num + 6] += virial_xy;
        g_virial[n * vatom_num + 7] += virial_zx;
        g_virial[n * vatom_num + 8] += virial_yz;
      }
    }
  }
}

__global__ void nepkk_zero_mean_charge2(const int N, NEP_FLOAT* g_values)
{
  int tid = threadIdx.x;
  int number_of_batches = (N - 1) / 1024 + 1;
  __shared__ double s_sum[1024];
  double sum = 0.0;
  for (int batch = 0; batch < number_of_batches; ++batch) {
    int n = tid + batch * 1024;
    if (n < N) sum += double(g_values[n]);
  }
  s_sum[tid] = sum;
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) s_sum[tid] += s_sum[tid + offset];
    __syncthreads();
  }
  const NEP_FLOAT mean = NEP_FLOAT(s_sum[0] / N);
  for (int batch = 0; batch < number_of_batches; ++batch) {
    int n = tid + batch * 1024;
    if (n < N) g_values[n] -= mean;
  }
}

__global__ void nepkk_subtract_value(const int N, const NEP_FLOAT value, NEP_FLOAT* g_values)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    g_values[n] -= value;
  }
}

inline void nepkk_zero_global_mean_charge2(
  const int N,
  const long long natoms_global,
  NEPKKAllreduceDouble allreduce_double,
  void* allreduce_context,
  GPU_Vector<NEP_FLOAT>& values)
{
  std::vector<NEP_FLOAT> h_values(N);
  values.copy_to_host(h_values.data(), N);
  double local_sum = 0.0;
  for (int n = 0; n < N; ++n) {
    local_sum += double(h_values[n]);
  }
  double global_sum = 0.0;
  allreduce_double(&local_sum, &global_sum, 1, allreduce_context);
  const NEP_FLOAT mean = NEP_FLOAT(global_sum / double(natoms_global));
  nepkk_subtract_value<<<(N - 1) / 256 + 1, 256>>>(N, mean, values.data());
  CUDA_CHECK_KERNEL
}

inline void nepkk_ewald_find_force_charge2(
  const int N,
  const int block_size,
  const int grid_size,
  const int num_kpoints_max,
  const NEP_FLOAT alpha,
  const NEP_FLOAT alpha_factor,
  const NEPKK_Box& box,
  const GPU_Vector<NEP_FLOAT>& charge,
  const GPU_Vector<NEP_FLOAT>& position,
  GPU_Vector<int>& num_kpoints,
  GPU_Vector<NEP_FLOAT>& kx,
  GPU_Vector<NEP_FLOAT>& ky,
  GPU_Vector<NEP_FLOAT>& kz,
  GPU_Vector<NEP_FLOAT>& G,
  GPU_Vector<NEP_FLOAT>& S_real,
  GPU_Vector<NEP_FLOAT>& S_imag,
  GPU_Vector<NEP_FLOAT>& D_real,
  double* force_per_atom,
  double* virial_per_atom,
  const int vflag_either,
  const int cvflag_atom,
  const int vatom_num,
  const int mpi_size,
  NEPKKAllreduceDouble allreduce_double,
  void* allreduce_context)
{
  (void)num_kpoints_max;
  const int cpu_num_kpoints = nepkk_find_k_and_G_charge2_host(
    alpha, alpha_factor, box, num_kpoints, kx, ky, kz, G, S_real, S_imag);
  const int k_grid_size = (cpu_num_kpoints - 1) / block_size + 1;
  nepkk_find_structure_factor_charge2<<<k_grid_size, block_size>>>(
    N, cpu_num_kpoints, charge.data(), position.data(), kx.data(), ky.data(), kz.data(), S_real.data(), S_imag.data());
  CUDA_CHECK_KERNEL

  if (mpi_size > 1 && cpu_num_kpoints > 0) {
    std::vector<NEP_FLOAT> h_S_real(cpu_num_kpoints);
    std::vector<NEP_FLOAT> h_S_imag(cpu_num_kpoints);
    S_real.copy_to_host(h_S_real.data(), cpu_num_kpoints);
    S_imag.copy_to_host(h_S_imag.data(), cpu_num_kpoints);

    std::vector<double> h_local(cpu_num_kpoints * 2);
    std::vector<double> h_global(cpu_num_kpoints * 2);
    for (int nk = 0; nk < cpu_num_kpoints; ++nk) {
      h_local[nk] = double(h_S_real[nk]);
      h_local[cpu_num_kpoints + nk] = double(h_S_imag[nk]);
    }
    allreduce_double(h_local.data(), h_global.data(), int(h_global.size()), allreduce_context);
    for (int nk = 0; nk < cpu_num_kpoints; ++nk) {
      h_S_real[nk] = NEP_FLOAT(h_global[nk]);
      h_S_imag[nk] = NEP_FLOAT(h_global[cpu_num_kpoints + nk]);
    }
    S_real.copy_from_host(h_S_real.data(), cpu_num_kpoints);
    S_imag.copy_from_host(h_S_imag.data(), cpu_num_kpoints);
  }

  nepkk_find_force_charge_reciprocal_space_charge2<<<grid_size, block_size>>>(
    N,
    cpu_num_kpoints,
    alpha_factor,
    charge.data(),
    position.data(),
    kx.data(),
    ky.data(),
    kz.data(),
    G.data(),
    S_real.data(),
    S_imag.data(),
    D_real.data(),
    force_per_atom,
    virial_per_atom,
    vflag_either,
    cvflag_atom,
    vatom_num);
  CUDA_CHECK_KERNEL
}
