#pragma once

#include "../utilities/common.cuh"
#include "../utilities/error.cuh"
#include "../utilities/gpu_vector.cuh"
#include "nepkk.cuh"
#include <cmath>
#include <cufft.h>
#include <iostream>
#include <vector>

namespace {

int nepkk_get_power2_pppm_K(const int m)
{
  int n = 16;
  while (n < m) {
    n *= 2;
  }
  return n;
}

bool nepkk_is_pppm_fft_friendly(int n)
{
  if (n < 1) return false;
  const int factors[4] = {2, 3, 5, 7};
  for (int i = 0; i < 4; ++i) {
    while (n % factors[i] == 0) n /= factors[i];
  }
  return n == 1;
}

int nepkk_get_friendly_pppm_K(const int m)
{
  int n = m < 16 ? 16 : m;
  while (!nepkk_is_pppm_fft_friendly(n)) {
    ++n;
  }
  return n;
}

double nepkk_box_volume(const NEPKK_Box& box)
{
  return box.h[0] * (box.h[4] * box.h[8] - box.h[5] * box.h[7]) +
         box.h[1] * (box.h[5] * box.h[6] - box.h[3] * box.h[8]) +
         box.h[2] * (box.h[3] * box.h[7] - box.h[4] * box.h[6]);
}

double nepkk_box_area(const NEPKK_Box& box, const int d)
{
  const double a1[3] = {box.h[0], box.h[3], box.h[6]};
  const double a2[3] = {box.h[1], box.h[4], box.h[7]};
  const double a3[3] = {box.h[2], box.h[5], box.h[8]};
  const double* b = d == 0 ? a2 : (d == 1 ? a3 : a1);
  const double* c = d == 0 ? a3 : (d == 1 ? a1 : a2);
  const double x = b[1] * c[2] - b[2] * c[1];
  const double y = b[2] * c[0] - b[0] * c[2];
  const double z = b[0] * c[1] - b[1] * c[0];
  return std::sqrt(x * x + y * y + z * z);
}

__constant__ float nepkk_pppm_sinc_coeff[6] = {
  1.0f, -1.6666667e-1f, 8.3333333e-3f, -1.9841270e-4f, 2.7557319e-6f, -2.5052108e-8f};
__constant__ float nepkk_pppm_G_coeff[5] = {
  1.0000000e+00f, -1.6666667e+00f, 7.7777778e-01f, -8.9947090e-02f, 7.0546737e-04f};
__constant__ float nepkk_pppm_W_coeff[5][5] = {
  {2.6041667e-03f, -2.0833333e-02f, 6.2500000e-02f, -8.3333333e-02f, 4.1666667e-02f},
  {1.9791667e-01f, -4.5833333e-01f, 2.5000000e-01f, 1.6666667e-01f, -1.6666667e-01f},
  {5.9895833e-01f, 0.0000000e+00f, -6.2500000e-01f, 0.0000000e+00f, 2.5000000e-01f},
  {1.9791667e-01f, 4.5833333e-01f, 2.5000000e-01f, -1.6666667e-01f, -1.6666667e-01f},
  {2.6041667e-03f, 2.0833333e-02f, 6.2500000e-02f, 8.3333333e-02f, 4.1666667e-02f}};

__device__ inline float nepkk_pppm_sinc(const float x)
{
  float y = 0.0f;
  if (x * x <= 1.0f) {
    float term = 1.0f;
    for (int i = 0; i < 6; ++i) {
      y += nepkk_pppm_sinc_coeff[i] * term;
      term *= x * x;
    }
  } else {
    y = sin(x) / x;
  }
  return y;
}

__device__ inline int nepkk_pppm_mesh_index(const int K, const int n)
{
  if (n >= K) return n - K;
  if (n < 0) return n + K;
  return n;
}

__global__ void nepkk_pppm_find_k_and_G(
  const NEPKK_PPPM_Para para,
  NEP_FLOAT* g_kx,
  NEP_FLOAT* g_ky,
  NEP_FLOAT* g_kz,
  NEP_FLOAT* g_G)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < para.K0K1K2) {
    int nk[3];
    nk[2] = n / para.K0K1;
    nk[1] = (n - nk[2] * para.K0K1) / para.K[0];
    nk[0] = n % para.K[0];

    NEP_FLOAT denominator[3] = {FLOAT_LIT(0.0)};
    for (int d = 0; d < 3; ++d) {
      if (nk[d] >= para.K_half[d]) {
        nk[d] -= para.K[d];
      }
      NEP_FLOAT t = sin(FLOAT_LIT(0.5) * para.two_pi_over_K[d] * nk[d]);
      t *= t;
      t = (((FLOAT_LIT(7.0546737e-04) * t + FLOAT_LIT(-8.9947090e-02)) * t + FLOAT_LIT(7.7777778e-01)) * t
           + FLOAT_LIT(-1.6666667e+00)) * t + FLOAT_LIT(1.0000000e+00);
      denominator[d] = t * t;
    }

    const NEP_FLOAT kx = nk[0] * para.b[0][0] + nk[1] * para.b[1][0] + nk[2] * para.b[2][0];
    const NEP_FLOAT ky = nk[0] * para.b[0][1] + nk[1] * para.b[1][1] + nk[2] * para.b[2][1];
    const NEP_FLOAT kz = nk[0] * para.b[0][2] + nk[1] * para.b[1][2] + nk[2] * para.b[2][2];
    g_kx[n] = kx;
    g_ky[n] = ky;
    g_kz[n] = kz;
    const NEP_FLOAT ksq = kx * kx + ky * ky + kz * kz;

    NEP_FLOAT numerator = nepkk_pppm_sinc(FLOAT_LIT(0.5) * para.two_pi_over_K[0] * nk[0]);
    numerator *= nepkk_pppm_sinc(FLOAT_LIT(0.5) * para.two_pi_over_K[1] * nk[1]);
    numerator *= nepkk_pppm_sinc(FLOAT_LIT(0.5) * para.two_pi_over_K[2] * nk[2]);
    numerator = numerator * numerator * numerator * numerator * numerator;
    numerator *= numerator;

    if (ksq == FLOAT_LIT(0.0)) {
      g_G[n] = FLOAT_LIT(0.0);
    } else {
      NEP_FLOAT G_opt = numerator * para.two_pi_over_V / ksq * exp(-ksq * para.alpha_factor);
      G_opt /= denominator[0] * denominator[1] * denominator[2];
      g_G[n] = G_opt;
    }
  }
}

__global__ void nepkk_pppm_set_mesh_to_zero(const NEPKK_PPPM_Para para, cufftComplex* g_mesh)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < para.K0K1K2) {
    g_mesh[n].x = 0.0f;
    g_mesh[n].y = 0.0f;
  }
}

__global__ void nepkk_pppm_find_mesh(
  const int N,
  const NEPKK_PPPM_Para para,
  const NEPKK_Box box,
  const NEP_FLOAT* g_charge,
  const NEP_FLOAT* g_pos,
  cufftComplex* g_mesh)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    const NEP_FLOAT x = g_pos[n * 3];
    const NEP_FLOAT y = g_pos[n * 3 + 1];
    const NEP_FLOAT z = g_pos[n * 3 + 2];
    const float q = float(g_charge[n]);
    const float sx = float((box.hi[0] * x + box.hi[1] * y + box.hi[2] * z) * para.K[0]);
    const float sy = float((box.hi[3] * x + box.hi[4] * y + box.hi[5] * z) * para.K[1]);
    const float sz = float((box.hi[6] * x + box.hi[7] * y + box.hi[8] * z) * para.K[2]);
    const int ix = int(sx + 0.5f);
    const int iy = int(sy + 0.5f);
    const int iz = int(sz + 0.5f);
    const float dx = sx - ix;
    const float dy = sy - iy;
    const float dz = sz - iz;
    float Wx[5] = {0.0f};
    float Wy[5] = {0.0f};
    float Wz[5] = {0.0f};
    for (int d = 0; d < 5; ++d) {
      Wx[d] = (((nepkk_pppm_W_coeff[d][4] * dx + nepkk_pppm_W_coeff[d][3]) * dx + nepkk_pppm_W_coeff[d][2]) * dx
               + nepkk_pppm_W_coeff[d][1]) * dx + nepkk_pppm_W_coeff[d][0];
      Wy[d] = (((nepkk_pppm_W_coeff[d][4] * dy + nepkk_pppm_W_coeff[d][3]) * dy + nepkk_pppm_W_coeff[d][2]) * dy
               + nepkk_pppm_W_coeff[d][1]) * dy + nepkk_pppm_W_coeff[d][0];
      Wz[d] = (((nepkk_pppm_W_coeff[d][4] * dz + nepkk_pppm_W_coeff[d][3]) * dz + nepkk_pppm_W_coeff[d][2]) * dz
               + nepkk_pppm_W_coeff[d][1]) * dz + nepkk_pppm_W_coeff[d][0];
    }
    for (int n0 = -2; n0 <= 2; ++n0) {
      const int neighbor0 = nepkk_pppm_mesh_index(para.K[0], ix + n0);
      for (int n1 = -2; n1 <= 2; ++n1) {
        const int neighbor1 = nepkk_pppm_mesh_index(para.K[1], iy + n1);
        for (int n2 = -2; n2 <= 2; ++n2) {
          const int neighbor2 = nepkk_pppm_mesh_index(para.K[2], iz + n2);
          const int idx = neighbor0 + para.K[0] * (neighbor1 + para.K[1] * neighbor2);
          const float W = Wx[n0 + 2] * Wy[n1 + 2] * Wz[n2 + 2];
          atomicAdd(&g_mesh[idx].x, q * W);
        }
      }
    }
  }
}

__global__ void nepkk_pppm_ik_times_mesh_times_G(
  const NEPKK_PPPM_Para para,
  const NEP_FLOAT* g_kx,
  const NEP_FLOAT* g_ky,
  const NEP_FLOAT* g_kz,
  const NEP_FLOAT* g_G,
  const cufftComplex* g_mesh_fft,
  cufftComplex* g_mesh_fft_x,
  cufftComplex* g_mesh_fft_y,
  cufftComplex* g_mesh_fft_z)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < para.K0K1K2) {
    const float kx = float(g_kx[n]);
    const float ky = float(g_ky[n]);
    const float kz = float(g_kz[n]);
    const float G = float(g_G[n]);
    const cufftComplex mesh_fft = g_mesh_fft[n];
    g_mesh_fft_x[n] = {mesh_fft.y * kx * G, -mesh_fft.x * kx * G};
    g_mesh_fft_y[n] = {mesh_fft.y * ky * G, -mesh_fft.x * ky * G};
    g_mesh_fft_z[n] = {mesh_fft.y * kz * G, -mesh_fft.x * kz * G};
  }
}

__global__ void nepkk_pppm_find_mesh_G(
  const NEPKK_PPPM_Para para,
  const NEP_FLOAT* g_G,
  const cufftComplex* g_mesh,
  cufftComplex* g_mesh_G)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < para.K0K1K2) {
    const float G = float(g_G[n]);
    const cufftComplex mesh = g_mesh[n];
    g_mesh_G[n] = {mesh.x * G, mesh.y * G};
  }
}

__global__ void nepkk_pppm_find_mesh_virial(
  const NEPKK_PPPM_Para para,
  const NEP_FLOAT* g_kx,
  const NEP_FLOAT* g_ky,
  const NEP_FLOAT* g_kz,
  const NEP_FLOAT* g_G,
  const cufftComplex* g_S,
  cufftComplex* g_mesh_virial_xx,
  cufftComplex* g_mesh_virial_yy,
  cufftComplex* g_mesh_virial_zz,
  cufftComplex* g_mesh_virial_xy,
  cufftComplex* g_mesh_virial_yz,
  cufftComplex* g_mesh_virial_zx)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < para.K0K1K2) {
    const float kx = float(g_kx[n]);
    const float ky = float(g_ky[n]);
    const float kz = float(g_kz[n]);
    const float ksq = kx * kx + ky * ky + kz * kz;
    if (ksq != 0.0f) {
      const float alpha_k_factor = 2.0f * float(para.alpha_factor) + 2.0f / ksq;
      const float G = float(g_G[n]);
      const cufftComplex S = g_S[n];
      const float GSx = G * S.x;
      const float GSy = G * S.y;
      float B = 1.0f - alpha_k_factor * kx * kx;
      g_mesh_virial_xx[n] = {B * GSx, B * GSy};
      B = 1.0f - alpha_k_factor * ky * ky;
      g_mesh_virial_yy[n] = {B * GSx, B * GSy};
      B = 1.0f - alpha_k_factor * kz * kz;
      g_mesh_virial_zz[n] = {B * GSx, B * GSy};
      B = -alpha_k_factor * kx * ky;
      g_mesh_virial_xy[n] = {B * GSx, B * GSy};
      B = -alpha_k_factor * ky * kz;
      g_mesh_virial_yz[n] = {B * GSx, B * GSy};
      B = -alpha_k_factor * kz * kx;
      g_mesh_virial_zx[n] = {B * GSx, B * GSy};
    }
  }
}

__global__ void nepkk_pppm_find_force_virial_from_field(
  const int N,
  const NEPKK_PPPM_Para para,
  const NEPKK_Box box,
  const NEP_FLOAT* g_charge,
  const NEP_FLOAT* g_pos,
  const cufftComplex* g_mesh_G,
  const cufftComplex* g_mesh_fft_x_ifft,
  const cufftComplex* g_mesh_fft_y_ifft,
  const cufftComplex* g_mesh_fft_z_ifft,
  const cufftComplex* g_mesh_virial_xx,
  const cufftComplex* g_mesh_virial_yy,
  const cufftComplex* g_mesh_virial_zz,
  const cufftComplex* g_mesh_virial_xy,
  const cufftComplex* g_mesh_virial_yz,
  const cufftComplex* g_mesh_virial_zx,
  NEP_FLOAT* g_D_real,
  double* g_f,
  double* g_virial,
  const int vflag_either,
  const int cvflag_atom,
  const int vatom_num)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    const NEP_FLOAT x = g_pos[n * 3];
    const NEP_FLOAT y = g_pos[n * 3 + 1];
    const NEP_FLOAT z = g_pos[n * 3 + 2];
    const float q = float(K_C_SP * g_charge[n]);
    const float sx = float((box.hi[0] * x + box.hi[1] * y + box.hi[2] * z) * para.K[0]);
    const float sy = float((box.hi[3] * x + box.hi[4] * y + box.hi[5] * z) * para.K[1]);
    const float sz = float((box.hi[6] * x + box.hi[7] * y + box.hi[8] * z) * para.K[2]);
    const int ix = int(sx + 0.5f);
    const int iy = int(sy + 0.5f);
    const int iz = int(sz + 0.5f);
    const float dx = sx - ix;
    const float dy = sy - iy;
    const float dz = sz - iz;
    float Wx[5] = {0.0f};
    float Wy[5] = {0.0f};
    float Wz[5] = {0.0f};
    for (int d = 0; d < 5; ++d) {
      Wx[d] = (((nepkk_pppm_W_coeff[d][4] * dx + nepkk_pppm_W_coeff[d][3]) * dx + nepkk_pppm_W_coeff[d][2]) * dx
               + nepkk_pppm_W_coeff[d][1]) * dx + nepkk_pppm_W_coeff[d][0];
      Wy[d] = (((nepkk_pppm_W_coeff[d][4] * dy + nepkk_pppm_W_coeff[d][3]) * dy + nepkk_pppm_W_coeff[d][2]) * dy
               + nepkk_pppm_W_coeff[d][1]) * dy + nepkk_pppm_W_coeff[d][0];
      Wz[d] = (((nepkk_pppm_W_coeff[d][4] * dz + nepkk_pppm_W_coeff[d][3]) * dz + nepkk_pppm_W_coeff[d][2]) * dz
               + nepkk_pppm_W_coeff[d][1]) * dz + nepkk_pppm_W_coeff[d][0];
    }
    float D_real = 0.0f;
    float E[3] = {0.0f};
    float V[6] = {0.0f};
    for (int n0 = -2; n0 <= 2; ++n0) {
      const int neighbor0 = nepkk_pppm_mesh_index(para.K[0], ix + n0);
      for (int n1 = -2; n1 <= 2; ++n1) {
        const int neighbor1 = nepkk_pppm_mesh_index(para.K[1], iy + n1);
        for (int n2 = -2; n2 <= 2; ++n2) {
          const int neighbor2 = nepkk_pppm_mesh_index(para.K[2], iz + n2);
          const int idx = neighbor0 + para.K[0] * (neighbor1 + para.K[1] * neighbor2);
          const float W = Wx[n0 + 2] * Wy[n1 + 2] * Wz[n2 + 2];
          D_real += W * g_mesh_G[idx].x;
          E[0] += W * g_mesh_fft_x_ifft[idx].x;
          E[1] += W * g_mesh_fft_y_ifft[idx].x;
          E[2] += W * g_mesh_fft_z_ifft[idx].x;
          V[0] += W * g_mesh_virial_xx[idx].x;
          V[1] += W * g_mesh_virial_yy[idx].x;
          V[2] += W * g_mesh_virial_zz[idx].x;
          V[3] += W * g_mesh_virial_xy[idx].x;
          V[4] += W * g_mesh_virial_yz[idx].x;
          V[5] += W * g_mesh_virial_zx[idx].x;
        }
      }
    }

    g_D_real[n] = NEP_FLOAT(2.0f * K_C_SP * D_real);
    g_f[n * 3] += double(2.0f * q * E[0]);
    g_f[n * 3 + 1] += double(2.0f * q * E[1]);
    g_f[n * 3 + 2] += double(2.0f * q * E[2]);

    if (vflag_either) {
      const double virial_xx = double(q * V[0]);
      const double virial_yy = double(q * V[1]);
      const double virial_zz = double(q * V[2]);
      const double virial_xy = double(q * V[3]);
      const double virial_yz = double(q * V[4]);
      const double virial_zx = double(q * V[5]);
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

inline void nepkk_pppm_allreduce_mesh(
  NEPKK_PPPM_Data& pppm,
  NEPKKAllreduceDouble allreduce_double,
  void* allreduce_context)
{
  const int mesh_size = pppm.para.K0K1K2;
  std::vector<cufftComplex> h_mesh(mesh_size);
  pppm.mesh.copy_to_host(h_mesh.data(), mesh_size);
  std::vector<double> h_local(mesh_size * 2);
  std::vector<double> h_global(mesh_size * 2);
  for (int i = 0; i < mesh_size; ++i) {
    h_local[i] = double(h_mesh[i].x);
    h_local[mesh_size + i] = double(h_mesh[i].y);
  }
  allreduce_double(h_local.data(), h_global.data(), int(h_global.size()), allreduce_context);
  for (int i = 0; i < mesh_size; ++i) {
    h_mesh[i].x = float(h_global[i]);
    h_mesh[i].y = float(h_global[mesh_size + i]);
  }
  pppm.mesh.copy_from_host(h_mesh.data(), mesh_size);
}

inline void nepkk_pppm_destroy(NEPKK_PPPM_Data& pppm)
{
  if (pppm.plan_initialized) {
    cufftDestroy(pppm.plan);
    pppm.plan_initialized = false;
  }
  if (pppm.plan_virial_initialized) {
    cufftDestroy(pppm.plan_virial);
    pppm.plan_virial_initialized = false;
  }
}

inline void nepkk_pppm_allocate_memory(NEPKK_PPPM_Data& pppm)
{
  nepkk_pppm_destroy(pppm);
  const int mesh_size = pppm.para.K0K1K2;
  // printf("------mesh_size of pppm %d------", mesh_size);
  pppm.kx.resize(mesh_size);
  pppm.ky.resize(mesh_size);
  pppm.kz.resize(mesh_size);
  pppm.G.resize(mesh_size);
  pppm.mesh.resize(mesh_size);
  pppm.mesh_G.resize(mesh_size);
  pppm.mesh_x.resize(mesh_size);
  pppm.mesh_y.resize(mesh_size);
  pppm.mesh_z.resize(mesh_size);
  pppm.mesh_virial.resize(mesh_size * 6);
  if (cufftPlan3d(&pppm.plan, pppm.para.K[2], pppm.para.K[1], pppm.para.K[0], CUFFT_C2C)
      != CUFFT_SUCCESS) {
    std::cout << "CUFFT error: NEPKK PPPM plan creation failed" << std::endl;
    exit(1);
  }
  pppm.plan_initialized = true;
  int n[3] = {pppm.para.K[2], pppm.para.K[1], pppm.para.K[0]};
  if (cufftPlanMany(&pppm.plan_virial, 3, n, NULL, 1, mesh_size, NULL, 1, mesh_size, CUFFT_C2C, 6)
      != CUFFT_SUCCESS) {
    std::cout << "CUFFT error: NEPKK PPPM virial plan creation failed" << std::endl;
    exit(1);
  }
  pppm.plan_virial_initialized = true;
}

inline void nepkk_pppm_find_para(
  NEPKK_PPPM_Data& pppm,
  const int N,
  const NEP_FLOAT alpha,
  const NEP_FLOAT alpha_factor,
  const NEPKK_Box& box,
  const NEP_FLOAT mesh_spacing,
  const int mesh_mode,
  const int user_mesh[3])
{
  NEPKK_PPPM_Para& para = pppm.para;
  const NEP_FLOAT two_pi = FLOAT_LIT(6.2831853);
  const double spacing = mesh_spacing > FLOAT_LIT(0.0) ? double(mesh_spacing) : 1.0;
  const double volume = nepkk_box_volume(box);
  // printf("box: %f %f %f %f %f %f %f %f %f\n", box.h[0], box.h[1], box.h[2], box.h[3], box.h[4], box.h[5], box.h[6], box.h[7], box.h[8]);
  para.alpha = alpha;
  para.alpha_factor = alpha_factor;
  para.two_pi_over_V = two_pi / NEP_FLOAT(volume);
  int K[3] = {0};
  for (int d = 0; d < 3; ++d) {
    if (user_mesh[d] > 0) {
      K[d] = user_mesh[d];
    } else {
      const double box_thickness = volume / nepkk_box_area(box, d);
      const double mesh_estimate = box_thickness / spacing;
      if (mesh_mode == 1) {
        K[d] = nepkk_get_friendly_pppm_K(int(ceil(mesh_estimate)));
      } else {
        K[d] = nepkk_get_power2_pppm_K(int(mesh_estimate));
      }
    }
    para.K_half[d] = K[d] / 2;
    para.two_pi_over_K[d] = two_pi / K[d];
  }
  para.K0K1 = K[0] * K[1];
  para.K0K1K2 = para.K0K1 * K[2];
  const bool need_allocate =
    !pppm.plan_initialized || K[0] != para.K[0] || K[1] != para.K[1] || K[2] != para.K[2];
  para.K[0] = K[0];
  para.K[1] = K[1];
  para.K[2] = K[2];
  for (int d = 0; d < 3; ++d) {
    para.b[0][d] = two_pi * NEP_FLOAT(box.hi[d]);
    para.b[1][d] = two_pi * NEP_FLOAT(box.hi[3 + d]);
    para.b[2][d] = two_pi * NEP_FLOAT(box.hi[6 + d]);
  }
  if (need_allocate) {
    nepkk_pppm_allocate_memory(pppm);
  }
}

inline void nepkk_pppm_find_force_charge2(
  NEPKK_PPPM_Data& pppm,
  const int N,
  const NEP_FLOAT alpha,
  const NEP_FLOAT alpha_factor,
  const NEPKK_Box& box,
  const NEP_FLOAT mesh_spacing,
  const int mesh_mode,
  const int user_mesh[3],
  const GPU_Vector<NEP_FLOAT>& charge,
  const GPU_Vector<NEP_FLOAT>& position,
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
  nepkk_pppm_find_para(pppm, N, alpha, alpha_factor, box, mesh_spacing, mesh_mode, user_mesh);
  const NEPKK_PPPM_Para para = pppm.para;
  const int mesh_grid_size = (para.K0K1K2 - 1) / 64 + 1;
  const int atom_grid_size = (N - 1) / 64 + 1;

  nepkk_pppm_find_k_and_G<<<mesh_grid_size, 64>>>(
    para, pppm.kx.data(), pppm.ky.data(), pppm.kz.data(), pppm.G.data());
  CUDA_CHECK_KERNEL

  nepkk_pppm_set_mesh_to_zero<<<mesh_grid_size, 64>>>(para, pppm.mesh.data());
  CUDA_CHECK_KERNEL

  nepkk_pppm_find_mesh<<<atom_grid_size, 64>>>(
    N, para, box, charge.data(), position.data(), pppm.mesh.data());
  CUDA_CHECK_KERNEL

  if (mpi_size > 1) {
    nepkk_pppm_allreduce_mesh(pppm, allreduce_double, allreduce_context);
  }

  if (cufftExecC2C(pppm.plan, pppm.mesh.data(), pppm.mesh.data(), CUFFT_FORWARD) != CUFFT_SUCCESS) {
    std::cout << "CUFFT error: NEPKK PPPM forward transform failed" << std::endl;
    exit(1);
  }

  nepkk_pppm_ik_times_mesh_times_G<<<mesh_grid_size, 64>>>(
    para,
    pppm.kx.data(),
    pppm.ky.data(),
    pppm.kz.data(),
    pppm.G.data(),
    pppm.mesh.data(),
    pppm.mesh_x.data(),
    pppm.mesh_y.data(),
    pppm.mesh_z.data());
  CUDA_CHECK_KERNEL

  nepkk_pppm_find_mesh_G<<<mesh_grid_size, 64>>>(
    para, pppm.G.data(), pppm.mesh.data(), pppm.mesh_G.data());
  CUDA_CHECK_KERNEL

  for (int d = 0; d < 6; ++d) {
    nepkk_pppm_set_mesh_to_zero<<<mesh_grid_size, 64>>>(para, pppm.mesh_virial.data() + para.K0K1K2 * d);
    CUDA_CHECK_KERNEL
  }
  nepkk_pppm_find_mesh_virial<<<mesh_grid_size, 64>>>(
    para,
    pppm.kx.data(),
    pppm.ky.data(),
    pppm.kz.data(),
    pppm.G.data(),
    pppm.mesh.data(),
    pppm.mesh_virial.data() + para.K0K1K2 * 0,
    pppm.mesh_virial.data() + para.K0K1K2 * 1,
    pppm.mesh_virial.data() + para.K0K1K2 * 2,
    pppm.mesh_virial.data() + para.K0K1K2 * 3,
    pppm.mesh_virial.data() + para.K0K1K2 * 4,
    pppm.mesh_virial.data() + para.K0K1K2 * 5);
  CUDA_CHECK_KERNEL

  if (cufftExecC2C(pppm.plan, pppm.mesh_G.data(), pppm.mesh_G.data(), CUFFT_INVERSE) != CUFFT_SUCCESS ||
      cufftExecC2C(pppm.plan, pppm.mesh_x.data(), pppm.mesh_x.data(), CUFFT_INVERSE) != CUFFT_SUCCESS ||
      cufftExecC2C(pppm.plan, pppm.mesh_y.data(), pppm.mesh_y.data(), CUFFT_INVERSE) != CUFFT_SUCCESS ||
      cufftExecC2C(pppm.plan, pppm.mesh_z.data(), pppm.mesh_z.data(), CUFFT_INVERSE) != CUFFT_SUCCESS ||
      cufftExecC2C(pppm.plan_virial, pppm.mesh_virial.data(), pppm.mesh_virial.data(), CUFFT_INVERSE) != CUFFT_SUCCESS) {
    std::cout << "CUFFT error: NEPKK PPPM inverse transform failed" << std::endl;
    exit(1);
  }

  nepkk_pppm_find_force_virial_from_field<<<atom_grid_size, 64>>>(
    N,
    para,
    box,
    charge.data(),
    position.data(),
    pppm.mesh_G.data(),
    pppm.mesh_x.data(),
    pppm.mesh_y.data(),
    pppm.mesh_z.data(),
    pppm.mesh_virial.data() + para.K0K1K2 * 0,
    pppm.mesh_virial.data() + para.K0K1K2 * 1,
    pppm.mesh_virial.data() + para.K0K1K2 * 2,
    pppm.mesh_virial.data() + para.K0K1K2 * 3,
    pppm.mesh_virial.data() + para.K0K1K2 * 4,
    pppm.mesh_virial.data() + para.K0K1K2 * 5,
    D_real.data(),
    force_per_atom,
    virial_per_atom,
    vflag_either,
    cvflag_atom,
    vatom_num);
  CUDA_CHECK_KERNEL
}

} // namespace
