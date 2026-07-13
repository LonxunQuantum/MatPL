/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.

***
This section removes all code related to USE_TABLE. 
Change the float to NEP_FLOAT(float or double)
  wuxingxing@pwmat.com and MatPL development team. 2026. Beijing Lonxun Quantum Co.,Ltd.
***
*/
#pragma once
#include "common.cuh"

const int NUM_OF_ABC = 24; // 3 + 5 + 7 + 9 for L_max = 4
__constant__ NEP_FLOAT C3B[NUM_OF_ABC] = {
  FLOAT_LIT(0.238732414637843), FLOAT_LIT(0.119366207318922), FLOAT_LIT(0.119366207318922), FLOAT_LIT(0.099471839432435),
  FLOAT_LIT(0.596831036594608), FLOAT_LIT(0.596831036594608), FLOAT_LIT(0.149207759148652), FLOAT_LIT(0.149207759148652),
  FLOAT_LIT(0.139260575205408), FLOAT_LIT(0.104445431404056), FLOAT_LIT(0.104445431404056), FLOAT_LIT(1.044454314040563),
  FLOAT_LIT(1.044454314040563), FLOAT_LIT(0.174075719006761), FLOAT_LIT(0.174075719006761), FLOAT_LIT(0.011190581936149),
  FLOAT_LIT(0.223811638722978), FLOAT_LIT(0.223811638722978), FLOAT_LIT(0.111905819361489), FLOAT_LIT(0.111905819361489),
  FLOAT_LIT(1.566681471060845), FLOAT_LIT(1.566681471060845), FLOAT_LIT(0.195835183882606), FLOAT_LIT(0.195835183882606)};
__constant__ NEP_FLOAT C4B[5] = {
  FLOAT_LIT(-0.007499480826664),
  FLOAT_LIT(-0.134990654879954),
  FLOAT_LIT(0.067495327439977),
  FLOAT_LIT(0.404971964639861),
  FLOAT_LIT(-0.809943929279723)};
__constant__ NEP_FLOAT C5B[3] = {FLOAT_LIT(0.026596810706114), FLOAT_LIT(0.053193621412227), FLOAT_LIT(0.026596810706114)};

__constant__ NEP_FLOAT Z_COEFFICIENT_1[2][2] = {
  {FLOAT_LIT(0.0), FLOAT_LIT(1.0)},
  {FLOAT_LIT(1.0), FLOAT_LIT(0.0)}};

__constant__ NEP_FLOAT Z_COEFFICIENT_2[3][3] = {
  {FLOAT_LIT(-1.0), FLOAT_LIT(0.0), FLOAT_LIT(3.0)},
  {FLOAT_LIT(0.0), FLOAT_LIT(1.0), FLOAT_LIT(0.0)},
  {FLOAT_LIT(1.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0)}};

__constant__ NEP_FLOAT Z_COEFFICIENT_3[4][4] = {
  {FLOAT_LIT(0.0), FLOAT_LIT(-3.0), FLOAT_LIT(0.0), FLOAT_LIT(5.0)},
  {FLOAT_LIT(-1.0), FLOAT_LIT(0.0), FLOAT_LIT(5.0), FLOAT_LIT(0.0)},
  {FLOAT_LIT(0.0), FLOAT_LIT(1.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0)},
  {FLOAT_LIT(1.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0)}};

__constant__ NEP_FLOAT Z_COEFFICIENT_4[5][5] = {
  {FLOAT_LIT(3.0), FLOAT_LIT(0.0), FLOAT_LIT(-30.0), FLOAT_LIT(0.0), FLOAT_LIT(35.0)},
  {FLOAT_LIT(0.0), FLOAT_LIT(-3.0), FLOAT_LIT(0.0), FLOAT_LIT(7.0), FLOAT_LIT(0.0)},
  {FLOAT_LIT(-1.0), FLOAT_LIT(0.0), FLOAT_LIT(7.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0)},
  {FLOAT_LIT(0.0), FLOAT_LIT(1.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0)},
  {FLOAT_LIT(1.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0)}};

__constant__ NEP_FLOAT Z_COEFFICIENT_5[6][6] = {
  {FLOAT_LIT(0.0), FLOAT_LIT(15.0), FLOAT_LIT(0.0), FLOAT_LIT(-70.0), FLOAT_LIT(0.0), FLOAT_LIT(63.0)},
  {FLOAT_LIT(1.0), FLOAT_LIT(0.0), FLOAT_LIT(-14.0), FLOAT_LIT(0.0), FLOAT_LIT(21.0), FLOAT_LIT(0.0)},
  {FLOAT_LIT(0.0), FLOAT_LIT(-1.0), FLOAT_LIT(0.0), FLOAT_LIT(3.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0)},
  {FLOAT_LIT(-1.0), FLOAT_LIT(0.0), FLOAT_LIT(9.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0)},
  {FLOAT_LIT(0.0), FLOAT_LIT(1.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0)},
  {FLOAT_LIT(1.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0)}};

__constant__ NEP_FLOAT Z_COEFFICIENT_6[7][7] = {
  {FLOAT_LIT(-5.0), FLOAT_LIT(0.0), FLOAT_LIT(105.0), FLOAT_LIT(0.0), FLOAT_LIT(-315.0), FLOAT_LIT(0.0), FLOAT_LIT(231.0)},
  {FLOAT_LIT(0.0), FLOAT_LIT(5.0), FLOAT_LIT(0.0), FLOAT_LIT(-30.0), FLOAT_LIT(0.0), FLOAT_LIT(33.0), FLOAT_LIT(0.0)},
  {FLOAT_LIT(1.0), FLOAT_LIT(0.0), FLOAT_LIT(-18.0), FLOAT_LIT(0.0), FLOAT_LIT(33.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0)},
  {FLOAT_LIT(0.0), FLOAT_LIT(-3.0), FLOAT_LIT(0.0), FLOAT_LIT(11.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0)},
  {FLOAT_LIT(-1.0), FLOAT_LIT(0.0), FLOAT_LIT(11.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0)},
  {FLOAT_LIT(0.0), FLOAT_LIT(1.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0)},
  {FLOAT_LIT(1.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0)}};

__constant__ NEP_FLOAT Z_COEFFICIENT_7[8][8] = {
  {FLOAT_LIT(0.0), FLOAT_LIT(-35.0), FLOAT_LIT(0.0), FLOAT_LIT(315.0), FLOAT_LIT(0.0), FLOAT_LIT(-693.0), FLOAT_LIT(0.0), FLOAT_LIT(429.0)},
  {FLOAT_LIT(-5.0), FLOAT_LIT(0.0), FLOAT_LIT(135.0), FLOAT_LIT(0.0), FLOAT_LIT(-495.0), FLOAT_LIT(0.0), FLOAT_LIT(429.0), FLOAT_LIT(0.0)},
  {FLOAT_LIT(0.0), FLOAT_LIT(15.0), FLOAT_LIT(0.0), FLOAT_LIT(-110.0), FLOAT_LIT(0.0), FLOAT_LIT(143.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0)},
  {FLOAT_LIT(3.0), FLOAT_LIT(0.0), FLOAT_LIT(-66.0), FLOAT_LIT(0.0), FLOAT_LIT(143.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0)},
  {FLOAT_LIT(0.0), FLOAT_LIT(-3.0), FLOAT_LIT(0.0), FLOAT_LIT(13.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0)},
  {FLOAT_LIT(-1.0), FLOAT_LIT(0.0), FLOAT_LIT(13.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0)},
  {FLOAT_LIT(0.0), FLOAT_LIT(1.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0)},
  {FLOAT_LIT(1.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0)}};

__constant__ NEP_FLOAT Z_COEFFICIENT_8[9][9] = {
  {FLOAT_LIT(35.0), FLOAT_LIT(0.0), FLOAT_LIT(-1260.0), FLOAT_LIT(0.0), FLOAT_LIT(6930.0), FLOAT_LIT(0.0), FLOAT_LIT(-12012.0), FLOAT_LIT(0.0), FLOAT_LIT(6435.0)},
  {FLOAT_LIT(0.0), FLOAT_LIT(-35.0), FLOAT_LIT(0.0), FLOAT_LIT(385.0), FLOAT_LIT(0.0), FLOAT_LIT(-1001.0), FLOAT_LIT(0.0), FLOAT_LIT(715.0), FLOAT_LIT(0.0)},
  {FLOAT_LIT(-1.0), FLOAT_LIT(0.0), FLOAT_LIT(33.0), FLOAT_LIT(0.0), FLOAT_LIT(-143.0), FLOAT_LIT(0.0), FLOAT_LIT(143.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0)},
  {FLOAT_LIT(0.0), FLOAT_LIT(3.0), FLOAT_LIT(0.0), FLOAT_LIT(-26.0), FLOAT_LIT(0.0), FLOAT_LIT(39.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0)},
  {FLOAT_LIT(1.0), FLOAT_LIT(0.0), FLOAT_LIT(-26.0), FLOAT_LIT(0.0), FLOAT_LIT(65.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0)},
  {FLOAT_LIT(0.0), FLOAT_LIT(-1.0), FLOAT_LIT(0.0), FLOAT_LIT(5.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0)},
  {FLOAT_LIT(-1.0), FLOAT_LIT(0.0), FLOAT_LIT(15.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0)},
  {FLOAT_LIT(0.0), FLOAT_LIT(1.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0)},
  {FLOAT_LIT(1.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0)}};

__constant__ NEP_FLOAT COVALENT_RADIUS[94] = {
  FLOAT_LIT(0.426667), FLOAT_LIT(0.613333), FLOAT_LIT(1.6),     FLOAT_LIT(1.25333), FLOAT_LIT(1.02667), FLOAT_LIT(1.0),     FLOAT_LIT(0.946667), FLOAT_LIT(0.84),    FLOAT_LIT(0.853333),
  FLOAT_LIT(0.893333), FLOAT_LIT(1.86667),  FLOAT_LIT(1.66667), FLOAT_LIT(1.50667), FLOAT_LIT(1.38667), FLOAT_LIT(1.46667), FLOAT_LIT(1.36),     FLOAT_LIT(1.32),    FLOAT_LIT(1.28),
  FLOAT_LIT(2.34667),  FLOAT_LIT(2.05333),  FLOAT_LIT(1.77333), FLOAT_LIT(1.62667), FLOAT_LIT(1.61333), FLOAT_LIT(1.46667), FLOAT_LIT(1.42667),  FLOAT_LIT(1.38667), FLOAT_LIT(1.33333),
  FLOAT_LIT(1.32),     FLOAT_LIT(1.34667),  FLOAT_LIT(1.45333), FLOAT_LIT(1.49333), FLOAT_LIT(1.45333), FLOAT_LIT(1.53333), FLOAT_LIT(1.46667),  FLOAT_LIT(1.52),    FLOAT_LIT(1.56),
  FLOAT_LIT(2.52),     FLOAT_LIT(2.22667),  FLOAT_LIT(1.96),    FLOAT_LIT(1.85333), FLOAT_LIT(1.76),    FLOAT_LIT(1.65333), FLOAT_LIT(1.53333),  FLOAT_LIT(1.50667), FLOAT_LIT(1.50667),
  FLOAT_LIT(1.44),     FLOAT_LIT(1.53333),  FLOAT_LIT(1.64),    FLOAT_LIT(1.70667), FLOAT_LIT(1.68),    FLOAT_LIT(1.68),    FLOAT_LIT(1.64),     FLOAT_LIT(1.76),    FLOAT_LIT(1.74667),
  FLOAT_LIT(2.78667),  FLOAT_LIT(2.34667),  FLOAT_LIT(2.16),    FLOAT_LIT(1.96),    FLOAT_LIT(2.10667), FLOAT_LIT(2.09333), FLOAT_LIT(2.08),     FLOAT_LIT(2.06667), FLOAT_LIT(2.01333),
  FLOAT_LIT(2.02667),  FLOAT_LIT(2.01333),  FLOAT_LIT(2.0),     FLOAT_LIT(1.98667), FLOAT_LIT(1.98667), FLOAT_LIT(1.97333), FLOAT_LIT(2.04),     FLOAT_LIT(1.94667), FLOAT_LIT(1.82667),
  FLOAT_LIT(1.74667),  FLOAT_LIT(1.64),     FLOAT_LIT(1.57333), FLOAT_LIT(1.54667), FLOAT_LIT(1.48),    FLOAT_LIT(1.49333), FLOAT_LIT(1.50667),  FLOAT_LIT(1.76),    FLOAT_LIT(1.73333),
  FLOAT_LIT(1.73333),  FLOAT_LIT(1.81333),  FLOAT_LIT(1.74667), FLOAT_LIT(1.84),    FLOAT_LIT(1.89333), FLOAT_LIT(2.68),    FLOAT_LIT(2.41333),  FLOAT_LIT(2.22667), FLOAT_LIT(2.10667),
  FLOAT_LIT(2.02667),  FLOAT_LIT(2.04),     FLOAT_LIT(2.05333), FLOAT_LIT(2.06667)};

const int SIZE_BOX_AND_INVERSE_BOX = 18; // (3 * 3) * 2
const int MAX_NUM_N_10 = 10;
const int MAX_NUM_N = 20;                // basis_size_radial+1 = 16+1
const int MAX_DIM = 103;                 // 13 + 9 * 10
const int MAX_DIM_ANGULAR = 90;          // 9 * 10
// const int MAX_NUM_N = 20;                // n_max+1 = 19+1
// const int MAX_DIM = MAX_NUM_N * 7;
// const int MAX_DIM_ANGULAR = MAX_NUM_N * 6;
__constant__ NEP_FLOAT Q_SCALER[MAX_DIM];

static __device__ __forceinline__ void
complex_product(const NEP_FLOAT a, const NEP_FLOAT b, NEP_FLOAT& real_part, NEP_FLOAT& imag_part)
{
  const NEP_FLOAT real_temp = real_part;
  real_part = a * real_temp - b * imag_part;
  imag_part = a * imag_part + b * real_temp;
}

static __device__ void apply_ann_one_layer(
  const int N_des,
  const int N_neu,
  const NEP_FLOAT* w0,
  const NEP_FLOAT* b0,
  const NEP_FLOAT* w1,
  const NEP_FLOAT* b1,
  NEP_FLOAT* q,
  NEP_FLOAT& energy,
  NEP_FLOAT* energy_derivative,
  const int atom_type)
{
  for (int n = 0; n < N_neu; ++n) {
    NEP_FLOAT w0_times_q = FLOAT_LIT(0.0);
    for (int d = 0; d < N_des; ++d) {
      w0_times_q += w0[n * N_des + d] * q[d];
    }
    NEP_FLOAT x1 = tanh(w0_times_q - b0[n]);
    NEP_FLOAT tanh_der = FLOAT_LIT(1.0) - x1 * x1;
    energy += w1[n] * x1;
    for (int d = 0; d < N_des; ++d) {
      NEP_FLOAT y1 = tanh_der * w0[n * N_des + d];
      energy_derivative[d] += w1[n] * y1;
    }
  }
  energy -= b1[atom_type];
}

static __device__ void apply_ann_one_layer_nep5(
  const int N_des,
  const int N_neu,
  const NEP_FLOAT* w0,
  const NEP_FLOAT* b0,
  const NEP_FLOAT* w1,
  const NEP_FLOAT* b1,
  NEP_FLOAT* q,
  NEP_FLOAT& energy,
  NEP_FLOAT* energy_derivative,
  const int atom_type)
{
  for (int n = 0; n < N_neu; ++n) {
    NEP_FLOAT w0_times_q = FLOAT_LIT(0.0);
    for (int d = 0; d < N_des; ++d) {
      w0_times_q += w0[n * N_des + d] * q[d];
    }
    NEP_FLOAT x1 = tanh(w0_times_q - b0[n]);
    NEP_FLOAT tanh_der = FLOAT_LIT(1.0) - x1 * x1;
    energy += w1[n] * x1;
    for (int d = 0; d < N_des; ++d) {
      NEP_FLOAT y1 = tanh_der * w0[n * N_des + d];
      energy_derivative[d] += w1[n] * y1;
    }
  }
  energy -= w1[N_neu] + b1[0]; // typewise bias + common bias
}

static __device__ void apply_ann_one_layer_charge(
  const int N_des,
  const int N_neu,
  const NEP_FLOAT* w0,
  const NEP_FLOAT* b0,
  const NEP_FLOAT* w1,
  const NEP_FLOAT* b1,
  NEP_FLOAT* q,
  NEP_FLOAT& energy,
  NEP_FLOAT* energy_derivative,
  NEP_FLOAT& charge,
  NEP_FLOAT* charge_derivative,
  const int model_version)
{
  for (int n = 0; n < N_neu; ++n) {
    NEP_FLOAT w0_times_q = FLOAT_LIT(0.0);
    for (int d = 0; d < N_des; ++d) {
      w0_times_q += w0[n * N_des + d] * q[d];
    }
    NEP_FLOAT x1 = tanh(w0_times_q - b0[n]);
    NEP_FLOAT tanh_der = FLOAT_LIT(1.0) - x1 * x1;
    energy += w1[n] * x1;
    charge += w1[n + N_neu] * x1;
    for (int d = 0; d < N_des; ++d) {
      NEP_FLOAT y1 = tanh_der * w0[n * N_des + d];
      energy_derivative[d] += w1[n] * y1;
      charge_derivative[d] += w1[n + N_neu] * y1;
    }
  }
  energy -= (model_version == 5 ? w1[N_neu * 2] + b1[0] : b1[0]);
}

static __device__ __forceinline__ void find_fc(NEP_FLOAT rc, NEP_FLOAT rcinv, NEP_FLOAT d12, NEP_FLOAT& fc)
{
  if (d12 < rc) {
    NEP_FLOAT x = d12 * rcinv;
    fc = FLOAT_LIT(0.5) * cos(FLOAT_LIT(3.1415927) * x) + FLOAT_LIT(0.5);
  } else {
    fc = FLOAT_LIT(0.0);
  }
}

static __device__ __host__ __forceinline__ void
find_fc_and_fcp(NEP_FLOAT rc, NEP_FLOAT rcinv, NEP_FLOAT d12, NEP_FLOAT& fc, NEP_FLOAT& fcp)
{
  if (d12 < rc) {
    NEP_FLOAT x = d12 * rcinv;
    fc = FLOAT_LIT(0.5) * cos(FLOAT_LIT(3.1415927) * x) + FLOAT_LIT(0.5);
    fcp = FLOAT_LIT(-1.5707963) * sin(FLOAT_LIT(3.1415927) * x);
    fcp *= rcinv;
  } else {
    fc = FLOAT_LIT(0.0);
    fcp = FLOAT_LIT(0.0);
  }
}

static __device__ __forceinline__ void
find_fc_and_fcp_zbl(NEP_FLOAT r1, NEP_FLOAT r2, NEP_FLOAT d12, NEP_FLOAT& fc, NEP_FLOAT& fcp)
{
  if (d12 < r1) {
    fc = FLOAT_LIT(1.0);
    fcp = FLOAT_LIT(0.0);
  } else if (d12 < r2) {
    NEP_FLOAT pi_factor = FLOAT_LIT(3.1415927) / (r2 - r1);
    fc = cos(pi_factor * (d12 - r1)) * FLOAT_LIT(0.5) + FLOAT_LIT(0.5);
    fcp = -sin(pi_factor * (d12 - r1)) * pi_factor * FLOAT_LIT(0.5);
  } else {
    fc = FLOAT_LIT(0.0);
    fcp = FLOAT_LIT(0.0);
  }
}

static __device__ __forceinline__ void
find_phi_and_phip_zbl(NEP_FLOAT a, NEP_FLOAT b, NEP_FLOAT x, NEP_FLOAT& phi, NEP_FLOAT& phip)
{
  NEP_FLOAT tmp = a * exp(-b * x);
  phi += tmp;
  phip -= b * tmp;
}

static __device__ __forceinline__ void find_f_and_fp_zbl(
  const NEP_FLOAT zizj,
  const NEP_FLOAT a_inv,
  const NEP_FLOAT rc_inner,
  const NEP_FLOAT rc_outer,
  const NEP_FLOAT d12,
  const NEP_FLOAT d12inv,
  NEP_FLOAT& f,
  NEP_FLOAT& fp)
{
  const NEP_FLOAT x = d12 * a_inv;
  f = fp = FLOAT_LIT(0.0);
  const NEP_FLOAT Zbl_para[8] = {
    FLOAT_LIT(0.18175), FLOAT_LIT(3.1998), FLOAT_LIT(0.50986), FLOAT_LIT(0.94229),
    FLOAT_LIT(0.28022), FLOAT_LIT(0.4029), FLOAT_LIT(0.02817), FLOAT_LIT(0.20162)};
  find_phi_and_phip_zbl(Zbl_para[0], Zbl_para[1], x, f, fp);
  find_phi_and_phip_zbl(Zbl_para[2], Zbl_para[3], x, f, fp);
  find_phi_and_phip_zbl(Zbl_para[4], Zbl_para[5], x, f, fp);
  find_phi_and_phip_zbl(Zbl_para[6], Zbl_para[7], x, f, fp);
  f *= zizj;
  fp *= zizj * a_inv;
  fp = fp * d12inv - f * d12inv * d12inv;
  f *= d12inv;
  NEP_FLOAT fc, fcp;
  find_fc_and_fcp_zbl(rc_inner, rc_outer, d12, fc, fcp);
  fp = fp * fc + f * fcp;
  f *= fc;
}

static __device__ __forceinline__ void find_f_and_fp_zbl(
  const NEP_FLOAT* zbl_para,
  const NEP_FLOAT zizj,
  const NEP_FLOAT a_inv,
  const NEP_FLOAT d12,
  const NEP_FLOAT d12inv,
  NEP_FLOAT& f,
  NEP_FLOAT& fp)
{
  const NEP_FLOAT x = d12 * a_inv;
  f = fp = FLOAT_LIT(0.0);
  find_phi_and_phip_zbl(zbl_para[2], zbl_para[3], x, f, fp);
  find_phi_and_phip_zbl(zbl_para[4], zbl_para[5], x, f, fp);
  find_phi_and_phip_zbl(zbl_para[6], zbl_para[7], x, f, fp);
  find_phi_and_phip_zbl(zbl_para[8], zbl_para[9], x, f, fp);
  f *= zizj;
  fp *= zizj * a_inv;
  fp = fp * d12inv - f * d12inv * d12inv;
  f *= d12inv;
  NEP_FLOAT fc, fcp;
  find_fc_and_fcp_zbl(zbl_para[0], zbl_para[1], d12, fc, fcp);
  fp = fp * fc + f * fcp;
  f *= fc;
}

static __device__ __forceinline__ void
find_fn(const int n, const NEP_FLOAT rcinv, const NEP_FLOAT d12, const NEP_FLOAT fc12, NEP_FLOAT& fn)
{
  if (n == 0) {
    fn = fc12;
  } else if (n == 1) {
    NEP_FLOAT x = FLOAT_LIT(2.0) * (d12 * rcinv - FLOAT_LIT(1.0)) * (d12 * rcinv - FLOAT_LIT(1.0)) - FLOAT_LIT(1.0);
    fn = (x + FLOAT_LIT(1.0)) * FLOAT_LIT(0.5) * fc12;
  } else {
    NEP_FLOAT x = FLOAT_LIT(2.0) * (d12 * rcinv - FLOAT_LIT(1.0)) * (d12 * rcinv - FLOAT_LIT(1.0)) - FLOAT_LIT(1.0);
    NEP_FLOAT t0 = FLOAT_LIT(1.0);
    NEP_FLOAT t1 = x;
    NEP_FLOAT t2;
    for (int m = 2; m <= n; ++m) {
      t2 = FLOAT_LIT(2.0) * x * t1 - t0;
      t0 = t1;
      t1 = t2;
    }
    fn = (t2 + FLOAT_LIT(1.0)) * FLOAT_LIT(0.5) * fc12;
  }
}

static __device__ __forceinline__ void find_fn_and_fnp(
  const int n,
  const NEP_FLOAT rcinv,
  const NEP_FLOAT d12,
  const NEP_FLOAT fc12,
  const NEP_FLOAT fcp12,
  NEP_FLOAT& fn,
  NEP_FLOAT& fnp)
{
  if (n == 0) {
    fn = fc12;
    fnp = fcp12;
  } else if (n == 1) {
    NEP_FLOAT x = FLOAT_LIT(2.0) * (d12 * rcinv - FLOAT_LIT(1.0)) * (d12 * rcinv - FLOAT_LIT(1.0)) - FLOAT_LIT(1.0);
    fn = (x + FLOAT_LIT(1.0)) * FLOAT_LIT(0.5);
    fnp = FLOAT_LIT(2.0) * (d12 * rcinv - FLOAT_LIT(1.0)) * rcinv * fc12 + fn * fcp12;
    fn *= fc12;
  } else {
    NEP_FLOAT x = FLOAT_LIT(2.0) * (d12 * rcinv - FLOAT_LIT(1.0)) * (d12 * rcinv - FLOAT_LIT(1.0)) - FLOAT_LIT(1.0);
    NEP_FLOAT t0 = FLOAT_LIT(1.0);
    NEP_FLOAT t1 = x;
    NEP_FLOAT t2;
    NEP_FLOAT u0 = FLOAT_LIT(1.0);
    NEP_FLOAT u1 = FLOAT_LIT(2.0) * x;
    NEP_FLOAT u2;
    for (int m = 2; m <= n; ++m) {
      t2 = FLOAT_LIT(2.0) * x * t1 - t0;
      t0 = t1;
      t1 = t2;
      u2 = FLOAT_LIT(2.0) * x * u1 - u0;
      u0 = u1;
      u1 = u2;
    }
    fn = (t2 + FLOAT_LIT(1.0)) * FLOAT_LIT(0.5);
    fnp = n * u0 * FLOAT_LIT(2.0) * (d12 * rcinv - FLOAT_LIT(1.0)) * rcinv;
    fnp = fnp * fc12 + fn * fcp12;
    fn *= fc12;
  }
}

static __device__ __forceinline__ void
find_fn(const int n_max, const NEP_FLOAT rcinv, const NEP_FLOAT d12, const NEP_FLOAT fc12, NEP_FLOAT* fn)
{
  NEP_FLOAT x = FLOAT_LIT(2.0) * (d12 * rcinv - FLOAT_LIT(1.0)) * (d12 * rcinv - FLOAT_LIT(1.0)) - FLOAT_LIT(1.0);
  NEP_FLOAT half_fc12 = FLOAT_LIT(0.5) * fc12;
  fn[0] = fc12;
  fn[1] = (x + FLOAT_LIT(1.0)) * half_fc12;
  NEP_FLOAT fn_m_minus_2 = FLOAT_LIT(1.0);
  NEP_FLOAT fn_m_minus_1 = x;
  NEP_FLOAT tmp = FLOAT_LIT(0.0);
  for (int m = 2; m <= n_max; ++m) {
    tmp = FLOAT_LIT(2.0) * x * fn_m_minus_1 - fn_m_minus_2;
    fn_m_minus_2 = fn_m_minus_1;
    fn_m_minus_1 = tmp;
    fn[m] = (tmp + FLOAT_LIT(1.0)) * half_fc12;
  }
}

static __device__ __forceinline__ NEP_FLOAT find_gn(
  const int basis_size,
  const NEP_FLOAT rcinv,
  const NEP_FLOAT d12,
  const NEP_FLOAT fc12,
  const NEP_FLOAT* coefficients)
{
  const NEP_FLOAT d12_mul_rcinv = d12 * rcinv;
  const NEP_FLOAT x = FLOAT_LIT(2.0) * (d12_mul_rcinv - FLOAT_LIT(1.0)) *
      (d12_mul_rcinv - FLOAT_LIT(1.0)) - FLOAT_LIT(1.0);
  const NEP_FLOAT half_fc12 = FLOAT_LIT(0.5) * fc12;
  NEP_FLOAT gn = fc12 * coefficients[0];
  if (basis_size >= 1) {
    gn += (x + FLOAT_LIT(1.0)) * half_fc12 * coefficients[1];
  }
  NEP_FLOAT fn_m_minus_2 = FLOAT_LIT(1.0);
  NEP_FLOAT fn_m_minus_1 = x;
  for (int m = 2; m <= basis_size; ++m) {
    const NEP_FLOAT fn_m = FLOAT_LIT(2.0) * x * fn_m_minus_1 - fn_m_minus_2;
    gn += (fn_m + FLOAT_LIT(1.0)) * half_fc12 * coefficients[m];
    fn_m_minus_2 = fn_m_minus_1;
    fn_m_minus_1 = fn_m;
  }
  return gn;
}

static __device__ __host__ __forceinline__ void find_fn_and_fnp(
  const int n_max,
  const NEP_FLOAT rcinv,
  const NEP_FLOAT d12,
  const NEP_FLOAT fc12,
  const NEP_FLOAT fcp12,
  NEP_FLOAT* fn,
  NEP_FLOAT* fnp)
{
  NEP_FLOAT d12_mul_rcinv = d12 * rcinv;
  NEP_FLOAT x = FLOAT_LIT(2.0) * (d12_mul_rcinv - FLOAT_LIT(1.0)) * (d12_mul_rcinv - FLOAT_LIT(1.0)) - FLOAT_LIT(1.0);
  fn[0] = fc12;
  fnp[0] = fcp12;
  fn[1] = (x + FLOAT_LIT(1.0)) * FLOAT_LIT(0.5) * fc12;
  fnp[1] = FLOAT_LIT(2.0) * (d12_mul_rcinv - FLOAT_LIT(1.0)) * rcinv * fc12 + (x + FLOAT_LIT(1.0)) * FLOAT_LIT(0.5) * fcp12;
  NEP_FLOAT u0 = FLOAT_LIT(1.0);
  NEP_FLOAT u1 = FLOAT_LIT(2.0) * x;
  NEP_FLOAT u2;
  NEP_FLOAT fn_m_minus_2 = FLOAT_LIT(1.0);
  NEP_FLOAT fn_m_minus_1 = x;
  for (int m = 2; m <= n_max; ++m) {
    NEP_FLOAT fn_tmp1 = FLOAT_LIT(2.0) * x * fn_m_minus_1 - fn_m_minus_2;
    fn_m_minus_2 = fn_m_minus_1;
    fn_m_minus_1 = fn_tmp1;
    NEP_FLOAT fnp_tmp = m * u1;
    u2 = FLOAT_LIT(2.0) * x * u1 - u0;
    u0 = u1;
    u1 = u2;

    NEP_FLOAT fn_tmp2 = (fn_tmp1 + FLOAT_LIT(1.0)) * FLOAT_LIT(0.5);
    fnp[m] = (fnp_tmp * FLOAT_LIT(2.0) * (d12 * rcinv - FLOAT_LIT(1.0)) * rcinv) * fc12 + fn_tmp2 * fcp12;
    fn[m] = fn_tmp2 * fc12;
  }
}

static __device__ __host__ __forceinline__ void find_fn_and_fnp_strided(
  const int n_max,
  const NEP_FLOAT rcinv,
  const NEP_FLOAT d12,
  const NEP_FLOAT fc12,
  const NEP_FLOAT fcp12,
  const int stride,
  NEP_FLOAT* fn,
  NEP_FLOAT* fnp)
{
  NEP_FLOAT d12_mul_rcinv = d12 * rcinv;
  NEP_FLOAT x = FLOAT_LIT(2.0) * (d12_mul_rcinv - FLOAT_LIT(1.0)) *
    (d12_mul_rcinv - FLOAT_LIT(1.0)) - FLOAT_LIT(1.0);
  fn[0] = fc12;
  fnp[0] = fcp12;
  fn[stride] = (x + FLOAT_LIT(1.0)) * FLOAT_LIT(0.5) * fc12;
  fnp[stride] = FLOAT_LIT(2.0) * (d12_mul_rcinv - FLOAT_LIT(1.0)) * rcinv * fc12 +
    (x + FLOAT_LIT(1.0)) * FLOAT_LIT(0.5) * fcp12;
  NEP_FLOAT u0 = FLOAT_LIT(1.0);
  NEP_FLOAT u1 = FLOAT_LIT(2.0) * x;
  NEP_FLOAT u2;
  NEP_FLOAT fn_m_minus_2 = FLOAT_LIT(1.0);
  NEP_FLOAT fn_m_minus_1 = x;
  for (int m = 2; m <= n_max; ++m) {
    NEP_FLOAT fn_tmp1 = FLOAT_LIT(2.0) * x * fn_m_minus_1 - fn_m_minus_2;
    fn_m_minus_2 = fn_m_minus_1;
    fn_m_minus_1 = fn_tmp1;
    NEP_FLOAT fnp_tmp = m * u1;
    u2 = FLOAT_LIT(2.0) * x * u1 - u0;
    u0 = u1;
    u1 = u2;

    NEP_FLOAT fn_tmp2 = (fn_tmp1 + FLOAT_LIT(1.0)) * FLOAT_LIT(0.5);
    fnp[m * stride] =
      (fnp_tmp * FLOAT_LIT(2.0) * (d12_mul_rcinv - FLOAT_LIT(1.0)) * rcinv) * fc12 +
      fn_tmp2 * fcp12;
    fn[m * stride] = fn_tmp2 * fc12;
  }
}

static __device__ __host__ __forceinline__ void find_fnp(
  const int n_max,
  const NEP_FLOAT rcinv,
  const NEP_FLOAT d12,
  const NEP_FLOAT fc12,
  const NEP_FLOAT fcp12,
  NEP_FLOAT* fnp)
{
  NEP_FLOAT d12_mul_rcinv = d12 * rcinv;
  NEP_FLOAT x = FLOAT_LIT(2.0) * (d12_mul_rcinv - FLOAT_LIT(1.0)) * (d12_mul_rcinv - FLOAT_LIT(1.0)) - FLOAT_LIT(1.0);
  fnp[0] = fcp12;
  fnp[1] = FLOAT_LIT(2.0) * (d12_mul_rcinv - FLOAT_LIT(1.0)) * rcinv * fc12 +
    (x + FLOAT_LIT(1.0)) * FLOAT_LIT(0.5) * fcp12;
  NEP_FLOAT u0 = FLOAT_LIT(1.0);
  NEP_FLOAT u1 = FLOAT_LIT(2.0) * x;
  NEP_FLOAT u2;
  NEP_FLOAT fn_m_minus_2 = FLOAT_LIT(1.0);
  NEP_FLOAT fn_m_minus_1 = x;
  for (int m = 2; m <= n_max; ++m) {
    NEP_FLOAT fn_tmp1 = FLOAT_LIT(2.0) * x * fn_m_minus_1 - fn_m_minus_2;
    fn_m_minus_2 = fn_m_minus_1;
    fn_m_minus_1 = fn_tmp1;
    NEP_FLOAT fnp_tmp = m * u1;
    u2 = FLOAT_LIT(2.0) * x * u1 - u0;
    u0 = u1;
    u1 = u2;

    NEP_FLOAT fn_tmp2 = (fn_tmp1 + FLOAT_LIT(1.0)) * FLOAT_LIT(0.5);
    fnp[m] = (fnp_tmp * FLOAT_LIT(2.0) * (d12_mul_rcinv - FLOAT_LIT(1.0)) * rcinv) * fc12 +
      fn_tmp2 * fcp12;
  }
}

static __device__ __host__ __forceinline__ void find_fnp_strided(
  const int n_max,
  const NEP_FLOAT rcinv,
  const NEP_FLOAT d12,
  const NEP_FLOAT fc12,
  const NEP_FLOAT fcp12,
  const int stride,
  NEP_FLOAT* fnp)
{
  NEP_FLOAT d12_mul_rcinv = d12 * rcinv;
  NEP_FLOAT x = FLOAT_LIT(2.0) * (d12_mul_rcinv - FLOAT_LIT(1.0)) * (d12_mul_rcinv - FLOAT_LIT(1.0)) - FLOAT_LIT(1.0);
  fnp[0] = fcp12;
  fnp[stride] = FLOAT_LIT(2.0) * (d12_mul_rcinv - FLOAT_LIT(1.0)) * rcinv * fc12 +
    (x + FLOAT_LIT(1.0)) * FLOAT_LIT(0.5) * fcp12;
  NEP_FLOAT u0 = FLOAT_LIT(1.0);
  NEP_FLOAT u1 = FLOAT_LIT(2.0) * x;
  NEP_FLOAT u2;
  NEP_FLOAT fn_m_minus_2 = FLOAT_LIT(1.0);
  NEP_FLOAT fn_m_minus_1 = x;
  for (int m = 2; m <= n_max; ++m) {
    NEP_FLOAT fn_tmp1 = FLOAT_LIT(2.0) * x * fn_m_minus_1 - fn_m_minus_2;
    fn_m_minus_2 = fn_m_minus_1;
    fn_m_minus_1 = fn_tmp1;
    NEP_FLOAT fnp_tmp = m * u1;
    u2 = FLOAT_LIT(2.0) * x * u1 - u0;
    u0 = u1;
    u1 = u2;

    NEP_FLOAT fn_tmp2 = (fn_tmp1 + FLOAT_LIT(1.0)) * FLOAT_LIT(0.5);
    fnp[m * stride] = (fnp_tmp * FLOAT_LIT(2.0) * (d12_mul_rcinv - FLOAT_LIT(1.0)) * rcinv) * fc12 +
      fn_tmp2 * fcp12;
  }
}

static __device__ __forceinline__ void get_f12_4body(
  const NEP_FLOAT d12,
  const NEP_FLOAT d12inv,
  const NEP_FLOAT fn,
  const NEP_FLOAT fnp,
  const NEP_FLOAT Fp,
  const NEP_FLOAT* s,
  const NEP_FLOAT* r12,
  NEP_FLOAT* f12)
{
  NEP_FLOAT fn_factor = Fp * fn;
  NEP_FLOAT fnp_factor = Fp * fnp * d12inv;
  NEP_FLOAT y20 = (FLOAT_LIT(3.0) * r12[2] * r12[2] - d12 * d12);

  // derivative wrt s[0]
  NEP_FLOAT tmp0 = C4B[0] * FLOAT_LIT(3.0) * s[0] * s[0] + C4B[1] * (s[1] * s[1] + s[2] * s[2]) +
                   C4B[2] * (s[3] * s[3] + s[4] * s[4]);
  NEP_FLOAT tmp1 = tmp0 * y20 * fnp_factor;
  NEP_FLOAT tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] - tmp2 * FLOAT_LIT(2.0) * r12[0];
  f12[1] += tmp1 * r12[1] - tmp2 * FLOAT_LIT(2.0) * r12[1];
  f12[2] += tmp1 * r12[2] + tmp2 * FLOAT_LIT(4.0) * r12[2];

  // derivative wrt s[1]
  tmp0 = C4B[1] * s[0] * s[1] * FLOAT_LIT(2.0) - C4B[3] * s[3] * s[1] * FLOAT_LIT(2.0) + C4B[4] * s[2] * s[4];
  tmp1 = tmp0 * r12[0] * r12[2] * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] + tmp2 * r12[2];
  f12[1] += tmp1 * r12[1];
  f12[2] += tmp1 * r12[2] + tmp2 * r12[0];

  // derivative wrt s[2]
  tmp0 = C4B[1] * s[0] * s[2] * FLOAT_LIT(2.0) + C4B[3] * s[3] * s[2] * FLOAT_LIT(2.0) + C4B[4] * s[1] * s[4];
  tmp1 = tmp0 * r12[1] * r12[2] * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0];
  f12[1] += tmp1 * r12[1] + tmp2 * r12[2];
  f12[2] += tmp1 * r12[2] + tmp2 * r12[1];

  // derivative wrt s[3]
  tmp0 = C4B[2] * s[0] * s[3] * FLOAT_LIT(2.0) + C4B[3] * (s[2] * s[2] - s[1] * s[1]);
  tmp1 = tmp0 * (r12[0] * r12[0] - r12[1] * r12[1]) * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] + tmp2 * FLOAT_LIT(2.0) * r12[0];
  f12[1] += tmp1 * r12[1] - tmp2 * FLOAT_LIT(2.0) * r12[1];
  f12[2] += tmp1 * r12[2];

  // derivative wrt s[4]
  tmp0 = C4B[2] * s[0] * s[4] * FLOAT_LIT(2.0) + C4B[4] * s[1] * s[2];
  tmp1 = tmp0 * (FLOAT_LIT(2.0) * r12[0] * r12[1]) * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] + tmp2 * FLOAT_LIT(2.0) * r12[1];
  f12[1] += tmp1 * r12[1] + tmp2 * FLOAT_LIT(2.0) * r12[0];
  f12[2] += tmp1 * r12[2];
}

static __device__ __forceinline__ void get_f12_5body(
  const NEP_FLOAT d12,
  const NEP_FLOAT d12inv,
  const NEP_FLOAT fn,
  const NEP_FLOAT fnp,
  const NEP_FLOAT Fp,
  const NEP_FLOAT* s,
  const NEP_FLOAT* r12,
  NEP_FLOAT* f12)
{
  NEP_FLOAT fn_factor = Fp * fn;
  NEP_FLOAT fnp_factor = Fp * fnp * d12inv;
  NEP_FLOAT s1_sq_plus_s2_sq = s[1] * s[1] + s[2] * s[2];

  // derivative wrt s[0]
  NEP_FLOAT tmp0 = C5B[0] * FLOAT_LIT(4.0) * s[0] * s[0] * s[0] + C5B[1] * s1_sq_plus_s2_sq * FLOAT_LIT(2.0) * s[0];
  NEP_FLOAT tmp1 = tmp0 * r12[2] * fnp_factor;
  NEP_FLOAT tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0];
  f12[1] += tmp1 * r12[1];
  f12[2] += tmp1 * r12[2] + tmp2;

  // derivative wrt s[1]
  tmp0 = C5B[1] * s[0] * s[0] * s[1] * FLOAT_LIT(2.0) + C5B[2] * s1_sq_plus_s2_sq * s[1] * FLOAT_LIT(4.0);
  tmp1 = tmp0 * r12[0] * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] + tmp2;
  f12[1] += tmp1 * r12[1];
  f12[2] += tmp1 * r12[2];

  // derivative wrt s[2]
  tmp0 = C5B[1] * s[0] * s[0] * s[2] * FLOAT_LIT(2.0) + C5B[2] * s1_sq_plus_s2_sq * s[2] * FLOAT_LIT(4.0);
  tmp1 = tmp0 * r12[1] * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0];
  f12[1] += tmp1 * r12[1] + tmp2;
  f12[2] += tmp1 * r12[2];
}

template <int L>
static __device__ __forceinline__ void calculate_s_one(
  const int n, const int n_max_angular_plus_1, const NEP_FLOAT* Fp, const NEP_FLOAT* sum_fxyz, NEP_FLOAT* s)
{
  const int L_minus_1 = L - 1;
  const int L_twice_plus_1 = 2 * L + 1;
  const int L_square_minus_1 = L * L - 1;
  NEP_FLOAT Fp_factor = FLOAT_LIT(2.0) * Fp[L_minus_1 * n_max_angular_plus_1 + n];
  s[0] = sum_fxyz[n * NUM_OF_ABC + L_square_minus_1] * C3B[L_square_minus_1] * Fp_factor;
  Fp_factor *= FLOAT_LIT(2.0);
  for (int k = 1; k < L_twice_plus_1; ++k) {
    s[k] = sum_fxyz[n * NUM_OF_ABC + L_square_minus_1 + k] * C3B[L_square_minus_1 + k] * Fp_factor;
  }
}

template <int L>
static __device__ __forceinline__ void accumulate_f12_one(
  const NEP_FLOAT d12inv, const NEP_FLOAT fn, const NEP_FLOAT fnp, const NEP_FLOAT* s, const NEP_FLOAT* r12, NEP_FLOAT* f12)
{
  const NEP_FLOAT dx[3] = {
    (FLOAT_LIT(1.0) - r12[0] * r12[0]) * d12inv, -r12[0] * r12[1] * d12inv, -r12[0] * r12[2] * d12inv};
  const NEP_FLOAT dy[3] = {
    -r12[0] * r12[1] * d12inv, (FLOAT_LIT(1.0) - r12[1] * r12[1]) * d12inv, -r12[1] * r12[2] * d12inv};
  const NEP_FLOAT dz[3] = {
    -r12[0] * r12[2] * d12inv, -r12[1] * r12[2] * d12inv, (FLOAT_LIT(1.0) - r12[2] * r12[2]) * d12inv};

  NEP_FLOAT z_pow[L + 1] = {FLOAT_LIT(1.0)};
  for (int n = 1; n <= L; ++n) {
    z_pow[n] = r12[2] * z_pow[n - 1];
  }

  NEP_FLOAT real_part = FLOAT_LIT(1.0);
  NEP_FLOAT imag_part = FLOAT_LIT(0.0);
  for (int n1 = 0; n1 <= L; ++n1) {
    int n2_start = (L + n1) % 2 == 0 ? 0 : 1;
    NEP_FLOAT z_factor = FLOAT_LIT(0.0);
    NEP_FLOAT dz_factor = FLOAT_LIT(0.0);
    for (int n2 = n2_start; n2 <= L - n1; n2 += 2) {
      if (L == 1) {
        z_factor += Z_COEFFICIENT_1[n1][n2] * z_pow[n2];
        if (n2 > 0) {
          dz_factor += Z_COEFFICIENT_1[n1][n2] * n2 * z_pow[n2 - 1];
        }
      }
      if (L == 2) {
        z_factor += Z_COEFFICIENT_2[n1][n2] * z_pow[n2];
        if (n2 > 0) {
          dz_factor += Z_COEFFICIENT_2[n1][n2] * n2 * z_pow[n2 - 1];
        }
      }
      if (L == 3) {
        z_factor += Z_COEFFICIENT_3[n1][n2] * z_pow[n2];
        if (n2 > 0) {
          dz_factor += Z_COEFFICIENT_3[n1][n2] * n2 * z_pow[n2 - 1];
        }
      }
      if (L == 4) {
        z_factor += Z_COEFFICIENT_4[n1][n2] * z_pow[n2];
        if (n2 > 0) {
          dz_factor += Z_COEFFICIENT_4[n1][n2] * n2 * z_pow[n2 - 1];
        }
      }
      if (L == 5) {
        z_factor += Z_COEFFICIENT_5[n1][n2] * z_pow[n2];
        if (n2 > 0) {
          dz_factor += Z_COEFFICIENT_5[n1][n2] * n2 * z_pow[n2 - 1];
        }
      }
      if (L == 6) {
        z_factor += Z_COEFFICIENT_6[n1][n2] * z_pow[n2];
        if (n2 > 0) {
          dz_factor += Z_COEFFICIENT_6[n1][n2] * n2 * z_pow[n2 - 1];
        }
      }
      if (L == 7) {
        z_factor += Z_COEFFICIENT_7[n1][n2] * z_pow[n2];
        if (n2 > 0) {
          dz_factor += Z_COEFFICIENT_7[n1][n2] * n2 * z_pow[n2 - 1];
        }
      }
      if (L == 8) {
        z_factor += Z_COEFFICIENT_8[n1][n2] * z_pow[n2];
        if (n2 > 0) {
          dz_factor += Z_COEFFICIENT_8[n1][n2] * n2 * z_pow[n2 - 1];
        }
      }
    }
    if (n1 == 0) {
      for (int d = 0; d < 3; ++d) {
        f12[d] += s[0] * (z_factor * fnp * r12[d] + fn * dz_factor * dz[d]);
      }
    } else {
      NEP_FLOAT real_part_n1 = n1 * real_part;
      NEP_FLOAT imag_part_n1 = n1 * imag_part;
      for (int d = 0; d < 3; ++d) {
        NEP_FLOAT real_part_dx = dx[d];
        NEP_FLOAT imag_part_dy = dy[d];
        complex_product(real_part_n1, imag_part_n1, real_part_dx, imag_part_dy);
        f12[d] += (s[2 * n1 - 1] * real_part_dx + s[2 * n1 - 0] * imag_part_dy) * z_factor * fn;
      }
      complex_product(r12[0], r12[1], real_part, imag_part);
      const NEP_FLOAT xy_temp = s[2 * n1 - 1] * real_part + s[2 * n1 - 0] * imag_part;
      for (int d = 0; d < 3; ++d) {
        f12[d] += xy_temp * (z_factor * fnp * r12[d] + fn * dz_factor * dz[d]);
      }
    }
  }
}

static __device__ __forceinline__ void accumulate_f12(
  const int L_max,
  const int num_L,
  const int n,
  const int n_max_angular_plus_1,
  const NEP_FLOAT d12,
  const NEP_FLOAT* r12,
  NEP_FLOAT fn,
  NEP_FLOAT fnp,
  const NEP_FLOAT* Fp,
  const NEP_FLOAT* sum_fxyz,
  NEP_FLOAT* f12)
{
  const NEP_FLOAT fn_original = fn;
  const NEP_FLOAT fnp_original = fnp;
  const NEP_FLOAT d12inv = FLOAT_LIT(1.0) / d12;
  const NEP_FLOAT r12unit[3] = {r12[0] * d12inv, r12[1] * d12inv, r12[2] * d12inv};

  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  if (num_L >= L_max + 2) {
    NEP_FLOAT s1[3] = {
      sum_fxyz[n * NUM_OF_ABC + 0], sum_fxyz[n * NUM_OF_ABC + 1], sum_fxyz[n * NUM_OF_ABC + 2]};
    get_f12_5body(d12, d12inv, fn, fnp, Fp[(L_max + 1) * n_max_angular_plus_1 + n], s1, r12, f12);
  }

  if (L_max >= 1) {
    NEP_FLOAT s1[3];
    calculate_s_one<1>(n, n_max_angular_plus_1, Fp, sum_fxyz, s1);
    accumulate_f12_one<1>(d12inv, fn_original, fnp_original, s1, r12unit, f12);
  }

  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  if (num_L >= L_max + 1) {
    NEP_FLOAT s2[5] = {
      sum_fxyz[n * NUM_OF_ABC + 3],
      sum_fxyz[n * NUM_OF_ABC + 4],
      sum_fxyz[n * NUM_OF_ABC + 5],
      sum_fxyz[n * NUM_OF_ABC + 6],
      sum_fxyz[n * NUM_OF_ABC + 7]};
    get_f12_4body(d12, d12inv, fn, fnp, Fp[L_max * n_max_angular_plus_1 + n], s2, r12, f12);
  }

  if (L_max >= 2) {
    NEP_FLOAT s2[5];
    calculate_s_one<2>(n, n_max_angular_plus_1, Fp, sum_fxyz, s2);
    accumulate_f12_one<2>(d12inv, fn_original, fnp_original, s2, r12unit, f12);
  }

  if (L_max >= 3) {
    NEP_FLOAT s3[7];
    calculate_s_one<3>(n, n_max_angular_plus_1, Fp, sum_fxyz, s3);
    accumulate_f12_one<3>(d12inv, fn_original, fnp_original, s3, r12unit, f12);
  }

  if (L_max >= 4) {
    NEP_FLOAT s4[9];
    calculate_s_one<4>(n, n_max_angular_plus_1, Fp, sum_fxyz, s4);
    accumulate_f12_one<4>(d12inv, fn_original, fnp_original, s4, r12unit, f12);
  }

  if (L_max >= 5) {
    NEP_FLOAT s5[11];
    calculate_s_one<5>(n, n_max_angular_plus_1, Fp, sum_fxyz, s5);
    accumulate_f12_one<5>(d12inv, fn_original, fnp_original, s5, r12unit, f12);
  }

  if (L_max >= 6) {
    NEP_FLOAT s6[13];
    calculate_s_one<6>(n, n_max_angular_plus_1, Fp, sum_fxyz, s6);
    accumulate_f12_one<6>(d12inv, fn_original, fnp_original, s6, r12unit, f12);
  }

  if (L_max >= 7) {
    NEP_FLOAT s7[15];
    calculate_s_one<7>(n, n_max_angular_plus_1, Fp, sum_fxyz, s7);
    accumulate_f12_one<7>(d12inv, fn_original, fnp_original, s7, r12unit, f12);
  }

  if (L_max >= 8) {
    NEP_FLOAT s8[17];
    calculate_s_one<8>(n, n_max_angular_plus_1, Fp, sum_fxyz, s8);
    accumulate_f12_one<8>(d12inv, fn_original, fnp_original, s8, r12unit, f12);
  }
}

template <int L>
static __device__ __forceinline__ void
accumulate_s_one(const NEP_FLOAT x12, const NEP_FLOAT y12, const NEP_FLOAT z12, const NEP_FLOAT fn, NEP_FLOAT* s)
{
  int s_index = L * L - 1;
  NEP_FLOAT z_pow[L + 1] = {FLOAT_LIT(1.0)};
  for (int n = 1; n <= L; ++n) {
    z_pow[n] = z12 * z_pow[n - 1];
  }
  NEP_FLOAT real_part = x12;
  NEP_FLOAT imag_part = y12;
  for (int n1 = 0; n1 <= L; ++n1) {
    int n2_start = (L + n1) % 2 == 0 ? 0 : 1;
    NEP_FLOAT z_factor = FLOAT_LIT(0.0);
    for (int n2 = n2_start; n2 <= L - n1; n2 += 2) {
      if (L == 1) {
        z_factor += Z_COEFFICIENT_1[n1][n2] * z_pow[n2];
      }
      if (L == 2) {
        z_factor += Z_COEFFICIENT_2[n1][n2] * z_pow[n2];
      }
      if (L == 3) {
        z_factor += Z_COEFFICIENT_3[n1][n2] * z_pow[n2];
      }
      if (L == 4) {
        z_factor += Z_COEFFICIENT_4[n1][n2] * z_pow[n2];
      }
      if (L == 5) {
        z_factor += Z_COEFFICIENT_5[n1][n2] * z_pow[n2];
      }
      if (L == 6) {
        z_factor += Z_COEFFICIENT_6[n1][n2] * z_pow[n2];
      }
      if (L == 7) {
        z_factor += Z_COEFFICIENT_7[n1][n2] * z_pow[n2];
      }
      if (L == 8) {
        z_factor += Z_COEFFICIENT_8[n1][n2] * z_pow[n2];
      }
    }
    z_factor *= fn;
    if (n1 == 0) {
      s[s_index++] += z_factor;
    } else {
      s[s_index++] += z_factor * real_part;
      s[s_index++] += z_factor * imag_part;
      complex_product(x12, y12, real_part, imag_part);
    }
  }
}

static __device__ __forceinline__ void accumulate_s(
  const int L_max, const NEP_FLOAT d12, NEP_FLOAT x12, NEP_FLOAT y12, NEP_FLOAT z12, const NEP_FLOAT fn, NEP_FLOAT* s)
{
  NEP_FLOAT d12inv = FLOAT_LIT(1.0) / d12;
  x12 *= d12inv;
  y12 *= d12inv;
  z12 *= d12inv;
  if (L_max >= 1) {
    accumulate_s_one<1>(x12, y12, z12, fn, s);
  }
  if (L_max >= 2) {
    accumulate_s_one<2>(x12, y12, z12, fn, s);
  }
  if (L_max >= 3) {
    accumulate_s_one<3>(x12, y12, z12, fn, s);
  }
  if (L_max >= 4) {
    accumulate_s_one<4>(x12, y12, z12, fn, s);
  }
  if (L_max >= 5) {
    accumulate_s_one<5>(x12, y12, z12, fn, s);
  }
  if (L_max >= 6) {
    accumulate_s_one<6>(x12, y12, z12, fn, s);
  }
  if (L_max >= 7) {
    accumulate_s_one<7>(x12, y12, z12, fn, s);
  }
  if (L_max >= 8) {
    accumulate_s_one<8>(x12, y12, z12, fn, s);
  }
}

template <int L>
static __device__ __forceinline__ NEP_FLOAT find_q_one(const NEP_FLOAT* s)
{
  const int start_index = L * L - 1;
  const int num_terms = 2 * L + 1;
  NEP_FLOAT q = FLOAT_LIT(0.0);
  for (int k = 1; k < num_terms; ++k) {
    q += C3B[start_index + k] * s[start_index + k] * s[start_index + k];
  }
  q *= FLOAT_LIT(2.0);
  q += C3B[start_index] * s[start_index] * s[start_index];
  return q;
}

static __device__ __forceinline__ void find_q(
  const int L_max,
  const int num_L,
  const int n_max_angular_plus_1,
  const int n,
  const NEP_FLOAT* s,
  NEP_FLOAT* q,
  const int q_stride)
{
  if (L_max >= 1) {
    q[(0 * n_max_angular_plus_1 + n) * q_stride] = find_q_one<1>(s);
  }
  if (L_max >= 2) {
    q[(1 * n_max_angular_plus_1 + n) * q_stride] = find_q_one<2>(s);
  }
  if (L_max >= 3) {
    q[(2 * n_max_angular_plus_1 + n) * q_stride] = find_q_one<3>(s);
  }
  if (L_max >= 4) {
    q[(3 * n_max_angular_plus_1 + n) * q_stride] = find_q_one<4>(s);
  }
  if (L_max >= 5) {
    q[(4 * n_max_angular_plus_1 + n) * q_stride] = find_q_one<5>(s);
  }
  if (L_max >= 6) {
    q[(5 * n_max_angular_plus_1 + n) * q_stride] = find_q_one<6>(s);
  }
  if (L_max >= 7) {
    q[(6 * n_max_angular_plus_1 + n) * q_stride] = find_q_one<7>(s);
  }
  if (L_max >= 8) {
    q[(7 * n_max_angular_plus_1 + n) * q_stride] = find_q_one<8>(s);
  }
  if (num_L >= L_max + 1) {
    q[(L_max * n_max_angular_plus_1 + n) * q_stride] =
      C4B[0] * s[3] * s[3] * s[3] + C4B[1] * s[3] * (s[4] * s[4] + s[5] * s[5]) +
      C4B[2] * s[3] * (s[6] * s[6] + s[7] * s[7]) + C4B[3] * s[6] * (s[5] * s[5] - s[4] * s[4]) +
      C4B[4] * s[4] * s[5] * s[7];
  }
  if (num_L >= L_max + 2) {
    NEP_FLOAT s0_sq = s[0] * s[0];
    NEP_FLOAT s1_sq_plus_s2_sq = s[1] * s[1] + s[2] * s[2];
    q[((L_max + 1) * n_max_angular_plus_1 + n) * q_stride] =
      C5B[0] * s0_sq * s0_sq + C5B[1] * s0_sq * s1_sq_plus_s2_sq +
      C5B[2] * s1_sq_plus_s2_sq * s1_sq_plus_s2_sq;
  }
}
