/*
    Copyright 2017 Zheyong Fan, Ville Vierimaa, Mikko Ervasti, and Ari Harju
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
*/

#pragma once
#include "nep_utilities.cuh"


static __device__ __forceinline__ void
scd_accumulate_blm_rij(const double d12, 
                    const double x, 
                    const double y, 
                    const double z, 
                    double* blm,
                    double* rij_blm,
                    double* dblm_x,
                    double* dblm_y,
                    double* dblm_z,
                    double* dblm_r)
{
  double d12inv = 1.0 / d12;
  double x12 = x * d12inv;
  double y12 = y * d12inv;
  double z12 = z * d12inv;
  double x2 = x * x;
  double y2 = y * y;
  double z2 = z * z;
  double xy = x * y;
  double xz = x * z;
  double yz = y * z;
  double r2 = d12 * d12;
  double x2my2 = x2 - y2;
  double xyz = x * yz;
  double x3 = x * x2;
  double y3 = y * y2;
  double z3 = z * z2;
  double r3 = d12 * r2;
  double x12sq = x12 * x12;
  double y12sq = y12 * y12;
  double z12sq = z12 * z12;
  double x12sq_minus_y12sq = x12sq - y12sq;
  // L = 1 s0
  blm[0]     = z;                                             // Y10 blm
  dblm_r[0]  = 0.0;                                           // Y10 blm/dr
  dblm_x[0]  = 0.0;                                           // Y10 blm/dx
  dblm_y[0]  = 0.0;                                           // Y10 blm/dy
  dblm_z[0]  = 1.0;                                           // Y10 blm/dz
  rij_blm[0] = z12;                                           // Y10 sij_blm
  // s1
  blm[1]     = x;                                             // Y11_real
  dblm_r[1]  = 0.0;                                           
  dblm_x[1]  = 1.0;                                           
  dblm_y[1]  = 0.0;                                           
  dblm_z[1]  = 0.0;                                           
  rij_blm[1] = x12;                                           
  // s2
  blm[2]     = y;                                             // Y11_imag
  dblm_r[2]  = 0.0;                                           
  dblm_x[2]  = 0.0;                                           
  dblm_y[2]  = 1.0;                                           
  dblm_z[2]  = 0.0;                                           
  rij_blm[2] = y12;                                           
  
  // L = 2 s0
  blm[3]     = 3.0 * z2- d12 * d12;                           // Y20
  dblm_r[3]  = -2.0 * d12;                                           
  dblm_x[3]  = 0.0;                                           
  dblm_y[3]  = 0.0;                                           
  dblm_z[3]  = 6.0 * z;                                           
  rij_blm[3] = 3.0 * z12sq - 1.0;                                           
  //s1
  blm[4]    = xz;                                             // Y21_real
  dblm_r[4] = 0.0;
  dblm_x[4] = z;
  dblm_y[4] = 0.0;
  dblm_z[4] = x;
  rij_blm[4]= x12 * z12;
  //s2
  blm[5]    = yz;                                             // Y21_imag
  dblm_r[5] = 0.0;
  dblm_x[5] = 0.0;
  dblm_y[5] = z;
  dblm_z[5] = y;
  rij_blm[5]= y12 * z12;
  //s3
  blm[6]    = x2 - y2;                                        // Y22_real
  dblm_r[6] = 0.0;
  dblm_x[6] = 2.0 * x;
  dblm_y[6] = -2.0 * y;
  dblm_z[6] = 0.0;
  rij_blm[6]= x12sq_minus_y12sq;
  //s4
  blm[7]     = 2.0 * xy;                                      // Y22_imag
  dblm_r[7]  = 0.0;
  dblm_x[7]  = 2.0 * y;
  dblm_y[7]  = 2.0 * x;
  dblm_z[7]  = 0.0;
  rij_blm[7] = 2.0 * x12 * y12;

  // L = 3 s0
  blm[8]     = (5.0 * z2 - 3.0 * r2) * z;                     // Y30
  dblm_r[8]  = -6.0 * z * d12;
  dblm_x[8]  = 0.0;
  dblm_y[8]  = 0.0;
  dblm_z[8]  = 15 * z2 - 3 * r2;
  rij_blm[8] = (5.0 * z12sq - 3.0) * z12;
  //s1
  blm[9]     = (5.0 * z2 - r2) * x;                          // Y31_real
  dblm_r[9]  = -2.0 * x * d12;
  dblm_x[9]  = 5.0 * z2 - r2;
  dblm_y[9]  = 0.0;
  dblm_z[9]  = 10.0 * xz;
  rij_blm[9] = (5.0 * z12sq - 1.0) * x12;
  //s2
  blm[10]    = (5.0 * z2 - r2) * y;                          // Y31_imag
  dblm_r[10] = -2.0 * y * d12;
  dblm_x[10] = 0.0;
  dblm_y[10] = 5.0 * z2 - r2;
  dblm_z[10] = 10.0 * yz;
  rij_blm[10]= (5.0 * z12sq - 1.0) * y12;
  //s3
  blm[11]    = (x2 - y2) * z;                                // Y32_real
  dblm_r[11] = 0.0;
  dblm_x[11] = 2.0 * xz;
  dblm_y[11] = -2.0 * yz;
  dblm_z[11] = x2 - y2;
  rij_blm[11]= x12sq_minus_y12sq * z12;
  //s4
  blm[12]     = 2.0 * xyz;                                // Y32_imag
  dblm_r[12]  = 0.0;
  dblm_x[12]  = 2.0 * yz;
  dblm_y[12]  = 2.0 * xz;
  dblm_z[12]  = 2.0 * xy;
  rij_blm[12] = 2.0 * x12 * y12 * z12;
  //s5
  blm[13]    = (x2 - 3.0 * y2) * x;                           // Y33_real
  dblm_r[13] = 0.0;
  dblm_x[13] = 3.0 * (x2 - y2);
  dblm_y[13] = -6.0 * xy;
  dblm_z[13] = 0.0;
  rij_blm[13]= (x12 * x12 - 3.0 * y12 * y12) * x12;
  //s6
  blm[14]    = (3.0 * x2 - y2) * y;                           // Y33_imag
  dblm_r[14] = 0.0;
  dblm_x[14] = 6.0 * xy;
  dblm_y[14] = 3.0 * (x2 - y2);
  dblm_z[14] = 0.0;
  rij_blm[14]= (3.0 * x12 * x12 - y12 * y12) * y12;

  //L = 4 s0 检查下标 核对公式
  blm[15]    = (35.0 * z2 - 30.0 * r2) * z2 + 3.0 * r2 * r2;   // Y40
  dblm_r[15] = (-60.0) * z2 * d12 + 12.0 * r3;
  dblm_x[15] = 0.0;
  dblm_y[15] = 0.0;
  dblm_z[15] = 140.0 * z3 - 60.0 * z * r2;
  rij_blm[15]= ((35.0 * z12sq - 30.0) * z12sq + 3.0);
  //s1
  blm[16]    = (7.0 * z2 - 3.0 * r2) * xz;                    // Y41_real
  dblm_r[16] = -6.0 * xz * d12;
  dblm_x[16] = 7.0 * z3 - 3.0 * z * r2;
  dblm_y[16] = 0.0;
  dblm_z[16]  = 21.0 * x * z2 - 3.0 * x * r2;
  rij_blm[16] = (7.0 * z12sq - 3.0) * x12 * z12;
  //s2
  blm[17]    = (7.0 * z2 - 3.0 * r2) * yz;                    // Y41_iamg
  dblm_r[17] = -6.0 * yz * d12;
  dblm_x[17] = 0.0;
  dblm_y[17] = 7.0 * z3 - 3.0 * z * r2;
  dblm_z[17] = 21.0 * y * z2 - 3.0 * y * r2;
  rij_blm[17]= (7.0 * z12sq - 3.0) * y12 * z12;
  //s3
  blm[18]    = (7.0 * z2 - r2) * x2my2;                       // Y42_real
  dblm_r[18] = 2.0 * d12 * (y2 - x2);
  dblm_x[18] = 14.0 * x * z2 - 2.0 * x * r2;
  dblm_y[18] = 2.0 * y *(r2 - 7.0 * z2);
  dblm_z[18] = 14.0 * x2 * z - 14.0 * y2 * z;
  rij_blm[18]= (7.0 * z12sq - 1.0) * x12sq_minus_y12sq;
  //s4
  blm[19]    = (7.0 * z2 - r2) * 2.0 * xy;                    // Y42_imag
  dblm_r[19] = -4.0 * xy * d12;
  dblm_x[19] = 2.0 * y * (7.0 * z2 - r2);
  dblm_y[19] = 2.0 * x * (7.0 * z2 - r2);
  dblm_z[19] = 28.0 * xyz;
  rij_blm[19]= (7.0 * z12sq - 1.0) * x12 * y12 * 2.0;
  //s5
  blm[20]    = (x2 - 3.0 * y2) * xz;                          // Y43_real
  dblm_r[20] = 0.0;
  dblm_x[20] = 3.0 * z * (x2 - y2);
  dblm_y[20] = -6.0 * xyz;
  dblm_z[20] = x3 - 3.0 * x * y2;
  rij_blm[20]= (x12sq - 3.0 * y12sq) * x12 * z12;
  //s6
  blm[21]    = (3.0 * x2 - y2) * yz;                         // Y43_imag
  dblm_r[21] = 0.0;
  dblm_x[21] = 6.0 * xyz;
  dblm_y[21] = 3.0 * x2 * z - 3.0 * y2 * z;
  dblm_z[21] = 3.0 * y * x2 - y3;
  rij_blm[21]= (3.0 * x12sq - y12sq) * y12 * z12;
  //s7
  blm[22]    = x2my2 * x2my2 - 4.0 * x2 * y2;                // Y44_real
  dblm_r[22] = 0.0;
  dblm_x[22] = 4.0 * x3 - 12.0 * x * y2;
  dblm_y[22] = 4.0 * y3 - 12.0 * x2 * y;
  dblm_z[22] = 0.0;
  rij_blm[22]= (x12sq_minus_y12sq * x12sq_minus_y12sq - 4.0 * x12sq * y12sq);
  //s8
  blm[23]    = 4.0 * x2my2 * xy;                            // Y44_imag
  dblm_r[23] = 0.0;
  dblm_x[23] = 12.0 * x2 * y - 4.0 * y3;
  dblm_y[23] = 4.0 * x3 - 12.0 * x * y2;
  dblm_z[23] = 0.0;
  rij_blm[23]= (4.0 * x12 * y12 * x12sq_minus_y12sq);
}


static __device__ __forceinline__ void scd_get_f12_1(
  const double *fn12,
  const double *fnp12,
  const double *blm,
  const double *rij_blm,
  const double *dblm_x,
  const double *dblm_y,
  const double *dblm_z,
  const double *dblm_r,
  const double *scd_r12,
  const double *dsnlm_dc,
  const double *s,
  const double *r12,
  const double d12inv,
  const double rij_Lsq, 
  const double rij_L2sq, 
  const double fn,
  const double fnp,
  const double Fp,
  const int n_base_angular, 
  const int dsnlm_start_idx,
  const int type_j,
  const int ntypes,
  const int n1, 
  const int n2,
  double *f12k)
{ //l = 1
  int k_start_id = type_j * n_base_angular * 4;
  int k_idx = 0;
  double dfk = 0.0; // dgn(rij)/dc
  int dsnlm_idx = dsnlm_start_idx + type_j * n_base_angular * NUM_OF_ABC;
  for(int k=0; k < n_base_angular; k++) {
    int dsnlm_i = dsnlm_idx + k * NUM_OF_ABC;
    double tmpr = 0.0, tmpx = 0.0, tmpy = 0.0, tmpz = 0.0;
    double rr0 = 0.0, rr1 = 0.0, rr2 = 0.0;
    double rrr0 = 0.0, rrr1 = 0.0, rrr2=0.0;
    k_idx = k_start_id + k * 4;
    // 左边项 rij
    rr0 =       C3B[0] * dsnlm_dc[dsnlm_i]   * fnp * blm[0]; 
    rr1 = 2.0 * C3B[1] * dsnlm_dc[dsnlm_i+1] * fnp * blm[1];
    rr2 = 2.0 * C3B[2] * dsnlm_dc[dsnlm_i+2] * fnp * blm[2];
    // 右边项
    dfk = fnp12[k] * rij_Lsq - fn12[k] * rij_L2sq;
    rrr0 =       s[0] * dfk * blm[0]; 
    rrr1 = 2.0 * s[1] * dfk * blm[1]; 
    rrr2 = 2.0 * s[2] * dfk * blm[2]; // 后项 dblm/drij L=1 时 为0
    tmpr = rr0 + rr1 + rr2 + rrr0 + rrr1 + rrr2;
    f12k[k_idx + 0] += 2.0 * Fp * scd_r12[0] * tmpr;// 可以考虑移出去

    // xij
    tmpx += 2.0 * C3B[1] * dsnlm_dc[dsnlm_i+1] * fn; // dblm/dxij b10 b12 为0
    tmpx += 2.0 * s[1] * fn12[k] * rij_Lsq; // dblm/dxij b10 b12 为0
    f12k[k_idx + 1] += 2.0 * Fp * scd_r12[1] * tmpx;

    // yij
    tmpy += 2.0 * C3B[2] * dsnlm_dc[dsnlm_i+2] * fn;
    tmpy += 2.0 * s[2] * fn12[k] * rij_Lsq;
    f12k[k_idx + 2] += 2.0 * Fp * scd_r12[2] * tmpy;

    // zij
    tmpz += C3B[0] * dsnlm_dc[dsnlm_i] * fn;
    tmpz += s[0] * fn12[k] * rij_Lsq;
    f12k[k_idx + 3] += 2.0 * Fp * scd_r12[3] * tmpz;

    // if (n1==0 and n2==0 and k == 0){
    //   printf("\tscd L=1 n1=%d n2=%d k=%d s0=%lf s1=%lf s2=%lf fnp=%lf fn=%lf Fp=%lf fn12[%d]=%lf fnp12[%d]=%lf rij_Lsq=%lf rij_L2sq=%lf b10=%lf b11=%lf b12=%lf dqr=%lf dqx=%lf dqy=%lf dqz=%lf r0=%f r1=%f r2=%f rr1=%f rr2=%f rr3=%f\n", 
    //           n1, n2, k, s[0], s[1]*2.0, s[2]*2.0, fnp, fn, Fp, k, fn12[k], k, fnp12[k], rij_Lsq, rij_L2sq, blm[0], blm[1], blm[2], tmpr, tmpx, tmpy, tmpz, rr0, rr1, rr2, rrr0, rrr1, rrr2);
    // }
  }
}

static __device__ __forceinline__ void scd_get_f12_2(
  const double *fn12,
  const double *fnp12,
  const double *blm,
  const double *rij_blm,
  const double *dblm_x,
  const double *dblm_y,
  const double *dblm_z,
  const double *dblm_r,
  const double *scd_r12,
  const double *dsnlm_dc,
  const double* s,
  const double* r12,
  const double d12inv,
  const double rij_Lsq, 
  const double rij_L2sq, 
  const double fn,
  const double fnp,
  const double Fp,
  const int n_base_angular, 
  const int dsnlm_start_idx,
  const int type_j,
  const int ntypes,
  const int n1, 
  const int n2,
  double *f12k
  )
{
  // L = 2 c3b 3 4 5 6 7
  int k_start_id = type_j * n_base_angular * 4;
  int k_idx = 0;
  int dsnlm_idx = dsnlm_start_idx + type_j * n_base_angular * NUM_OF_ABC;
  for(int k=0; k < n_base_angular; k++) {
    int dsnlm_i = dsnlm_idx + k * NUM_OF_ABC;
    double tmpr = 0.0, tmpx = 0.0, tmpy = 0.0, tmpz = 0.0;
    k_idx = k_start_id + k * 4;
    // 左边项 rij
    tmpr +=  C3B[3] * dsnlm_dc[dsnlm_i+3] * (fnp * blm[3] + fn * dblm_r[3]) + 
                  2.0 * C3B[4] * dsnlm_dc[dsnlm_i+4] * fnp * blm[4] + 
                  2.0 * C3B[5] * dsnlm_dc[dsnlm_i+5] * fnp * blm[5] + 
                  2.0 * C3B[6] * dsnlm_dc[dsnlm_i+6] * fnp * blm[6] +
                  2.0 * C3B[7] * dsnlm_dc[dsnlm_i+7] * fnp * blm[7]; 
    // 右边项
    tmpr += s[0] * ((fnp12[k] * rij_Lsq - 2.0 * fn12[k] * rij_L2sq) * blm[3] + fn12[k] * rij_Lsq * dblm_r[3]) +  
            2.0 * s[1] * (fnp12[k] * rij_Lsq - 2.0 * fn12[k] * rij_L2sq) * blm[4] +
            2.0 * s[2] * (fnp12[k] * rij_Lsq - 2.0 * fn12[k] * rij_L2sq) * blm[5] +
            2.0 * s[3] * (fnp12[k] * rij_Lsq - 2.0 * fn12[k] * rij_L2sq) * blm[6] +
            2.0 * s[4] * (fnp12[k] * rij_Lsq - 2.0 * fn12[k] * rij_L2sq) * blm[7]; 
    f12k[k_idx + 0] += 2.0 * Fp * scd_r12[0] * tmpr;// 可以考虑移出去

    // 左边项 xij
    tmpx += 
                  2.0 * C3B[4] * dsnlm_dc[dsnlm_i+4] * fn * dblm_x[4] + 
                  2.0 * C3B[6] * dsnlm_dc[dsnlm_i+6] * fn * dblm_x[6] +
                  2.0 * C3B[7] * dsnlm_dc[dsnlm_i+7] * fn * dblm_x[7];     
    // 右边项
    tmpx += 
            2.0 * s[1] * fn12[k] * rij_Lsq * dblm_x[4] +
            2.0 * s[3] * fn12[k] * rij_Lsq * dblm_x[6] +
            2.0 * s[4] * fn12[k] * rij_Lsq * dblm_x[7];
    f12k[k_idx + 1] += 2.0 * Fp * scd_r12[1] * tmpx;

    // 左边项 yij
    tmpy += 
                  2.0 * C3B[5] * dsnlm_dc[dsnlm_i+5] * fn * dblm_y[5] + 
                  2.0 * C3B[6] * dsnlm_dc[dsnlm_i+6] * fn * dblm_y[6] +
                  2.0 * C3B[7] * dsnlm_dc[dsnlm_i+7] * fn * dblm_y[7];     
    // 右边项
    tmpy += 
            2.0 * s[2] * fn12[k] * rij_Lsq * dblm_y[5] + 
            2.0 * s[3] * fn12[k] * rij_Lsq * dblm_y[6] +
            2.0 * s[4] * fn12[k] * rij_Lsq * dblm_y[7];    
    f12k[k_idx + 2] += 2.0 * Fp * scd_r12[2] * tmpy;
    
    // 左边项 zij
    tmpz +=  C3B[3] * dsnlm_dc[dsnlm_i+3] * fn * dblm_z[3] + 
                  2.0 * C3B[4] * dsnlm_dc[dsnlm_i+4] * fn * dblm_z[4] + 
                  2.0 * C3B[5] * dsnlm_dc[dsnlm_i+5] * fn * dblm_z[5];     
    // 右边项
    tmpz +=  s[0] * fn12[k] * rij_Lsq * dblm_z[3] +  
                  2.0 * s[1] * fn12[k] * rij_Lsq * dblm_z[4] +
                  2.0 * s[2] * fn12[k] * rij_Lsq * dblm_z[5];
    f12k[k_idx + 3] += 2.0 * Fp * scd_r12[3] * tmpz;

    // if (n1==0 and n2==0 and k == 0){
    //   printf("\tscd L=2 n1=%d n2=%d s0=%lf s1=%lf s2=%lf s3=%lf s4=%lf fnp=%lf fn=%lf Fp=%lf b20=%lf b21=%lf b22=%lf b23=%lf b24=%lf dqr=%lf dqx=%lf dqy=%lf dqz=%lf\n", 
    //           n1, n2, s[0], s[1]*2.0, s[2]*2.0, s[3]*2.0, s[4]*2.0, fnp, fn, Fp, blm[3], blm[4], blm[5], blm[6], blm[7], tmpr, tmpx, tmpy, tmpz);
    // }
  }
}

static __device__ __forceinline__ void scd_get_f12_4body(
  const double *fn12,
  const double *fnp12,
  const double *blm,
  const double *rij_blm,
  const double *dblm_x,
  const double *dblm_y,
  const double *dblm_z,
  const double *dblm_r,
  const double *scd_r12,
  const double *dsnlm_dc,
  const double* s,
  const double* r12,
  const double d12inv,
  const double rij_Lsq, 
  const double rij_L2sq, 
  const double fn,
  const double fnp,
  const double Fp,
  const int n_base_angular,
  const int dsnlm_start_idx, 
  const int type_j,
  const int ntypes,
  const int n1, 
  const int n2,
  double *f12k)
{
  // L = 2 c3b 3 4 5 6 7

  int k_start_id = type_j * n_base_angular * 4;
  int dsnlm_idx = dsnlm_start_idx + type_j * n_base_angular * NUM_OF_ABC;
  int k_idx = 0;

  double dnlm_drij[5] = {0.0};
  dnlm_drij[0] = fnp * blm[3] + fn * dblm_r[3];
  dnlm_drij[1] = fnp * blm[4];
  dnlm_drij[2] = fnp * blm[5];
  dnlm_drij[3] = fnp * blm[6];
  dnlm_drij[4] = fnp * blm[7];
  double dnlm_dxij[5] = {0.0};
  dnlm_dxij[0] = 0.0;
  dnlm_dxij[1] = fn * dblm_x[4];
  dnlm_dxij[2] = 0.0;
  dnlm_dxij[3] = fn * dblm_x[6];
  dnlm_dxij[4] = fn * dblm_x[7];
  double dnlm_dyij[5] = {0.0};
  dnlm_dyij[0] = 0.0;
  dnlm_dyij[1] = 0.0;
  dnlm_dyij[2] = fn * dblm_y[5];
  dnlm_dyij[3] = fn * dblm_y[6];
  dnlm_dyij[4] = fn * dblm_y[7];
  double dnlm_dzij[5] = {0.0};
  dnlm_dzij[0] = fn * dblm_z[3];
  dnlm_dzij[1] = fn * dblm_z[4];
  dnlm_dzij[2] = fn * dblm_z[5];
  dnlm_dzij[3] = 0.0;
  dnlm_dzij[4] = 0.0;
  double dnlm_dc[5] = {0.0};
  double dnlm_drij_dc[5] = {0.0};
  double dnlm_dxij_dc[5] = {0.0};
  double dnlm_dyij_dc[5] = {0.0};
  double dnlm_dzij_dc[5] = {0.0};

  double s2[5] = {0.0};
  s2[0] = s[0] * s[0];
  s2[1] = s[1] * s[1];
  s2[2] = s[2] * s[2];
  s2[3] = s[3] * s[3];
  s2[4] = s[4] * s[4];
  
  for(int k=0; k < n_base_angular; k++) {
    int dsnlm_i = dsnlm_idx + k * NUM_OF_ABC;
    double tmpr = 0.0, tmpx = 0.0, tmpy = 0.0, tmpz = 0.0;
    k_idx = k_start_id + k * 4;
    dnlm_dc[0] = dsnlm_dc[dsnlm_i + 3];
    dnlm_dc[1] = dsnlm_dc[dsnlm_i + 4];
    dnlm_dc[2] = dsnlm_dc[dsnlm_i + 5];
    dnlm_dc[3] = dsnlm_dc[dsnlm_i + 6];
    dnlm_dc[4] = dsnlm_dc[dsnlm_i + 7];
    
    dnlm_drij_dc[0] = (fnp12[k] * rij_Lsq - 2.0 * fn12[k] * rij_L2sq) * blm[3] + fn12[k] * rij_Lsq * dblm_r[3]; 
    dnlm_drij_dc[1] = (fnp12[k] * rij_Lsq - 2.0 * fn12[k] * rij_L2sq) * blm[4];
    dnlm_drij_dc[2] = (fnp12[k] * rij_Lsq - 2.0 * fn12[k] * rij_L2sq) * blm[5];
    dnlm_drij_dc[3] = (fnp12[k] * rij_Lsq - 2.0 * fn12[k] * rij_L2sq) * blm[6];
    dnlm_drij_dc[4] = (fnp12[k] * rij_Lsq - 2.0 * fn12[k] * rij_L2sq) * blm[7];
    
    dnlm_dxij_dc[0] = 0.0;
    dnlm_dxij_dc[1] = fn12[k] * rij_Lsq * dblm_x[4];
    dnlm_dxij_dc[2] = 0.0;
    dnlm_dxij_dc[3] = fn12[k] * rij_Lsq * dblm_x[6];
    dnlm_dxij_dc[4] = fn12[k] * rij_Lsq * dblm_x[7];

    dnlm_dyij_dc[0] = 0.0;
    dnlm_dyij_dc[1] = 0.0;
    dnlm_dyij_dc[2] = fn12[k] * rij_Lsq * dblm_y[5];
    dnlm_dyij_dc[3] = fn12[k] * rij_Lsq * dblm_y[6];
    dnlm_dyij_dc[4] = fn12[k] * rij_Lsq * dblm_y[7];

    dnlm_dzij_dc[0] = fn12[k] * rij_Lsq * dblm_z[3];
    dnlm_dzij_dc[1] = fn12[k] * rij_Lsq * dblm_z[4];
    dnlm_dzij_dc[2] = fn12[k] * rij_Lsq * dblm_z[5];
    dnlm_dzij_dc[3] = 0.0;
    dnlm_dzij_dc[4] = 0.0;
    // drij
    // d0
    tmpr += 3.0 * C4B[0] * (
      2.0 * s[0] * dnlm_dc[0] * dnlm_drij[0] + s2[0] * dnlm_drij_dc[0]);
    // d1
    tmpr += C4B[1] * (
      dnlm_drij_dc[0] * (s2[1] + s2[2]) + dnlm_drij[0] * 2.0 * (s[1] * dnlm_dc[1] + s[2] * dnlm_dc[2])
    );
    tmpr += 2.0 * C4B[1] * (
      dnlm_dc[0] * (s[1] * dnlm_drij[1] + s[2] * dnlm_drij[2]) + 
      s[0] * (dnlm_dc[1] * dnlm_drij[1] + s[1] * dnlm_drij_dc[1] + 
                dnlm_dc[2] * dnlm_drij[2] + s[2] * dnlm_drij_dc[2])
    );
    // d2
    tmpr += C4B[2] * (
      dnlm_drij_dc[0] * (s2[3] + s2[4]) + dnlm_drij[0] * 2.0 * (s[3] * dnlm_dc[3] + s[4] * dnlm_dc[4]));
    tmpr += 2.0 * C4B[2] * (
      dnlm_dc[0] * (s[3] * dnlm_drij[3] + s[4] * dnlm_drij[4]) + 
      s[0] * (dnlm_dc[3] * dnlm_drij[3] + s[3] * dnlm_drij_dc[3] + 
                dnlm_dc[4] * dnlm_drij[4] + s[4] * dnlm_drij_dc[4])  
    );
    // d3
    tmpr += C4B[3] * (
      dnlm_drij_dc[3] * (s2[2] - s2[1]) + dnlm_drij[3] * 2.0 * (s[2] * dnlm_dc[2] - s[1] * dnlm_dc[1]));
    tmpr += 2.0 * C4B[3] * (
      dnlm_dc[3] * (s[2] * dnlm_drij[2] - s[1] * dnlm_drij[1]) +
        s[3] * (dnlm_dc[2] * dnlm_drij[2] + s[2] * dnlm_drij_dc[2] -
                dnlm_dc[1] * dnlm_drij[1] - s[1] * dnlm_drij_dc[1])
    );
    // d4
    tmpr += C4B[4] * (
      dnlm_drij_dc[1] * s[2] * s[4] + dnlm_drij[1] * dnlm_dc[2] * s[4] + dnlm_drij[1] * s[2] * dnlm_dc[4] +
      dnlm_dc[1] * dnlm_drij[2] * s[4] + s[1] * dnlm_drij_dc[2] * s[4] + s[1] * dnlm_drij[2] * dnlm_dc[4] +
      dnlm_dc[1] * s[2] * dnlm_drij[4] + s[1] * dnlm_dc[2] * dnlm_drij[4] + s[1] * s[2] * dnlm_drij_dc[4]
    );
    f12k[k_idx + 0] += Fp * scd_r12[0] * tmpr;
    // dxij
    // d0
    tmpx += 3.0 * C4B[0] * (
      2.0 * s[0] * dnlm_dc[0] * dnlm_dxij[0] + s2[0] * dnlm_dxij_dc[0]);
    // d1
    tmpx += C4B[1] * (
      dnlm_dxij_dc[0] * (s2[1] + s2[2]) + dnlm_dxij[0] * 2.0 * (s[1] * dnlm_dc[1] + s[2] * dnlm_dc[2])
    );
    tmpx += 2.0 * C4B[1] * (
      dnlm_dc[0] * (s[1] * dnlm_dxij[1] + s[2] * dnlm_dxij[2]) + 
      s[0] * (dnlm_dc[1] * dnlm_dxij[1] + s[1] * dnlm_dxij_dc[1] + 
                dnlm_dc[2] * dnlm_dxij[2] + s[2] * dnlm_dxij_dc[2])
    );
    // d2
    tmpx += C4B[2] * (
      dnlm_dxij_dc[0] * (s2[3] + s2[4]) + dnlm_dxij[0] * 2.0 * (s[3] * dnlm_dc[3] + s[4] * dnlm_dc[4]));
    tmpx += 2.0 * C4B[2] * (
      dnlm_dc[0] * (s[3] * dnlm_dxij[3] + s[4] * dnlm_dxij[4]) + 
      s[0] * (dnlm_dc[3] * dnlm_dxij[3] + s[3] * dnlm_dxij_dc[3] + 
                dnlm_dc[4] * dnlm_dxij[4] + s[4] * dnlm_dxij_dc[4])  
    );
    // d3
    tmpx += C4B[3] * (
      dnlm_dxij_dc[3] * (s2[2] - s2[1]) + dnlm_dxij[3] * 2.0 * (s[2] * dnlm_dc[2] - s[1] * dnlm_dc[1]));
    tmpx += 2.0 * C4B[3] * (
      dnlm_dc[3] * (s[2] * dnlm_dxij[2] - s[1] * dnlm_dxij[1]) +
        s[3] * (dnlm_dc[2] * dnlm_dxij[2] + s[2] * dnlm_dxij_dc[2] -
                dnlm_dc[1] * dnlm_dxij[1] - s[1] * dnlm_dxij_dc[1])
    );
    // d4
    tmpx += C4B[4] * (
      dnlm_dxij_dc[1] * s[2] * s[4] + dnlm_dxij[1] * dnlm_dc[2] * s[4] + dnlm_dxij[1] * s[2] * dnlm_dc[4] +
      dnlm_dc[1] * dnlm_dxij[2] * s[4] + s[1] * dnlm_dxij_dc[2] * s[4] + s[1] * dnlm_dxij[2] * dnlm_dc[4] +
      dnlm_dc[1] * s[2] * dnlm_dxij[4] + s[1] * dnlm_dc[2] * dnlm_dxij[4] + s[1] * s[2] * dnlm_dxij_dc[4]
    );    
    f12k[k_idx + 1] += Fp * scd_r12[1] * tmpx;

    // dyij
    // d0
    tmpy += 3.0 * C4B[0] * (
      2.0 * s[0] * dnlm_dc[0] * dnlm_dyij[0] + s2[0] * dnlm_dyij_dc[0]);
    // d1
    tmpy += C4B[1] * (
      dnlm_dyij_dc[0] * (s2[1] + s2[2]) + dnlm_dyij[0] * 2.0 * (s[1] * dnlm_dc[1] + s[2] * dnlm_dc[2])
    );
    tmpy += 2.0 * C4B[1] * (
      dnlm_dc[0] * (s[1] * dnlm_dyij[1] + s[2] * dnlm_dyij[2]) + 
      s[0] * (dnlm_dc[1] * dnlm_dyij[1] + s[1] * dnlm_dyij_dc[1] + 
                dnlm_dc[2] * dnlm_dyij[2] + s[2] * dnlm_dyij_dc[2])
    );
    // d2
    tmpy += C4B[2] * (
      dnlm_dyij_dc[0] * (s2[3] + s2[4]) + dnlm_dyij[0] * 2.0 * (s[3] * dnlm_dc[3] + s[4] * dnlm_dc[4]));
    tmpy += 2.0 * C4B[2] * (
      dnlm_dc[0] * (s[3] * dnlm_dyij[3] + s[4] * dnlm_dyij[4]) + 
      s[0] * (dnlm_dc[3] * dnlm_dyij[3] + s[3] * dnlm_dyij_dc[3] + 
                dnlm_dc[4] * dnlm_dyij[4] + s[4] * dnlm_dyij_dc[4])  
    );
    // d3
    tmpy += C4B[3] * (
      dnlm_dyij_dc[3] * (s2[2] - s2[1]) + dnlm_dyij[3] * 2.0 * (s[2] * dnlm_dc[2] - s[1] * dnlm_dc[1]));
    tmpy += 2.0 * C4B[3] * (
      dnlm_dc[3] * (s[2] * dnlm_dyij[2] - s[1] * dnlm_dyij[1]) +
        s[3] * (dnlm_dc[2] * dnlm_dyij[2] + s[2] * dnlm_dyij_dc[2] -
                dnlm_dc[1] * dnlm_dyij[1] - s[1] * dnlm_dyij_dc[1])
    );
    // d4
    tmpy += C4B[4] * (
      dnlm_dyij_dc[1] * s[2] * s[4] + dnlm_dyij[1] * dnlm_dc[2] * s[4] + dnlm_dyij[1] * s[2] * dnlm_dc[4] +
      dnlm_dc[1] * dnlm_dyij[2] * s[4] + s[1] * dnlm_dyij_dc[2] * s[4] + s[1] * dnlm_dyij[2] * dnlm_dc[4] +
      dnlm_dc[1] * s[2] * dnlm_dyij[4] + s[1] * dnlm_dc[2] * dnlm_dyij[4] + s[1] * s[2] * dnlm_dyij_dc[4]
    );    
    f12k[k_idx + 2] += Fp * scd_r12[2] * tmpy;

    // dzij
    // d0
    tmpz += 3.0 * C4B[0] * (
      2.0 * s[0] * dnlm_dc[0] * dnlm_dzij[0] + s2[0] * dnlm_dzij_dc[0]);
    // d1
    tmpz += C4B[1] * (
      dnlm_dzij_dc[0] * (s2[1] + s2[2]) + dnlm_dzij[0] * 2.0 * (s[1] * dnlm_dc[1] + s[2] * dnlm_dc[2])
    );
    tmpz += 2.0 * C4B[1] * (
      dnlm_dc[0] * (s[1] * dnlm_dzij[1] + s[2] * dnlm_dzij[2]) + 
      s[0] * (dnlm_dc[1] * dnlm_dzij[1] + s[1] * dnlm_dzij_dc[1] + 
                dnlm_dc[2] * dnlm_dzij[2] + s[2] * dnlm_dzij_dc[2])
    );
    // d2
    tmpz += C4B[2] * (
      dnlm_dzij_dc[0] * (s2[3] + s2[4]) + dnlm_dzij[0] * 2.0 * (s[3] * dnlm_dc[3] + s[4] * dnlm_dc[4]));
    tmpz += 2.0 * C4B[2] * (
      dnlm_dc[0] * (s[3] * dnlm_dzij[3] + s[4] * dnlm_dzij[4]) + 
      s[0] * (dnlm_dc[3] * dnlm_dzij[3] + s[3] * dnlm_dzij_dc[3] + 
                dnlm_dc[4] * dnlm_dzij[4] + s[4] * dnlm_dzij_dc[4])  
    );
    // d3
    tmpz += C4B[3] * (
      dnlm_dzij_dc[3] * (s2[2] - s2[1]) + dnlm_dzij[3] * 2.0 * (s[2] * dnlm_dc[2] - s[1] * dnlm_dc[1]));
    tmpz += 2.0 * C4B[3] * (
      dnlm_dc[3] * (s[2] * dnlm_dzij[2] - s[1] * dnlm_dzij[1]) +
        s[3] * (dnlm_dc[2] * dnlm_dzij[2] + s[2] * dnlm_dzij_dc[2] -
                dnlm_dc[1] * dnlm_dzij[1] - s[1] * dnlm_dzij_dc[1])
    );
    // d4
    tmpz += C4B[4] * (
      dnlm_dzij_dc[1] * s[2] * s[4] + dnlm_dzij[1] * dnlm_dc[2] * s[4] + dnlm_dzij[1] * s[2] * dnlm_dc[4] +
      dnlm_dc[1] * dnlm_dzij[2] * s[4] + s[1] * dnlm_dzij_dc[2] * s[4] + s[1] * dnlm_dzij[2] * dnlm_dc[4] +
      dnlm_dc[1] * s[2] * dnlm_dzij[4] + s[1] * dnlm_dc[2] * dnlm_dzij[4] + s[1] * s[2] * dnlm_dzij_dc[4]
    );    
    f12k[k_idx + 3] += Fp * scd_r12[3] * tmpz;
  }
}

static __device__ __forceinline__ void scd_get_f12_5body(
  const double *fn12,
  const double *fnp12,
  const double *blm,
  const double *rij_blm,
  const double *dblm_x,
  const double *dblm_y,
  const double *dblm_z,
  const double *dblm_r,
  const double *scd_r12,
  const double *dsnlm_dc,
  const double* s,
  const double* r12,
  const double d12inv,
  const double rij_Lsq, 
  const double rij_L2sq, 
  const double fn,
  const double fnp,
  const double Fp,
  const int n_base_angular,
  const int dsnlm_start_idx,
  const int type_j,
  const int ntypes,
  const int n1, 
  const int n2,
  double *f12k
)
{
  // L = 1
  int k_start_id = type_j * n_base_angular * 4;
  int dsnlm_idx = dsnlm_start_idx + type_j * n_base_angular * NUM_OF_ABC;
  int k_idx = 0;
  double dnlm_drij[3] = {0.0};
  dnlm_drij[0] = fnp * blm[0]; // fn * dblm/drij = 0
  dnlm_drij[1] = fnp * blm[1];
  dnlm_drij[2] = fnp * blm[2];
  double dnlm_dxij[3] = {0.0};
  dnlm_dxij[0] = 0.0;
  dnlm_dxij[1] = fn; //  dblm_x[1] = 1.0
  dnlm_dxij[2] = 0.0;
  double dnlm_dyij[3] = {0.0};
  dnlm_dyij[0] = 0.0;
  dnlm_dyij[1] = 0.0;
  dnlm_dyij[2] = fn; //  dblm_y[2] = 1.0
  double dnlm_dzij[3] = {0.0};
  dnlm_dzij[0] = fn; // dblm_z[0] = 1.0
  dnlm_dzij[1] = 0.0;
  dnlm_dzij[2] = 0.0;
  double dnlm_dc[3] = {0.0};
  double dnlm_drij_dc[3] = {0.0};
  double dnlm_dxij_dc[3] = {0.0};
  double dnlm_dyij_dc[3] = {0.0};
  double dnlm_dzij_dc[3] = {0.0};

  double s2[3] = {0.0};
  s2[0] = s[0] * s[0];
  s2[1] = s[1] * s[1];
  s2[2] = s[2] * s[2];
  
  double ds1s2 = 0.0;
  double ds1s2_c = 0.0;
  double d_tmp = 0.0;
  for(int k=0; k < n_base_angular; k++) {
    int dsnlm_i = dsnlm_idx + k * NUM_OF_ABC;
    double tmpr = 0.0, tmpx = 0.0, tmpy = 0.0, tmpz = 0.0;
    k_idx = k_start_id + k * 4;
    dnlm_dc[0] = dsnlm_dc[dsnlm_i + 0];
    dnlm_dc[1] = dsnlm_dc[dsnlm_i + 1];
    dnlm_dc[2] = dsnlm_dc[dsnlm_i + 2];
    
    dnlm_drij_dc[0] = (fnp12[k] * rij_Lsq - fn12[k] * rij_L2sq) * blm[0]; 
    dnlm_drij_dc[1] = (fnp12[k] * rij_Lsq - fn12[k] * rij_L2sq) * blm[1];
    dnlm_drij_dc[2] = (fnp12[k] * rij_Lsq - fn12[k] * rij_L2sq) * blm[2];
    
    // dnlm_dxij_dc[0] = 0.0;
    dnlm_dxij_dc[1] = fn12[k] * rij_Lsq;// dblm_x[1] = 1.0
    // dnlm_dxij_dc[2] = 0.0;

    // dnlm_dyij_dc[0] = 0.0;
    // dnlm_dyij_dc[1] = 0.0;
    dnlm_dyij_dc[2] = fn12[k] * rij_Lsq;// dblm_y[2] = 1.0

    dnlm_dzij_dc[0] = fn12[k] * rij_Lsq;// dblm_z[0] = 1.0
    // dnlm_dzij_dc[1] = 0.0;
    // dnlm_dzij_dc[2] = 0.0;

    // drij
    // d0
    tmpr += 4.0 * C5B[0] * (3.0 * s2[0] * dnlm_dc[0] * dnlm_drij[0] + s2[0] * s[0] * dnlm_drij_dc[0]);
    // d1
    ds1s2 = s[1] * dnlm_drij[1] + s[2] * dnlm_drij[2];
    ds1s2_c = 2.0 * s[1] * dnlm_dc[1] + 2.0 * s[2] * dnlm_dc[2];
    d_tmp = dnlm_dc[1] * dnlm_drij[1] + s[1] * dnlm_drij_dc[1] + dnlm_dc[2] * dnlm_drij[2] + s[2] * dnlm_drij_dc[2];
    tmpr += 2.0 * C5B[1] * (
      dnlm_dc[0] * dnlm_drij[0] * (s2[1] + s2[2]) + s[0] * dnlm_drij_dc[0] * (s2[1] + s2[2]) + 
      s[0] * dnlm_drij[0] * ds1s2_c + 2.0 * s[0] * dnlm_dc[0] * ds1s2 + s2[0] * d_tmp);
    // d2
    tmpr += 4.0 * C5B[2] * (ds1s2_c * ds1s2 + (s2[1] + s2[2]) * d_tmp);
    f12k[k_idx + 0] += Fp * scd_r12[0] * tmpr;

    // dxij
    // d0
    tmpx += 4.0 * C5B[0] * (3.0 * s2[0] * dnlm_dc[0] * dnlm_dxij[0] + s2[0] * s[0] * dnlm_dxij_dc[0]);
    // d1
    ds1s2 = s[1] * dnlm_dxij[1] + s[2] * dnlm_dxij[2];
    ds1s2_c = 2.0 * s[1] * dnlm_dc[1] + 2.0 * s[2] * dnlm_dc[2];
    d_tmp = dnlm_dc[1] * dnlm_dxij[1] + s[1] * dnlm_dxij_dc[1] + dnlm_dc[2] * dnlm_dxij[2] + s[2] * dnlm_dxij_dc[2];
    tmpx += 2.0 * C5B[1] * (
      dnlm_dc[0] * dnlm_dxij[0] * (s2[1] + s2[2]) + s[0] * dnlm_dxij_dc[0] * (s2[1] + s2[2]) + 
      s[0] * dnlm_dxij[0] * ds1s2_c + 2.0 * s[0] * dnlm_dc[0] * ds1s2 + s2[0] * d_tmp);
    // d2
    tmpx += 4.0 * C5B[2] * (ds1s2_c * ds1s2 + (s2[1] + s2[2]) * d_tmp);
    f12k[k_idx + 1] += Fp * scd_r12[1] * tmpx;

    // dyij
    // d0
    tmpy += 4.0 * C5B[0] * (3.0 * s2[0] * dnlm_dc[0] * dnlm_dyij[0] + s2[0] * s[0] * dnlm_dyij_dc[0]);
    // d1
    ds1s2 = s[1] * dnlm_dyij[1] + s[2] * dnlm_dyij[2];
    ds1s2_c = 2.0 * s[1] * dnlm_dc[1] + 2.0 * s[2] * dnlm_dc[2];
    d_tmp = dnlm_dc[1] * dnlm_dyij[1] + s[1] * dnlm_dyij_dc[1] + dnlm_dc[2] * dnlm_dyij[2] + s[2] * dnlm_dyij_dc[2];
    tmpy += 2.0 * C5B[1] * (
      dnlm_dc[0] * dnlm_dyij[0] * (s2[1] + s2[2]) + s[0] * dnlm_dyij_dc[0] * (s2[1] + s2[2]) + 
      s[0] * dnlm_dyij[0] * ds1s2_c + 2.0 * s[0] * dnlm_dc[0] * ds1s2 + s2[0] * d_tmp);
    // d2
    tmpy += 4.0 * C5B[2] * (ds1s2_c * ds1s2 + (s2[1] + s2[2]) * d_tmp);
    f12k[k_idx + 2] += Fp * scd_r12[2] * tmpy;

    // dzij
    // d0
    tmpz += 4.0 * C5B[0] * (3.0 * s2[0] * dnlm_dc[0] * dnlm_dzij[0] + s2[0] * s[0] * dnlm_dzij_dc[0]);
    // d1
    ds1s2 = s[1] * dnlm_dzij[1] + s[2] * dnlm_dzij[2];
    ds1s2_c = 2.0 * s[1] * dnlm_dc[1] + 2.0 * s[2] * dnlm_dc[2];
    d_tmp = dnlm_dc[1] * dnlm_dzij[1] + s[1] * dnlm_dzij_dc[1] + dnlm_dc[2] * dnlm_dzij[2] + s[2] * dnlm_dzij_dc[2];
    tmpz += 2.0 * C5B[1] * (
      dnlm_dc[0] * dnlm_dzij[0] * (s2[1] + s2[2]) + s[0] * dnlm_dzij_dc[0] * (s2[1] + s2[2]) + 
      s[0] * dnlm_dzij[0] * ds1s2_c + 2.0 * s[0] * dnlm_dc[0] * ds1s2 + s2[0] * d_tmp);
    // d2
    tmpz += 4.0 * C5B[2] * (ds1s2_c * ds1s2 + (s2[1] + s2[2]) * d_tmp);
    f12k[k_idx + 3] += Fp * scd_r12[3] * tmpz;
  }

  // if (n1==0 and n2==0){
  //   printf("\t5bL=2 n1=%d n2=%d s0=%lf s1=%lf s2=%lf s3=%lf s4=%lf fnp=%lf fn=%lf Fp=%lf r12=%lf x=%lf y=%lf z=%lf dqr=%lf dqx=%lf dqy=%lf dqz=%lf\n", 
  //           n1, n2, s[0], s[1], s[2], s[3], s[4], fnp, fn, Fp, d12, r12[0], r12[1], r12[2], tmp, tmpx, tmpy, tmpz);
  // }
}

static __device__ __forceinline__ void scd_get_f12_3(
  const double *fn12,
  const double *fnp12,
  const double *blm,
  const double *rij_blm,
  const double *dblm_x,
  const double *dblm_y,
  const double *dblm_z,
  const double *dblm_r,
  const double *scd_r12,
  const double *dsnlm_dc,
  const double* s,
  const double* r12,
  const double d12inv,
  const double rij_Lsq, 
  const double rij_L2sq, 
  const double fn,
  const double fnp,
  const double Fp,
  const int n_base_angular, 
  const int dsnlm_start_idx,
  const int type_j,
  const int ntypes,
  const int n1, 
  const int n2,
  double *f12k)
{
  // L = 3 c3b 8 9 10 11 12 13 14  s 0 1 2 3 4 5 6
  int k_start_id = type_j * n_base_angular * 4;
  int k_idx = 0;
  int dsnlm_idx = dsnlm_start_idx + type_j * n_base_angular * NUM_OF_ABC;
  for(int k=0; k < n_base_angular; k++) {
    int dsnlm_i = dsnlm_idx + k * NUM_OF_ABC;
    double tmpr = 0.0, tmpx = 0.0, tmpy = 0.0, tmpz = 0.0;
    k_idx = k_start_id + k * 4;
    // 左边项 rij
    tmpr +=         C3B[8]  * dsnlm_dc[dsnlm_i+8] * (fnp * blm[8]  + fn * dblm_r[8]) + 
              2.0 * C3B[9]  * dsnlm_dc[dsnlm_i+9] * (fnp * blm[9]  + fn * dblm_r[9]) + 
              2.0 * C3B[10] * dsnlm_dc[dsnlm_i+10] * (fnp * blm[10] + fn * dblm_r[10])+ 
              2.0 * C3B[11] * dsnlm_dc[dsnlm_i+11] *  fnp * blm[11] + // dblm/drij = 0
              2.0 * C3B[12] * dsnlm_dc[dsnlm_i+12] *  fnp * blm[12] + // dblm/drij = 0
              2.0 * C3B[13] * dsnlm_dc[dsnlm_i+13] *  fnp * blm[13] + // dblm/drij = 0
              2.0 * C3B[14] * dsnlm_dc[dsnlm_i+14] *  fnp * blm[14];  // dblm/drij = 0
    // 右边项
    tmpr +=       s[0] * ((fnp12[k] * rij_Lsq - 3.0 * fn12[k] * rij_L2sq) * blm[8]  + fn12[k] * rij_Lsq * dblm_r[8]) +
            2.0 * s[1] * ((fnp12[k] * rij_Lsq - 3.0 * fn12[k] * rij_L2sq) * blm[9]  + fn12[k] * rij_Lsq * dblm_r[9]) +
            2.0 * s[2] * ((fnp12[k] * rij_Lsq - 3.0 * fn12[k] * rij_L2sq) * blm[10] + fn12[k] * rij_Lsq * dblm_r[10])+
            2.0 * s[3] *  (fnp12[k] * rij_Lsq - 3.0 * fn12[k] * rij_L2sq) * blm[11] +   //dblm/drij = 0
            2.0 * s[4] *  (fnp12[k] * rij_Lsq - 3.0 * fn12[k] * rij_L2sq) * blm[12] +   //dblm/drij = 0
            2.0 * s[5] *  (fnp12[k] * rij_Lsq - 3.0 * fn12[k] * rij_L2sq) * blm[13] +   //dblm/drij = 0
            2.0 * s[6] *  (fnp12[k] * rij_Lsq - 3.0 * fn12[k] * rij_L2sq) * blm[14];    //dblm/drij = 0
    f12k[k_idx + 0] += 2.0 * Fp * scd_r12[0] * tmpr;// 可以考虑移出去

    // 左边项 xij
    tmpx +=       //C3B[8]  * dsnlm_dc[dsnlm_i+8]  * fn * dblm_x[8] +  
              2.0 * C3B[9]  * dsnlm_dc[dsnlm_i+9]  * fn * dblm_x[9] + 
              // 2.0 * C3B[10] * dsnlm_dc[dsnlm_i+10] * fn * dblm_x[10] + 
              2.0 * C3B[11] * dsnlm_dc[dsnlm_i+11] * fn * dblm_x[11] + 
              2.0 * C3B[12] * dsnlm_dc[dsnlm_i+12] * fn * dblm_x[12] + 
              2.0 * C3B[13] * dsnlm_dc[dsnlm_i+13] * fn * dblm_x[13] +
              2.0 * C3B[14] * dsnlm_dc[dsnlm_i+14] * fn * dblm_x[14];     
    // 右边项
    tmpx +=      // s[0] * fn12[k] * rij_Lsq * dblm_x[8]  + //db30/dx = 0
              2.0 * s[1] * fn12[k] * rij_Lsq * dblm_x[9]  + 
              // 2.0 * s[2] * fn12[k] * rij_Lsq * dblm_x[10] + //db33/dx = 0
              2.0 * s[3] * fn12[k] * rij_Lsq * dblm_x[11] + 
              2.0 * s[4] * fn12[k] * rij_Lsq * dblm_x[12] + 
              2.0 * s[5] * fn12[k] * rij_Lsq * dblm_x[13] + 
              2.0 * s[6] * fn12[k] * rij_Lsq * dblm_x[14];
    f12k[k_idx + 1] += 2.0 * Fp * scd_r12[1] * tmpx;

    // 左边项 yij
    tmpy +=   //       C3B[8] * dsnlm_dc[dsnlm_i+8]  * fn * dblm_y[8] +  
              // 2.0 * C3B[9] * dsnlm_dc[dsnlm_i+9]  * fn * dblm_y[9] + 
              2.0 * C3B[10] * dsnlm_dc[dsnlm_i+10] * fn * dblm_y[10] +
              2.0 * C3B[11] * dsnlm_dc[dsnlm_i+11] * fn * dblm_y[11] +
              2.0 * C3B[12] * dsnlm_dc[dsnlm_i+12] * fn * dblm_y[12] +
              2.0 * C3B[13] * dsnlm_dc[dsnlm_i+13] * fn * dblm_y[13] +
              2.0 * C3B[14] * dsnlm_dc[dsnlm_i+14] * fn * dblm_y[14];     
    // 右边项
    tmpy +=   // s[0] * fn12[k] * rij_Lsq * dblm_y[8] + 0.0 +
              // 2.0 * s[1] * fn12[k] * rij_Lsq * dblm_y[9] + 0.0 +
              2.0 * s[2] * fn12[k] * rij_Lsq * dblm_y[10] +
              2.0 * s[3] * fn12[k] * rij_Lsq * dblm_y[11] +
              2.0 * s[4] * fn12[k] * rij_Lsq * dblm_y[12] +
              2.0 * s[5] * fn12[k] * rij_Lsq * dblm_y[13] +
              2.0 * s[6] * fn12[k] * rij_Lsq * dblm_y[14];
    f12k[k_idx + 2] += 2.0 * Fp * scd_r12[2] * tmpy;

    // 左边项 zij
    tmpz +=         C3B[8]  * dsnlm_dc[dsnlm_i+8]  * fn * dblm_z[8] +  
              2.0 * C3B[9]  * dsnlm_dc[dsnlm_i+9]  * fn * dblm_z[9] +  
              2.0 * C3B[10] * dsnlm_dc[dsnlm_i+10] * fn * dblm_z[10] +
              2.0 * C3B[11] * dsnlm_dc[dsnlm_i+11] * fn * dblm_z[11] +
              2.0 * C3B[12] * dsnlm_dc[dsnlm_i+12] * fn * dblm_z[12];
              // 2.0 * C3B[13] * dsnlm_dc[dsnlm_i+13] * fn  * dblm_z[13] + * 0.0 +
              // 2.0 * C3B[14] * dsnlm_dc[dsnlm_i+14] * fn  * dblm_z[14] + * 0.0;     
    // 右边项
    tmpz +=         s[0] * fn12[k] * rij_Lsq * dblm_z[8] + 
              2.0 * s[1] * fn12[k] * rij_Lsq * dblm_z[9] + 
              2.0 * s[2] * fn12[k] * rij_Lsq * dblm_z[10]+ 
              2.0 * s[3] * fn12[k] * rij_Lsq * dblm_z[11]+ 
              2.0 * s[4] * fn12[k] * rij_Lsq * dblm_z[12];
            // 2.0 * s[5] * fn12[k] * rij_Lsq * dblm_z[13] +  0.0 +
            // 2.0 * s[6] * fn12[k] * rij_Lsq * dblm_z[14] +  0.0;
    f12k[k_idx + 3] += 2.0 * Fp * scd_r12[3] * tmpz;
    // if (n1==0 and n2==0 and k == 0){
    //   printf("\tscd L=3 n1=%d n2=%d s0=%lf s1=%lf s2=%lf s3=%lf s4=%lf s5=%lf s6=%lf fnp=%lf fn=%lf Fp=%lf b30=%lf b31=%lf b32=%lf b33=%lf b34=%lf b35=%lf b36=%lf dqr=%lf dqx=%lf dqy=%lf dqz=%lf\n", 
    //           n1, n2, s[0], s[1]*2.0, s[2]*2.0, s[3]*2.0, s[4]*2.0, s[5]*2.0, s[6]*2.0, fnp, fn, Fp, blm[8], blm[9], blm[10], blm[11], blm[12], blm[13], blm[14], tmpr, tmpx, tmpy, tmpz);
    // }
  }
}

static __device__ __forceinline__ void scd_get_f12_4(
  const double *fn12,
  const double *fnp12,
  const double *blm,
  const double *rij_blm,
  const double *dblm_x,
  const double *dblm_y,
  const double *dblm_z,
  const double *dblm_r,
  const double *scd_r12,
  const double *dsnlm_dc,
  const double* s,
  const double* r12,
  const double d12inv,
  const double rij_Lsq, 
  const double rij_L2sq, 
  const double fn,
  const double fnp,
  const double Fp,
  const int n_base_angular, 
  const int dsnlm_start_idx,
  const int type_j,
  const int ntypes,
  const int n1, 
  const int n2,
  double *f12k)
{
  int k_start_id = type_j * n_base_angular * 4;
  int k_idx = 0;
  int dsnlm_idx = dsnlm_start_idx + type_j * n_base_angular * NUM_OF_ABC;
  for(int k=0; k < n_base_angular; k++) {
    int dsnlm_i = dsnlm_idx + k * NUM_OF_ABC;
    double tmpr = 0.0, tmpx = 0.0, tmpy = 0.0, tmpz = 0.0;
    k_idx = k_start_id + k * 4;
    // 左边项 rij
    tmpr +=       C3B[15] * dsnlm_dc[dsnlm_i+15] * (fnp * blm[15] + fn * dblm_r[15]) + 
            2.0 * C3B[16] * dsnlm_dc[dsnlm_i+16] * (fnp * blm[16] + fn * dblm_r[16]) + 
            2.0 * C3B[17] * dsnlm_dc[dsnlm_i+17] * (fnp * blm[17] + fn * dblm_r[17]) + 
            2.0 * C3B[18] * dsnlm_dc[dsnlm_i+18] * (fnp * blm[18] + fn * dblm_r[18]) + 
            2.0 * C3B[19] * dsnlm_dc[dsnlm_i+19] * (fnp * blm[19] + fn * dblm_r[19]) + 
            2.0 * C3B[20] * dsnlm_dc[dsnlm_i+20] *  fnp * blm[20] +
            2.0 * C3B[21] * dsnlm_dc[dsnlm_i+21] *  fnp * blm[21] +
            2.0 * C3B[22] * dsnlm_dc[dsnlm_i+22] *  fnp * blm[22] +
            2.0 * C3B[23] * dsnlm_dc[dsnlm_i+23] *  fnp * blm[23];
    // 右边项
    tmpr +=       s[0] * ((fnp12[k] * rij_Lsq - 4.0 * fn12[k] * rij_L2sq) * blm[15] + fn12[k]* rij_Lsq * dblm_r[15]) +  
            2.0 * s[1] * ((fnp12[k] * rij_Lsq - 4.0 * fn12[k] * rij_L2sq) * blm[16] + fn12[k]* rij_Lsq * dblm_r[16]) +
            2.0 * s[2] * ((fnp12[k] * rij_Lsq - 4.0 * fn12[k] * rij_L2sq) * blm[17] + fn12[k]* rij_Lsq * dblm_r[17]) +
            2.0 * s[3] * ((fnp12[k] * rij_Lsq - 4.0 * fn12[k] * rij_L2sq) * blm[18] + fn12[k]* rij_Lsq * dblm_r[18]) +
            2.0 * s[4] * ((fnp12[k] * rij_Lsq - 4.0 * fn12[k] * rij_L2sq) * blm[19] + fn12[k]* rij_Lsq * dblm_r[19]) +
            2.0 * s[5] * ((fnp12[k] * rij_Lsq - 4.0 * fn12[k] * rij_L2sq) * blm[20]) +
            2.0 * s[6] * ((fnp12[k] * rij_Lsq - 4.0 * fn12[k] * rij_L2sq) * blm[21]) +
            2.0 * s[7] * ((fnp12[k] * rij_Lsq - 4.0 * fn12[k] * rij_L2sq) * blm[22]) +
            2.0 * s[8] * ((fnp12[k] * rij_Lsq - 4.0 * fn12[k] * rij_L2sq) * blm[23]); 
    f12k[k_idx + 0] += 2.0 * Fp * scd_r12[0] * tmpr;// 可以考虑移出去

    // 左边项 xij
    tmpx +=       C3B[15] * dsnlm_dc[dsnlm_i+15] * fn * dblm_x[15] + 
            2.0 * C3B[16] * dsnlm_dc[dsnlm_i+16] * fn * dblm_x[16] + 
            2.0 * C3B[17] * dsnlm_dc[dsnlm_i+17] * fn * dblm_x[17] + 
            2.0 * C3B[18] * dsnlm_dc[dsnlm_i+18] * fn * dblm_x[18] + 
            2.0 * C3B[19] * dsnlm_dc[dsnlm_i+19] * fn * dblm_x[19] + 
            2.0 * C3B[20] * dsnlm_dc[dsnlm_i+20] * fn * dblm_x[20] + 
            2.0 * C3B[21] * dsnlm_dc[dsnlm_i+21] * fn * dblm_x[21] + 
            2.0 * C3B[22] * dsnlm_dc[dsnlm_i+22] * fn * dblm_x[22] + 
            2.0 * C3B[23] * dsnlm_dc[dsnlm_i+23] * fn * dblm_x[23];

    // 右边项 xij
    tmpx +=       s[0] * fn12[k] * rij_Lsq * dblm_x[15] + 
            2.0 * s[1] * fn12[k] * rij_Lsq * dblm_x[16] + 
            2.0 * s[2] * fn12[k] * rij_Lsq * dblm_x[17] + 
            2.0 * s[3] * fn12[k] * rij_Lsq * dblm_x[18] + 
            2.0 * s[4] * fn12[k] * rij_Lsq * dblm_x[19] + 
            2.0 * s[5] * fn12[k] * rij_Lsq * dblm_x[20] + 
            2.0 * s[6] * fn12[k] * rij_Lsq * dblm_x[21] + 
            2.0 * s[7] * fn12[k] * rij_Lsq * dblm_x[22] + 
            2.0 * s[8] * fn12[k] * rij_Lsq * dblm_x[23];
    f12k[k_idx + 1] += 2.0 * Fp * scd_r12[1] * tmpx;

    // 左边项 yij
    tmpy +=       C3B[15] * dsnlm_dc[dsnlm_i+15] * fn * dblm_y[15] + 
            2.0 * C3B[16] * dsnlm_dc[dsnlm_i+16] * fn * dblm_y[16] + 
            2.0 * C3B[17] * dsnlm_dc[dsnlm_i+17] * fn * dblm_y[17] + 
            2.0 * C3B[18] * dsnlm_dc[dsnlm_i+18] * fn * dblm_y[18] + 
            2.0 * C3B[19] * dsnlm_dc[dsnlm_i+19] * fn * dblm_y[19] + 
            2.0 * C3B[20] * dsnlm_dc[dsnlm_i+20] * fn * dblm_y[20] + 
            2.0 * C3B[21] * dsnlm_dc[dsnlm_i+21] * fn * dblm_y[21] + 
            2.0 * C3B[22] * dsnlm_dc[dsnlm_i+22] * fn * dblm_y[22] + 
            2.0 * C3B[23] * dsnlm_dc[dsnlm_i+23] * fn * dblm_y[23];

    // 右边项 yij
    tmpy +=       s[0] * fn12[k] * rij_Lsq * dblm_y[15] + 
            2.0 * s[1] * fn12[k] * rij_Lsq * dblm_y[16] + 
            2.0 * s[2] * fn12[k] * rij_Lsq * dblm_y[17] + 
            2.0 * s[3] * fn12[k] * rij_Lsq * dblm_y[18] + 
            2.0 * s[4] * fn12[k] * rij_Lsq * dblm_y[19] + 
            2.0 * s[5] * fn12[k] * rij_Lsq * dblm_y[20] + 
            2.0 * s[6] * fn12[k] * rij_Lsq * dblm_y[21] + 
            2.0 * s[7] * fn12[k] * rij_Lsq * dblm_y[22] + 
            2.0 * s[8] * fn12[k] * rij_Lsq * dblm_y[23];
    f12k[k_idx + 2] += 2.0 * Fp * scd_r12[2] * tmpy;

    // 左边项 zij
    tmpz +=       C3B[15] * dsnlm_dc[dsnlm_i+15] * fn * dblm_z[15] + 
            2.0 * C3B[16] * dsnlm_dc[dsnlm_i+16] * fn * dblm_z[16] + 
            2.0 * C3B[17] * dsnlm_dc[dsnlm_i+17] * fn * dblm_z[17] + 
            2.0 * C3B[18] * dsnlm_dc[dsnlm_i+18] * fn * dblm_z[18] + 
            2.0 * C3B[19] * dsnlm_dc[dsnlm_i+19] * fn * dblm_z[19] + 
            2.0 * C3B[20] * dsnlm_dc[dsnlm_i+20] * fn * dblm_z[20] + 
            2.0 * C3B[21] * dsnlm_dc[dsnlm_i+21] * fn * dblm_z[21] + 
            2.0 * C3B[22] * dsnlm_dc[dsnlm_i+22] * fn * dblm_z[22] + 
            2.0 * C3B[23] * dsnlm_dc[dsnlm_i+23] * fn * dblm_z[23];

    // 右边项 zij
    tmpz +=       s[0] * fn12[k] * rij_Lsq * dblm_z[15] + 
            2.0 * s[1] * fn12[k] * rij_Lsq * dblm_z[16] + 
            2.0 * s[2] * fn12[k] * rij_Lsq * dblm_z[17] + 
            2.0 * s[3] * fn12[k] * rij_Lsq * dblm_z[18] + 
            2.0 * s[4] * fn12[k] * rij_Lsq * dblm_z[19] + 
            2.0 * s[5] * fn12[k] * rij_Lsq * dblm_z[20] + 
            2.0 * s[6] * fn12[k] * rij_Lsq * dblm_z[21] + 
            2.0 * s[7] * fn12[k] * rij_Lsq * dblm_z[22] + 
            2.0 * s[8] * fn12[k] * rij_Lsq * dblm_z[23];
    f12k[k_idx + 3] += 2.0 * Fp * scd_r12[3] * tmpz;

    // if (n1==0 and n2==0 and k == 0){
    //   printf("\tscd L=4 n1=%d n2=%d s0=%lf s1=%lf s2=%lf s3=%lf s4=%lf s5=%lf s6=%lf s7=%lf s8=%lf fnp=%lf fn=%lf Fp=%lf b40=%lf b41=%lf b42=%lf b43=%lf b44=%lf b45=%lf b46=%lf b47=%lf b48=%lf dqr=%lf dqx=%lf dqy=%lf dqz=%lf\n", 
    //           n1, n2, s[0], s[1]*2.0, s[2]*2.0, s[3]*2.0, s[4]*2.0, s[5]*2.0, s[6]*2.0,  s[7]*2.0, s[8]*2.0, fnp, fn, Fp, blm[15], blm[16], blm[17], blm[18], blm[19], blm[20], blm[21], blm[23], blm[23], tmpr, tmpx, tmpy, tmpz);
    // }
  }
}


// others J of C_nk_iJ
static __device__ __forceinline__ void scd_get_f12_1_J(
  const double *fn12,
  const double *fnp12,
  const double *blm,
  const double *rij_blm,
  const double *dblm_x,
  const double *dblm_y,
  const double *dblm_z,
  const double *dblm_r,
  const double *scd_r12,
  const double *dsnlm_dc,
  const double *s,
  const double *r12,
  const double d12inv,
  const double rij_Lsq, 
  const double rij_L2sq, 
  const double fn,
  const double fnp,
  const double Fp,
  const int n_base_angular, 
  const int dsnlm_start_idx,
  const int type_j,
  const int ntypes,
  const int n1, 
  const int n2,
  double *f12k)
{ //l = 1
  int k_idx = 0;
  for (int j = 0; j < ntypes; j++){
    if (type_j == j) continue;
    int dsnlm_idx = dsnlm_start_idx + j * n_base_angular * NUM_OF_ABC;
    int k_start_idx = j * n_base_angular * 4;
    for(int k=0; k < n_base_angular; k++) {
      int dsnlm_i = dsnlm_idx + k * NUM_OF_ABC;
      double tmpr = 0.0, tmpx = 0.0, tmpy = 0.0, tmpz = 0.0;
      k_idx = k_start_idx + k * 4;
      // 左边项 rij
      tmpr +=       C3B[0] * dsnlm_dc[dsnlm_i]   * fnp * blm[0]; 
      tmpr += 2.0 * C3B[1] * dsnlm_dc[dsnlm_i+1] * fnp * blm[1];
      tmpr += 2.0 * C3B[2] * dsnlm_dc[dsnlm_i+2] * fnp * blm[2];
      f12k[k_idx + 0] += 2.0 * Fp * scd_r12[0] * tmpr;// 可以考虑移出去

      // xij
      tmpx += 2.0 * C3B[1] * dsnlm_dc[dsnlm_i+1] * fn; // dblm/dxij b10 b12 为0
      f12k[k_idx + 1] += 2.0 * Fp * scd_r12[1] * tmpx;

      // yij
      tmpy += 2.0 * C3B[2] * dsnlm_dc[dsnlm_i+2] * fn;
      f12k[k_idx + 2] += 2.0 * Fp * scd_r12[2] * tmpy;

      // zij
      tmpz += C3B[0] * dsnlm_dc[dsnlm_i] * fn;
      f12k[k_idx + 3] += 2.0 * Fp * scd_r12[3] * tmpz;

      // if (n1==0 and n2==0 and k == 0){
      //   printf("\tscd_J L=1 n1=%d n2=%d k=%d s0=%lf s1=%lf s2=%lf fnp=%lf fn=%lf Fp=%lf fn12[%d]=%lf fnp12[%d]=%lf rij_Lsq=%lf rij_L2sq=%lf b10=%lf b11=%lf b12=%lf dqr=%lf dqx=%lf dqy=%lf dqz=%lf\n", 
      //           n1, n2, k, s[0], s[1]*2.0, s[2]*2.0, fnp, fn, Fp, k, fn12[k], k, fnp12[k], rij_Lsq, rij_L2sq, blm[0], blm[1], blm[2], tmpr, tmpx, tmpy, tmpz);
      // }
    }
  }
}

static __device__ __forceinline__ void scd_get_f12_4body_J(
  const double *fn12,
  const double *fnp12,
  const double *blm,
  const double *rij_blm,
  const double *dblm_x,
  const double *dblm_y,
  const double *dblm_z,
  const double *dblm_r,
  const double *scd_r12,
  const double *dsnlm_dc,
  const double* s,
  const double* r12,
  const double d12inv,
  const double rij_Lsq, 
  const double rij_L2sq, 
  const double fn,
  const double fnp,
  const double Fp,
  const int n_base_angular,
  const int dsnlm_start_idx, 
  const int type_j,
  const int ntypes,
  const int n1, 
  const int n2,
  double *f12k)
{
  // L = 2 c3b 3 4 5 6 7
  double dnlm_drij[5] = {0.0};
  dnlm_drij[0] = fnp * blm[3] + fn * dblm_r[3];
  dnlm_drij[1] = fnp * blm[4];
  dnlm_drij[2] = fnp * blm[5];
  dnlm_drij[3] = fnp * blm[6];
  dnlm_drij[4] = fnp * blm[7];
  double dnlm_dxij[5] = {0.0};
  dnlm_dxij[0] = 0.0;
  dnlm_dxij[1] = fn * dblm_x[4];
  dnlm_dxij[2] = 0.0;
  dnlm_dxij[3] = fn * dblm_x[6];
  dnlm_dxij[4] = fn * dblm_x[7];
  double dnlm_dyij[5] = {0.0};
  dnlm_dyij[0] = 0.0;
  dnlm_dyij[1] = 0.0;
  dnlm_dyij[2] = fn * dblm_y[5];
  dnlm_dyij[3] = fn * dblm_y[6];
  dnlm_dyij[4] = fn * dblm_y[7];
  double dnlm_dzij[5] = {0.0};
  dnlm_dzij[0] = fn * dblm_z[3];
  dnlm_dzij[1] = fn * dblm_z[4];
  dnlm_dzij[2] = fn * dblm_z[5];
  dnlm_dzij[3] = 0.0;
  dnlm_dzij[4] = 0.0;
  double dnlm_dc[5] = {0.0};
  int k_idx = 0;
  for (int j = 0; j < ntypes; j++){  
    if (type_j == j) continue;
    int dsnlm_idx = dsnlm_start_idx + j * n_base_angular * NUM_OF_ABC;
    int k_start_id = j * n_base_angular * 4;
    for(int k=0; k < n_base_angular; k++) {
      int dsnlm_i = dsnlm_idx + k * NUM_OF_ABC;
      double tmpr = 0.0, tmpx = 0.0, tmpy = 0.0, tmpz = 0.0;
      k_idx = k_start_id + k * 4;
      dnlm_dc[0] = dsnlm_dc[dsnlm_i + 3];
      dnlm_dc[1] = dsnlm_dc[dsnlm_i + 4];
      dnlm_dc[2] = dsnlm_dc[dsnlm_i + 5];
      dnlm_dc[3] = dsnlm_dc[dsnlm_i + 6];
      dnlm_dc[4] = dsnlm_dc[dsnlm_i + 7];

      // drij
      // d0
      tmpr += 3.0 * C4B[0] * (
        2.0 * s[0] * dnlm_dc[0] * dnlm_drij[0]);
      // d1
      tmpr += C4B[1] * (
        dnlm_drij[0] * 2.0 * (s[1] * dnlm_dc[1] + s[2] * dnlm_dc[2])
      );
      tmpr += 2.0 * C4B[1] * (
        dnlm_dc[0] * (s[1] * dnlm_drij[1] + s[2] * dnlm_drij[2]) + 
        s[0] * (dnlm_dc[1] * dnlm_drij[1] + 
                  dnlm_dc[2] * dnlm_drij[2])
      );
      // d2
      tmpr += C4B[2] * (
        dnlm_drij[0] * 2.0 * (s[3] * dnlm_dc[3] + s[4] * dnlm_dc[4]));
      tmpr += 2.0 * C4B[2] * (
        dnlm_dc[0] * (s[3] * dnlm_drij[3] + s[4] * dnlm_drij[4]) + 
        s[0] * (dnlm_dc[3] * dnlm_drij[3] +  
                  dnlm_dc[4] * dnlm_drij[4])  
      );
      // d3
      tmpr += C4B[3] * (
        dnlm_drij[3] * 2.0 * (s[2] * dnlm_dc[2] - s[1] * dnlm_dc[1]));
      tmpr += 2.0 * C4B[3] * (
        dnlm_dc[3] * (s[2] * dnlm_drij[2] - s[1] * dnlm_drij[1]) +
          s[3] * (dnlm_dc[2] * dnlm_drij[2] -
                  dnlm_dc[1] * dnlm_drij[1])
      );
      // d4
      tmpr += C4B[4] * (
        dnlm_drij[1] * dnlm_dc[2] * s[4] + dnlm_drij[1] * s[2] * dnlm_dc[4] +
        dnlm_dc[1] * dnlm_drij[2] * s[4] + s[1] * dnlm_drij[2] * dnlm_dc[4] +
        dnlm_dc[1] * s[2] * dnlm_drij[4] + s[1] * dnlm_dc[2] * dnlm_drij[4]
      );
      f12k[k_idx + 0] += Fp * scd_r12[0] * tmpr;
      // dxij
      // d0
      tmpx += 3.0 * C4B[0] * (
        2.0 * s[0] * dnlm_dc[0] * dnlm_dxij[0]);
      // d1
      tmpx += C4B[1] * (
        dnlm_dxij[0] * 2.0 * (s[1] * dnlm_dc[1] + s[2] * dnlm_dc[2])
      );
      tmpx += 2.0 * C4B[1] * (
        dnlm_dc[0] * (s[1] * dnlm_dxij[1] + s[2] * dnlm_dxij[2]) + 
        s[0] * (dnlm_dc[1] * dnlm_dxij[1] + 
                  dnlm_dc[2] * dnlm_dxij[2])
      );
      // d2
      tmpx += C4B[2] * (
        dnlm_dxij[0] * 2.0 * (s[3] * dnlm_dc[3] + s[4] * dnlm_dc[4]));
      tmpx += 2.0 * C4B[2] * (
        dnlm_dc[0] * (s[3] * dnlm_dxij[3] + s[4] * dnlm_dxij[4]) + 
        s[0] * (dnlm_dc[3] * dnlm_dxij[3] + 
                  dnlm_dc[4] * dnlm_dxij[4])  
      );
      // d3
      tmpx += C4B[3] * (
        dnlm_dxij[3] * 2.0 * (s[2] * dnlm_dc[2] - s[1] * dnlm_dc[1]));
      tmpx += 2.0 * C4B[3] * (
        dnlm_dc[3] * (s[2] * dnlm_dxij[2] - s[1] * dnlm_dxij[1]) +
          s[3] * (dnlm_dc[2] * dnlm_dxij[2] -
                  dnlm_dc[1] * dnlm_dxij[1])
      );
      // d4
      tmpx += C4B[4] * (
        dnlm_dxij[1] * dnlm_dc[2] * s[4] + dnlm_dxij[1] * s[2] * dnlm_dc[4] +
        dnlm_dc[1] * dnlm_dxij[2] * s[4] + s[1] * dnlm_dxij[2] * dnlm_dc[4] +
        dnlm_dc[1] * s[2] * dnlm_dxij[4] + s[1] * dnlm_dc[2] * dnlm_dxij[4]
      );    
      f12k[k_idx + 1] += Fp * scd_r12[1] * tmpx;

      // dyij
      // d0
      tmpy += 3.0 * C4B[0] * (
        2.0 * s[0] * dnlm_dc[0] * dnlm_dyij[0]);
      // d1
      tmpy += C4B[1] * (
        dnlm_dyij[0] * 2.0 * (s[1] * dnlm_dc[1] + s[2] * dnlm_dc[2])
      );
      tmpy += 2.0 * C4B[1] * (
        dnlm_dc[0] * (s[1] * dnlm_dyij[1] + s[2] * dnlm_dyij[2]) + 
        s[0] * (dnlm_dc[1] * dnlm_dyij[1] + 
                  dnlm_dc[2] * dnlm_dyij[2])
      );
      // d2
      tmpy += C4B[2] * (
        dnlm_dyij[0] * 2.0 * (s[3] * dnlm_dc[3] + s[4] * dnlm_dc[4]));
      tmpy += 2.0 * C4B[2] * (
        dnlm_dc[0] * (s[3] * dnlm_dyij[3] + s[4] * dnlm_dyij[4]) + 
        s[0] * (dnlm_dc[3] * dnlm_dyij[3] + 
                  dnlm_dc[4] * dnlm_dyij[4])  
      );
      // d3
      tmpy += C4B[3] * (
        dnlm_dyij[3] * 2.0 * (s[2] * dnlm_dc[2] - s[1] * dnlm_dc[1]));
      tmpy += 2.0 * C4B[3] * (
        dnlm_dc[3] * (s[2] * dnlm_dyij[2] - s[1] * dnlm_dyij[1]) +
          s[3] * (dnlm_dc[2] * dnlm_dyij[2] -
                  dnlm_dc[1] * dnlm_dyij[1])
      );
      // d4
      tmpy += C4B[4] * (
        dnlm_dyij[1] * dnlm_dc[2] * s[4] + dnlm_dyij[1] * s[2] * dnlm_dc[4] +
        dnlm_dc[1] * dnlm_dyij[2] * s[4] + s[1] * dnlm_dyij[2] * dnlm_dc[4] +
        dnlm_dc[1] * s[2] * dnlm_dyij[4] + s[1] * dnlm_dc[2] * dnlm_dyij[4]
      );    
      f12k[k_idx + 2] += Fp * scd_r12[2] * tmpy;

      // dzij
      // d0
      tmpz += 3.0 * C4B[0] * (
        2.0 * s[0] * dnlm_dc[0] * dnlm_dzij[0]);
      // d1
      tmpz += C4B[1] * (
        dnlm_dzij[0] * 2.0 * (s[1] * dnlm_dc[1] + s[2] * dnlm_dc[2])
      );
      tmpz += 2.0 * C4B[1] * (
        dnlm_dc[0] * (s[1] * dnlm_dzij[1] + s[2] * dnlm_dzij[2]) + 
        s[0] * (dnlm_dc[1] * dnlm_dzij[1] + 
                  dnlm_dc[2] * dnlm_dzij[2])
      );
      // d2
      tmpz += C4B[2] * (
        dnlm_dzij[0] * 2.0 * (s[3] * dnlm_dc[3] + s[4] * dnlm_dc[4]));
      tmpz += 2.0 * C4B[2] * (
        dnlm_dc[0] * (s[3] * dnlm_dzij[3] + s[4] * dnlm_dzij[4]) + 
        s[0] * (dnlm_dc[3] * dnlm_dzij[3] + 
                  dnlm_dc[4] * dnlm_dzij[4])  
      );
      // d3
      tmpz += C4B[3] * (
        dnlm_dzij[3] * 2.0 * (s[2] * dnlm_dc[2] - s[1] * dnlm_dc[1]));
      tmpz += 2.0 * C4B[3] * (
        dnlm_dc[3] * (s[2] * dnlm_dzij[2] - s[1] * dnlm_dzij[1]) +
          s[3] * (dnlm_dc[2] * dnlm_dzij[2] -
                  dnlm_dc[1] * dnlm_dzij[1])
      );
      // d4
      tmpz += C4B[4] * (
        dnlm_dzij[1] * dnlm_dc[2] * s[4] + dnlm_dzij[1] * s[2] * dnlm_dc[4] +
        dnlm_dc[1] * dnlm_dzij[2] * s[4] + s[1] * dnlm_dzij[2] * dnlm_dc[4] +
        dnlm_dc[1] * s[2] * dnlm_dzij[4] + s[1] * dnlm_dc[2] * dnlm_dzij[4]
      );    
      f12k[k_idx + 3] += Fp * scd_r12[3] * tmpz;
    }
  }
}

static __device__ __forceinline__ void scd_get_f12_5body_J(
  const double *fn12,
  const double *fnp12,
  const double *blm,
  const double *rij_blm,
  const double *dblm_x,
  const double *dblm_y,
  const double *dblm_z,
  const double *dblm_r,
  const double *scd_r12,
  const double *dsnlm_dc,
  const double* s,
  const double* r12,
  const double d12inv,
  const double rij_Lsq, 
  const double rij_L2sq, 
  const double fn,
  const double fnp,
  const double Fp,
  const int n_base_angular,
  const int dsnlm_start_idx,
  const int type_j,
  const int ntypes,
  const int n1, 
  const int n2,
  double *f12k
)
{
  // L = 1
  double dnlm_drij[3] = {0.0};
  dnlm_drij[0] = fnp * blm[0]; // fn * dblm/drij = 0
  dnlm_drij[1] = fnp * blm[1];
  dnlm_drij[2] = fnp * blm[2];
  double dnlm_dxij[3] = {0.0};
  dnlm_dxij[0] = 0.0;
  dnlm_dxij[1] = fn; //  dblm_x[1] = 1.0
  dnlm_dxij[2] = 0.0;
  double dnlm_dyij[3] = {0.0};
  dnlm_dyij[0] = 0.0;
  dnlm_dyij[1] = 0.0;
  dnlm_dyij[2] = fn; //  dblm_y[2] = 1.0
  double dnlm_dzij[3] = {0.0};
  dnlm_dzij[0] = fn; // dblm_z[0] = 1.0
  dnlm_dzij[1] = 0.0;
  dnlm_dzij[2] = 0.0;
  double dnlm_dc[3] = {0.0};

  double s2[3] = {0.0};
  s2[0] = s[0] * s[0];
  s2[1] = s[1] * s[1];
  s2[2] = s[2] * s[2];
  
  double ds1s2 = 0.0;
  double ds1s2_c = 0.0;
  double d_tmp = 0.0;
  int k_idx = 0;
  for (int j = 0; j < ntypes; j++){  
    if (type_j == j) continue;
    int dsnlm_idx = dsnlm_start_idx + j * n_base_angular * NUM_OF_ABC;
    int k_start_id = j * n_base_angular * 4;
    for(int k=0; k < n_base_angular; k++) {
      int dsnlm_i = dsnlm_idx + k * NUM_OF_ABC;
      double tmpr = 0.0, tmpx = 0.0, tmpy = 0.0, tmpz = 0.0;
      k_idx = k_start_id + k * 4;
      dnlm_dc[0] = dsnlm_dc[dsnlm_i + 0];
      dnlm_dc[1] = dsnlm_dc[dsnlm_i + 1];
      dnlm_dc[2] = dsnlm_dc[dsnlm_i + 2];
      // drij
      // d0
      tmpr += 4.0 * C5B[0] * (3.0 * s2[0] * dnlm_dc[0] * dnlm_drij[0]);
      // d1
      ds1s2 = s[1] * dnlm_drij[1] + s[2] * dnlm_drij[2];
      ds1s2_c = 2.0 * s[1] * dnlm_dc[1] + 2.0 * s[2] * dnlm_dc[2];
      d_tmp = dnlm_dc[1] * dnlm_drij[1] + dnlm_dc[2] * dnlm_drij[2];
      tmpr += 2.0 * C5B[1] * (
        dnlm_dc[0] * dnlm_drij[0] * (s2[1] + s2[2]) +  
        s[0] * dnlm_drij[0] * ds1s2_c + 2.0 * s[0] * dnlm_dc[0] * ds1s2 + s2[0] * d_tmp);
      // d2
      tmpr += 4.0 * C5B[2] * (ds1s2_c * ds1s2 + (s2[1] + s2[2]) * d_tmp);
      f12k[k_idx + 0] += Fp * scd_r12[0] * tmpr;

      // dxij
      // d0
      tmpx += 4.0 * C5B[0] * (3.0 * s2[0] * dnlm_dc[0] * dnlm_dxij[0]);
      // d1
      ds1s2 = s[1] * dnlm_dxij[1] + s[2] * dnlm_dxij[2];
      ds1s2_c = 2.0 * s[1] * dnlm_dc[1] + 2.0 * s[2] * dnlm_dc[2];
      d_tmp = dnlm_dc[1] * dnlm_dxij[1] + dnlm_dc[2] * dnlm_dxij[2];
      tmpx += 2.0 * C5B[1] * (
        dnlm_dc[0] * dnlm_dxij[0] * (s2[1] + s2[2]) +  
        s[0] * dnlm_dxij[0] * ds1s2_c + 2.0 * s[0] * dnlm_dc[0] * ds1s2 + s2[0] * d_tmp);
      // d2
      tmpx += 4.0 * C5B[2] * (ds1s2_c * ds1s2 + (s2[1] + s2[2]) * d_tmp);
      f12k[k_idx + 1] += Fp * scd_r12[1] * tmpx;

      // dyij
      // d0
      tmpy += 4.0 * C5B[0] * (3.0 * s2[0] * dnlm_dc[0] * dnlm_dyij[0]);
      // d1
      ds1s2 = s[1] * dnlm_dyij[1] + s[2] * dnlm_dyij[2];
      ds1s2_c = 2.0 * s[1] * dnlm_dc[1] + 2.0 * s[2] * dnlm_dc[2];
      d_tmp = dnlm_dc[1] * dnlm_dyij[1] + dnlm_dc[2] * dnlm_dyij[2];
      tmpy += 2.0 * C5B[1] * (
        dnlm_dc[0] * dnlm_dyij[0] * (s2[1] + s2[2]) +  
        s[0] * dnlm_dyij[0] * ds1s2_c + 2.0 * s[0] * dnlm_dc[0] * ds1s2 + s2[0] * d_tmp);
      // d2
      tmpy += 4.0 * C5B[2] * (ds1s2_c * ds1s2 + (s2[1] + s2[2]) * d_tmp);
      f12k[k_idx + 2] += Fp * scd_r12[2] * tmpy;

      // dzij
      // d0
      tmpz += 4.0 * C5B[0] * (3.0 * s2[0] * dnlm_dc[0] * dnlm_dzij[0]);
      // d1
      ds1s2 = s[1] * dnlm_dzij[1] + s[2] * dnlm_dzij[2];
      ds1s2_c = 2.0 * s[1] * dnlm_dc[1] + 2.0 * s[2] * dnlm_dc[2];
      d_tmp = dnlm_dc[1] * dnlm_dzij[1] + dnlm_dc[2] * dnlm_dzij[2];
      tmpz += 2.0 * C5B[1] * (
        dnlm_dc[0] * dnlm_dzij[0] * (s2[1] + s2[2]) +  
        s[0] * dnlm_dzij[0] * ds1s2_c + 2.0 * s[0] * dnlm_dc[0] * ds1s2 + s2[0] * d_tmp);
      // d2
      tmpz += 4.0 * C5B[2] * (ds1s2_c * ds1s2 + (s2[1] + s2[2]) * d_tmp);
      f12k[k_idx + 3] += Fp * scd_r12[3] * tmpz;
    }
  }
  // if (n1==0 and n2==0){
  //   printf("\t5bL=2 n1=%d n2=%d s0=%lf s1=%lf s2=%lf s3=%lf s4=%lf fnp=%lf fn=%lf Fp=%lf r12=%lf x=%lf y=%lf z=%lf dqr=%lf dqx=%lf dqy=%lf dqz=%lf\n", 
  //           n1, n2, s[0], s[1], s[2], s[3], s[4], fnp, fn, Fp, d12, r12[0], r12[1], r12[2], tmp, tmpx, tmpy, tmpz);
  // }
}

static __device__ __forceinline__ void scd_get_f12_2_J(
  const double *fn12,
  const double *fnp12,
  const double *blm,
  const double *rij_blm,
  const double *dblm_x,
  const double *dblm_y,
  const double *dblm_z,
  const double *dblm_r,
  const double *scd_r12,
  const double *dsnlm_dc,
  const double* s,
  const double* r12,
  const double d12inv,
  const double rij_Lsq, 
  const double rij_L2sq, 
  const double fn,
  const double fnp,
  const double Fp,
  const int n_base_angular, 
  const int dsnlm_start_idx,
  const int type_j,
  const int ntypes,
  const int n1, 
  const int n2,
  double *f12k
  )
{
  // L = 2 c3b 3 4 5 6 7
  int k_idx = 0;
  for (int j = 0; j < ntypes; j++){  
    if (type_j == j) continue;
    int dsnlm_idx = dsnlm_start_idx + j * n_base_angular * NUM_OF_ABC;
    int k_start_idx = j * n_base_angular * 4;    
    for(int k=0; k < n_base_angular; k++) {
      int dsnlm_i = dsnlm_idx + k * NUM_OF_ABC;
      double tmpr = 0.0, tmpx = 0.0, tmpy = 0.0, tmpz = 0.0;
      k_idx = k_start_idx + k * 4;
      // 左边项 rij
      tmpr +=  C3B[3] * dsnlm_dc[dsnlm_i+3] * (fnp * blm[3] + fn * dblm_r[3]) + 
                    2.0 * C3B[4] * dsnlm_dc[dsnlm_i+4] * fnp * blm[4] + 
                    2.0 * C3B[5] * dsnlm_dc[dsnlm_i+5] * fnp * blm[5] + 
                    2.0 * C3B[6] * dsnlm_dc[dsnlm_i+6] * fnp * blm[6] +
                    2.0 * C3B[7] * dsnlm_dc[dsnlm_i+7] * fnp * blm[7]; 
      f12k[k_idx + 0] += 2.0 * Fp * scd_r12[0] * tmpr;// 可以考虑移出去

      // 左边项 xij
      tmpx += 
                    2.0 * C3B[4] * dsnlm_dc[dsnlm_i+4] * fn * dblm_x[4] + 
                    2.0 * C3B[6] * dsnlm_dc[dsnlm_i+6] * fn * dblm_x[6] +
                    2.0 * C3B[7] * dsnlm_dc[dsnlm_i+7] * fn * dblm_x[7];     
      f12k[k_idx + 1] += 2.0 * Fp * scd_r12[1] * tmpx;

      // 左边项 yij
      tmpy += 
                    2.0 * C3B[5] * dsnlm_dc[dsnlm_i+5] * fn * dblm_y[5] + 
                    2.0 * C3B[6] * dsnlm_dc[dsnlm_i+6] * fn * dblm_y[6] +
                    2.0 * C3B[7] * dsnlm_dc[dsnlm_i+7] * fn * dblm_y[7];     
      f12k[k_idx + 2] += 2.0 * Fp * scd_r12[2] * tmpy;
      
      // 左边项 zij
      tmpz +=  C3B[3] * dsnlm_dc[dsnlm_i+3] * fn * dblm_z[3] + 
                    2.0 * C3B[4] * dsnlm_dc[dsnlm_i+4] * fn * dblm_z[4] + 
                    2.0 * C3B[5] * dsnlm_dc[dsnlm_i+5] * fn * dblm_z[5];     
      f12k[k_idx + 3] += 2.0 * Fp * scd_r12[3] * tmpz;

      // if (n1==0 and n2==0 and k == 0){
      //   printf("\tscd_J L=2 n1=%d n2=%d s0=%lf s1=%lf s2=%lf s3=%lf s4=%lf fnp=%lf fn=%lf Fp=%lf b20=%lf b21=%lf b22=%lf b23=%lf b24=%lf dqr=%lf dqx=%lf dqy=%lf dqz=%lf\n", 
      //           n1, n2, s[0], s[1]*2.0, s[2]*2.0, s[3]*2.0, s[4]*2.0, fnp, fn, Fp, blm[3], blm[4], blm[5], blm[6], blm[7], tmpr, tmpx, tmpy, tmpz);
      // }
    }
  }
}


static __device__ __forceinline__ void scd_get_f12_3_J(
  const double *fn12,
  const double *fnp12,
  const double *blm,
  const double *rij_blm,
  const double *dblm_x,
  const double *dblm_y,
  const double *dblm_z,
  const double *dblm_r,
  const double *scd_r12,
  const double *dsnlm_dc,
  const double* s,
  const double* r12,
  const double d12inv,
  const double rij_Lsq, 
  const double rij_L2sq, 
  const double fn,
  const double fnp,
  const double Fp,
  const int n_base_angular, 
  const int dsnlm_start_idx,
  const int type_j,
  const int ntypes,
  const int n1, 
  const int n2,
  double *f12k)
{
  // L = 3 c3b 8 9 10 11 12 13 14  s 0 1 2 3 4 5 6
  int k_idx = 0;
  for (int j = 0; j < ntypes; j++){  
    if (type_j == j) continue;
    int dsnlm_idx = dsnlm_start_idx + j * n_base_angular * NUM_OF_ABC;
    int k_start_idx = j * n_base_angular * 4;   
    for(int k=0; k < n_base_angular; k++) {
      int dsnlm_i = dsnlm_idx + k * NUM_OF_ABC;
      double tmpr = 0.0, tmpx = 0.0, tmpy = 0.0, tmpz = 0.0;
      k_idx = k_start_idx + k * 4;
      // 左边项 rij
      tmpr +=         C3B[8]  * dsnlm_dc[dsnlm_i+8] * (fnp * blm[8]  + fn * dblm_r[8]) + 
                2.0 * C3B[9]  * dsnlm_dc[dsnlm_i+9] * (fnp * blm[9]  + fn * dblm_r[9]) + 
                2.0 * C3B[10] * dsnlm_dc[dsnlm_i+10] * (fnp * blm[10] + fn * dblm_r[10])+ 
                2.0 * C3B[11] * dsnlm_dc[dsnlm_i+11] *  fnp * blm[11] + // dblm/drij = 0
                2.0 * C3B[12] * dsnlm_dc[dsnlm_i+12] *  fnp * blm[12] + // dblm/drij = 0
                2.0 * C3B[13] * dsnlm_dc[dsnlm_i+13] *  fnp * blm[13] + // dblm/drij = 0
                2.0 * C3B[14] * dsnlm_dc[dsnlm_i+14] *  fnp * blm[14];  // dblm/drij = 0
      f12k[k_idx + 0] += 2.0 * Fp * scd_r12[0] * tmpr;// 可以考虑移出去

      // 左边项 xij
      tmpx +=       //C3B[8]  * dsnlm_dc[dsnlm_i+8]  * fn * dblm_x[8] +  
                2.0 * C3B[9]  * dsnlm_dc[dsnlm_i+9]  * fn * dblm_x[9] + 
                // 2.0 * C3B[10] * dsnlm_dc[dsnlm_i+10] * fn * dblm_x[10] + 
                2.0 * C3B[11] * dsnlm_dc[dsnlm_i+11] * fn * dblm_x[11] + 
                2.0 * C3B[12] * dsnlm_dc[dsnlm_i+12] * fn * dblm_x[12] + 
                2.0 * C3B[13] * dsnlm_dc[dsnlm_i+13] * fn * dblm_x[13] +
                2.0 * C3B[14] * dsnlm_dc[dsnlm_i+14] * fn * dblm_x[14];     
      f12k[k_idx + 1] += 2.0 * Fp * scd_r12[1] * tmpx;

      // 左边项 yij
      tmpy +=   //       C3B[8] * dsnlm_dc[dsnlm_i+8]  * fn * dblm_y[8] +  
                // 2.0 * C3B[9] * dsnlm_dc[dsnlm_i+9]  * fn * dblm_y[9] + 
                2.0 * C3B[10] * dsnlm_dc[dsnlm_i+10] * fn * dblm_y[10] +
                2.0 * C3B[11] * dsnlm_dc[dsnlm_i+11] * fn * dblm_y[11] +
                2.0 * C3B[12] * dsnlm_dc[dsnlm_i+12] * fn * dblm_y[12] +
                2.0 * C3B[13] * dsnlm_dc[dsnlm_i+13] * fn * dblm_y[13] +
                2.0 * C3B[14] * dsnlm_dc[dsnlm_i+14] * fn * dblm_y[14];     
      f12k[k_idx + 2] += 2.0 * Fp * scd_r12[2] * tmpy;

      // 左边项 zij
      tmpz +=         C3B[8]  * dsnlm_dc[dsnlm_i+8]  * fn * dblm_z[8] +  
                2.0 * C3B[9]  * dsnlm_dc[dsnlm_i+9]  * fn * dblm_z[9] +  
                2.0 * C3B[10] * dsnlm_dc[dsnlm_i+10] * fn * dblm_z[10] +
                2.0 * C3B[11] * dsnlm_dc[dsnlm_i+11] * fn * dblm_z[11] +
                2.0 * C3B[12] * dsnlm_dc[dsnlm_i+12] * fn * dblm_z[12];
                // 2.0 * C3B[13] * dsnlm_dc[dsnlm_i+13] * fn  * dblm_z[13] + * 0.0 +
                // 2.0 * C3B[14] * dsnlm_dc[dsnlm_i+14] * fn  * dblm_z[14] + * 0.0;     
      f12k[k_idx + 3] += 2.0 * Fp * scd_r12[3] * tmpz;
      // if (n1==0 and n2==0 and k == 0){
      //   printf("\tscd_J L=3 n1=%d n2=%d s0=%lf s1=%lf s2=%lf s3=%lf s4=%lf s5=%lf s6=%lf fnp=%lf fn=%lf Fp=%lf b30=%lf b31=%lf b32=%lf b33=%lf b34=%lf b35=%lf b36=%lf dqr=%lf dqx=%lf dqy=%lf dqz=%lf\n", 
      //           n1, n2, s[0], s[1]*2.0, s[2]*2.0, s[3]*2.0, s[4]*2.0, s[5]*2.0, s[6]*2.0, fnp, fn, Fp, blm[8], blm[9], blm[10], blm[11], blm[12], blm[13], blm[14], tmpr, tmpx, tmpy, tmpz);
      // }
    }
  }
}


static __device__ __forceinline__ void scd_get_f12_4_J(
  const double *fn12,
  const double *fnp12,
  const double *blm,
  const double *rij_blm,
  const double *dblm_x,
  const double *dblm_y,
  const double *dblm_z,
  const double *dblm_r,
  const double *scd_r12,
  const double *dsnlm_dc,
  const double* s,
  const double* r12,
  const double d12inv,
  const double rij_Lsq, 
  const double rij_L2sq, 
  const double fn,
  const double fnp,
  const double Fp,
  const int n_base_angular, 
  const int dsnlm_start_idx,
  const int type_j,
  const int ntypes,
  const int n1, 
  const int n2,
  double *f12k)
{
  int k_idx = 0;
  for (int j = 0; j < ntypes; j++){  
    if (type_j == j) continue;
    int dsnlm_idx = dsnlm_start_idx + j * n_base_angular * NUM_OF_ABC;
    int k_start_idx = j * n_base_angular * 4;   
    for(int k=0; k < n_base_angular; k++) {
      int dsnlm_i = dsnlm_idx + k * NUM_OF_ABC;
      double tmpr = 0.0, tmpx = 0.0, tmpy = 0.0, tmpz = 0.0;
      k_idx = k_start_idx + k * 4;
      // 左边项 rij
      tmpr +=       C3B[15] * dsnlm_dc[dsnlm_i+15] * (fnp * blm[15] + fn * dblm_r[15]) + 
              2.0 * C3B[16] * dsnlm_dc[dsnlm_i+16] * (fnp * blm[16] + fn * dblm_r[16]) + 
              2.0 * C3B[17] * dsnlm_dc[dsnlm_i+17] * (fnp * blm[17] + fn * dblm_r[17]) + 
              2.0 * C3B[18] * dsnlm_dc[dsnlm_i+18] * (fnp * blm[18] + fn * dblm_r[18]) + 
              2.0 * C3B[19] * dsnlm_dc[dsnlm_i+19] * (fnp * blm[19] + fn * dblm_r[19]) + 
              2.0 * C3B[20] * dsnlm_dc[dsnlm_i+20] *  fnp * blm[20] +
              2.0 * C3B[21] * dsnlm_dc[dsnlm_i+21] *  fnp * blm[21] +
              2.0 * C3B[22] * dsnlm_dc[dsnlm_i+22] *  fnp * blm[22] +
              2.0 * C3B[23] * dsnlm_dc[dsnlm_i+23] *  fnp * blm[23];
      f12k[k_idx + 0] += 2.0 * Fp * scd_r12[0] * tmpr;// 可以考虑移出去
      // 左边项 xij
      tmpx +=       C3B[15] * dsnlm_dc[dsnlm_i+15] * fn * dblm_x[15] + 
              2.0 * C3B[16] * dsnlm_dc[dsnlm_i+16] * fn * dblm_x[16] + 
              2.0 * C3B[17] * dsnlm_dc[dsnlm_i+17] * fn * dblm_x[17] + 
              2.0 * C3B[18] * dsnlm_dc[dsnlm_i+18] * fn * dblm_x[18] + 
              2.0 * C3B[19] * dsnlm_dc[dsnlm_i+19] * fn * dblm_x[19] + 
              2.0 * C3B[20] * dsnlm_dc[dsnlm_i+20] * fn * dblm_x[20] + 
              2.0 * C3B[21] * dsnlm_dc[dsnlm_i+21] * fn * dblm_x[21] + 
              2.0 * C3B[22] * dsnlm_dc[dsnlm_i+22] * fn * dblm_x[22] + 
              2.0 * C3B[23] * dsnlm_dc[dsnlm_i+23] * fn * dblm_x[23];

      f12k[k_idx + 1] += 2.0 * Fp * scd_r12[1] * tmpx;
      // 左边项 yij
      tmpy +=       C3B[15] * dsnlm_dc[dsnlm_i+15] * fn * dblm_y[15] + 
              2.0 * C3B[16] * dsnlm_dc[dsnlm_i+16] * fn * dblm_y[16] + 
              2.0 * C3B[17] * dsnlm_dc[dsnlm_i+17] * fn * dblm_y[17] + 
              2.0 * C3B[18] * dsnlm_dc[dsnlm_i+18] * fn * dblm_y[18] + 
              2.0 * C3B[19] * dsnlm_dc[dsnlm_i+19] * fn * dblm_y[19] + 
              2.0 * C3B[20] * dsnlm_dc[dsnlm_i+20] * fn * dblm_y[20] + 
              2.0 * C3B[21] * dsnlm_dc[dsnlm_i+21] * fn * dblm_y[21] + 
              2.0 * C3B[22] * dsnlm_dc[dsnlm_i+22] * fn * dblm_y[22] + 
              2.0 * C3B[23] * dsnlm_dc[dsnlm_i+23] * fn * dblm_y[23];

      f12k[k_idx + 2] += 2.0 * Fp * scd_r12[2] * tmpy;
      // 左边项 zij
      tmpz +=       C3B[15] * dsnlm_dc[dsnlm_i+15] * fn * dblm_z[15] + 
              2.0 * C3B[16] * dsnlm_dc[dsnlm_i+16] * fn * dblm_z[16] + 
              2.0 * C3B[17] * dsnlm_dc[dsnlm_i+17] * fn * dblm_z[17] + 
              2.0 * C3B[18] * dsnlm_dc[dsnlm_i+18] * fn * dblm_z[18] + 
              2.0 * C3B[19] * dsnlm_dc[dsnlm_i+19] * fn * dblm_z[19] + 
              2.0 * C3B[20] * dsnlm_dc[dsnlm_i+20] * fn * dblm_z[20] + 
              2.0 * C3B[21] * dsnlm_dc[dsnlm_i+21] * fn * dblm_z[21] + 
              2.0 * C3B[22] * dsnlm_dc[dsnlm_i+22] * fn * dblm_z[22] + 
              2.0 * C3B[23] * dsnlm_dc[dsnlm_i+23] * fn * dblm_z[23];
      f12k[k_idx + 3] += 2.0 * Fp * scd_r12[3] * tmpz;

      // if (n1==0 and n2==0 and k == 0){
      //   printf("\tscd_J L=4 n1=%d n2=%d s0=%lf s1=%lf s2=%lf s3=%lf s4=%lf s5=%lf s6=%lf s7=%lf s8=%lf fnp=%lf fn=%lf Fp=%lf b40=%lf b41=%lf b42=%lf b43=%lf b44=%lf b45=%lf b46=%lf b47=%lf b48=%lf dqr=%lf dqx=%lf dqy=%lf dqz=%lf\n", 
      //           n1, n2, s[0], s[1]*2.0, s[2]*2.0, s[3]*2.0, s[4]*2.0, s[5]*2.0, s[6]*2.0,  s[7]*2.0, s[8]*2.0, fnp, fn, Fp, blm[15], blm[16], blm[17], blm[18], blm[19], blm[20], blm[21], blm[23], blm[23], tmpr, tmpx, tmpy, tmpz);
      // }
    }
  }
}

static __device__ __forceinline__ void scd_accumulate_f12(
  const int n,
  const double d12,
  const double* r12,
  double fn,
  double fnp,
  const double* Fp,// dEi/dq
  const double* dsnlm_dc, // dsnlm/dc 与Nmax无关
  const double* sum_fxyz, //[i, n, 24]
  double* blm,
  double* rij_blm,
  double* dblm_x,
  double* dblm_y,
  double* dblm_z,
  double* dblm_r,
  double* f12,//rij坐标的导数
  double* f12k,// rij坐标的导数，以nbase3b展开
  double* scd_r12,//对坐标的二阶导
  double* fn12,// fk
  double* fnp12,// fk对rij的导
  const int type_j,//邻居类型
  const int ntypes,
  const int lmax_3,
  const int n_max_angular,
  const int n_base_angular,
  const int dc_start_idx,
  const int dsnlm_start_idx,
  const int n1,
  const int n2) // i-> [ntype, nmax, nbase]-> [ntyp, ]
{
  const double d12inv = 1.0 / d12;
  // l = 1
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  double rij_Lsq = d12inv;
  double rij_L2sq= d12inv  * d12inv;
  double s1[3] = {
    sum_fxyz[n * NUM_OF_ABC + 0] * C3B[0],
    sum_fxyz[n * NUM_OF_ABC + 1] * C3B[1],
    sum_fxyz[n * NUM_OF_ABC + 2] * C3B[2]};
  scd_get_f12_1(fn12, fnp12, 
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s1, r12, 
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n*lmax_3],
                n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k);

  scd_get_f12_1_J(fn12, fnp12, 
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s1, r12, 
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n*lmax_3],
                n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k);

  // l = 2
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  rij_Lsq = rij_L2sq;
  rij_L2sq = rij_L2sq * d12inv;
  double s2[5] = {
    sum_fxyz[n * NUM_OF_ABC + 3] * C3B[3],
    sum_fxyz[n * NUM_OF_ABC + 4] * C3B[4],
    sum_fxyz[n * NUM_OF_ABC + 5] * C3B[5],
    sum_fxyz[n * NUM_OF_ABC + 6] * C3B[6],
    sum_fxyz[n * NUM_OF_ABC + 7] * C3B[7]};

  scd_get_f12_2(fn12, fnp12, 
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s2, r12, 
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n*lmax_3+1],
                n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k);

  scd_get_f12_2_J(fn12, fnp12, 
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s2, r12, 
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n*lmax_3+1],
                n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k);

  // l = 3
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  rij_Lsq = rij_L2sq;
  rij_L2sq = rij_L2sq * d12inv;  
  double s3[7] = {
    sum_fxyz[n * NUM_OF_ABC + 8] * C3B[8],
    sum_fxyz[n * NUM_OF_ABC + 9] * C3B[9],
    sum_fxyz[n * NUM_OF_ABC + 10] * C3B[10],
    sum_fxyz[n * NUM_OF_ABC + 11] * C3B[11],
    sum_fxyz[n * NUM_OF_ABC + 12] * C3B[12],
    sum_fxyz[n * NUM_OF_ABC + 13] * C3B[13],
    sum_fxyz[n * NUM_OF_ABC + 14] * C3B[14]};
  scd_get_f12_3(fn12, fnp12, 
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s3, r12, 
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n*lmax_3+2],
                n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k);

  scd_get_f12_3_J(fn12, fnp12, 
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s3, r12, 
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n*lmax_3+2],
                n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k);

  // l = 4
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  rij_Lsq = rij_L2sq;
  rij_L2sq = rij_L2sq * d12inv;
  double s4[9] = {
    sum_fxyz[n * NUM_OF_ABC + 15] * C3B[15],
    sum_fxyz[n * NUM_OF_ABC + 16] * C3B[16],
    sum_fxyz[n * NUM_OF_ABC + 17] * C3B[17],
    sum_fxyz[n * NUM_OF_ABC + 18] * C3B[18],
    sum_fxyz[n * NUM_OF_ABC + 19] * C3B[19],
    sum_fxyz[n * NUM_OF_ABC + 20] * C3B[20],
    sum_fxyz[n * NUM_OF_ABC + 21] * C3B[21],
    sum_fxyz[n * NUM_OF_ABC + 22] * C3B[22],
    sum_fxyz[n * NUM_OF_ABC + 23] * C3B[23]};
  scd_get_f12_4(fn12, fnp12, 
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s4, r12, 
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n*lmax_3+3], 
                n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k);

  scd_get_f12_4_J(fn12, fnp12, 
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s4, r12, 
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n*lmax_3+3], 
                n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k);

}

static __device__ __forceinline__ void scd_accumulate_f12_with_4body(
  const int n,
  const double d12,
  const double* r12,
  double fn,
  double fnp,
  const double* Fp,
  const double* dsnlm_dc, // dsnlm/dc 与Nmax无关
  const double* sum_fxyz,
  double* blm,
  double* rij_blm,
  double* dblm_x,
  double* dblm_y,
  double* dblm_z,
  double* dblm_r,
  double* f12,//rij坐标的导数
  double* f12k,// rij坐标的导数，以nbase3b展开
  double* scd_r12,//对坐标的二阶导
  double* fn12,
  double* fnp12,
  const int type_j,
  const int ntypes,
  const int lmax_3,
  const int n_max_angular,
  const int n_base_angular,
  const int dc_start_idx,
  const int dsnlm_start_idx,
  const int n1,
  const int n2)
{
  const double d12inv = 1.0 / d12;
  double rij_Lsq = d12inv;
  double rij_L2sq= d12inv  * d12inv;
  // l = 1
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  double s1[3] = {
    sum_fxyz[n * NUM_OF_ABC + 0] * C3B[0],
    sum_fxyz[n * NUM_OF_ABC + 1] * C3B[1],
    sum_fxyz[n * NUM_OF_ABC + 2] * C3B[2]};
  scd_get_f12_1(fn12, fnp12, 
              blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
              scd_r12, dsnlm_dc, s1, r12, 
              d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n*lmax_3],
              n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k);

  scd_get_f12_1_J(fn12, fnp12, 
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s1, r12, 
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n*lmax_3],
                n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k);

  // l = 2
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  rij_Lsq = rij_L2sq;
  rij_L2sq = rij_L2sq * d12inv;
  double s2[5] = {
    sum_fxyz[n * NUM_OF_ABC + 3],
    sum_fxyz[n * NUM_OF_ABC + 4],
    sum_fxyz[n * NUM_OF_ABC + 5],
    sum_fxyz[n * NUM_OF_ABC + 6],
    sum_fxyz[n * NUM_OF_ABC + 7]};

  scd_get_f12_4body(fn12, fnp12, 
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s2, r12, 
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n_max_angular * lmax_3 + n],
                n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k);

  scd_get_f12_4body_J(fn12, fnp12, 
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s2, r12, 
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n_max_angular * lmax_3 + n],
                n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k);

  s2[0] *= C3B[3];
  s2[1] *= C3B[4];
  s2[2] *= C3B[5];
  s2[3] *= C3B[6];
  s2[4] *= C3B[7];
  scd_get_f12_2(fn12, fnp12, 
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s2, r12, 
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n*lmax_3+1],
                n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k); 

  scd_get_f12_2_J(fn12, fnp12, 
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s2, r12, 
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n*lmax_3+1],
                n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k);
  // l = 3
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  rij_Lsq = rij_L2sq;
  rij_L2sq = rij_L2sq * d12inv;  
  double s3[7] = {
    sum_fxyz[n * NUM_OF_ABC + 8] * C3B[8],
    sum_fxyz[n * NUM_OF_ABC + 9] * C3B[9],
    sum_fxyz[n * NUM_OF_ABC + 10] * C3B[10],
    sum_fxyz[n * NUM_OF_ABC + 11] * C3B[11],
    sum_fxyz[n * NUM_OF_ABC + 12] * C3B[12],
    sum_fxyz[n * NUM_OF_ABC + 13] * C3B[13],
    sum_fxyz[n * NUM_OF_ABC + 14] * C3B[14]};
  scd_get_f12_3(fn12, fnp12, 
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s3, r12, 
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n*lmax_3+2],
                n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k);

  scd_get_f12_3_J(fn12, fnp12, 
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s3, r12, 
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n*lmax_3+2],
                n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k);

  // l = 4
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  rij_Lsq = rij_L2sq;
  rij_L2sq = rij_L2sq * d12inv;
  double s4[9] = {
    sum_fxyz[n * NUM_OF_ABC + 15] * C3B[15],
    sum_fxyz[n * NUM_OF_ABC + 16] * C3B[16],
    sum_fxyz[n * NUM_OF_ABC + 17] * C3B[17],
    sum_fxyz[n * NUM_OF_ABC + 18] * C3B[18],
    sum_fxyz[n * NUM_OF_ABC + 19] * C3B[19],
    sum_fxyz[n * NUM_OF_ABC + 20] * C3B[20],
    sum_fxyz[n * NUM_OF_ABC + 21] * C3B[21],
    sum_fxyz[n * NUM_OF_ABC + 22] * C3B[22],
    sum_fxyz[n * NUM_OF_ABC + 23] * C3B[23]};
  scd_get_f12_4(fn12, fnp12, 
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s4, r12, 
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n*lmax_3+3], 
                n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k);

  scd_get_f12_4_J(fn12, fnp12, 
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s4, r12, 
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n*lmax_3+3], 
                n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k);
}

static __device__ __forceinline__ void scd_accumulate_f12_with_5body(
  const int n,
  const double d12,
  const double* r12,
  double fn,
  double fnp,
  const double* Fp,
  const double* dsnlm_dc, // dsnlm/dc 与Nmax无关
  const double* sum_fxyz,
  double* blm,
  double* rij_blm,
  double* dblm_x,
  double* dblm_y,
  double* dblm_z,
  double* dblm_r,
  double* f12,//rij坐标的导数
  double* f12k,// rij坐标的导数，以nbase3b展开
  double* scd_r12,//对坐标的二阶导
  double* fn12,
  double* fnp12,
  const int type_j,
  const int ntypes,
  const int lmax_3,
  const int n_max_angular,
  const int n_base_angular,
  const int dc_start_idx,
  const int dsnlm_start_idx,
  const int n1,
  const int n2)
{
  const double d12inv = 1.0 / d12;
  double rij_Lsq = d12inv;
  double rij_L2sq= d12inv  * d12inv;
  // l = 1
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  double s1[3] = {
    sum_fxyz[n * NUM_OF_ABC + 0], sum_fxyz[n * NUM_OF_ABC + 1], sum_fxyz[n * NUM_OF_ABC + 2]};
  scd_get_f12_5body(fn12, fnp12, 
              blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
              scd_r12, dsnlm_dc, s1, r12, 
              d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n_max_angular * lmax_3 + n_max_angular + n], 
              n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k);

  scd_get_f12_5body_J(fn12, fnp12, 
              blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
              scd_r12, dsnlm_dc, s1, r12, 
              d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n_max_angular * lmax_3 + n_max_angular + n], 
              n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k);
  
  s1[0] *= C3B[0];
  s1[1] *= C3B[1];
  s1[2] *= C3B[2];
  scd_get_f12_1(fn12, fnp12, 
              blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
              scd_r12, dsnlm_dc, s1, r12, 
              d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n*lmax_3],
              n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k);

  scd_get_f12_1_J(fn12, fnp12, 
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s1, r12, 
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n*lmax_3],
                n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k);

  // l = 2
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  rij_Lsq = rij_L2sq;
  rij_L2sq = rij_L2sq * d12inv;
  double s2[5] = {
    sum_fxyz[n * NUM_OF_ABC + 3],
    sum_fxyz[n * NUM_OF_ABC + 4],
    sum_fxyz[n * NUM_OF_ABC + 5],
    sum_fxyz[n * NUM_OF_ABC + 6],
    sum_fxyz[n * NUM_OF_ABC + 7]};
  scd_get_f12_4body(fn12, fnp12, 
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s2, r12, 
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n_max_angular * lmax_3 + n],
                n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k);

  scd_get_f12_4body_J(fn12, fnp12, 
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s2, r12, 
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n_max_angular * lmax_3 + n],
                n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k);

  s2[0] *= C3B[3];
  s2[1] *= C3B[4];
  s2[2] *= C3B[5];
  s2[3] *= C3B[6];
  s2[4] *= C3B[7];
  scd_get_f12_2(fn12, fnp12, 
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s2, r12, 
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n*lmax_3+1],
                n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k); 

  scd_get_f12_2_J(fn12, fnp12, 
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s2, r12, 
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n*lmax_3+1],
                n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k);

  // l = 3
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  rij_Lsq = rij_L2sq;
  rij_L2sq = rij_L2sq * d12inv;
  double s3[7] = {
    sum_fxyz[n * NUM_OF_ABC + 8] * C3B[8],
    sum_fxyz[n * NUM_OF_ABC + 9] * C3B[9],
    sum_fxyz[n * NUM_OF_ABC + 10] * C3B[10],
    sum_fxyz[n * NUM_OF_ABC + 11] * C3B[11],
    sum_fxyz[n * NUM_OF_ABC + 12] * C3B[12],
    sum_fxyz[n * NUM_OF_ABC + 13] * C3B[13],
    sum_fxyz[n * NUM_OF_ABC + 14] * C3B[14]};
  scd_get_f12_3(fn12, fnp12, 
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s3, r12, 
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n*lmax_3+2],
                n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k);

  scd_get_f12_3_J(fn12, fnp12, 
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s3, r12, 
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n*lmax_3+2],
                n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k);

  // l = 4
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  rij_Lsq = rij_L2sq;
  rij_L2sq = rij_L2sq * d12inv;
  double s4[9] = {
    sum_fxyz[n * NUM_OF_ABC + 15] * C3B[15],
    sum_fxyz[n * NUM_OF_ABC + 16] * C3B[16],
    sum_fxyz[n * NUM_OF_ABC + 17] * C3B[17],
    sum_fxyz[n * NUM_OF_ABC + 18] * C3B[18],
    sum_fxyz[n * NUM_OF_ABC + 19] * C3B[19],
    sum_fxyz[n * NUM_OF_ABC + 20] * C3B[20],
    sum_fxyz[n * NUM_OF_ABC + 21] * C3B[21],
    sum_fxyz[n * NUM_OF_ABC + 22] * C3B[22],
    sum_fxyz[n * NUM_OF_ABC + 23] * C3B[23]};
  scd_get_f12_4(fn12, fnp12, 
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s4, r12, 
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n*lmax_3+3], 
                n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k);
  
  scd_get_f12_4_J(fn12, fnp12, 
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s4, r12, 
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n*lmax_3+3], 
                n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k);

}
