// 从当前受检HIP数学实现生成，完整保留GPUMD版权声明。
#pragma once
static const char* kNepMbSecondGradRtcSource=R"MATPLMBRTC(
#include <hip/hip_runtime.h>
#ifndef INT_MAX
#define INT_MAX 2147483647
#endif
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


const int NUM_OF_ABC = 24; // 3 + 5 + 7 + 9 for L_max = 4
// 除了c10 c20 c30 c40 其他的都少2倍
__constant__ double C3B[NUM_OF_ABC] = {
  0.238732414637843, 0.119366207318922, 0.119366207318922, 0.099471839432435,   //c10 c11 c12 c20
  0.596831036594608, 0.596831036594608, 0.149207759148652, 0.149207759148652,   //c21 c22 c23 c24
  0.139260575205408, 0.104445431404056, 0.104445431404056, 1.044454314040563,   //c30 c31 c32 c33
  1.044454314040563, 0.174075719006761, 0.174075719006761, 0.011190581936149,   //c34 c35 c36 c40
  0.223811638722978, 0.223811638722978, 0.111905819361489, 0.111905819361489,   //C41 C42 C43 C44
  1.566681471060845, 1.566681471060845, 0.195835183882606, 0.195835183882606};  //C45 C46 C47 C48
__constant__ double C4B[5] = {
  -0.007499480826664,
  -0.134990654879954,
  0.067495327439977,
  0.404971964639861,
  -0.809943929279723};
__constant__ double C5B[3] = {0.026596810706114, 0.053193621412227, 0.026596810706114};

const int SIZE_BOX_AND_INVERSE_BOX = 18; // (3 * 3) * 2
const int MAX_NUM_N = 20;                // n_max+1 = 19+1

const int MAX_LMAX = 6; // 4 + 1 + 1
const int MAX_DIM = MAX_NUM_N * 7;
const int MAX_DIM_ANGULAR = MAX_NUM_N * 6;

const double PI = 3.14159265358979323846;
const double HALF_PI = 1.5707963267948966;

static __device__ void dev_apply_mic(double const* __restrict__ h, double& x12, double& y12, double& z12)
{
    double sx12 = h[9] * x12 + h[10] * y12 + h[11] * z12;
    double sy12 = h[12] * x12 + h[13] * y12 + h[14] * z12;
    double sz12 = h[15] * x12 + h[16] * y12 + h[17] * z12;
    sx12 -= nearbyint(sx12);
    sy12 -= nearbyint(sy12);
    sz12 -= nearbyint(sz12);
    x12 = h[0] * sx12 + h[1] * sy12 + h[2] * sz12;
    y12 = h[3] * sx12 + h[4] * sy12 + h[5] * sz12;
    z12 = h[6] * sx12 + h[7] * sy12 + h[8] * sz12;
}

static __device__ __forceinline__ void find_fc(double rc, double rcinv, double d12, double& fc)
{
  if (d12 < rc) {
    double x = d12 * rcinv;
    fc = 0.5 * cos(PI * x) + 0.5;
  } else {
    fc = 0.0;
  }
}

__device__ __host__ __forceinline__ void find_fc_and_fcp(
        double rc, double rcinv, double d12, double& fc, double& fcp) {
    if (d12 < rc) {
        double x = d12 * rcinv;
        fc = 0.5 * cos(PI * x) + 0.5;
        fcp = -HALF_PI * sin(PI * x) * rcinv;
    } else {
        fc = 0.0;
        fcp = 0.0;
    }
}

static __device__ __forceinline__ void
find_fc_and_fcp_zbl(double r1, double r2, double d12, double& fc, double& fcp)
{
  if (d12 < r1) {
    fc = 1.0;
    fcp = 0.0;
  } else if (d12 < r2) {
    double pi_factor = PI / (r2 - r1);
    fc = cos(pi_factor * (d12 - r1)) * 0.5 + 0.5;
    fcp = -sin(pi_factor * (d12 - r1)) * pi_factor * 0.5;
  } else {
    fc = 0.0;
    fcp = 0.0;
  }
}

static __device__ __forceinline__ void
find_phi_and_phip_zbl(double a, double b, double x, double& phi, double& phip)
{
  double tmp = a * exp(-b * x);
  phi += tmp;
  phip -= b * tmp;
}

static __device__ __forceinline__ void find_f_and_fp_zbl(
  const double zizj,
  const double a_inv,
  const double rc_inner,
  const double rc_outer,
  const double d12,
  const double d12inv,
  double& f,
  double& fp)
{
  const double x = d12 * a_inv;
  f = fp = 0.0;
  const double Zbl_para[8] = {
    0.18175, 3.1998, 0.50986, 0.94229, 0.28022, 0.4029, 0.02817, 0.20162};
  find_phi_and_phip_zbl(Zbl_para[0], Zbl_para[1], x, f, fp);
  find_phi_and_phip_zbl(Zbl_para[2], Zbl_para[3], x, f, fp);
  find_phi_and_phip_zbl(Zbl_para[4], Zbl_para[5], x, f, fp);
  find_phi_and_phip_zbl(Zbl_para[6], Zbl_para[7], x, f, fp);
  f *= zizj;
  fp *= zizj * a_inv;
  fp = fp * d12inv - f * d12inv * d12inv;
  f *= d12inv;
  double fc, fcp;
  find_fc_and_fcp_zbl(rc_inner, rc_outer, d12, fc, fcp);
  fp = fp * fc + f * fcp;
  f *= fc;
}

static __device__ __forceinline__ void find_f_and_fp_zbl(
  const double* zbl_para,
  const double zizj,
  const double a_inv,
  const double d12,
  const double d12inv,
  double& f,
  double& fp)
{
  const double x = d12 * a_inv;
  f = fp = 0.0;
  find_phi_and_phip_zbl(zbl_para[2], zbl_para[3], x, f, fp);
  find_phi_and_phip_zbl(zbl_para[4], zbl_para[5], x, f, fp);
  find_phi_and_phip_zbl(zbl_para[6], zbl_para[7], x, f, fp);
  find_phi_and_phip_zbl(zbl_para[8], zbl_para[9], x, f, fp);
  f *= zizj;
  fp *= zizj * a_inv;
  fp = fp * d12inv - f * d12inv * d12inv;
  f *= d12inv;
  double fc, fcp;
  find_fc_and_fcp_zbl(zbl_para[0], zbl_para[1], d12, fc, fcp);
  fp = fp * fc + f * fcp;
  f *= fc;
}

static __device__ __forceinline__ void
find_fn(const int n, const double rcinv, const double d12, const double fc12, double& fn)
{
  if (n == 0) {
    fn = fc12;
  } else if (n == 1) {
    double x = 2.0 * (d12 * rcinv - 1.0) * (d12 * rcinv - 1.0) - 1.0;
    fn = (x + 1.0) * 0.5 * fc12;
  } else {
    double x = 2.0 * (d12 * rcinv - 1.0) * (d12 * rcinv - 1.0) - 1.0;
    double t0 = 1.0;
    double t1 = x;
    double t2;
    for (int m = 2; m <= n; ++m) {
      t2 = 2.0 * x * t1 - t0;
      t0 = t1;
      t1 = t2;
    }
    fn = (t2 + 1.0) * 0.5 * fc12;
  }
}

static __device__ __forceinline__ void find_fn_and_fnp(
  const int n,
  const double rcinv,
  const double d12,
  const double fc12,
  const double fcp12,
  double& fn,
  double& fnp)
{
  if (n == 0) {
    fn = fc12;
    fnp = fcp12;
  } else if (n == 1) {
    double x = 2.0 * (d12 * rcinv - 1.0) * (d12 * rcinv - 1.0) - 1.0;
    fn = (x + 1.0) * 0.5;
    fnp = 2.0 * (d12 * rcinv - 1.0) * rcinv * fc12 + fn * fcp12;
    fn *= fc12;
  } else {
    double x = 2.0 * (d12 * rcinv - 1.0) * (d12 * rcinv - 1.0) - 1.0;
    double t0 = 1.0;
    double t1 = x;
    double t2;
    double u0 = 1.0;
    double u1 = 2.0 * x;
    double u2;
    for (int m = 2; m < n; ++m) {
      t2 = 2.0 * x * t1 - t0;
      t0 = t1;
      t1 = t2;
      u2 = 2.0 * x * u1 - u0;
      u0 = u1;
      u1 = u2;
    }
    fn = (t2 + 1.0) * 0.5;
    fnp = n * u0 * 2.0 * (d12 * rcinv - 1.0) * rcinv;
    fnp = fnp * fc12 + fn * fcp12;
    fn *= fc12;
  }
}

static __device__ __forceinline__ void
find_fn(const int n_max, const double rcinv, const double d12, const double fc12, double* fn)
{
  double x = 2.0 * (d12 * rcinv - 1.0) * (d12 * rcinv - 1.0) - 1.0;
  fn[0] = 1.0;
  fn[1] = x;
  for (int m = 2; m < n_max; ++m) {
    fn[m] = 2.0 * x * fn[m - 1] - fn[m - 2];
  }
  for (int m = 0; m < n_max; ++m) {
    fn[m] = (fn[m] + 1.0) * 0.5 * fc12;
  }
}

__device__ __host__ __forceinline__ void find_fn_and_fnp(
        const int n_max, const double rcinv, const double d12, const double fc12, const double fcp12, double* fn, double* fnp) {
    double x = 2.0 * (d12 * rcinv - 1.0) * (d12 * rcinv - 1.0) - 1.0;
    fn[0] = 1.0;
    fnp[0] = 0.0;
    fn[1] = x;
    fnp[1] = 1.0;
    double u0 = 1.0;
    double u1 = 2.0 * x;
    double u2;
    for (int m = 2; m < n_max; ++m) {
        fn[m] = 2.0 * x * fn[m - 1] - fn[m - 2];
        fnp[m] = m * u1;
        u2 = 2.0 * x * u1 - u0;
        u0 = u1;
        u1 = u2;
    }
    for (int m = 0; m < n_max; ++m) {
        fn[m] = (fn[m] + 1.0) * 0.5;
        fnp[m] *= 2.0 * (d12 * rcinv - 1.0) * rcinv;
        fnp[m] = fnp[m] * fc12 + fn[m] * fcp12;
        fn[m] *= fc12;
    }
}

static __device__ __forceinline__ void get_f12_1(
  const double d12inv,
  const double fn,
  const double fnp,
  const double Fp,
  const double* s,
  const double* r12,
  double* f12,
  double* f12d,
  const int n1,
  const int n2)
{
  double tmp = 2.0 * fnp * (s[0] * r12[2] + 2.0 * (s[1] * r12[0] + s[2] * r12[1]));
  f12[3] += tmp * Fp;
  f12d[3] += tmp;

  double tmpx = 2.0 * fn * s[1] * 2.0; // 把 c11 和 c12少的2倍补上
  f12[0] += tmpx * Fp;
  f12d[0] += tmpx;

  double tmpy = 2.0 * fn * s[2] * 2.0;
  f12[1] += tmpy * Fp;
  f12d[1] += tmpy;

  double tmpz = 2.0 * fn * s[0];
  f12[2] += tmpz * Fp;
  f12d[2] += tmpz;
  // if (n1==0 and n2==0){
  //   printf("\tL=1 n1=%d n2=%d s0=%f s1=%f s2=%f fnp=%f fn=%f Fp=%f b10=%f b11=%f b12=%f dqr=%f dqx=%f dqy=%f dqz=%f\n",
  //           n1, n2, s[0], s[1]*2.0, s[2]*2.0, fnp, fn, Fp, r12[2], r12[0], r12[1], tmp, tmpx, tmpy, tmpz);
  // }
}

static __device__ __forceinline__ void get_f12_2(
  const double d12,
  const double d12inv,
  const double fn,
  const double fnp,
  const double Fp,
  const double* s,
  const double* r12,
  double* f12,
  double* f12d,
  const int n1,
  const int n2)
{
  double tmpx = 2.0 * ( // 2.0 来自平方导数
    2.0 * fn * (s[1] * r12[2] + s[3] * 2.0 * r12[0] + s[4] * 2.0 * r12[1]) //第一个2.0来自clm少的2倍
  );
  f12[0] += Fp * tmpx;
  f12d[0] += tmpx;

  double tmpy = 2.0 * (
    2.0 * fn * (s[2] * r12[2] + s[3] * (-2.0 * r12[1]) + s[4] * 2.0 * r12[0])
  );
  f12[1] += Fp * tmpy;
  f12d[1] += tmpy;

  double tmpz = 2.0 * (
    s[0] * fn * 6.0 * r12[2] + 2.0 * s[1] * fn * r12[0] + 2.0 * fn * s[2] * r12[1]
  );
  f12[2] += Fp * tmpz;
  f12d[2] += tmpz;

  double tmp = s[0] * (fnp * (3.0 * r12[2] * r12[2]- d12 * d12) - fn * 2.0 * d12) +
                2.0 * fnp * (s[1] * r12[0] * r12[2] +
                             s[2] * r12[1] * r12[2] +
                             s[3] * (r12[0] * r12[0]-r12[1] * r12[1]) +
                             s[4] * 2.0 * r12[0] * r12[1]
                             );

  // if (n1==0 and n2==0){
  //   printf("\tL=2 b0=%f b1=%f b2=%f b3=%f b4=%f s0=%f s1=%f s2=%f s3=%f s4=%f x=%f y=%f z=%f r=%f fnp=%f fn=%f tmp=%f\n",
  //           b0, b1, b2, b3, b4, s[0], s[1]*2.0, s[2]*2.0, s[3]*2.0, s[4]*2.0, r12[0], r12[1], r12[2], d12,  fnp, fn, tmp);
  // }
  tmp = 2.0 * tmp;
  f12[3] += Fp * tmp;
  f12d[3] += tmp;

  // if (n1==0 and n2==0){
  //   printf("\tL=2 n1=%d n2=%d s0=%f s1=%f s2=%f s3=%f s4=%f fnp=%f fn=%f Fp=%f r12=%f x=%f y=%f z=%f dqr=%f dqx=%f dqy=%f dqz=%f\n",
  //          n1, n2, s[0], s[1]*2.0, s[2]*2.0, s[3]*2.0, s[4]*2.0, fnp, fn, Fp, d12, r12[0], r12[1], r12[2], tmp, tmpx, tmpy, tmpz);
  // }
}

static __device__ __forceinline__ void get_f12_4body(
  const double d12,
  const double d12inv,
  const double fn,
  const double fnp,
  const double Fp,
  const double* s,
  const double* r12,
  double* f12,
  double* f12d,
  const int n1,
  const int n2)
{
  // rij
  double ds0_r = (fnp * (3.0 * r12[2] * r12[2]- d12 * d12) - fn * 2.0 * d12);
  double ds1_r = fnp * r12[0] * r12[2];
  double ds2_r = fnp * r12[1] * r12[2];
  double ds3_r = fnp * (r12[0] * r12[0] - r12[1] * r12[1]);
  double ds4_r = fnp * 2.0 * r12[0] * r12[1];

  double s02 = s[0] * s[0];
  double s12 = s[1] * s[1];
  double s22 = s[2] * s[2];
  double s32 = s[3] * s[3];
  double s42 = s[4] * s[4];

  double tmp = 3.0 * C4B[0] * s02 * ds0_r +
              C4B[1] * ds0_r * (s12 + s22) + C4B[1] * s[0] * (2.0 * s[1] * ds1_r + 2.0 * s[2] * ds2_r) +
              C4B[2] * ds0_r * (s32 + s42) + C4B[2] * s[0] * (2.0 * s[3] * ds3_r + 2.0 * s[4] * ds4_r) +
              C4B[3] * ds3_r * (s22 - s12) + C4B[3] * s[3] * (2.0 * s[2] * ds2_r - 2.0 * s[1] * ds1_r) +
              C4B[4] *(ds1_r * s[2] * s[4] + s[1] * ds2_r * s[4] + s[1] * s[2] * ds4_r);
  f12[3] += Fp * tmp;
  f12d[3] += tmp;
  //xij
  ds0_r = 0.0;
  ds1_r = fn * r12[2];
  ds2_r = 0.0;
  ds3_r = fn * 2.0 * r12[0];
  ds4_r = fn * 2.0 * r12[1];

  double tmpx = 3.0 * C4B[0] * s02 * ds0_r +
              C4B[1] * ds0_r * (s12 + s22) + C4B[1] * s[0] * (2.0 * s[1] * ds1_r + 2.0 * s[2] * ds2_r) +
              C4B[2] * ds0_r * (s32 + s42) + C4B[2] * s[0] * (2.0 * s[3] * ds3_r + 2.0 * s[4] * ds4_r) +
              C4B[3] * ds3_r * (s22 - s12) + C4B[3] * s[3] * (2.0 * s[2] * ds2_r - 2.0 * s[1] * ds1_r) +
              C4B[4] *(ds1_r * s[2] * s[4] + s[1] * ds2_r * s[4] + s[1] * s[2] * ds4_r);
  f12[0] += Fp * tmpx;
  f12d[0] += tmpx;

  //yij
  ds0_r = 0.0;
  ds1_r = 0.0;
  ds2_r = fn * r12[2];
  ds3_r = fn * (-2.0 * r12[1]);
  ds4_r = fn * 2.0 * r12[0];
  double tmpy = 3.0 * C4B[0] * s02 * ds0_r +
              C4B[1] * ds0_r * (s12 + s22) + C4B[1] * s[0] * (2.0 * s[1] * ds1_r + 2.0 * s[2] * ds2_r) +
              C4B[2] * ds0_r * (s32 + s42) + C4B[2] * s[0] * (2.0 * s[3] * ds3_r + 2.0 * s[4] * ds4_r) +
              C4B[3] * ds3_r * (s22 - s12) + C4B[3] * s[3] * (2.0 * s[2] * ds2_r - 2.0 * s[1] * ds1_r) +
              C4B[4] *(ds1_r * s[2] * s[4] + s[1] * ds2_r * s[4] + s[1] * s[2] * ds4_r);
  f12[1] += Fp * tmpy;
  f12d[1] += tmpy;

  //zij
  ds0_r = fn * 6.0 * r12[2];
  ds1_r = fn * r12[0];
  ds2_r = fn * r12[1];
  ds3_r = 0.0;
  ds4_r = 0.0;
  double tmpz = 3.0 * C4B[0] * s02 * ds0_r +
              C4B[1] * ds0_r * (s12 + s22) + C4B[1] * s[0] * (2.0 * s[1] * ds1_r + 2.0 * s[2] * ds2_r) +
              C4B[2] * ds0_r * (s32 + s42) + C4B[2] * s[0] * (2.0 * s[3] * ds3_r + 2.0 * s[4] * ds4_r) +
              C4B[3] * ds3_r * (s22 - s12) + C4B[3] * s[3] * (2.0 * s[2] * ds2_r - 2.0 * s[1] * ds1_r) +
              C4B[4] *(ds1_r * s[2] * s[4] + s[1] * ds2_r * s[4] + s[1] * s[2] * ds4_r);
  f12[2] += Fp * tmpz;
  f12d[2] += tmpz;

  // if (n1==0 and n2==0){
  //   printf("\t4bL=2 n1=%d n2=%d s0=%f s1=%f s2=%f s3=%f s4=%f fnp=%f fn=%f Fp=%f r12=%f x=%f y=%f z=%f dqr=%f dqx=%f dqy=%f dqz=%f\n",
  //           n1, n2, s[0], s[1], s[2], s[3], s[4], fnp, fn, Fp, d12, r12[0], r12[1], r12[2], tmp, tmpx, tmpy, tmpz);
  // }
}

static __device__ __forceinline__ void get_f12_5body(
  const double d12,
  const double d12inv,
  const double fn,
  const double fnp,
  const double Fp,
  const double* s,
  const double* r12,
  double* f12,
  double* f12d,
  const int n1,
  const int n2)
{
  //rij
  double ds0_r = fnp * r12[2];
  double ds1_r = fnp * r12[0];
  double ds2_r = fnp * r12[1];

  double s02 = s[0] * s[0];
  double s12 = s[1] * s[1];
  double s22 = s[2] * s[2];
  double s12ms22 = s12 + s22;

  double tmp = 4.0 * C5B[0] * s02 * s[0] * ds0_r + 2.0 * C5B[1] * s[0] * ds0_r * s12ms22 +
              C5B[1] * s02 * 2.0 * (s[1] * ds1_r + s[2] * ds2_r) +
              4.0 * C5B[2] * s12ms22 * (s[1] * ds1_r + s[2] * ds2_r);
  f12[3] += Fp * tmp;
  f12d[3] += tmp;
  //xij
  // ds0_r = 0.0;
  ds1_r = fn;
  // ds2_r = 0.0;
  double tmpx = C5B[1] * s02 * 2.0 * (s[1] * ds1_r) + 4.0 * C5B[2] * s12ms22 * (s[1] * ds1_r);
  f12[0] += Fp * tmpx;
  f12d[0] += tmpx;

  //yij
  // ds0_r = 0.0;
  // ds1_r = 0.0;
  ds2_r = fn;
  double tmpy = C5B[1] * s02 * 2.0 * s[2] * ds2_r + 4.0 * C5B[2] * s12ms22 * s[2] * ds2_r;
  f12[1] += Fp * tmpy;
  f12d[1] += tmpy;

  //zij
  ds0_r = fn;
  // ds1_r = 0.0;
  // ds2_r = 0.0;
  double tmpz = 4.0 * C5B[0] * s02 * s[0] * ds0_r + 2.0 * C5B[1] * s[0] * ds0_r * s12ms22;
  f12[2] += Fp * tmpz;
  f12d[2] += tmpz;

  // if (n1==0 and n2==0){
  //   printf("\t5bL=2 n1=%d n2=%d s0=%f s1=%f s2=%f s3=%f s4=%f fnp=%f fn=%f Fp=%f r12=%f x=%f y=%f z=%f dqr=%f dqx=%f dqy=%f dqz=%f\n",
  //           n1, n2, s[0], s[1], s[2], s[3], s[4], fnp, fn, Fp, d12, r12[0], r12[1], r12[2], tmp, tmpx, tmpy, tmpz);
  // }
}

static __device__ __forceinline__ void get_f12_3(
  const double d12,
  const double d12inv,
  const double fn,
  const double fnp,
  const double Fp,
  const double* s,
  const double* r12,
  double* f12,
  double* f12d,
  const int n1,
  const int n2)
{
  double r2 = d12 * d12;
  double x2 = r12[0] * r12[0];
  double y2 = r12[1] * r12[1];
  double z2 = r12[2] * r12[2];
  double xy = r12[0] * r12[1];
  double xz = r12[0] * r12[2];
  double yz = r12[1] * r12[2];
  double x = r12[0];
  double y = r12[1];
  double z = r12[2];

  double tmp = s[0] * (fnp * (5.0 * z2 - 3.0 * r2) * z + fn * (-6.0 * z * d12)) +
    2.0 * (
          s[1] * (fnp * (5.0 * z2 - r2) * x + fn * (-2.0 * x * d12)) +
          s[2] * (fnp * (5.0 * z2 - r2) * y + fn * (-2.0 * y * d12)) +
          s[3] * (fnp * (x2 - y2) * z) +
          s[4] * (fnp * 2.0 * xy * z) +
          s[5] * (fnp * (x2 - 3.0 * y2) * x) +
          s[6] * (fnp * (3.0 * x2 - y2) * y)
          );
  tmp = tmp * 2.0;
  f12[3] += Fp * tmp;
  f12d[3] += tmp;
  // dx
  double tmpx = 2.0 * ( // 2.0  是clm少的2倍
          s[1] * (5.0 * z2 - r2) +
          s[3] * 2.0 * xz +
          s[4] * 2.0 * yz +
          s[5] * 3.0 * (x2 - y2) +
          s[6] * 6.0 * xy
          );
  tmpx = tmpx * 2.0 * fn;
  f12[0] += Fp * tmpx;
  f12d[0] += tmpx;

  //dy
  double tmpy = 2.0 * (
        s[2] * (5.0 * z2 - r2) +
        s[3] * (-2.0 * yz) +
        s[4] * 2.0 * xz -
        s[5] * 6.0 * xy +
        s[6] * 3.0 * (x2 - y2)
  );
  tmpy = tmpy * 2.0 * fn;
  f12[1] += Fp * tmpy;
  f12d[1] += tmpy;
  //dz
  double tmpz = s[0] * (15 * z2 - 3 * r2) +
      2.0 * (
        s[1] * 10.0 * xz +
        s[2] * 10.0 * yz +
        s[3] * (x2 - y2) +
        s[4] * 2.0 * xy
    );
  tmpz = tmpz * 2.0 * fn;
  f12[2] += Fp * tmpz;
  f12d[2] += tmpz;

  // if(n1==0 and n2==0){
  //   printf("\tL=3 n1=%d n2=%d s0=%f s1=%f s2=%f s3=%f s4=%f s5=%f s6=%f fnp=%f fn=%f Fp=%f r12=%f x=%f y=%f z=%f dqr=%f dqx=%f dqy=%f dqz=%f\n",
  //           n1, n2, s[0], s[1]*2.0, s[2]*2.0, s[3]*2.0, s[4]*2.0, s[5]*2.0, s[6]*2.0, fnp, fn, Fp, d12, r12[0], r12[1], r12[2], tmp, tmpx, tmpy, tmpz);
  // }

}

static __device__ __forceinline__ void get_f12_4(
  const double x,
  const double y,
  const double z,
  const double r,
  const double rinv,
  const double fn,
  const double fnp,
  const double Fp,
  const double* s,
  double* f12,
  double* f12d,
  const int n1,
  const int n2)
{
  const double r2 = r * r;
  const double x2 = x * x;
  const double y2 = y * y;
  const double z2 = z * z;
  const double xy = x * y;
  const double xz = x * z;
  const double yz = y * z;
  const double xyz = x * yz;
  const double x2my2 = x2 - y2;
  const double x3 = x * x2;
  const double y3 = y * y2;
  const double z3 = z * z2;
  const double r3 = r * r2;
  // rij
  double tmp = s[0] * (fnp * ((35.0 * z2 - 30.0 * r2) * z2 + 3.0 * r2 * r2) +
                        fn * ((-60.0) * z2 * r + 12.0 * r3)) +
              2.0 * (
                s[1] * (fnp * (7.0 * z2 - 3.0 * r2) * xz - fn * 6.0 * xz * r) +
                s[2] * (fnp * (7.0 * z2 - 3.0 * r2) * yz - fn * 6.0 * yz * r) +
                s[3] * (fnp * (7.0 * z2 - r2) * x2my2  + fn * 2.0 * r * (y2 - x2)) +
                s[4] * (fnp * (7.0 * z2 - r2) * 2.0 * xy - fn * 4.0 * xy * r) +
                s[5] * (fnp * (x2 - 3.0 * y2) * xz) +
                s[6] * (fnp * (3.0 * x2 - y2) * yz) +
                s[7] * (fnp * (x2my2 * x2my2 - 4.0 * x2 * y2)) +
                s[8] * (fnp * 4.0 * x2my2 * xy)
              );
  tmp = tmp * 2.0;
  f12[3] += Fp * tmp;
  f12d[3] += tmp;
  // x
  double tmpx = 2.0 * (
              s[1] * (7.0 * z3 - 3.0 * z * r2) +
              s[3] * (14.0 * x * z2 - 2.0 * x * r2) +
              s[4] * 2.0 * y * (7.0 * z2 - r2) +
              s[5] * 3.0 * z * (x2 - y2) +
              s[6] * 6.0 * xyz +
              s[7] * (4.0 * x3 - 12.0 * x * y2) +
              s[8] * (12.0 * x2 * y - 4.0 * y3)
  );
  tmpx = 2.0 * fn * tmpx;
  f12[0] += Fp * tmpx;
  f12d[0] += tmpx;

  double tmpy = 2.0 * (
              s[2] * (7.0 * z3 - 3.0 * z * r2) +
              s[3] * 2.0 * y *(r2 - 7.0 * z2) +
              s[4] * 2.0 * x * (7.0 * z2 - r2) -
              s[5] * (6.0) * xyz +
              s[6] * (3.0 * x2 * z - 3.0 * y2 * z) +
              s[7] * (4.0 * y3 - 12.0 * x2 * y) +
              s[8] * (4.0 * x3 - 12.0 * x * y2)
  );
  tmpy = 2.0 * fn * tmpy;
  f12[1] += Fp * tmpy;
  f12d[1] += tmpy;

  double tmpz = s[0] * (140.0 * z3 - 60.0 * z * r2) +
        2.0 * (
            s[1] * (21.0 * x * z2 - 3.0 * x * r2) +
            s[2] * (21.0 * y * z2 - 3.0 * y * r2) +
            s[3] * (14.0 * x2 * z - 14.0 * y2 * z) +
            s[4] * 28.0 * xyz +
            s[5] * (x3 - 3.0 * x * y2) +
            s[6] * (3.0 * y * x2 - y3)
        );
  tmpz = 2.0 * fn * tmpz;
  f12[2] += Fp * tmpz;
  f12d[2] += tmpz;

  // if(n1==0 and n2==0){
  //   printf("\tL=4 n1=%d n2=%d s0=%f s1=%f s2=%f s3=%f s4=%f s5=%f s6=%f s7=%f s8=%f fnp=%f fn=%f Fp=%f r12=%f x=%f y=%f z=%f dqr=%f dqx=%f dqy=%f dqz=%f\n\n\n",
  //           n1, n2, s[0], s[1]*2.0, s[2]*2.0, s[3]*2.0, s[4]*2.0, s[5]*2.0, s[6]*2.0, s[7]*2.0, s[8]*2.0, fnp, fn, Fp, r, x, y, z, tmp, tmpx, tmpy, tmpz);
  // }
}

static __device__ __forceinline__ void accumulate_f12(
  const int n,
  const double d12,
  const double* r12,
  double fn,
  double fnp,
  const double* Fp,
  const double* sum_fxyz,
  const double* s_rij_blm,
  double* f12,
  double* f12d,
  double* dfeat_c3,
  double* fn12,
  double* fnp12,
  const int type_j,
  const int ntypes,
  const int lmax_3,
  const int n_max_angular,
  const int n_base_angular,
  const int dc_start_idx,
  const int n1,
  const int n2) // i-> [ntype, nmax, nbase]-> [ntyp, ]
{
  const double d12inv = 1.0 / d12;
  // l = 1
  // double gn12 = fn; //for dc
  // double gn12p = fnp;
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  double s1[3] = {
    sum_fxyz[n * NUM_OF_ABC + 0] * C3B[0],
    sum_fxyz[n * NUM_OF_ABC + 1] * C3B[1],
    sum_fxyz[n * NUM_OF_ABC + 2] * C3B[2]};
  get_f12_1(d12inv, fn, fnp, Fp[n*lmax_3], s1, r12, f12, f12d, n1, n2);

  // l = 2
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  double s2[5] = {
    sum_fxyz[n * NUM_OF_ABC + 3] * C3B[3],
    sum_fxyz[n * NUM_OF_ABC + 4] * C3B[4],
    sum_fxyz[n * NUM_OF_ABC + 5] * C3B[5],
    sum_fxyz[n * NUM_OF_ABC + 6] * C3B[6],
    sum_fxyz[n * NUM_OF_ABC + 7] * C3B[7]};
  get_f12_2(d12, d12inv, fn, fnp, Fp[n*lmax_3+1], s2, r12, f12, f12d+4, n1, n2);
  // l = 3
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  double s3[7] = {
    sum_fxyz[n * NUM_OF_ABC + 8] * C3B[8],
    sum_fxyz[n * NUM_OF_ABC + 9] * C3B[9],
    sum_fxyz[n * NUM_OF_ABC + 10] * C3B[10],
    sum_fxyz[n * NUM_OF_ABC + 11] * C3B[11],
    sum_fxyz[n * NUM_OF_ABC + 12] * C3B[12],
    sum_fxyz[n * NUM_OF_ABC + 13] * C3B[13],
    sum_fxyz[n * NUM_OF_ABC + 14] * C3B[14]};
  get_f12_3(d12, d12inv, fn, fnp, Fp[n*lmax_3+2], s3, r12, f12, f12d+8, n1, n2);
  // l = 4
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
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
  get_f12_4(
    r12[0], r12[1], r12[2], d12, d12inv, fn, fnp, Fp[n*lmax_3+3], s4, f12, f12d+12, n1, n2);
  for(int kk=0; kk < n_base_angular; ++kk) {
    // l = 1
    double tmp1 = s1[0] * s_rij_blm[0] +
                  s1[1] * s_rij_blm[1] * 2.0 +
                  s1[2] * s_rij_blm[2] * 2.0;
    // l = 2
    double tmp2 =
                  s2[0] * s_rij_blm[3] +
           2.0 * (s2[1] * s_rij_blm[4] +
                  s2[2] * s_rij_blm[5] +
                  s2[3] * s_rij_blm[6] +
                  s2[4] * s_rij_blm[7] );
    // l = 3
    double tmp3 =
                  s3[0] * s_rij_blm[8] +
           2.0 * (s3[1] * s_rij_blm[9] +
                  s3[2] * s_rij_blm[10] +
                  s3[3] * s_rij_blm[11] +
                  s3[4] * s_rij_blm[12] +
                  s3[5] * s_rij_blm[13] +
                  s3[6] * s_rij_blm[14] );
    // l = 4
    double tmp4 = s4[0] * s_rij_blm[15] +
           2.0 * (s4[1] * s_rij_blm[16] +
                  s4[2] * s_rij_blm[17] +
                  s4[3] * s_rij_blm[18] +
                  s4[4] * s_rij_blm[19] +
                  s4[5] * s_rij_blm[20] +
                  s4[6] * s_rij_blm[21] +
                  s4[7] * s_rij_blm[22] +
                  s4[8] * s_rij_blm[23] );

    tmp1 = Fp[n*lmax_3]*tmp1 + Fp[n*lmax_3+1]*tmp2 + Fp[n*lmax_3+2]*tmp3 + Fp[n*lmax_3+3]*tmp4;
    tmp1 = tmp1 * 2.0 * fn12[kk];
    int dc_id = dc_start_idx + type_j * n_max_angular * n_base_angular + n*n_base_angular + kk;
    dfeat_c3[dc_id] += tmp1;
  }
}

static __device__ __forceinline__ void accumulate_f12_with_4body(
  const int n,
  const double d12,
  const double* r12,
  double fn,
  double fnp,
  const double* Fp,
  const double* sum_fxyz,
  const double* s_rij_blm,
  double* f12,
  double* f12d,
  double* dfeat_c3,
  double* fn12,
  double* fnp12,
  const int type_j,
  const int ntypes,
  const int lmax_3,
  const int n_max_angular,
  const int n_base_angular,
  const int dc_start_idx,
  const int n1,
  const int n2)
{
  const double d12inv = 1.0 / d12;
  // l = 1
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  double s1[3] = {
    sum_fxyz[n * NUM_OF_ABC + 0] * C3B[0],
    sum_fxyz[n * NUM_OF_ABC + 1] * C3B[1],
    sum_fxyz[n * NUM_OF_ABC + 2] * C3B[2]};
  get_f12_1(d12inv, fn, fnp, Fp[n*lmax_3], s1, r12, f12, f12d, n1, n2);
  // l = 2
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  double s2[5] = {
    sum_fxyz[n * NUM_OF_ABC + 3],
    sum_fxyz[n * NUM_OF_ABC + 4],
    sum_fxyz[n * NUM_OF_ABC + 5],
    sum_fxyz[n * NUM_OF_ABC + 6],
    sum_fxyz[n * NUM_OF_ABC + 7]};

  double s24b[5] = {
    sum_fxyz[n * NUM_OF_ABC + 3],
    sum_fxyz[n * NUM_OF_ABC + 4],
    sum_fxyz[n * NUM_OF_ABC + 5],
    sum_fxyz[n * NUM_OF_ABC + 6],
    sum_fxyz[n * NUM_OF_ABC + 7]};

  double s24bsq[5] = {
    s24b[0] * s24b[0],
    s24b[1] * s24b[1],
    s24b[2] * s24b[2],
    s24b[3] * s24b[3],
    s24b[4] * s24b[4]};

  get_f12_4body(d12, d12inv, fn, fnp, Fp[n_max_angular * lmax_3 + n], s2, r12, f12, f12d + 16,n1, n2);
  s2[0] *= C3B[3];
  s2[1] *= C3B[4];
  s2[2] *= C3B[5];
  s2[3] *= C3B[6];
  s2[4] *= C3B[7];
  get_f12_2(d12, d12inv, fn, fnp, Fp[n*lmax_3+1], s2, r12, f12, f12d + 4,  n1, n2);
  // l = 3
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  double s3[7] = {
    sum_fxyz[n * NUM_OF_ABC + 8] * C3B[8],
    sum_fxyz[n * NUM_OF_ABC + 9] * C3B[9],
    sum_fxyz[n * NUM_OF_ABC + 10] * C3B[10],
    sum_fxyz[n * NUM_OF_ABC + 11] * C3B[11],
    sum_fxyz[n * NUM_OF_ABC + 12] * C3B[12],
    sum_fxyz[n * NUM_OF_ABC + 13] * C3B[13],
    sum_fxyz[n * NUM_OF_ABC + 14] * C3B[14]};
  get_f12_3(d12, d12inv, fn, fnp, Fp[n*lmax_3+2], s3, r12, f12, f12d + 8, n1, n2);
  // l = 4
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
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
  get_f12_4(
    r12[0], r12[1], r12[2], d12, d12inv, fn, fnp, Fp[n*lmax_3+3], s4, f12, f12d + 12, n1, n2);

  // for c3 param
  double ds0_r = 0.0;
  double ds1_r = 0.0;
  double ds2_r = 0.0;
  double ds3_r = 0.0;
  double ds4_r = 0.0;
  for(int kk=0; kk < n_base_angular; ++kk) {
    // l = 1
    double tmp1 = s1[0] * s_rij_blm[0] +
                  s1[1] * s_rij_blm[1] * 2.0 +
                  s1[2] * s_rij_blm[2] * 2.0;

    // l = 2
    ds0_r = s_rij_blm[3] * fn12[kk];
    ds1_r = s_rij_blm[4] * fn12[kk];
    ds2_r = s_rij_blm[5] * fn12[kk];
    ds3_r = s_rij_blm[6] * fn12[kk];
    ds4_r = s_rij_blm[7] * fn12[kk];

    double tmp2_4b = 3.0 * C4B[0] * s24bsq[0] * ds0_r +
              C4B[1] * ds0_r * (s24bsq[1] + s24bsq[2]) + C4B[1] * s24b[0] * (2.0 * s24b[1] * ds1_r + 2.0 * s24b[2] * ds2_r) +
              C4B[2] * ds0_r * (s24bsq[3] + s24bsq[4]) + C4B[2] * s24b[0] * (2.0 * s24b[3] * ds3_r + 2.0 * s24b[4] * ds4_r) +
              C4B[3] * ds3_r * (s24bsq[2] - s24bsq[1]) + C4B[3] * s24b[3] * (2.0 * s24b[2] * ds2_r - 2.0 * s24b[1] * ds1_r) +
              C4B[4] *(ds1_r * s24b[2] * s24b[4] + s24b[1] * ds2_r * s24b[4] + s24b[1] * s24b[2] * ds4_r);

    double tmp2 =
                  s2[0] * s_rij_blm[3] +
           2.0 * (s2[1] * s_rij_blm[4] +
                  s2[2] * s_rij_blm[5] +
                  s2[3] * s_rij_blm[6] +
                  s2[4] * s_rij_blm[7] );

    // l = 3
    double tmp3 =
                  s3[0] * s_rij_blm[8] +
           2.0 * (s3[1] * s_rij_blm[9] +
                  s3[2] * s_rij_blm[10] +
                  s3[3] * s_rij_blm[11] +
                  s3[4] * s_rij_blm[12] +
                  s3[5] * s_rij_blm[13] +
                  s3[6] * s_rij_blm[14] );
    // l = 4
    double tmp4 = s4[0] * s_rij_blm[15] +
           2.0 * (s4[1] * s_rij_blm[16] +
                  s4[2] * s_rij_blm[17] +
                  s4[3] * s_rij_blm[18] +
                  s4[4] * s_rij_blm[19] +
                  s4[5] * s_rij_blm[20] +
                  s4[6] * s_rij_blm[21] +
                  s4[7] * s_rij_blm[22] +
                  s4[8] * s_rij_blm[23] );

    tmp1 = Fp[n*lmax_3]*tmp1 + Fp[n*lmax_3+1]*tmp2 + Fp[n*lmax_3+2]*tmp3 + Fp[n*lmax_3+3]*tmp4;
    tmp1 = tmp1 * 2.0 * fn12[kk];
    tmp1 = tmp1 + tmp2_4b * Fp[n_max_angular * lmax_3 + n];
    int dc_id = dc_start_idx + type_j * n_max_angular * n_base_angular + n*n_base_angular + kk;
    dfeat_c3[dc_id] += tmp1;
  }
}

static __device__ __forceinline__ void accumulate_f12_with_5body(
  const int n,
  const double d12,
  const double* r12,
  double fn,
  double fnp,
  const double* Fp,
  const double* sum_fxyz,
  const double* s_rij_blm,
  double* f12,
  double* f12d,
  double* dfeat_c3,
  double* fn12,
  double* fnp12,
  const int type_j,
  const int ntypes,
  const int lmax_3,
  const int n_max_angular,
  const int n_base_angular,
  const int dc_start_idx,
  const int n1,
  const int n2)
{
  const double d12inv = 1.0 / d12;
  // l = 1
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  double s1[3] = {
    sum_fxyz[n * NUM_OF_ABC + 0], sum_fxyz[n * NUM_OF_ABC + 1], sum_fxyz[n * NUM_OF_ABC + 2]};
  double s15b[3] = {
    sum_fxyz[n * NUM_OF_ABC + 0], sum_fxyz[n * NUM_OF_ABC + 1], sum_fxyz[n * NUM_OF_ABC + 2]};
  double s15bsq[3] = {s15b[0] * s15b[0], s15b[1] * s15b[1], s15b[2] * s15b[2]};

  get_f12_5body(d12, d12inv, fn, fnp, Fp[n_max_angular * lmax_3 + n_max_angular + n], s1, r12, f12, f12d+20, n1, n2);

  s1[0] *= C3B[0];
  s1[1] *= C3B[1];
  s1[2] *= C3B[2];
  get_f12_1(d12inv, fn, fnp, Fp[n*lmax_3], s1, r12, f12, f12d, n1, n2);

  // l = 2
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  double s2[5] = {
    sum_fxyz[n * NUM_OF_ABC + 3],
    sum_fxyz[n * NUM_OF_ABC + 4],
    sum_fxyz[n * NUM_OF_ABC + 5],
    sum_fxyz[n * NUM_OF_ABC + 6],
    sum_fxyz[n * NUM_OF_ABC + 7]};

  double s24b[5] = {
    sum_fxyz[n * NUM_OF_ABC + 3],
    sum_fxyz[n * NUM_OF_ABC + 4],
    sum_fxyz[n * NUM_OF_ABC + 5],
    sum_fxyz[n * NUM_OF_ABC + 6],
    sum_fxyz[n * NUM_OF_ABC + 7]};

  double s24bsq[5] = {
    s24b[0] * s24b[0],
    s24b[1] * s24b[1],
    s24b[2] * s24b[2],
    s24b[3] * s24b[3],
    s24b[4] * s24b[4]};

  get_f12_4body(d12, d12inv, fn, fnp, Fp[n_max_angular * lmax_3 + n], s2, r12, f12, f12d+16, n1, n2);
  s2[0] *= C3B[3];
  s2[1] *= C3B[4];
  s2[2] *= C3B[5];
  s2[3] *= C3B[6];
  s2[4] *= C3B[7];
  get_f12_2(d12, d12inv, fn, fnp, Fp[n*lmax_3+1], s2, r12, f12, f12d+4, n1, n2);

  // l = 3
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  double s3[7] = {
    sum_fxyz[n * NUM_OF_ABC + 8] * C3B[8],
    sum_fxyz[n * NUM_OF_ABC + 9] * C3B[9],
    sum_fxyz[n * NUM_OF_ABC + 10] * C3B[10],
    sum_fxyz[n * NUM_OF_ABC + 11] * C3B[11],
    sum_fxyz[n * NUM_OF_ABC + 12] * C3B[12],
    sum_fxyz[n * NUM_OF_ABC + 13] * C3B[13],
    sum_fxyz[n * NUM_OF_ABC + 14] * C3B[14]};
  get_f12_3(d12, d12inv, fn, fnp, Fp[n*lmax_3+2], s3, r12, f12, f12d+8, n1, n2);

  // l = 4
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
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
  get_f12_4(
    r12[0], r12[1], r12[2], d12, d12inv, fn, fnp, Fp[n*lmax_3+3], s4, f12, f12d+12, n1, n2);

  // for c3 param
  double ds0_r = 0.0;
  double ds1_r = 0.0;
  double ds2_r = 0.0;
  double ds3_r = 0.0;
  double ds4_r = 0.0;
  double ds5b0_r = 0.0;
  double ds5b1_r = 0.0;
  double ds5b2_r = 0.0;
  for(int kk=0; kk < n_base_angular; ++kk) {
    // l = 1
    double tmp1 = s1[0] * s_rij_blm[0] +
                  s1[1] * s_rij_blm[1] * 2.0 +
                  s1[2] * s_rij_blm[2] * 2.0;
    // l = 1 with 5b
    ds5b0_r = s_rij_blm[0] * fn12[kk];
    ds5b1_r = s_rij_blm[1] * fn12[kk];
    ds5b2_r = s_rij_blm[2] * fn12[kk];

    double tmp1_5b = 4.0 * C5B[0] * s15bsq[0] * s15b[0] * ds5b0_r + 2.0 * C5B[1] * s15b[0] * ds5b0_r * (s15bsq[1] + s15bsq[2]) +
            C5B[1] * s15bsq[0] * 2.0 * (s15b[1] * ds5b1_r + s15b[2] * ds5b2_r) +
            4.0 * C5B[2] * (s15bsq[1] + s15bsq[2]) * (s15b[1] * ds5b1_r + s15b[2] * ds5b2_r);

    // l = 2 with 4b
    ds0_r = s_rij_blm[3] * fn12[kk];
    ds1_r = s_rij_blm[4] * fn12[kk];
    ds2_r = s_rij_blm[5] * fn12[kk];
    ds3_r = s_rij_blm[6] * fn12[kk];
    ds4_r = s_rij_blm[7] * fn12[kk];
    double tmp2_4b = 3.0 * C4B[0] * s24bsq[0] * ds0_r +
              C4B[1] * ds0_r * (s24bsq[1] + s24bsq[2]) + C4B[1] * s24b[0] * (2.0 * s24b[1] * ds1_r + 2.0 * s24b[2] * ds2_r) +
              C4B[2] * ds0_r * (s24bsq[3] + s24bsq[4]) + C4B[2] * s24b[0] * (2.0 * s24b[3] * ds3_r + 2.0 * s24b[4] * ds4_r) +
              C4B[3] * ds3_r * (s24bsq[2] - s24bsq[1]) + C4B[3] * s24b[3] * (2.0 * s24b[2] * ds2_r - 2.0 * s24b[1] * ds1_r) +
              C4B[4] *(ds1_r * s24b[2] * s24b[4] + s24b[1] * ds2_r * s24b[4] + s24b[1] * s24b[2] * ds4_r);

    double tmp2 =
                  s2[0] * s_rij_blm[3] +
           2.0 * (s2[1] * s_rij_blm[4] +
                  s2[2] * s_rij_blm[5] +
                  s2[3] * s_rij_blm[6] +
                  s2[4] * s_rij_blm[7] );

    // l = 3
    double tmp3 =
                  s3[0] * s_rij_blm[8] +
           2.0 * (s3[1] * s_rij_blm[9] +
                  s3[2] * s_rij_blm[10] +
                  s3[3] * s_rij_blm[11] +
                  s3[4] * s_rij_blm[12] +
                  s3[5] * s_rij_blm[13] +
                  s3[6] * s_rij_blm[14] );
    // l = 4
    double tmp4 = s4[0] * s_rij_blm[15] +
           2.0 * (s4[1] * s_rij_blm[16] +
                  s4[2] * s_rij_blm[17] +
                  s4[3] * s_rij_blm[18] +
                  s4[4] * s_rij_blm[19] +
                  s4[5] * s_rij_blm[20] +
                  s4[6] * s_rij_blm[21] +
                  s4[7] * s_rij_blm[22] +
                  s4[8] * s_rij_blm[23] );

    tmp1 = Fp[n*lmax_3]*tmp1 + Fp[n*lmax_3+1]*tmp2 + Fp[n*lmax_3+2]*tmp3 + Fp[n*lmax_3+3]*tmp4;
    tmp1 = tmp1 * 2.0 * fn12[kk];
    tmp1 = tmp1 + tmp2_4b * Fp[n_max_angular * lmax_3 + n];
    tmp1 = tmp1 + tmp1_5b * Fp[n_max_angular * lmax_3 + n_max_angular + n];
    int dc_id = dc_start_idx + type_j * n_max_angular * n_base_angular + n*n_base_angular + kk;
    dfeat_c3[dc_id] += tmp1;
  }
}

static __device__ __forceinline__ void
accumulate_s(const double d12, double x12, double y12, double z12, const double fn, double* s)
{
  double d12inv = 1.0 / d12;
  x12 *= d12inv;
  y12 *= d12inv;
  z12 *= d12inv;
  double x12sq = x12 * x12;
  double y12sq = y12 * y12;
  double z12sq = z12 * z12;
  double x12sq_minus_y12sq = x12sq - y12sq;
  s[0] += z12 * fn;                                                            // Y10
  s[1] += x12 * fn;                                                            // Y11_real
  s[2] += y12 * fn;                                                            // Y11_imag
  s[3] += (3.0 * z12sq - 1.0) * fn;                                            // Y20
  s[4] += x12 * z12 * fn;                                                      // Y21_real
  s[5] += y12 * z12 * fn;                                                      // Y21_imag
  s[6] += x12sq_minus_y12sq * fn;                                              // Y22_real
  s[7] += 2.0 * x12 * y12 * fn;                                                // Y22_imag
  s[8] += (5.0 * z12sq - 3.0) * z12 * fn;                                      // Y30
  s[9] += (5.0 * z12sq - 1.0) * x12 * fn;                                      // Y31_real
  s[10] += (5.0 * z12sq - 1.0) * y12 * fn;                                     // Y31_imag
  s[11] += x12sq_minus_y12sq * z12 * fn;                                       // Y32_real
  s[12] += 2.0 * x12 * y12 * z12 * fn;                                         // Y32_imag
  s[13] += (x12 * x12 - 3.0 * y12 * y12) * x12 * fn;                           // Y33_real
  s[14] += (3.0 * x12 * x12 - y12 * y12) * y12 * fn;                           // Y33_imag
  s[15] += ((35.0 * z12sq - 30.0) * z12sq + 3.0) * fn;                         // Y40
  s[16] += (7.0 * z12sq - 3.0) * x12 * z12 * fn;                               // Y41_real
  s[17] += (7.0 * z12sq - 3.0) * y12 * z12 * fn;                               // Y41_iamg
  s[18] += (7.0 * z12sq - 1.0) * x12sq_minus_y12sq * fn;                       // Y42_real
  s[19] += (7.0 * z12sq - 1.0) * x12 * y12 * 2.0 * fn;                         // Y42_imag
  s[20] += (x12sq - 3.0 * y12sq) * x12 * z12 * fn;                             // Y43_real
  s[21] += (3.0 * x12sq - y12sq) * y12 * z12 * fn;                             // Y43_imag
  s[22] += (x12sq_minus_y12sq * x12sq_minus_y12sq - 4.0 * x12sq * y12sq) * fn; // Y44_real
  s[23] += (4.0 * x12 * y12 * x12sq_minus_y12sq) * fn;                         // Y44_imag
}

static __device__ __forceinline__ void
accumulate_blm_rij(const double d12,
                    const double x,
                    const double y,
                    const double z,
                    double* s)
{
  double d12inv = 1.0 / d12;
  double x12 = x*d12inv;
  double y12 = y * d12inv;
  double z12 = z * d12inv;
  double x12sq = x12 * x12;
  double y12sq = y12 * y12;
  double z12sq = z12 * z12;
  double x12sq_minus_y12sq = x12sq - y12sq;
  s[0] += z12;                                                            // Y10
  s[1] += x12;                                                            // Y11_real
  s[2] += y12;                                                            // Y11_imag
  s[3] += (3.0 * z12sq - 1.0);                                            // Y20
  s[4] += x12 * z12;                                                      // Y21_real
  s[5] += y12 * z12;                                                      // Y21_imag
  s[6] += x12sq_minus_y12sq;                                              // Y22_real
  s[7] += 2.0 * x12 * y12;                                                // Y22_imag
  s[8] += (5.0 * z12sq - 3.0) * z12;                                      // Y30
  s[9] += (5.0 * z12sq - 1.0) * x12;                                      // Y31_real
  s[10] += (5.0 * z12sq - 1.0) * y12;                                     // Y31_imag
  s[11] += x12sq_minus_y12sq * z12;                                       // Y32_real
  s[12] += 2.0 * x12 * y12 * z12;                                         // Y32_imag
  s[13] += (x12 * x12 - 3.0 * y12 * y12) * x12;                           // Y33_real
  s[14] += (3.0 * x12 * x12 - y12 * y12) * y12;                           // Y33_imag
  s[15] += ((35.0 * z12sq - 30.0) * z12sq + 3.0);                         // Y40
  s[16] += (7.0 * z12sq - 3.0) * x12 * z12;                               // Y41_real
  s[17] += (7.0 * z12sq - 3.0) * y12 * z12;                               // Y41_iamg
  s[18] += (7.0 * z12sq - 1.0) * x12sq_minus_y12sq;                       // Y42_real
  s[19] += (7.0 * z12sq - 1.0) * x12 * y12 * 2.0;                         // Y42_imag
  s[20] += (x12sq - 3.0 * y12sq) * x12 * z12;                             // Y43_real
  s[21] += (3.0 * x12sq - y12sq) * y12 * z12;                             // Y43_imag
  s[22] += (x12sq_minus_y12sq * x12sq_minus_y12sq - 4.0 * x12sq * y12sq); // Y44_real
  s[23] += (4.0 * x12 * y12 * x12sq_minus_y12sq);                    // Y44_imag
}

static __device__ __forceinline__ void
find_q(const int n_max_angular_plus_1, const int n, const double* s, double* q)
{
  q[n] = C3B[0] * s[0] * s[0] + 2.0 * (C3B[1] * s[1] * s[1] + C3B[2] * s[2] * s[2]);
  q[n_max_angular_plus_1 + n] =
    C3B[3] * s[3] * s[3] + 2.0 * (C3B[4] * s[4] * s[4] + C3B[5] * s[5] * s[5] +
                                   C3B[6] * s[6] * s[6] + C3B[7] * s[7] * s[7]);
  q[2 * n_max_angular_plus_1 + n] =
    C3B[8] * s[8] * s[8] +
    2.0 * (C3B[9] * s[9] * s[9] + C3B[10] * s[10] * s[10] + C3B[11] * s[11] * s[11] +
            C3B[12] * s[12] * s[12] + C3B[13] * s[13] * s[13] + C3B[14] * s[14] * s[14]);
  q[3 * n_max_angular_plus_1 + n] =
    C3B[15] * s[15] * s[15] +
    2.0 * (C3B[16] * s[16] * s[16] + C3B[17] * s[17] * s[17] + C3B[18] * s[18] * s[18] +
            C3B[19] * s[19] * s[19] + C3B[20] * s[20] * s[20] + C3B[21] * s[21] * s[21] +
            C3B[22] * s[22] * s[22] + C3B[23] * s[23] * s[23]);
}

static __device__ __forceinline__ void
find_q_with_4body(const int n_max_angular_plus_1, const int n, const double* s, double* q)
{
  find_q(n_max_angular_plus_1, n, s, q);
  q[4 * n_max_angular_plus_1 + n] =
    C4B[0] * s[3] * s[3] * s[3] + C4B[1] * s[3] * (s[4] * s[4] + s[5] * s[5]) +
    C4B[2] * s[3] * (s[6] * s[6] + s[7] * s[7]) + C4B[3] * s[6] * (s[5] * s[5] - s[4] * s[4]) +
    C4B[4] * s[4] * s[5] * s[7];
}

static __device__ __forceinline__ void
find_q_with_5body(const int n_max_angular_plus_1, const int n, const double* s, double* q)
{
  find_q_with_4body(n_max_angular_plus_1, n, s, q);
  double s0_sq = s[0] * s[0];
  double s1_sq_plus_s2_sq = s[1] * s[1] + s[2] * s[2];
  q[5 * n_max_angular_plus_1 + n] = C5B[0] * s0_sq * s0_sq + C5B[1] * s0_sq * s1_sq_plus_s2_sq +
                                    C5B[2] * s1_sq_plus_s2_sq * s1_sq_plus_s2_sq;
}
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
  // int k_start_id = type_j * n_base_angular * 4;
  const int k_start_id = 0;
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
  // int k_start_id = type_j * n_base_angular * 4;
  const int k_start_id = 0;
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

  // int k_start_id = type_j * n_base_angular * 4;
  const int k_start_id = 0;
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
  // int k_start_id = type_j * n_base_angular * 4;
  const int k_start_id = 0;
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
  // int k_start_id = type_j * n_base_angular * 4;
  const int k_start_id = 0;
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
  // int k_start_id = type_j * n_base_angular * 4;
  const int k_start_id = 0;
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
  const int j = type_j;
  // for (int j = 0; j < ntypes; j++){
  //   if (type_j == j) continue;
    int dsnlm_idx = dsnlm_start_idx + j * n_base_angular * NUM_OF_ABC;
    // int k_start_idx = j * n_base_angular * 4;
    const int k_start_idx = 0;
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
  // }
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
  const int j = type_j;
  // for (int j = 0; j < ntypes; j++){
  //   if (type_j == j) continue;
    int dsnlm_idx = dsnlm_start_idx + j * n_base_angular * NUM_OF_ABC;
    // int k_start_id = j * n_base_angular * 4;
    const int k_start_id = 0;
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
  // }
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
  const int j = type_j;
  // for (int j = 0; j < ntypes; j++){
  //   if (type_j == j) continue;
    int dsnlm_idx = dsnlm_start_idx + j * n_base_angular * NUM_OF_ABC;
    // int k_start_id = j * n_base_angular * 4;
    const int k_start_id = 0;
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
  // }
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
  const int j = type_j;
  // for (int j = 0; j < ntypes; j++){
  //   if (type_j == j) continue;
    int dsnlm_idx = dsnlm_start_idx + j * n_base_angular * NUM_OF_ABC;
    // int k_start_idx = j * n_base_angular * 4;
    const int k_start_idx = 0;
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
  // }
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
  const int j = type_j;
  // for (int j = 0; j < ntypes; j++){
  //   if (type_j == j) continue;
    int dsnlm_idx = dsnlm_start_idx + j * n_base_angular * NUM_OF_ABC;
    // int k_start_idx = j * n_base_angular * 4;
    const int k_start_idx = 0;
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
  // }
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
  const int j = type_j;
  // for (int j = 0; j < ntypes; j++){
  //   if (type_j == j) continue;
    int dsnlm_idx = dsnlm_start_idx + j * n_base_angular * NUM_OF_ABC;
    // int k_start_idx = j * n_base_angular * 4;
    const int k_start_idx = 0;
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
  // }
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
  const int n2, // i-> [ntype, nmax, nbase]-> [ntyp, ]
  const bool same_type = false)
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
  if (same_type) {
  scd_get_f12_1(fn12, fnp12,
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s1, r12,
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n*lmax_3],
                n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k);
  } else {
  scd_get_f12_1_J(fn12, fnp12,
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s1, r12,
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n*lmax_3],
                n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k);
  }

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

  if (same_type) {
  scd_get_f12_2(fn12, fnp12,
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s2, r12,
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n*lmax_3+1],
                n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k);
  } else {
  scd_get_f12_2_J(fn12, fnp12,
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s2, r12,
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n*lmax_3+1],
                n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k);
  }

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
  if (same_type) {
  scd_get_f12_3(fn12, fnp12,
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s3, r12,
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n*lmax_3+2],
                n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k);
  } else {
  scd_get_f12_3_J(fn12, fnp12,
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s3, r12,
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n*lmax_3+2],
                n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k);
  }

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
  if (same_type) {
  scd_get_f12_4(fn12, fnp12,
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s4, r12,
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n*lmax_3+3],
                n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k);
  } else {
  scd_get_f12_4_J(fn12, fnp12,
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s4, r12,
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n*lmax_3+3],
                n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k);
  }
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
  const int n2,
  const bool same_type = false)
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
  if (same_type) {
  scd_get_f12_1(fn12, fnp12,
              blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
              scd_r12, dsnlm_dc, s1, r12,
              d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n*lmax_3],
              n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k);
  } else {
  scd_get_f12_1_J(fn12, fnp12,
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s1, r12,
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n*lmax_3],
                n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k);
  }

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

  if (same_type) {
  scd_get_f12_4body(fn12, fnp12,
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s2, r12,
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n_max_angular * lmax_3 + n],
                n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k);
  } else {
  scd_get_f12_4body_J(fn12, fnp12,
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s2, r12,
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n_max_angular * lmax_3 + n],
                n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k);
  }

  s2[0] *= C3B[3];
  s2[1] *= C3B[4];
  s2[2] *= C3B[5];
  s2[3] *= C3B[6];
  s2[4] *= C3B[7];
  if (same_type) {
  scd_get_f12_2(fn12, fnp12,
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s2, r12,
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n*lmax_3+1],
                n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k);
  } else {
  scd_get_f12_2_J(fn12, fnp12,
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s2, r12,
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n*lmax_3+1],
                n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k);
  }

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
  if (same_type) {
  scd_get_f12_3(fn12, fnp12,
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s3, r12,
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n*lmax_3+2],
                n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k);
  } else {
  scd_get_f12_3_J(fn12, fnp12,
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s3, r12,
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n*lmax_3+2],
                n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k);
  }

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
  if (same_type) {
  scd_get_f12_4(fn12, fnp12,
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s4, r12,
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n*lmax_3+3],
                n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k);
  } else {
  scd_get_f12_4_J(fn12, fnp12,
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s4, r12,
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n*lmax_3+3],
                n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k);
  }
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
  const int n2,
  const bool same_type = false)
{
  const double d12inv = 1.0 / d12;
  double rij_Lsq = d12inv;
  double rij_L2sq= d12inv  * d12inv;
  // l = 1
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  double s1[3] = {
    sum_fxyz[n * NUM_OF_ABC + 0], sum_fxyz[n * NUM_OF_ABC + 1], sum_fxyz[n * NUM_OF_ABC + 2]};
  if (same_type) {
  scd_get_f12_5body(fn12, fnp12,
              blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
              scd_r12, dsnlm_dc, s1, r12,
              d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n_max_angular * lmax_3 + n_max_angular + n],
              n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k);
  } else {
  scd_get_f12_5body_J(fn12, fnp12,
              blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
              scd_r12, dsnlm_dc, s1, r12,
              d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n_max_angular * lmax_3 + n_max_angular + n],
              n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k);
  }

  s1[0] *= C3B[0];
  s1[1] *= C3B[1];
  s1[2] *= C3B[2];
  if (same_type) {
  scd_get_f12_1(fn12, fnp12,
              blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
              scd_r12, dsnlm_dc, s1, r12,
              d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n*lmax_3],
              n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k);
  } else {
  scd_get_f12_1_J(fn12, fnp12,
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s1, r12,
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n*lmax_3],
                n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k);
  }

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
  if (same_type) {
  scd_get_f12_4body(fn12, fnp12,
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s2, r12,
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n_max_angular * lmax_3 + n],
                n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k);
  } else {
  scd_get_f12_4body_J(fn12, fnp12,
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s2, r12,
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n_max_angular * lmax_3 + n],
                n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k);
  }

  s2[0] *= C3B[3];
  s2[1] *= C3B[4];
  s2[2] *= C3B[5];
  s2[3] *= C3B[6];
  s2[4] *= C3B[7];
  if (same_type) {
  scd_get_f12_2(fn12, fnp12,
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s2, r12,
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n*lmax_3+1],
                n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k);
  } else {
  scd_get_f12_2_J(fn12, fnp12,
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s2, r12,
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n*lmax_3+1],
                n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k);
  }

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
  if (same_type) {
  scd_get_f12_3(fn12, fnp12,
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s3, r12,
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n*lmax_3+2],
                n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k);
  } else {
  scd_get_f12_3_J(fn12, fnp12,
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s3, r12,
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n*lmax_3+2],
                n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k);
  }

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
  if (same_type) {
  scd_get_f12_4(fn12, fnp12,
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s4, r12,
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n*lmax_3+3],
                n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k);
  } else {
  scd_get_f12_4_J(fn12, fnp12,
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s4, r12,
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n*lmax_3+3],
                n_base_angular, dsnlm_start_idx, type_j, ntypes, n1, n2, f12k);
  }
}

// NEP supports every currently named element, H (1) through Og (118).
constexpr int NEP_MAX_ELEMENT_TYPES = 118;


// FP64 OMat24 path with per-angular-order scratch lifetimes. Numerical expressions below are ported from
// nep_utilities_mb_secondc.cuh; only indexing and accumulation destinations differ.
struct NepMbSecondGradArgs {
  const double* grad_second;
  const double* d12;
  const int64_t* neighbor_list;
  const double* de_dfeat;
  const double* sum_fxyz;
  const int64_t* atom_type;
  const double* coeff3;
  const double* angular_workspace;
  const double* weighted_sums;
  double* gradsecond_c3;
  double rcut;
  double rcut_inv;
  int atom_count;
  int max_neighbors;
  int atom_types;
  int feat_2b_count;
  int many_body_feat_count;
};

namespace nep_mb_secondgrad_opt {

__host__ __device__ constexpr size_t align_up(size_t offset, size_t alignment) {
  return (offset + alignment - 1) & ~(alignment - 1);
}

template<int NMAX, int NBASIS, int TYPE_TILE>
struct SharedLayout {
  static constexpr size_t bits_offset = 0;
  static constexpr size_t types_offset = align_up(bits_offset + 4 * sizeof(unsigned int), alignof(double));
  static constexpr size_t count_offset = align_up(types_offset + NEP_MAX_ELEMENT_TYPES * sizeof(int), alignof(double));
  static constexpr size_t fp_offset = align_up(count_offset + sizeof(int), alignof(double));
  static constexpr size_t sums_offset = align_up(fp_offset + NMAX * 6 * sizeof(double), alignof(double));
  static constexpr size_t dsnlm_offset = align_up(sums_offset + NMAX * 24 * sizeof(double), alignof(double));
  static constexpr size_t output_offset = align_up(dsnlm_offset + TYPE_TILE * NBASIS * 24 * sizeof(double), alignof(double));
  static constexpr size_t bytes = align_up(output_offset + TYPE_TILE * NMAX * NBASIS * sizeof(double), alignof(double));

  static_assert(bits_offset + 4 * sizeof(unsigned int) <= bytes, "bitset bounds");
  static_assert(types_offset + NEP_MAX_ELEMENT_TYPES * sizeof(int) <= bytes, "type list bounds");
  static_assert(count_offset + sizeof(int) <= bytes, "type count bounds");
  static_assert(fp_offset + NMAX * 6 * sizeof(double) <= bytes, "Fp bounds");
  static_assert(sums_offset + NMAX * 24 * sizeof(double) <= bytes, "sum bounds");
  static_assert(dsnlm_offset + TYPE_TILE * NBASIS * 24 * sizeof(double) <= bytes, "dsnlm bounds");
  static_assert(output_offset + TYPE_TILE * NMAX * NBASIS * sizeof(double) <= bytes, "output bounds");

  unsigned int* type_bits;
  int* local_types;
  int* local_type_count;
  double* fp;
  double* sum_fxyz;
  double* dsnlm_tile;
  double* output_tile;

  __device__ explicit SharedLayout(unsigned char* raw)
      : type_bits(reinterpret_cast<unsigned int*>(raw + bits_offset)),
        local_types(reinterpret_cast<int*>(raw + types_offset)),
        local_type_count(reinterpret_cast<int*>(raw + count_offset)),
        fp(reinterpret_cast<double*>(raw + fp_offset)),
        sum_fxyz(reinterpret_cast<double*>(raw + sums_offset)),
        dsnlm_tile(reinterpret_cast<double*>(raw + dsnlm_offset)),
        output_tile(reinterpret_cast<double*>(raw + output_offset)) {}
};

template<int NMAX, int NBASIS, int TYPE_TILE>
struct SharedTileSink {
  double* values;
  double* accumulator = nullptr;
  mutable double directional_sum = 0.0;
  mutable int direction_count = 0;

  __device__ __forceinline__ void add(int type_slot, int n, int basis, double value) const {
    directional_sum += value;
    if (++direction_count == 4) {
      if (accumulator != nullptr) {
        *accumulator += directional_sum;
      } else {
        atomicAdd(&values[(type_slot * NMAX + n) * NBASIS + basis],
                  directional_sum);
      }
    }
  }
};

// One addressable per-neighbor context crosses the angular call boundary.
// Passing the FP64 state as a single reference avoids keeping a wide argument
// list live in caller registers while an angular specialization runs. This is
// thread-private storage; the cooperative shared-memory layout is unchanged.
template<int NBASIS>
struct NeighborContext {
  double d12;
  double scd_r12[4];
  const double* fn12;
  const double* fnp12;
  double fn, fnp, d12inv, rij_Lsq, rij_L2sq;
  const double* Fp;
  const double* dsnlm_dc;
  const double* sum_fxyz;
  const double* weighted_sums;
  const double* angular;
  double* output;
  int n, type_slot, tile_count;
};

// Each angular order owns only its 2L+1 FP64 components. In particular, do
// not rebuild the legacy six 24-element arrays in the neighbor scope.
template<int L>
struct AngularScratch {
  static_assert(L >= 1 && L <= 4, "Supported angular orders");
  double blm[2 * L + 1];
  double dblm_x[2 * L + 1];
  double dblm_y[2 * L + 1];
  double dblm_z[2 * L + 1];
  double dblm_r[2 * L + 1];
};

template<typename T>
struct AngularWorkspaceView {
  T* blm;
  T* dblm_x;
  T* dblm_y;
  T* dblm_z;
  T* dblm_r;
};

template<typename T>
__device__ __forceinline__ AngularWorkspaceView<T> angular_workspace_view(
    T* base, int offset) {
  return {base + offset, base + 24 + offset, base + 48 + offset,
          base + 72 + offset, base + 96 + offset};
}

template<int L, typename Scratch>
__device__ __forceinline__ void build_angular_scratch(
    double d12, double x, double y, double z, Scratch scratch) {

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
  if constexpr (L == 1) {
    scratch.blm[0]     = z;                                             // Y10 blm
    scratch.dblm_r[0]  = 0.0;                                           // Y10 blm/dr
    scratch.dblm_x[0]  = 0.0;                                           // Y10 blm/dx
    scratch.dblm_y[0]  = 0.0;                                           // Y10 blm/dy
    scratch.dblm_z[0]  = 1.0;                                           // Y10 blm/dz
    scratch.blm[1]     = x;                                             // Y11_real
    scratch.dblm_r[1]  = 0.0;
    scratch.dblm_x[1]  = 1.0;
    scratch.dblm_y[1]  = 0.0;
    scratch.dblm_z[1]  = 0.0;
    scratch.blm[2]     = y;                                             // Y11_imag
    scratch.dblm_r[2]  = 0.0;
    scratch.dblm_x[2]  = 0.0;
    scratch.dblm_y[2]  = 1.0;
    scratch.dblm_z[2]  = 0.0;
  }
  if constexpr (L == 2) {
    scratch.blm[0]     = 3.0 * z2- d12 * d12;                           // Y20
    scratch.dblm_r[0]  = -2.0 * d12;
    scratch.dblm_x[0]  = 0.0;
    scratch.dblm_y[0]  = 0.0;
    scratch.dblm_z[0]  = 6.0 * z;
    scratch.blm[1]    = xz;                                             // Y21_real
    scratch.dblm_r[1] = 0.0;
    scratch.dblm_x[1] = z;
    scratch.dblm_y[1] = 0.0;
    scratch.dblm_z[1] = x;
    scratch.blm[2]    = yz;                                             // Y21_imag
    scratch.dblm_r[2] = 0.0;
    scratch.dblm_x[2] = 0.0;
    scratch.dblm_y[2] = z;
    scratch.dblm_z[2] = y;
    scratch.blm[3]    = x2 - y2;                                        // Y22_real
    scratch.dblm_r[3] = 0.0;
    scratch.dblm_x[3] = 2.0 * x;
    scratch.dblm_y[3] = -2.0 * y;
    scratch.dblm_z[3] = 0.0;
    scratch.blm[4]     = 2.0 * xy;                                      // Y22_imag
    scratch.dblm_r[4]  = 0.0;
    scratch.dblm_x[4]  = 2.0 * y;
    scratch.dblm_y[4]  = 2.0 * x;
    scratch.dblm_z[4]  = 0.0;
  }
  if constexpr (L == 3) {
    scratch.blm[0]     = (5.0 * z2 - 3.0 * r2) * z;                     // Y30
    scratch.dblm_r[0]  = -6.0 * z * d12;
    scratch.dblm_x[0]  = 0.0;
    scratch.dblm_y[0]  = 0.0;
    scratch.dblm_z[0]  = 15 * z2 - 3 * r2;
    scratch.blm[1]     = (5.0 * z2 - r2) * x;                          // Y31_real
    scratch.dblm_r[1]  = -2.0 * x * d12;
    scratch.dblm_x[1]  = 5.0 * z2 - r2;
    scratch.dblm_y[1]  = 0.0;
    scratch.dblm_z[1]  = 10.0 * xz;
    scratch.blm[2]    = (5.0 * z2 - r2) * y;                          // Y31_imag
    scratch.dblm_r[2] = -2.0 * y * d12;
    scratch.dblm_x[2] = 0.0;
    scratch.dblm_y[2] = 5.0 * z2 - r2;
    scratch.dblm_z[2] = 10.0 * yz;
    scratch.blm[3]    = (x2 - y2) * z;                                // Y32_real
    scratch.dblm_r[3] = 0.0;
    scratch.dblm_x[3] = 2.0 * xz;
    scratch.dblm_y[3] = -2.0 * yz;
    scratch.dblm_z[3] = x2 - y2;
    scratch.blm[4]     = 2.0 * xyz;                                // Y32_imag
    scratch.dblm_r[4]  = 0.0;
    scratch.dblm_x[4]  = 2.0 * yz;
    scratch.dblm_y[4]  = 2.0 * xz;
    scratch.dblm_z[4]  = 2.0 * xy;
    scratch.blm[5]    = (x2 - 3.0 * y2) * x;                           // Y33_real
    scratch.dblm_r[5] = 0.0;
    scratch.dblm_x[5] = 3.0 * (x2 - y2);
    scratch.dblm_y[5] = -6.0 * xy;
    scratch.dblm_z[5] = 0.0;
    scratch.blm[6]    = (3.0 * x2 - y2) * y;                           // Y33_imag
    scratch.dblm_r[6] = 0.0;
    scratch.dblm_x[6] = 6.0 * xy;
    scratch.dblm_y[6] = 3.0 * (x2 - y2);
    scratch.dblm_z[6] = 0.0;
  }
  if constexpr (L == 4) {
    scratch.blm[0]    = (35.0 * z2 - 30.0 * r2) * z2 + 3.0 * r2 * r2;   // Y40
    scratch.dblm_r[0] = (-60.0) * z2 * d12 + 12.0 * r3;
    scratch.dblm_x[0] = 0.0;
    scratch.dblm_y[0] = 0.0;
    scratch.dblm_z[0] = 140.0 * z3 - 60.0 * z * r2;
    scratch.blm[1]    = (7.0 * z2 - 3.0 * r2) * xz;                    // Y41_real
    scratch.dblm_r[1] = -6.0 * xz * d12;
    scratch.dblm_x[1] = 7.0 * z3 - 3.0 * z * r2;
    scratch.dblm_y[1] = 0.0;
    scratch.dblm_z[1]  = 21.0 * x * z2 - 3.0 * x * r2;
    scratch.blm[2]    = (7.0 * z2 - 3.0 * r2) * yz;                    // Y41_iamg
    scratch.dblm_r[2] = -6.0 * yz * d12;
    scratch.dblm_x[2] = 0.0;
    scratch.dblm_y[2] = 7.0 * z3 - 3.0 * z * r2;
    scratch.dblm_z[2] = 21.0 * y * z2 - 3.0 * y * r2;
    scratch.blm[3]    = (7.0 * z2 - r2) * x2my2;                       // Y42_real
    scratch.dblm_r[3] = 2.0 * d12 * (y2 - x2);
    scratch.dblm_x[3] = 14.0 * x * z2 - 2.0 * x * r2;
    scratch.dblm_y[3] = 2.0 * y *(r2 - 7.0 * z2);
    scratch.dblm_z[3] = 14.0 * x2 * z - 14.0 * y2 * z;
    scratch.blm[4]    = (7.0 * z2 - r2) * 2.0 * xy;                    // Y42_imag
    scratch.dblm_r[4] = -4.0 * xy * d12;
    scratch.dblm_x[4] = 2.0 * y * (7.0 * z2 - r2);
    scratch.dblm_y[4] = 2.0 * x * (7.0 * z2 - r2);
    scratch.dblm_z[4] = 28.0 * xyz;
    scratch.blm[5]    = (x2 - 3.0 * y2) * xz;                          // Y43_real
    scratch.dblm_r[5] = 0.0;
    scratch.dblm_x[5] = 3.0 * z * (x2 - y2);
    scratch.dblm_y[5] = -6.0 * xyz;
    scratch.dblm_z[5] = x3 - 3.0 * x * y2;
    scratch.blm[6]    = (3.0 * x2 - y2) * yz;                         // Y43_imag
    scratch.dblm_r[6] = 0.0;
    scratch.dblm_x[6] = 6.0 * xyz;
    scratch.dblm_y[6] = 3.0 * x2 * z - 3.0 * y2 * z;
    scratch.dblm_z[6] = 3.0 * y * x2 - y3;
    scratch.blm[7]    = x2my2 * x2my2 - 4.0 * x2 * y2;                // Y44_real
    scratch.dblm_r[7] = 0.0;
    scratch.dblm_x[7] = 4.0 * x3 - 12.0 * x * y2;
    scratch.dblm_y[7] = 4.0 * y3 - 12.0 * x2 * y;
    scratch.dblm_z[7] = 0.0;
    scratch.blm[8]    = 4.0 * x2my2 * xy;                            // Y44_imag
    scratch.dblm_r[8] = 0.0;
    scratch.dblm_x[8] = 12.0 * x2 * y - 4.0 * y3;
    scratch.dblm_y[8] = 4.0 * x3 - 12.0 * x * y2;
    scratch.dblm_z[8] = 0.0;
  }
}

// Borrowed inputs for one basis contribution. Its arrays remain owned by the
// current angular order; no angular values escape into another L scope.
// The noinline basis helpers bound algebra temporaries to one k. Their callers
// keep fixed, unrolled k loops and issue the same r/x/y/z atomics in that order.
struct AngularContribution {
  const double *fn12, *fnp12, *blm;
  const double *dblm_x, *dblm_y, *dblm_z, *dblm_r;
  const double *scd_r12, *dsnlm_dc, *s;
  double rij_Lsq, rij_L2sq, fn, fnp, Fp;
  int type_slot, n;
  double* output;
  double* accumulator = nullptr;
};

template<int L, int NMAX, int NBASIS, int TYPE_TILE>
__device__ __forceinline__ void accumulate_direct_basis(const AngularContribution& c, int k, int uj)
{
  const double* fn12 = c.fn12;
  const double* fnp12 = c.fnp12;
  const double* blm = c.blm;
  const double* dblm_x = c.dblm_x;
  const double* dblm_y = c.dblm_y;
  const double* dblm_z = c.dblm_z;
  const double* dblm_r = c.dblm_r;
  const double* scd_r12 = c.scd_r12;
  const double* dsnlm_dc = c.dsnlm_dc;
  const double* s = c.s;
  const double rij_Lsq = c.rij_Lsq;
  const double rij_L2sq = c.rij_L2sq;
  const double fn = c.fn;
  const double fnp = c.fnp;
  const double Fp = c.Fp;
  const int type_slot = c.type_slot;
  const int n = c.n;
  const SharedTileSink<NMAX, NBASIS, TYPE_TILE> sink{c.output, c.accumulator};

  if constexpr (L == 1) {
    if (type_slot < 0) return;

    double dfk = 0.0;
    int dsnlm_idx = 0 + type_slot * NBASIS * NUM_OF_ABC;

      int dsnlm_i = dsnlm_idx + k * NUM_OF_ABC;
      double tmpr = 0.0, tmpx = 0.0, tmpy = 0.0, tmpz = 0.0;
      double rr0 = 0.0, rr1 = 0.0, rr2 = 0.0;
      double rrr0 = 0.0, rrr1 = 0.0, rrr2=0.0;
      rr0 =       C3B[0] * dsnlm_dc[dsnlm_i]   * fnp * blm[0 - (L * L - 1)];
      rr1 = 2.0 * C3B[1] * dsnlm_dc[dsnlm_i+1] * fnp * blm[1 - (L * L - 1)];
      rr2 = 2.0 * C3B[2] * dsnlm_dc[dsnlm_i+2] * fnp * blm[2 - (L * L - 1)];
      dfk = fnp12[k] * rij_Lsq - fn12[k] * rij_L2sq;
      rrr0 =       s[0] * dfk * blm[0 - (L * L - 1)];
      rrr1 = 2.0 * s[1] * dfk * blm[1 - (L * L - 1)];
      rrr2 = 2.0 * s[2] * dfk * blm[2 - (L * L - 1)];
      tmpr = rr0 + rr1 + rr2 + rrr0 + rrr1 + rrr2;
      sink.add(type_slot, n, k, 2.0 * Fp * scd_r12[0] * tmpr);
      tmpx += 2.0 * C3B[1] * dsnlm_dc[dsnlm_i+1] * fn;
      tmpx += 2.0 * s[1] * fn12[k] * rij_Lsq;
      sink.add(type_slot, n, k, 2.0 * Fp * scd_r12[1] * tmpx);
      tmpy += 2.0 * C3B[2] * dsnlm_dc[dsnlm_i+2] * fn;
      tmpy += 2.0 * s[2] * fn12[k] * rij_Lsq;
      sink.add(type_slot, n, k, 2.0 * Fp * scd_r12[2] * tmpy);
      tmpz += C3B[0] * dsnlm_dc[dsnlm_i] * fn;
      tmpz += s[0] * fn12[k] * rij_Lsq;
      sink.add(type_slot, n, k, 2.0 * Fp * scd_r12[3] * tmpz);


  }
  if constexpr (L == 2) {
    if (type_slot < 0) return;

    int dsnlm_idx = 0 + type_slot * NBASIS * NUM_OF_ABC;

      int dsnlm_i = dsnlm_idx + k * NUM_OF_ABC;
      double tmpr = 0.0, tmpx = 0.0, tmpy = 0.0, tmpz = 0.0;
      tmpr +=  C3B[3] * dsnlm_dc[dsnlm_i+3] * (fnp * blm[3 - (L * L - 1)] + fn * dblm_r[3 - (L * L - 1)]) +
                    2.0 * C3B[4] * dsnlm_dc[dsnlm_i+4] * fnp * blm[4 - (L * L - 1)] +
                    2.0 * C3B[5] * dsnlm_dc[dsnlm_i+5] * fnp * blm[5 - (L * L - 1)] +
                    2.0 * C3B[6] * dsnlm_dc[dsnlm_i+6] * fnp * blm[6 - (L * L - 1)] +
                    2.0 * C3B[7] * dsnlm_dc[dsnlm_i+7] * fnp * blm[7 - (L * L - 1)];
      tmpr += s[0] * ((fnp12[k] * rij_Lsq - 2.0 * fn12[k] * rij_L2sq) * blm[3 - (L * L - 1)] + fn12[k] * rij_Lsq * dblm_r[3 - (L * L - 1)]) +
              2.0 * s[1] * (fnp12[k] * rij_Lsq - 2.0 * fn12[k] * rij_L2sq) * blm[4 - (L * L - 1)] +
              2.0 * s[2] * (fnp12[k] * rij_Lsq - 2.0 * fn12[k] * rij_L2sq) * blm[5 - (L * L - 1)] +
              2.0 * s[3] * (fnp12[k] * rij_Lsq - 2.0 * fn12[k] * rij_L2sq) * blm[6 - (L * L - 1)] +
              2.0 * s[4] * (fnp12[k] * rij_Lsq - 2.0 * fn12[k] * rij_L2sq) * blm[7 - (L * L - 1)];
      sink.add(type_slot, n, k, 2.0 * Fp * scd_r12[0] * tmpr);
      tmpx +=
                    2.0 * C3B[4] * dsnlm_dc[dsnlm_i+4] * fn * dblm_x[4 - (L * L - 1)] +
                    2.0 * C3B[6] * dsnlm_dc[dsnlm_i+6] * fn * dblm_x[6 - (L * L - 1)] +
                    2.0 * C3B[7] * dsnlm_dc[dsnlm_i+7] * fn * dblm_x[7 - (L * L - 1)];
      tmpx +=
              2.0 * s[1] * fn12[k] * rij_Lsq * dblm_x[4 - (L * L - 1)] +
              2.0 * s[3] * fn12[k] * rij_Lsq * dblm_x[6 - (L * L - 1)] +
              2.0 * s[4] * fn12[k] * rij_Lsq * dblm_x[7 - (L * L - 1)];
      sink.add(type_slot, n, k, 2.0 * Fp * scd_r12[1] * tmpx);
      tmpy +=
                    2.0 * C3B[5] * dsnlm_dc[dsnlm_i+5] * fn * dblm_y[5 - (L * L - 1)] +
                    2.0 * C3B[6] * dsnlm_dc[dsnlm_i+6] * fn * dblm_y[6 - (L * L - 1)] +
                    2.0 * C3B[7] * dsnlm_dc[dsnlm_i+7] * fn * dblm_y[7 - (L * L - 1)];
      tmpy +=
              2.0 * s[2] * fn12[k] * rij_Lsq * dblm_y[5 - (L * L - 1)] +
              2.0 * s[3] * fn12[k] * rij_Lsq * dblm_y[6 - (L * L - 1)] +
              2.0 * s[4] * fn12[k] * rij_Lsq * dblm_y[7 - (L * L - 1)];
      sink.add(type_slot, n, k, 2.0 * Fp * scd_r12[2] * tmpy);
      tmpz +=  C3B[3] * dsnlm_dc[dsnlm_i+3] * fn * dblm_z[3 - (L * L - 1)] +
                    2.0 * C3B[4] * dsnlm_dc[dsnlm_i+4] * fn * dblm_z[4 - (L * L - 1)] +
                    2.0 * C3B[5] * dsnlm_dc[dsnlm_i+5] * fn * dblm_z[5 - (L * L - 1)];
      tmpz +=  s[0] * fn12[k] * rij_Lsq * dblm_z[3 - (L * L - 1)] +
                    2.0 * s[1] * fn12[k] * rij_Lsq * dblm_z[4 - (L * L - 1)] +
                    2.0 * s[2] * fn12[k] * rij_Lsq * dblm_z[5 - (L * L - 1)];
      sink.add(type_slot, n, k, 2.0 * Fp * scd_r12[3] * tmpz);


  }
  if constexpr (L == 3) {
    if (type_slot < 0) return;

    int dsnlm_idx = 0 + type_slot * NBASIS * NUM_OF_ABC;

      int dsnlm_i = dsnlm_idx + k * NUM_OF_ABC;
      double tmpr = 0.0, tmpx = 0.0, tmpy = 0.0, tmpz = 0.0;
      tmpr +=         C3B[8]  * dsnlm_dc[dsnlm_i+8] * (fnp * blm[8 - (L * L - 1)]  + fn * dblm_r[8 - (L * L - 1)]) +
                2.0 * C3B[9]  * dsnlm_dc[dsnlm_i+9] * (fnp * blm[9 - (L * L - 1)]  + fn * dblm_r[9 - (L * L - 1)]) +
                2.0 * C3B[10] * dsnlm_dc[dsnlm_i+10] * (fnp * blm[10 - (L * L - 1)] + fn * dblm_r[10 - (L * L - 1)])+
                2.0 * C3B[11] * dsnlm_dc[dsnlm_i+11] *  fnp * blm[11 - (L * L - 1)] +
                2.0 * C3B[12] * dsnlm_dc[dsnlm_i+12] *  fnp * blm[12 - (L * L - 1)] +
                2.0 * C3B[13] * dsnlm_dc[dsnlm_i+13] *  fnp * blm[13 - (L * L - 1)] +
                2.0 * C3B[14] * dsnlm_dc[dsnlm_i+14] *  fnp * blm[14 - (L * L - 1)];
      tmpr +=       s[0] * ((fnp12[k] * rij_Lsq - 3.0 * fn12[k] * rij_L2sq) * blm[8 - (L * L - 1)]  + fn12[k] * rij_Lsq * dblm_r[8 - (L * L - 1)]) +
              2.0 * s[1] * ((fnp12[k] * rij_Lsq - 3.0 * fn12[k] * rij_L2sq) * blm[9 - (L * L - 1)]  + fn12[k] * rij_Lsq * dblm_r[9 - (L * L - 1)]) +
              2.0 * s[2] * ((fnp12[k] * rij_Lsq - 3.0 * fn12[k] * rij_L2sq) * blm[10 - (L * L - 1)] + fn12[k] * rij_Lsq * dblm_r[10 - (L * L - 1)])+
              2.0 * s[3] *  (fnp12[k] * rij_Lsq - 3.0 * fn12[k] * rij_L2sq) * blm[11 - (L * L - 1)] +
              2.0 * s[4] *  (fnp12[k] * rij_Lsq - 3.0 * fn12[k] * rij_L2sq) * blm[12 - (L * L - 1)] +
              2.0 * s[5] *  (fnp12[k] * rij_Lsq - 3.0 * fn12[k] * rij_L2sq) * blm[13 - (L * L - 1)] +
              2.0 * s[6] *  (fnp12[k] * rij_Lsq - 3.0 * fn12[k] * rij_L2sq) * blm[14 - (L * L - 1)];
      sink.add(type_slot, n, k, 2.0 * Fp * scd_r12[0] * tmpr);
      tmpx +=
                2.0 * C3B[9]  * dsnlm_dc[dsnlm_i+9]  * fn * dblm_x[9 - (L * L - 1)] +
                2.0 * C3B[11] * dsnlm_dc[dsnlm_i+11] * fn * dblm_x[11 - (L * L - 1)] +
                2.0 * C3B[12] * dsnlm_dc[dsnlm_i+12] * fn * dblm_x[12 - (L * L - 1)] +
                2.0 * C3B[13] * dsnlm_dc[dsnlm_i+13] * fn * dblm_x[13 - (L * L - 1)] +
                2.0 * C3B[14] * dsnlm_dc[dsnlm_i+14] * fn * dblm_x[14 - (L * L - 1)];
      tmpx +=
                2.0 * s[1] * fn12[k] * rij_Lsq * dblm_x[9 - (L * L - 1)]  +
                2.0 * s[3] * fn12[k] * rij_Lsq * dblm_x[11 - (L * L - 1)] +
                2.0 * s[4] * fn12[k] * rij_Lsq * dblm_x[12 - (L * L - 1)] +
                2.0 * s[5] * fn12[k] * rij_Lsq * dblm_x[13 - (L * L - 1)] +
                2.0 * s[6] * fn12[k] * rij_Lsq * dblm_x[14 - (L * L - 1)];
      sink.add(type_slot, n, k, 2.0 * Fp * scd_r12[1] * tmpx);
      tmpy +=
                2.0 * C3B[10] * dsnlm_dc[dsnlm_i+10] * fn * dblm_y[10 - (L * L - 1)] +
                2.0 * C3B[11] * dsnlm_dc[dsnlm_i+11] * fn * dblm_y[11 - (L * L - 1)] +
                2.0 * C3B[12] * dsnlm_dc[dsnlm_i+12] * fn * dblm_y[12 - (L * L - 1)] +
                2.0 * C3B[13] * dsnlm_dc[dsnlm_i+13] * fn * dblm_y[13 - (L * L - 1)] +
                2.0 * C3B[14] * dsnlm_dc[dsnlm_i+14] * fn * dblm_y[14 - (L * L - 1)];
      tmpy +=
                2.0 * s[2] * fn12[k] * rij_Lsq * dblm_y[10 - (L * L - 1)] +
                2.0 * s[3] * fn12[k] * rij_Lsq * dblm_y[11 - (L * L - 1)] +
                2.0 * s[4] * fn12[k] * rij_Lsq * dblm_y[12 - (L * L - 1)] +
                2.0 * s[5] * fn12[k] * rij_Lsq * dblm_y[13 - (L * L - 1)] +
                2.0 * s[6] * fn12[k] * rij_Lsq * dblm_y[14 - (L * L - 1)];
      sink.add(type_slot, n, k, 2.0 * Fp * scd_r12[2] * tmpy);
      tmpz +=         C3B[8]  * dsnlm_dc[dsnlm_i+8]  * fn * dblm_z[8 - (L * L - 1)] +
                2.0 * C3B[9]  * dsnlm_dc[dsnlm_i+9]  * fn * dblm_z[9 - (L * L - 1)] +
                2.0 * C3B[10] * dsnlm_dc[dsnlm_i+10] * fn * dblm_z[10 - (L * L - 1)] +
                2.0 * C3B[11] * dsnlm_dc[dsnlm_i+11] * fn * dblm_z[11 - (L * L - 1)] +
                2.0 * C3B[12] * dsnlm_dc[dsnlm_i+12] * fn * dblm_z[12 - (L * L - 1)];
      tmpz +=         s[0] * fn12[k] * rij_Lsq * dblm_z[8 - (L * L - 1)] +
                2.0 * s[1] * fn12[k] * rij_Lsq * dblm_z[9 - (L * L - 1)] +
                2.0 * s[2] * fn12[k] * rij_Lsq * dblm_z[10 - (L * L - 1)]+
                2.0 * s[3] * fn12[k] * rij_Lsq * dblm_z[11 - (L * L - 1)]+
                2.0 * s[4] * fn12[k] * rij_Lsq * dblm_z[12 - (L * L - 1)];
      sink.add(type_slot, n, k, 2.0 * Fp * scd_r12[3] * tmpz);


  }
  if constexpr (L == 4) {
    if (type_slot < 0) return;

    int dsnlm_idx = 0 + type_slot * NBASIS * NUM_OF_ABC;

      int dsnlm_i = dsnlm_idx + k * NUM_OF_ABC;
      double tmpr = 0.0, tmpx = 0.0, tmpy = 0.0, tmpz = 0.0;
      tmpr +=       C3B[15] * dsnlm_dc[dsnlm_i+15] * (fnp * blm[15 - (L * L - 1)] + fn * dblm_r[15 - (L * L - 1)]) +
              2.0 * C3B[16] * dsnlm_dc[dsnlm_i+16] * (fnp * blm[16 - (L * L - 1)] + fn * dblm_r[16 - (L * L - 1)]) +
              2.0 * C3B[17] * dsnlm_dc[dsnlm_i+17] * (fnp * blm[17 - (L * L - 1)] + fn * dblm_r[17 - (L * L - 1)]) +
              2.0 * C3B[18] * dsnlm_dc[dsnlm_i+18] * (fnp * blm[18 - (L * L - 1)] + fn * dblm_r[18 - (L * L - 1)]) +
              2.0 * C3B[19] * dsnlm_dc[dsnlm_i+19] * (fnp * blm[19 - (L * L - 1)] + fn * dblm_r[19 - (L * L - 1)]) +
              2.0 * C3B[20] * dsnlm_dc[dsnlm_i+20] *  fnp * blm[20 - (L * L - 1)] +
              2.0 * C3B[21] * dsnlm_dc[dsnlm_i+21] *  fnp * blm[21 - (L * L - 1)] +
              2.0 * C3B[22] * dsnlm_dc[dsnlm_i+22] *  fnp * blm[22 - (L * L - 1)] +
              2.0 * C3B[23] * dsnlm_dc[dsnlm_i+23] *  fnp * blm[23 - (L * L - 1)];
      tmpr +=       s[0] * ((fnp12[k] * rij_Lsq - 4.0 * fn12[k] * rij_L2sq) * blm[15 - (L * L - 1)] + fn12[k]* rij_Lsq * dblm_r[15 - (L * L - 1)]) +
              2.0 * s[1] * ((fnp12[k] * rij_Lsq - 4.0 * fn12[k] * rij_L2sq) * blm[16 - (L * L - 1)] + fn12[k]* rij_Lsq * dblm_r[16 - (L * L - 1)]) +
              2.0 * s[2] * ((fnp12[k] * rij_Lsq - 4.0 * fn12[k] * rij_L2sq) * blm[17 - (L * L - 1)] + fn12[k]* rij_Lsq * dblm_r[17 - (L * L - 1)]) +
              2.0 * s[3] * ((fnp12[k] * rij_Lsq - 4.0 * fn12[k] * rij_L2sq) * blm[18 - (L * L - 1)] + fn12[k]* rij_Lsq * dblm_r[18 - (L * L - 1)]) +
              2.0 * s[4] * ((fnp12[k] * rij_Lsq - 4.0 * fn12[k] * rij_L2sq) * blm[19 - (L * L - 1)] + fn12[k]* rij_Lsq * dblm_r[19 - (L * L - 1)]) +
              2.0 * s[5] * ((fnp12[k] * rij_Lsq - 4.0 * fn12[k] * rij_L2sq) * blm[20 - (L * L - 1)]) +
              2.0 * s[6] * ((fnp12[k] * rij_Lsq - 4.0 * fn12[k] * rij_L2sq) * blm[21 - (L * L - 1)]) +
              2.0 * s[7] * ((fnp12[k] * rij_Lsq - 4.0 * fn12[k] * rij_L2sq) * blm[22 - (L * L - 1)]) +
              2.0 * s[8] * ((fnp12[k] * rij_Lsq - 4.0 * fn12[k] * rij_L2sq) * blm[23 - (L * L - 1)]);
      sink.add(type_slot, n, k, 2.0 * Fp * scd_r12[0] * tmpr);
      tmpx +=       C3B[15] * dsnlm_dc[dsnlm_i+15] * fn * dblm_x[15 - (L * L - 1)] +
              2.0 * C3B[16] * dsnlm_dc[dsnlm_i+16] * fn * dblm_x[16 - (L * L - 1)] +
              2.0 * C3B[17] * dsnlm_dc[dsnlm_i+17] * fn * dblm_x[17 - (L * L - 1)] +
              2.0 * C3B[18] * dsnlm_dc[dsnlm_i+18] * fn * dblm_x[18 - (L * L - 1)] +
              2.0 * C3B[19] * dsnlm_dc[dsnlm_i+19] * fn * dblm_x[19 - (L * L - 1)] +
              2.0 * C3B[20] * dsnlm_dc[dsnlm_i+20] * fn * dblm_x[20 - (L * L - 1)] +
              2.0 * C3B[21] * dsnlm_dc[dsnlm_i+21] * fn * dblm_x[21 - (L * L - 1)] +
              2.0 * C3B[22] * dsnlm_dc[dsnlm_i+22] * fn * dblm_x[22 - (L * L - 1)] +
              2.0 * C3B[23] * dsnlm_dc[dsnlm_i+23] * fn * dblm_x[23 - (L * L - 1)];
      tmpx +=       s[0] * fn12[k] * rij_Lsq * dblm_x[15 - (L * L - 1)] +
              2.0 * s[1] * fn12[k] * rij_Lsq * dblm_x[16 - (L * L - 1)] +
              2.0 * s[2] * fn12[k] * rij_Lsq * dblm_x[17 - (L * L - 1)] +
              2.0 * s[3] * fn12[k] * rij_Lsq * dblm_x[18 - (L * L - 1)] +
              2.0 * s[4] * fn12[k] * rij_Lsq * dblm_x[19 - (L * L - 1)] +
              2.0 * s[5] * fn12[k] * rij_Lsq * dblm_x[20 - (L * L - 1)] +
              2.0 * s[6] * fn12[k] * rij_Lsq * dblm_x[21 - (L * L - 1)] +
              2.0 * s[7] * fn12[k] * rij_Lsq * dblm_x[22 - (L * L - 1)] +
              2.0 * s[8] * fn12[k] * rij_Lsq * dblm_x[23 - (L * L - 1)];
      sink.add(type_slot, n, k, 2.0 * Fp * scd_r12[1] * tmpx);
      tmpy +=       C3B[15] * dsnlm_dc[dsnlm_i+15] * fn * dblm_y[15 - (L * L - 1)] +
              2.0 * C3B[16] * dsnlm_dc[dsnlm_i+16] * fn * dblm_y[16 - (L * L - 1)] +
              2.0 * C3B[17] * dsnlm_dc[dsnlm_i+17] * fn * dblm_y[17 - (L * L - 1)] +
              2.0 * C3B[18] * dsnlm_dc[dsnlm_i+18] * fn * dblm_y[18 - (L * L - 1)] +
              2.0 * C3B[19] * dsnlm_dc[dsnlm_i+19] * fn * dblm_y[19 - (L * L - 1)] +
              2.0 * C3B[20] * dsnlm_dc[dsnlm_i+20] * fn * dblm_y[20 - (L * L - 1)] +
              2.0 * C3B[21] * dsnlm_dc[dsnlm_i+21] * fn * dblm_y[21 - (L * L - 1)] +
              2.0 * C3B[22] * dsnlm_dc[dsnlm_i+22] * fn * dblm_y[22 - (L * L - 1)] +
              2.0 * C3B[23] * dsnlm_dc[dsnlm_i+23] * fn * dblm_y[23 - (L * L - 1)];
      tmpy +=       s[0] * fn12[k] * rij_Lsq * dblm_y[15 - (L * L - 1)] +
              2.0 * s[1] * fn12[k] * rij_Lsq * dblm_y[16 - (L * L - 1)] +
              2.0 * s[2] * fn12[k] * rij_Lsq * dblm_y[17 - (L * L - 1)] +
              2.0 * s[3] * fn12[k] * rij_Lsq * dblm_y[18 - (L * L - 1)] +
              2.0 * s[4] * fn12[k] * rij_Lsq * dblm_y[19 - (L * L - 1)] +
              2.0 * s[5] * fn12[k] * rij_Lsq * dblm_y[20 - (L * L - 1)] +
              2.0 * s[6] * fn12[k] * rij_Lsq * dblm_y[21 - (L * L - 1)] +
              2.0 * s[7] * fn12[k] * rij_Lsq * dblm_y[22 - (L * L - 1)] +
              2.0 * s[8] * fn12[k] * rij_Lsq * dblm_y[23 - (L * L - 1)];
      sink.add(type_slot, n, k, 2.0 * Fp * scd_r12[2] * tmpy);
      tmpz +=       C3B[15] * dsnlm_dc[dsnlm_i+15] * fn * dblm_z[15 - (L * L - 1)] +
              2.0 * C3B[16] * dsnlm_dc[dsnlm_i+16] * fn * dblm_z[16 - (L * L - 1)] +
              2.0 * C3B[17] * dsnlm_dc[dsnlm_i+17] * fn * dblm_z[17 - (L * L - 1)] +
              2.0 * C3B[18] * dsnlm_dc[dsnlm_i+18] * fn * dblm_z[18 - (L * L - 1)] +
              2.0 * C3B[19] * dsnlm_dc[dsnlm_i+19] * fn * dblm_z[19 - (L * L - 1)] +
              2.0 * C3B[20] * dsnlm_dc[dsnlm_i+20] * fn * dblm_z[20 - (L * L - 1)] +
              2.0 * C3B[21] * dsnlm_dc[dsnlm_i+21] * fn * dblm_z[21 - (L * L - 1)] +
              2.0 * C3B[22] * dsnlm_dc[dsnlm_i+22] * fn * dblm_z[22 - (L * L - 1)] +
              2.0 * C3B[23] * dsnlm_dc[dsnlm_i+23] * fn * dblm_z[23 - (L * L - 1)];
      tmpz +=       s[0] * fn12[k] * rij_Lsq * dblm_z[15 - (L * L - 1)] +
              2.0 * s[1] * fn12[k] * rij_Lsq * dblm_z[16 - (L * L - 1)] +
              2.0 * s[2] * fn12[k] * rij_Lsq * dblm_z[17 - (L * L - 1)] +
              2.0 * s[3] * fn12[k] * rij_Lsq * dblm_z[18 - (L * L - 1)] +
              2.0 * s[4] * fn12[k] * rij_Lsq * dblm_z[19 - (L * L - 1)] +
              2.0 * s[5] * fn12[k] * rij_Lsq * dblm_z[20 - (L * L - 1)] +
              2.0 * s[6] * fn12[k] * rij_Lsq * dblm_z[21 - (L * L - 1)] +
              2.0 * s[7] * fn12[k] * rij_Lsq * dblm_z[22 - (L * L - 1)] +
              2.0 * s[8] * fn12[k] * rij_Lsq * dblm_z[23 - (L * L - 1)];
      sink.add(type_slot, n, k, 2.0 * Fp * scd_r12[3] * tmpz);


  }
}

template<int L, int NMAX, int NBASIS, int TYPE_TILE>
__device__ __forceinline__ void accumulate_direct(
  const double* fn12, const double* fnp12,
  const double* blm,
  const double* dblm_x, const double* dblm_y, const double* dblm_z, const double* dblm_r,
  const double* scd_r12, const double* dsnlm_dc, const double* s,
  double rij_Lsq, double rij_L2sq, double fn, double fnp, double Fp,
  int type_slot, int n, SharedTileSink<NMAX, NBASIS, TYPE_TILE> sink)
{
  const AngularContribution c{fn12, fnp12, blm,
      dblm_x, dblm_y, dblm_z, dblm_r, scd_r12, dsnlm_dc, s,
      rij_Lsq, rij_L2sq, fn, fnp, Fp, type_slot, n, sink.values};
  if (type_slot < 0) return;
  #pragma unroll
  for (int k = 0; k < NBASIS; ++k) {
    accumulate_direct_basis<L, NMAX, NBASIS, TYPE_TILE>(c, k, 0);
  }
}


template<int L, int NMAX, int NBASIS, int TYPE_TILE>
__device__ __forceinline__ void accumulate_cross_basis(const AngularContribution& c, int k, int uj)
{
  const double* blm = c.blm;
  const double* dblm_x = c.dblm_x;
  const double* dblm_y = c.dblm_y;
  const double* dblm_z = c.dblm_z;
  const double* dblm_r = c.dblm_r;
  const double* scd_r12 = c.scd_r12;
  const double* dsnlm_dc = c.dsnlm_dc;
  const double fn = c.fn;
  const double fnp = c.fnp;
  const double Fp = c.Fp;
  const int type_slot = c.type_slot;
  const int n = c.n;
  const SharedTileSink<NMAX, NBASIS, TYPE_TILE> sink{c.output, c.accumulator};

  if constexpr (L == 1) {

      int j = uj;
      if (type_slot == j) return;
      int dsnlm_idx = 0 + j * NBASIS * NUM_OF_ABC;


        int dsnlm_i = dsnlm_idx + k * NUM_OF_ABC;
        double tmpr = 0.0, tmpx = 0.0, tmpy = 0.0, tmpz = 0.0;

        tmpr +=       C3B[0] * dsnlm_dc[dsnlm_i]   * fnp * blm[0 - (L * L - 1)];
        tmpr += 2.0 * C3B[1] * dsnlm_dc[dsnlm_i+1] * fnp * blm[1 - (L * L - 1)];
        tmpr += 2.0 * C3B[2] * dsnlm_dc[dsnlm_i+2] * fnp * blm[2 - (L * L - 1)];
        sink.add(uj, n, k, 2.0 * Fp * scd_r12[0] * tmpr);
        tmpx += 2.0 * C3B[1] * dsnlm_dc[dsnlm_i+1] * fn;
        sink.add(uj, n, k, 2.0 * Fp * scd_r12[1] * tmpx);
        tmpy += 2.0 * C3B[2] * dsnlm_dc[dsnlm_i+2] * fn;
        sink.add(uj, n, k, 2.0 * Fp * scd_r12[2] * tmpy);
        tmpz += C3B[0] * dsnlm_dc[dsnlm_i] * fn;
        sink.add(uj, n, k, 2.0 * Fp * scd_r12[3] * tmpz);



  }
  if constexpr (L == 2) {

      int j = uj;
      if (type_slot == j) return;
      int dsnlm_idx = 0 + j * NBASIS * NUM_OF_ABC;



        int dsnlm_i = dsnlm_idx + k * NUM_OF_ABC;
        double tmpr = 0.0, tmpx = 0.0, tmpy = 0.0, tmpz = 0.0;

        tmpr +=  C3B[3] * dsnlm_dc[dsnlm_i+3] * (fnp * blm[3 - (L * L - 1)] + fn * dblm_r[3 - (L * L - 1)]) +
                      2.0 * C3B[4] * dsnlm_dc[dsnlm_i+4] * fnp * blm[4 - (L * L - 1)] +
                      2.0 * C3B[5] * dsnlm_dc[dsnlm_i+5] * fnp * blm[5 - (L * L - 1)] +
                      2.0 * C3B[6] * dsnlm_dc[dsnlm_i+6] * fnp * blm[6 - (L * L - 1)] +
                      2.0 * C3B[7] * dsnlm_dc[dsnlm_i+7] * fnp * blm[7 - (L * L - 1)];
        sink.add(uj, n, k, 2.0 * Fp * scd_r12[0] * tmpr);
        tmpx +=
                      2.0 * C3B[4] * dsnlm_dc[dsnlm_i+4] * fn * dblm_x[4 - (L * L - 1)] +
                      2.0 * C3B[6] * dsnlm_dc[dsnlm_i+6] * fn * dblm_x[6 - (L * L - 1)] +
                      2.0 * C3B[7] * dsnlm_dc[dsnlm_i+7] * fn * dblm_x[7 - (L * L - 1)];
        sink.add(uj, n, k, 2.0 * Fp * scd_r12[1] * tmpx);
        tmpy +=
                      2.0 * C3B[5] * dsnlm_dc[dsnlm_i+5] * fn * dblm_y[5 - (L * L - 1)] +
                      2.0 * C3B[6] * dsnlm_dc[dsnlm_i+6] * fn * dblm_y[6 - (L * L - 1)] +
                      2.0 * C3B[7] * dsnlm_dc[dsnlm_i+7] * fn * dblm_y[7 - (L * L - 1)];
        sink.add(uj, n, k, 2.0 * Fp * scd_r12[2] * tmpy);
        tmpz +=  C3B[3] * dsnlm_dc[dsnlm_i+3] * fn * dblm_z[3 - (L * L - 1)] +
                      2.0 * C3B[4] * dsnlm_dc[dsnlm_i+4] * fn * dblm_z[4 - (L * L - 1)] +
                      2.0 * C3B[5] * dsnlm_dc[dsnlm_i+5] * fn * dblm_z[5 - (L * L - 1)];
        sink.add(uj, n, k, 2.0 * Fp * scd_r12[3] * tmpz);



  }
  if constexpr (L == 3) {

      int j = uj;
      if (type_slot == j) return;
      int dsnlm_idx = 0 + j * NBASIS * NUM_OF_ABC;


        int dsnlm_i = dsnlm_idx + k * NUM_OF_ABC;
        double tmpr = 0.0, tmpx = 0.0, tmpy = 0.0, tmpz = 0.0;

        tmpr +=         C3B[8]  * dsnlm_dc[dsnlm_i+8] * (fnp * blm[8 - (L * L - 1)]  + fn * dblm_r[8 - (L * L - 1)]) +
                  2.0 * C3B[9]  * dsnlm_dc[dsnlm_i+9] * (fnp * blm[9 - (L * L - 1)]  + fn * dblm_r[9 - (L * L - 1)]) +
                  2.0 * C3B[10] * dsnlm_dc[dsnlm_i+10] * (fnp * blm[10 - (L * L - 1)] + fn * dblm_r[10 - (L * L - 1)])+
                  2.0 * C3B[11] * dsnlm_dc[dsnlm_i+11] *  fnp * blm[11 - (L * L - 1)] +
                  2.0 * C3B[12] * dsnlm_dc[dsnlm_i+12] *  fnp * blm[12 - (L * L - 1)] +
                  2.0 * C3B[13] * dsnlm_dc[dsnlm_i+13] *  fnp * blm[13 - (L * L - 1)] +
                  2.0 * C3B[14] * dsnlm_dc[dsnlm_i+14] *  fnp * blm[14 - (L * L - 1)];
        sink.add(uj, n, k, 2.0 * Fp * scd_r12[0] * tmpr);
        tmpx +=
                  2.0 * C3B[9]  * dsnlm_dc[dsnlm_i+9]  * fn * dblm_x[9 - (L * L - 1)] +
                  2.0 * C3B[11] * dsnlm_dc[dsnlm_i+11] * fn * dblm_x[11 - (L * L - 1)] +
                  2.0 * C3B[12] * dsnlm_dc[dsnlm_i+12] * fn * dblm_x[12 - (L * L - 1)] +
                  2.0 * C3B[13] * dsnlm_dc[dsnlm_i+13] * fn * dblm_x[13 - (L * L - 1)] +
                  2.0 * C3B[14] * dsnlm_dc[dsnlm_i+14] * fn * dblm_x[14 - (L * L - 1)];
        sink.add(uj, n, k, 2.0 * Fp * scd_r12[1] * tmpx);
        tmpy +=
                  2.0 * C3B[10] * dsnlm_dc[dsnlm_i+10] * fn * dblm_y[10 - (L * L - 1)] +
                  2.0 * C3B[11] * dsnlm_dc[dsnlm_i+11] * fn * dblm_y[11 - (L * L - 1)] +
                  2.0 * C3B[12] * dsnlm_dc[dsnlm_i+12] * fn * dblm_y[12 - (L * L - 1)] +
                  2.0 * C3B[13] * dsnlm_dc[dsnlm_i+13] * fn * dblm_y[13 - (L * L - 1)] +
                  2.0 * C3B[14] * dsnlm_dc[dsnlm_i+14] * fn * dblm_y[14 - (L * L - 1)];
        sink.add(uj, n, k, 2.0 * Fp * scd_r12[2] * tmpy);
        tmpz +=         C3B[8]  * dsnlm_dc[dsnlm_i+8]  * fn * dblm_z[8 - (L * L - 1)] +
                  2.0 * C3B[9]  * dsnlm_dc[dsnlm_i+9]  * fn * dblm_z[9 - (L * L - 1)] +
                  2.0 * C3B[10] * dsnlm_dc[dsnlm_i+10] * fn * dblm_z[10 - (L * L - 1)] +
                  2.0 * C3B[11] * dsnlm_dc[dsnlm_i+11] * fn * dblm_z[11 - (L * L - 1)] +
                  2.0 * C3B[12] * dsnlm_dc[dsnlm_i+12] * fn * dblm_z[12 - (L * L - 1)];
        sink.add(uj, n, k, 2.0 * Fp * scd_r12[3] * tmpz);



  }
  if constexpr (L == 4) {

      int j = uj;
      if (type_slot == j) return;
      int dsnlm_idx = 0 + j * NBASIS * NUM_OF_ABC;


        int dsnlm_i = dsnlm_idx + k * NUM_OF_ABC;
        double tmpr = 0.0, tmpx = 0.0, tmpy = 0.0, tmpz = 0.0;

        tmpr +=       C3B[15] * dsnlm_dc[dsnlm_i+15] * (fnp * blm[15 - (L * L - 1)] + fn * dblm_r[15 - (L * L - 1)]) +
                2.0 * C3B[16] * dsnlm_dc[dsnlm_i+16] * (fnp * blm[16 - (L * L - 1)] + fn * dblm_r[16 - (L * L - 1)]) +
                2.0 * C3B[17] * dsnlm_dc[dsnlm_i+17] * (fnp * blm[17 - (L * L - 1)] + fn * dblm_r[17 - (L * L - 1)]) +
                2.0 * C3B[18] * dsnlm_dc[dsnlm_i+18] * (fnp * blm[18 - (L * L - 1)] + fn * dblm_r[18 - (L * L - 1)]) +
                2.0 * C3B[19] * dsnlm_dc[dsnlm_i+19] * (fnp * blm[19 - (L * L - 1)] + fn * dblm_r[19 - (L * L - 1)]) +
                2.0 * C3B[20] * dsnlm_dc[dsnlm_i+20] *  fnp * blm[20 - (L * L - 1)] +
                2.0 * C3B[21] * dsnlm_dc[dsnlm_i+21] *  fnp * blm[21 - (L * L - 1)] +
                2.0 * C3B[22] * dsnlm_dc[dsnlm_i+22] *  fnp * blm[22 - (L * L - 1)] +
                2.0 * C3B[23] * dsnlm_dc[dsnlm_i+23] *  fnp * blm[23 - (L * L - 1)];
        sink.add(uj, n, k, 2.0 * Fp * scd_r12[0] * tmpr);
        tmpx +=       C3B[15] * dsnlm_dc[dsnlm_i+15] * fn * dblm_x[15 - (L * L - 1)] +
                2.0 * C3B[16] * dsnlm_dc[dsnlm_i+16] * fn * dblm_x[16 - (L * L - 1)] +
                2.0 * C3B[17] * dsnlm_dc[dsnlm_i+17] * fn * dblm_x[17 - (L * L - 1)] +
                2.0 * C3B[18] * dsnlm_dc[dsnlm_i+18] * fn * dblm_x[18 - (L * L - 1)] +
                2.0 * C3B[19] * dsnlm_dc[dsnlm_i+19] * fn * dblm_x[19 - (L * L - 1)] +
                2.0 * C3B[20] * dsnlm_dc[dsnlm_i+20] * fn * dblm_x[20 - (L * L - 1)] +
                2.0 * C3B[21] * dsnlm_dc[dsnlm_i+21] * fn * dblm_x[21 - (L * L - 1)] +
                2.0 * C3B[22] * dsnlm_dc[dsnlm_i+22] * fn * dblm_x[22 - (L * L - 1)] +
                2.0 * C3B[23] * dsnlm_dc[dsnlm_i+23] * fn * dblm_x[23 - (L * L - 1)];
        sink.add(uj, n, k, 2.0 * Fp * scd_r12[1] * tmpx);
        tmpy +=       C3B[15] * dsnlm_dc[dsnlm_i+15] * fn * dblm_y[15 - (L * L - 1)] +
                2.0 * C3B[16] * dsnlm_dc[dsnlm_i+16] * fn * dblm_y[16 - (L * L - 1)] +
                2.0 * C3B[17] * dsnlm_dc[dsnlm_i+17] * fn * dblm_y[17 - (L * L - 1)] +
                2.0 * C3B[18] * dsnlm_dc[dsnlm_i+18] * fn * dblm_y[18 - (L * L - 1)] +
                2.0 * C3B[19] * dsnlm_dc[dsnlm_i+19] * fn * dblm_y[19 - (L * L - 1)] +
                2.0 * C3B[20] * dsnlm_dc[dsnlm_i+20] * fn * dblm_y[20 - (L * L - 1)] +
                2.0 * C3B[21] * dsnlm_dc[dsnlm_i+21] * fn * dblm_y[21 - (L * L - 1)] +
                2.0 * C3B[22] * dsnlm_dc[dsnlm_i+22] * fn * dblm_y[22 - (L * L - 1)] +
                2.0 * C3B[23] * dsnlm_dc[dsnlm_i+23] * fn * dblm_y[23 - (L * L - 1)];
        sink.add(uj, n, k, 2.0 * Fp * scd_r12[2] * tmpy);
        tmpz +=       C3B[15] * dsnlm_dc[dsnlm_i+15] * fn * dblm_z[15 - (L * L - 1)] +
                2.0 * C3B[16] * dsnlm_dc[dsnlm_i+16] * fn * dblm_z[16 - (L * L - 1)] +
                2.0 * C3B[17] * dsnlm_dc[dsnlm_i+17] * fn * dblm_z[17 - (L * L - 1)] +
                2.0 * C3B[18] * dsnlm_dc[dsnlm_i+18] * fn * dblm_z[18 - (L * L - 1)] +
                2.0 * C3B[19] * dsnlm_dc[dsnlm_i+19] * fn * dblm_z[19 - (L * L - 1)] +
                2.0 * C3B[20] * dsnlm_dc[dsnlm_i+20] * fn * dblm_z[20 - (L * L - 1)] +
                2.0 * C3B[21] * dsnlm_dc[dsnlm_i+21] * fn * dblm_z[21 - (L * L - 1)] +
                2.0 * C3B[22] * dsnlm_dc[dsnlm_i+22] * fn * dblm_z[22 - (L * L - 1)] +
                2.0 * C3B[23] * dsnlm_dc[dsnlm_i+23] * fn * dblm_z[23 - (L * L - 1)];
        sink.add(uj, n, k, 2.0 * Fp * scd_r12[3] * tmpz);



  }
}

template<int L, int NMAX, int NBASIS, int TYPE_TILE>
__device__ __forceinline__ void accumulate_cross(
  const double* fn12, const double* fnp12,
  const double* blm,
  const double* dblm_x, const double* dblm_y, const double* dblm_z, const double* dblm_r,
  const double* scd_r12, const double* dsnlm_dc, const double* s,
  double rij_Lsq, double rij_L2sq, double fn, double fnp, double Fp,
  int type_slot, int tile_count, int n, SharedTileSink<NMAX, NBASIS, TYPE_TILE> sink)
{
  const AngularContribution c{fn12, fnp12, blm,
      dblm_x, dblm_y, dblm_z, dblm_r, scd_r12, dsnlm_dc, s,
      rij_Lsq, rij_L2sq, fn, fnp, Fp, type_slot, n, sink.values};
  for (int uj = 0; uj < tile_count; ++uj) {
    if (type_slot == uj) continue;
    #pragma unroll
    for (int k = 0; k < NBASIS; ++k) {
      accumulate_cross_basis<L, NMAX, NBASIS, TYPE_TILE>(c, k, uj);
    }
  }
}


// Prepare each direction inside its basis contribution so derivative arrays
// are released immediately after their r/x/y/z atomic, in the legacy order.
template<bool CROSS, int NMAX, int NBASIS, int TYPE_TILE>
__device__ __forceinline__ void accumulate_four_body_basis(const AngularContribution& c, int k, int uj)
{
  const double* fn12 = c.fn12;
  const double* fnp12 = c.fnp12;
  const double* blm = c.blm;
  const double* dblm_x = c.dblm_x;
  const double* dblm_y = c.dblm_y;
  const double* dblm_z = c.dblm_z;
  const double* dblm_r = c.dblm_r;
  const double* scd_r12 = c.scd_r12;
  const double* dsnlm_dc = c.dsnlm_dc;
  const double* s = c.s;
  const double rij_Lsq = c.rij_Lsq;
  const double rij_L2sq = c.rij_L2sq;
  const double fn = c.fn;
  const double fnp = c.fnp;
  const double Fp = c.Fp;
  const int type_slot = c.type_slot;
  const int n = c.n;
  const SharedTileSink<NMAX, NBASIS, TYPE_TILE> sink{c.output, c.accumulator};

  if constexpr (CROSS) {
    double dnlm_dc[5] = {0.0};


      int j = uj;
      if (type_slot == j) return;
      int dsnlm_idx = 0 + j * NBASIS * NUM_OF_ABC;


        int dsnlm_i = dsnlm_idx + k * NUM_OF_ABC;

        dnlm_dc[0] = dsnlm_dc[dsnlm_i + 3];
        dnlm_dc[1] = dsnlm_dc[dsnlm_i + 4];
        dnlm_dc[2] = dsnlm_dc[dsnlm_i + 5];
        dnlm_dc[3] = dsnlm_dc[dsnlm_i + 6];
        dnlm_dc[4] = dsnlm_dc[dsnlm_i + 7];
        {
          double tmpr = 0.0;
          double dnlm_drij[5] = {0.0};
          dnlm_drij[0] = fnp * blm[0] + fn * dblm_r[0];
          dnlm_drij[1] = fnp * blm[1];
          dnlm_drij[2] = fnp * blm[2];
          dnlm_drij[3] = fnp * blm[3];
          dnlm_drij[4] = fnp * blm[4];
          tmpr += 3.0 * C4B[0] * (
            2.0 * s[0] * dnlm_dc[0] * dnlm_drij[0]);
          tmpr += C4B[1] * (
            dnlm_drij[0] * 2.0 * (s[1] * dnlm_dc[1] + s[2] * dnlm_dc[2])
          );
          tmpr += 2.0 * C4B[1] * (
            dnlm_dc[0] * (s[1] * dnlm_drij[1] + s[2] * dnlm_drij[2]) +
            s[0] * (dnlm_dc[1] * dnlm_drij[1] +
                      dnlm_dc[2] * dnlm_drij[2])
          );
          tmpr += C4B[2] * (
            dnlm_drij[0] * 2.0 * (s[3] * dnlm_dc[3] + s[4] * dnlm_dc[4]));
          tmpr += 2.0 * C4B[2] * (
            dnlm_dc[0] * (s[3] * dnlm_drij[3] + s[4] * dnlm_drij[4]) +
            s[0] * (dnlm_dc[3] * dnlm_drij[3] +
                      dnlm_dc[4] * dnlm_drij[4])
          );
          tmpr += C4B[3] * (
            dnlm_drij[3] * 2.0 * (s[2] * dnlm_dc[2] - s[1] * dnlm_dc[1]));
          tmpr += 2.0 * C4B[3] * (
            dnlm_dc[3] * (s[2] * dnlm_drij[2] - s[1] * dnlm_drij[1]) +
              s[3] * (dnlm_dc[2] * dnlm_drij[2] -
                      dnlm_dc[1] * dnlm_drij[1])
          );
          tmpr += C4B[4] * (
            dnlm_drij[1] * dnlm_dc[2] * s[4] + dnlm_drij[1] * s[2] * dnlm_dc[4] +
            dnlm_dc[1] * dnlm_drij[2] * s[4] + s[1] * dnlm_drij[2] * dnlm_dc[4] +
            dnlm_dc[1] * s[2] * dnlm_drij[4] + s[1] * dnlm_dc[2] * dnlm_drij[4]
          );
          sink.add(uj, n, k, Fp * scd_r12[0] * tmpr);
        }
        {
          double tmpx = 0.0;
          double dnlm_dxij[5] = {0.0};
          dnlm_dxij[0] = 0.0;
          dnlm_dxij[1] = fn * dblm_x[1];
          dnlm_dxij[2] = 0.0;
          dnlm_dxij[3] = fn * dblm_x[3];
          dnlm_dxij[4] = fn * dblm_x[4];
          tmpx += 3.0 * C4B[0] * (
            2.0 * s[0] * dnlm_dc[0] * dnlm_dxij[0]);
          tmpx += C4B[1] * (
            dnlm_dxij[0] * 2.0 * (s[1] * dnlm_dc[1] + s[2] * dnlm_dc[2])
          );
          tmpx += 2.0 * C4B[1] * (
            dnlm_dc[0] * (s[1] * dnlm_dxij[1] + s[2] * dnlm_dxij[2]) +
            s[0] * (dnlm_dc[1] * dnlm_dxij[1] +
                      dnlm_dc[2] * dnlm_dxij[2])
          );
          tmpx += C4B[2] * (
            dnlm_dxij[0] * 2.0 * (s[3] * dnlm_dc[3] + s[4] * dnlm_dc[4]));
          tmpx += 2.0 * C4B[2] * (
            dnlm_dc[0] * (s[3] * dnlm_dxij[3] + s[4] * dnlm_dxij[4]) +
            s[0] * (dnlm_dc[3] * dnlm_dxij[3] +
                      dnlm_dc[4] * dnlm_dxij[4])
          );
          tmpx += C4B[3] * (
            dnlm_dxij[3] * 2.0 * (s[2] * dnlm_dc[2] - s[1] * dnlm_dc[1]));
          tmpx += 2.0 * C4B[3] * (
            dnlm_dc[3] * (s[2] * dnlm_dxij[2] - s[1] * dnlm_dxij[1]) +
              s[3] * (dnlm_dc[2] * dnlm_dxij[2] -
                      dnlm_dc[1] * dnlm_dxij[1])
          );
          tmpx += C4B[4] * (
            dnlm_dxij[1] * dnlm_dc[2] * s[4] + dnlm_dxij[1] * s[2] * dnlm_dc[4] +
            dnlm_dc[1] * dnlm_dxij[2] * s[4] + s[1] * dnlm_dxij[2] * dnlm_dc[4] +
            dnlm_dc[1] * s[2] * dnlm_dxij[4] + s[1] * dnlm_dc[2] * dnlm_dxij[4]
          );
          sink.add(uj, n, k, Fp * scd_r12[1] * tmpx);
        }
        {
          double tmpy = 0.0;
          double dnlm_dyij[5] = {0.0};
          dnlm_dyij[0] = 0.0;
          dnlm_dyij[1] = 0.0;
          dnlm_dyij[2] = fn * dblm_y[2];
          dnlm_dyij[3] = fn * dblm_y[3];
          dnlm_dyij[4] = fn * dblm_y[4];
          tmpy += 3.0 * C4B[0] * (
            2.0 * s[0] * dnlm_dc[0] * dnlm_dyij[0]);
          tmpy += C4B[1] * (
            dnlm_dyij[0] * 2.0 * (s[1] * dnlm_dc[1] + s[2] * dnlm_dc[2])
          );
          tmpy += 2.0 * C4B[1] * (
            dnlm_dc[0] * (s[1] * dnlm_dyij[1] + s[2] * dnlm_dyij[2]) +
            s[0] * (dnlm_dc[1] * dnlm_dyij[1] +
                      dnlm_dc[2] * dnlm_dyij[2])
          );
          tmpy += C4B[2] * (
            dnlm_dyij[0] * 2.0 * (s[3] * dnlm_dc[3] + s[4] * dnlm_dc[4]));
          tmpy += 2.0 * C4B[2] * (
            dnlm_dc[0] * (s[3] * dnlm_dyij[3] + s[4] * dnlm_dyij[4]) +
            s[0] * (dnlm_dc[3] * dnlm_dyij[3] +
                      dnlm_dc[4] * dnlm_dyij[4])
          );
          tmpy += C4B[3] * (
            dnlm_dyij[3] * 2.0 * (s[2] * dnlm_dc[2] - s[1] * dnlm_dc[1]));
          tmpy += 2.0 * C4B[3] * (
            dnlm_dc[3] * (s[2] * dnlm_dyij[2] - s[1] * dnlm_dyij[1]) +
              s[3] * (dnlm_dc[2] * dnlm_dyij[2] -
                      dnlm_dc[1] * dnlm_dyij[1])
          );
          tmpy += C4B[4] * (
            dnlm_dyij[1] * dnlm_dc[2] * s[4] + dnlm_dyij[1] * s[2] * dnlm_dc[4] +
            dnlm_dc[1] * dnlm_dyij[2] * s[4] + s[1] * dnlm_dyij[2] * dnlm_dc[4] +
            dnlm_dc[1] * s[2] * dnlm_dyij[4] + s[1] * dnlm_dc[2] * dnlm_dyij[4]
          );
          sink.add(uj, n, k, Fp * scd_r12[2] * tmpy);
        }
        {
          double tmpz = 0.0;
          double dnlm_dzij[5] = {0.0};
          dnlm_dzij[0] = fn * dblm_z[0];
          dnlm_dzij[1] = fn * dblm_z[1];
          dnlm_dzij[2] = fn * dblm_z[2];
          dnlm_dzij[3] = 0.0;
          dnlm_dzij[4] = 0.0;
          tmpz += 3.0 * C4B[0] * (
            2.0 * s[0] * dnlm_dc[0] * dnlm_dzij[0]);
          tmpz += C4B[1] * (
            dnlm_dzij[0] * 2.0 * (s[1] * dnlm_dc[1] + s[2] * dnlm_dc[2])
          );
          tmpz += 2.0 * C4B[1] * (
            dnlm_dc[0] * (s[1] * dnlm_dzij[1] + s[2] * dnlm_dzij[2]) +
            s[0] * (dnlm_dc[1] * dnlm_dzij[1] +
                      dnlm_dc[2] * dnlm_dzij[2])
          );
          tmpz += C4B[2] * (
            dnlm_dzij[0] * 2.0 * (s[3] * dnlm_dc[3] + s[4] * dnlm_dc[4]));
          tmpz += 2.0 * C4B[2] * (
            dnlm_dc[0] * (s[3] * dnlm_dzij[3] + s[4] * dnlm_dzij[4]) +
            s[0] * (dnlm_dc[3] * dnlm_dzij[3] +
                      dnlm_dc[4] * dnlm_dzij[4])
          );
          tmpz += C4B[3] * (
            dnlm_dzij[3] * 2.0 * (s[2] * dnlm_dc[2] - s[1] * dnlm_dc[1]));
          tmpz += 2.0 * C4B[3] * (
            dnlm_dc[3] * (s[2] * dnlm_dzij[2] - s[1] * dnlm_dzij[1]) +
              s[3] * (dnlm_dc[2] * dnlm_dzij[2] -
                      dnlm_dc[1] * dnlm_dzij[1])
          );
          tmpz += C4B[4] * (
            dnlm_dzij[1] * dnlm_dc[2] * s[4] + dnlm_dzij[1] * s[2] * dnlm_dc[4] +
            dnlm_dc[1] * dnlm_dzij[2] * s[4] + s[1] * dnlm_dzij[2] * dnlm_dc[4] +
            dnlm_dc[1] * s[2] * dnlm_dzij[4] + s[1] * dnlm_dc[2] * dnlm_dzij[4]
          );
          sink.add(uj, n, k, Fp * scd_r12[3] * tmpz);
        }



  } else {
    if (type_slot < 0) return;

    int dsnlm_idx = 0 + type_slot * NBASIS * NUM_OF_ABC;
    double dnlm_dc[5] = {0.0};
    double s2[5] = {0.0};
    s2[0] = s[0] * s[0];
    s2[1] = s[1] * s[1];
    s2[2] = s[2] * s[2];
    s2[3] = s[3] * s[3];
    s2[4] = s[4] * s[4];

      int dsnlm_i = dsnlm_idx + k * NUM_OF_ABC;
      dnlm_dc[0] = dsnlm_dc[dsnlm_i + 3];
      dnlm_dc[1] = dsnlm_dc[dsnlm_i + 4];
      dnlm_dc[2] = dsnlm_dc[dsnlm_i + 5];
      dnlm_dc[3] = dsnlm_dc[dsnlm_i + 6];
      dnlm_dc[4] = dsnlm_dc[dsnlm_i + 7];
      {
        double tmpr = 0.0;
        double dnlm_drij[5] = {0.0};
        dnlm_drij[0] = fnp * blm[0] + fn * dblm_r[0];
        dnlm_drij[1] = fnp * blm[1];
        dnlm_drij[2] = fnp * blm[2];
        dnlm_drij[3] = fnp * blm[3];
        dnlm_drij[4] = fnp * blm[4];
        double dnlm_drij_dc[5] = {0.0};
        dnlm_drij_dc[0] = (fnp12[k] * rij_Lsq - 2.0 * fn12[k] * rij_L2sq) * blm[0] + fn12[k] * rij_Lsq * dblm_r[0];
        dnlm_drij_dc[1] = (fnp12[k] * rij_Lsq - 2.0 * fn12[k] * rij_L2sq) * blm[1];
        dnlm_drij_dc[2] = (fnp12[k] * rij_Lsq - 2.0 * fn12[k] * rij_L2sq) * blm[2];
        dnlm_drij_dc[3] = (fnp12[k] * rij_Lsq - 2.0 * fn12[k] * rij_L2sq) * blm[3];
        dnlm_drij_dc[4] = (fnp12[k] * rij_Lsq - 2.0 * fn12[k] * rij_L2sq) * blm[4];
        tmpr += 3.0 * C4B[0] * (
          2.0 * s[0] * dnlm_dc[0] * dnlm_drij[0] + s2[0] * dnlm_drij_dc[0]);
        tmpr += C4B[1] * (
          dnlm_drij_dc[0] * (s2[1] + s2[2]) + dnlm_drij[0] * 2.0 * (s[1] * dnlm_dc[1] + s[2] * dnlm_dc[2])
        );
        tmpr += 2.0 * C4B[1] * (
          dnlm_dc[0] * (s[1] * dnlm_drij[1] + s[2] * dnlm_drij[2]) +
          s[0] * (dnlm_dc[1] * dnlm_drij[1] + s[1] * dnlm_drij_dc[1] +
                    dnlm_dc[2] * dnlm_drij[2] + s[2] * dnlm_drij_dc[2])
        );
        tmpr += C4B[2] * (
          dnlm_drij_dc[0] * (s2[3] + s2[4]) + dnlm_drij[0] * 2.0 * (s[3] * dnlm_dc[3] + s[4] * dnlm_dc[4]));
        tmpr += 2.0 * C4B[2] * (
          dnlm_dc[0] * (s[3] * dnlm_drij[3] + s[4] * dnlm_drij[4]) +
          s[0] * (dnlm_dc[3] * dnlm_drij[3] + s[3] * dnlm_drij_dc[3] +
                    dnlm_dc[4] * dnlm_drij[4] + s[4] * dnlm_drij_dc[4])
        );
        tmpr += C4B[3] * (
          dnlm_drij_dc[3] * (s2[2] - s2[1]) + dnlm_drij[3] * 2.0 * (s[2] * dnlm_dc[2] - s[1] * dnlm_dc[1]));
        tmpr += 2.0 * C4B[3] * (
          dnlm_dc[3] * (s[2] * dnlm_drij[2] - s[1] * dnlm_drij[1]) +
            s[3] * (dnlm_dc[2] * dnlm_drij[2] + s[2] * dnlm_drij_dc[2] -
                    dnlm_dc[1] * dnlm_drij[1] - s[1] * dnlm_drij_dc[1])
        );
        tmpr += C4B[4] * (
          dnlm_drij_dc[1] * s[2] * s[4] + dnlm_drij[1] * dnlm_dc[2] * s[4] + dnlm_drij[1] * s[2] * dnlm_dc[4] +
          dnlm_dc[1] * dnlm_drij[2] * s[4] + s[1] * dnlm_drij_dc[2] * s[4] + s[1] * dnlm_drij[2] * dnlm_dc[4] +
          dnlm_dc[1] * s[2] * dnlm_drij[4] + s[1] * dnlm_dc[2] * dnlm_drij[4] + s[1] * s[2] * dnlm_drij_dc[4]
        );
        sink.add(type_slot, n, k, Fp * scd_r12[0] * tmpr);
      }
      {
        double tmpx = 0.0;
        double dnlm_dxij[5] = {0.0};
        dnlm_dxij[0] = 0.0;
        dnlm_dxij[1] = fn * dblm_x[1];
        dnlm_dxij[2] = 0.0;
        dnlm_dxij[3] = fn * dblm_x[3];
        dnlm_dxij[4] = fn * dblm_x[4];
        double dnlm_dxij_dc[5] = {0.0};
        dnlm_dxij_dc[0] = 0.0;
        dnlm_dxij_dc[1] = fn12[k] * rij_Lsq * dblm_x[1];
        dnlm_dxij_dc[2] = 0.0;
        dnlm_dxij_dc[3] = fn12[k] * rij_Lsq * dblm_x[3];
        dnlm_dxij_dc[4] = fn12[k] * rij_Lsq * dblm_x[4];
        tmpx += 3.0 * C4B[0] * (
          2.0 * s[0] * dnlm_dc[0] * dnlm_dxij[0] + s2[0] * dnlm_dxij_dc[0]);
        tmpx += C4B[1] * (
          dnlm_dxij_dc[0] * (s2[1] + s2[2]) + dnlm_dxij[0] * 2.0 * (s[1] * dnlm_dc[1] + s[2] * dnlm_dc[2])
        );
        tmpx += 2.0 * C4B[1] * (
          dnlm_dc[0] * (s[1] * dnlm_dxij[1] + s[2] * dnlm_dxij[2]) +
          s[0] * (dnlm_dc[1] * dnlm_dxij[1] + s[1] * dnlm_dxij_dc[1] +
                    dnlm_dc[2] * dnlm_dxij[2] + s[2] * dnlm_dxij_dc[2])
        );
        tmpx += C4B[2] * (
          dnlm_dxij_dc[0] * (s2[3] + s2[4]) + dnlm_dxij[0] * 2.0 * (s[3] * dnlm_dc[3] + s[4] * dnlm_dc[4]));
        tmpx += 2.0 * C4B[2] * (
          dnlm_dc[0] * (s[3] * dnlm_dxij[3] + s[4] * dnlm_dxij[4]) +
          s[0] * (dnlm_dc[3] * dnlm_dxij[3] + s[3] * dnlm_dxij_dc[3] +
                    dnlm_dc[4] * dnlm_dxij[4] + s[4] * dnlm_dxij_dc[4])
        );
        tmpx += C4B[3] * (
          dnlm_dxij_dc[3] * (s2[2] - s2[1]) + dnlm_dxij[3] * 2.0 * (s[2] * dnlm_dc[2] - s[1] * dnlm_dc[1]));
        tmpx += 2.0 * C4B[3] * (
          dnlm_dc[3] * (s[2] * dnlm_dxij[2] - s[1] * dnlm_dxij[1]) +
            s[3] * (dnlm_dc[2] * dnlm_dxij[2] + s[2] * dnlm_dxij_dc[2] -
                    dnlm_dc[1] * dnlm_dxij[1] - s[1] * dnlm_dxij_dc[1])
        );
        tmpx += C4B[4] * (
          dnlm_dxij_dc[1] * s[2] * s[4] + dnlm_dxij[1] * dnlm_dc[2] * s[4] + dnlm_dxij[1] * s[2] * dnlm_dc[4] +
          dnlm_dc[1] * dnlm_dxij[2] * s[4] + s[1] * dnlm_dxij_dc[2] * s[4] + s[1] * dnlm_dxij[2] * dnlm_dc[4] +
          dnlm_dc[1] * s[2] * dnlm_dxij[4] + s[1] * dnlm_dc[2] * dnlm_dxij[4] + s[1] * s[2] * dnlm_dxij_dc[4]
        );
        sink.add(type_slot, n, k, Fp * scd_r12[1] * tmpx);
      }
      {
        double tmpy = 0.0;
        double dnlm_dyij[5] = {0.0};
        dnlm_dyij[0] = 0.0;
        dnlm_dyij[1] = 0.0;
        dnlm_dyij[2] = fn * dblm_y[2];
        dnlm_dyij[3] = fn * dblm_y[3];
        dnlm_dyij[4] = fn * dblm_y[4];
        double dnlm_dyij_dc[5] = {0.0};
        dnlm_dyij_dc[0] = 0.0;
        dnlm_dyij_dc[1] = 0.0;
        dnlm_dyij_dc[2] = fn12[k] * rij_Lsq * dblm_y[2];
        dnlm_dyij_dc[3] = fn12[k] * rij_Lsq * dblm_y[3];
        dnlm_dyij_dc[4] = fn12[k] * rij_Lsq * dblm_y[4];
        tmpy += 3.0 * C4B[0] * (
          2.0 * s[0] * dnlm_dc[0] * dnlm_dyij[0] + s2[0] * dnlm_dyij_dc[0]);
        tmpy += C4B[1] * (
          dnlm_dyij_dc[0] * (s2[1] + s2[2]) + dnlm_dyij[0] * 2.0 * (s[1] * dnlm_dc[1] + s[2] * dnlm_dc[2])
        );
        tmpy += 2.0 * C4B[1] * (
          dnlm_dc[0] * (s[1] * dnlm_dyij[1] + s[2] * dnlm_dyij[2]) +
          s[0] * (dnlm_dc[1] * dnlm_dyij[1] + s[1] * dnlm_dyij_dc[1] +
                    dnlm_dc[2] * dnlm_dyij[2] + s[2] * dnlm_dyij_dc[2])
        );
        tmpy += C4B[2] * (
          dnlm_dyij_dc[0] * (s2[3] + s2[4]) + dnlm_dyij[0] * 2.0 * (s[3] * dnlm_dc[3] + s[4] * dnlm_dc[4]));
        tmpy += 2.0 * C4B[2] * (
          dnlm_dc[0] * (s[3] * dnlm_dyij[3] + s[4] * dnlm_dyij[4]) +
          s[0] * (dnlm_dc[3] * dnlm_dyij[3] + s[3] * dnlm_dyij_dc[3] +
                    dnlm_dc[4] * dnlm_dyij[4] + s[4] * dnlm_dyij_dc[4])
        );
        tmpy += C4B[3] * (
          dnlm_dyij_dc[3] * (s2[2] - s2[1]) + dnlm_dyij[3] * 2.0 * (s[2] * dnlm_dc[2] - s[1] * dnlm_dc[1]));
        tmpy += 2.0 * C4B[3] * (
          dnlm_dc[3] * (s[2] * dnlm_dyij[2] - s[1] * dnlm_dyij[1]) +
            s[3] * (dnlm_dc[2] * dnlm_dyij[2] + s[2] * dnlm_dyij_dc[2] -
                    dnlm_dc[1] * dnlm_dyij[1] - s[1] * dnlm_dyij_dc[1])
        );
        tmpy += C4B[4] * (
          dnlm_dyij_dc[1] * s[2] * s[4] + dnlm_dyij[1] * dnlm_dc[2] * s[4] + dnlm_dyij[1] * s[2] * dnlm_dc[4] +
          dnlm_dc[1] * dnlm_dyij[2] * s[4] + s[1] * dnlm_dyij_dc[2] * s[4] + s[1] * dnlm_dyij[2] * dnlm_dc[4] +
          dnlm_dc[1] * s[2] * dnlm_dyij[4] + s[1] * dnlm_dc[2] * dnlm_dyij[4] + s[1] * s[2] * dnlm_dyij_dc[4]
        );
        sink.add(type_slot, n, k, Fp * scd_r12[2] * tmpy);
      }
      {
        double tmpz = 0.0;
        double dnlm_dzij[5] = {0.0};
        dnlm_dzij[0] = fn * dblm_z[0];
        dnlm_dzij[1] = fn * dblm_z[1];
        dnlm_dzij[2] = fn * dblm_z[2];
        dnlm_dzij[3] = 0.0;
        dnlm_dzij[4] = 0.0;
        double dnlm_dzij_dc[5] = {0.0};
        dnlm_dzij_dc[0] = fn12[k] * rij_Lsq * dblm_z[0];
        dnlm_dzij_dc[1] = fn12[k] * rij_Lsq * dblm_z[1];
        dnlm_dzij_dc[2] = fn12[k] * rij_Lsq * dblm_z[2];
        dnlm_dzij_dc[3] = 0.0;
        dnlm_dzij_dc[4] = 0.0;
        tmpz += 3.0 * C4B[0] * (
          2.0 * s[0] * dnlm_dc[0] * dnlm_dzij[0] + s2[0] * dnlm_dzij_dc[0]);
        tmpz += C4B[1] * (
          dnlm_dzij_dc[0] * (s2[1] + s2[2]) + dnlm_dzij[0] * 2.0 * (s[1] * dnlm_dc[1] + s[2] * dnlm_dc[2])
        );
        tmpz += 2.0 * C4B[1] * (
          dnlm_dc[0] * (s[1] * dnlm_dzij[1] + s[2] * dnlm_dzij[2]) +
          s[0] * (dnlm_dc[1] * dnlm_dzij[1] + s[1] * dnlm_dzij_dc[1] +
                    dnlm_dc[2] * dnlm_dzij[2] + s[2] * dnlm_dzij_dc[2])
        );
        tmpz += C4B[2] * (
          dnlm_dzij_dc[0] * (s2[3] + s2[4]) + dnlm_dzij[0] * 2.0 * (s[3] * dnlm_dc[3] + s[4] * dnlm_dc[4]));
        tmpz += 2.0 * C4B[2] * (
          dnlm_dc[0] * (s[3] * dnlm_dzij[3] + s[4] * dnlm_dzij[4]) +
          s[0] * (dnlm_dc[3] * dnlm_dzij[3] + s[3] * dnlm_dzij_dc[3] +
                    dnlm_dc[4] * dnlm_dzij[4] + s[4] * dnlm_dzij_dc[4])
        );
        tmpz += C4B[3] * (
          dnlm_dzij_dc[3] * (s2[2] - s2[1]) + dnlm_dzij[3] * 2.0 * (s[2] * dnlm_dc[2] - s[1] * dnlm_dc[1]));
        tmpz += 2.0 * C4B[3] * (
          dnlm_dc[3] * (s[2] * dnlm_dzij[2] - s[1] * dnlm_dzij[1]) +
            s[3] * (dnlm_dc[2] * dnlm_dzij[2] + s[2] * dnlm_dzij_dc[2] -
                    dnlm_dc[1] * dnlm_dzij[1] - s[1] * dnlm_dzij_dc[1])
        );
        tmpz += C4B[4] * (
          dnlm_dzij_dc[1] * s[2] * s[4] + dnlm_dzij[1] * dnlm_dc[2] * s[4] + dnlm_dzij[1] * s[2] * dnlm_dc[4] +
          dnlm_dc[1] * dnlm_dzij[2] * s[4] + s[1] * dnlm_dzij_dc[2] * s[4] + s[1] * dnlm_dzij[2] * dnlm_dc[4] +
          dnlm_dc[1] * s[2] * dnlm_dzij[4] + s[1] * dnlm_dc[2] * dnlm_dzij[4] + s[1] * s[2] * dnlm_dzij_dc[4]
        );
        sink.add(type_slot, n, k, Fp * scd_r12[3] * tmpz);
      }


  }
}

template<bool CROSS, int NMAX, int NBASIS, int TYPE_TILE>
__device__ __forceinline__ void accumulate_four_body(
  const double* fn12, const double* fnp12,
  const double* blm,
  const double* dblm_x, const double* dblm_y, const double* dblm_z, const double* dblm_r,
  const double* scd_r12, const double* dsnlm_dc, const double* s,
  double rij_Lsq, double rij_L2sq, double fn, double fnp, double Fp,
  int type_slot, int tile_count, int n, SharedTileSink<NMAX, NBASIS, TYPE_TILE> sink)
{
  const AngularContribution c{fn12, fnp12, blm,
      dblm_x, dblm_y, dblm_z, dblm_r, scd_r12, dsnlm_dc, s,
      rij_Lsq, rij_L2sq, fn, fnp, Fp, type_slot, n, sink.values};
  if constexpr (CROSS) {
    for (int uj = 0; uj < tile_count; ++uj) {
      if (type_slot == uj) continue;
      #pragma unroll
      for (int k = 0; k < NBASIS; ++k) {
        accumulate_four_body_basis<CROSS, NMAX, NBASIS, TYPE_TILE>(c, k, uj);
      }
    }
  } else {
    if (type_slot < 0) return;
    #pragma unroll
    for (int k = 0; k < NBASIS; ++k) {
      accumulate_four_body_basis<CROSS, NMAX, NBASIS, TYPE_TILE>(c, k, 0);
    }
  }
}


template<bool CROSS, int NMAX, int NBASIS, int TYPE_TILE>
__device__ __forceinline__ void accumulate_five_body_basis(const AngularContribution& c, int k, int uj)
{
  const double* fn12 = c.fn12;
  const double* fnp12 = c.fnp12;
  const double* blm = c.blm;
  const double* scd_r12 = c.scd_r12;
  const double* dsnlm_dc = c.dsnlm_dc;
  const double* s = c.s;
  const double rij_Lsq = c.rij_Lsq;
  const double rij_L2sq = c.rij_L2sq;
  const double fn = c.fn;
  const double fnp = c.fnp;
  const double Fp = c.Fp;
  const int type_slot = c.type_slot;
  const int n = c.n;
  const SharedTileSink<NMAX, NBASIS, TYPE_TILE> sink{c.output, c.accumulator};

  if constexpr (CROSS) {
    double dnlm_dc[3] = {0.0};
    double s2[3] = {0.0};
    s2[0] = s[0] * s[0];
    s2[1] = s[1] * s[1];
    s2[2] = s[2] * s[2];
    double ds1s2 = 0.0;
    double ds1s2_c = 0.0;
    double d_tmp = 0.0;


      int j = uj;
      if (type_slot == j) return;
      int dsnlm_idx = 0 + j * NBASIS * NUM_OF_ABC;


        int dsnlm_i = dsnlm_idx + k * NUM_OF_ABC;

        dnlm_dc[0] = dsnlm_dc[dsnlm_i + 0];
        dnlm_dc[1] = dsnlm_dc[dsnlm_i + 1];
        dnlm_dc[2] = dsnlm_dc[dsnlm_i + 2];
        {
          double tmpr = 0.0;
          double dnlm_drij[3] = {0.0};
          dnlm_drij[0] = fnp * blm[0];
          dnlm_drij[1] = fnp * blm[1];
          dnlm_drij[2] = fnp * blm[2];
          tmpr += 4.0 * C5B[0] * (3.0 * s2[0] * dnlm_dc[0] * dnlm_drij[0]);
          ds1s2 = s[1] * dnlm_drij[1] + s[2] * dnlm_drij[2];
          ds1s2_c = 2.0 * s[1] * dnlm_dc[1] + 2.0 * s[2] * dnlm_dc[2];
          d_tmp = dnlm_dc[1] * dnlm_drij[1] + dnlm_dc[2] * dnlm_drij[2];
          tmpr += 2.0 * C5B[1] * (
            dnlm_dc[0] * dnlm_drij[0] * (s2[1] + s2[2]) +
            s[0] * dnlm_drij[0] * ds1s2_c + 2.0 * s[0] * dnlm_dc[0] * ds1s2 + s2[0] * d_tmp);
          tmpr += 4.0 * C5B[2] * (ds1s2_c * ds1s2 + (s2[1] + s2[2]) * d_tmp);
          sink.add(uj, n, k, Fp * scd_r12[0] * tmpr);
        }
        {
          double tmpx = 0.0;
          double dnlm_dxij[3] = {0.0};
          dnlm_dxij[0] = 0.0;
          dnlm_dxij[1] = fn;
          dnlm_dxij[2] = 0.0;
          tmpx += 4.0 * C5B[0] * (3.0 * s2[0] * dnlm_dc[0] * dnlm_dxij[0]);
          ds1s2 = s[1] * dnlm_dxij[1] + s[2] * dnlm_dxij[2];
          ds1s2_c = 2.0 * s[1] * dnlm_dc[1] + 2.0 * s[2] * dnlm_dc[2];
          d_tmp = dnlm_dc[1] * dnlm_dxij[1] + dnlm_dc[2] * dnlm_dxij[2];
          tmpx += 2.0 * C5B[1] * (
            dnlm_dc[0] * dnlm_dxij[0] * (s2[1] + s2[2]) +
            s[0] * dnlm_dxij[0] * ds1s2_c + 2.0 * s[0] * dnlm_dc[0] * ds1s2 + s2[0] * d_tmp);
          tmpx += 4.0 * C5B[2] * (ds1s2_c * ds1s2 + (s2[1] + s2[2]) * d_tmp);
          sink.add(uj, n, k, Fp * scd_r12[1] * tmpx);
        }
        {
          double tmpy = 0.0;
          double dnlm_dyij[3] = {0.0};
          dnlm_dyij[0] = 0.0;
          dnlm_dyij[1] = 0.0;
          dnlm_dyij[2] = fn;
          tmpy += 4.0 * C5B[0] * (3.0 * s2[0] * dnlm_dc[0] * dnlm_dyij[0]);
          ds1s2 = s[1] * dnlm_dyij[1] + s[2] * dnlm_dyij[2];
          ds1s2_c = 2.0 * s[1] * dnlm_dc[1] + 2.0 * s[2] * dnlm_dc[2];
          d_tmp = dnlm_dc[1] * dnlm_dyij[1] + dnlm_dc[2] * dnlm_dyij[2];
          tmpy += 2.0 * C5B[1] * (
            dnlm_dc[0] * dnlm_dyij[0] * (s2[1] + s2[2]) +
            s[0] * dnlm_dyij[0] * ds1s2_c + 2.0 * s[0] * dnlm_dc[0] * ds1s2 + s2[0] * d_tmp);
          tmpy += 4.0 * C5B[2] * (ds1s2_c * ds1s2 + (s2[1] + s2[2]) * d_tmp);
          sink.add(uj, n, k, Fp * scd_r12[2] * tmpy);
        }
        {
          double tmpz = 0.0;
          double dnlm_dzij[3] = {0.0};
          dnlm_dzij[0] = fn;
          dnlm_dzij[1] = 0.0;
          dnlm_dzij[2] = 0.0;
          tmpz += 4.0 * C5B[0] * (3.0 * s2[0] * dnlm_dc[0] * dnlm_dzij[0]);
          ds1s2 = s[1] * dnlm_dzij[1] + s[2] * dnlm_dzij[2];
          ds1s2_c = 2.0 * s[1] * dnlm_dc[1] + 2.0 * s[2] * dnlm_dc[2];
          d_tmp = dnlm_dc[1] * dnlm_dzij[1] + dnlm_dc[2] * dnlm_dzij[2];
          tmpz += 2.0 * C5B[1] * (
            dnlm_dc[0] * dnlm_dzij[0] * (s2[1] + s2[2]) +
            s[0] * dnlm_dzij[0] * ds1s2_c + 2.0 * s[0] * dnlm_dc[0] * ds1s2 + s2[0] * d_tmp);
          tmpz += 4.0 * C5B[2] * (ds1s2_c * ds1s2 + (s2[1] + s2[2]) * d_tmp);
          sink.add(uj, n, k, Fp * scd_r12[3] * tmpz);
        }



  } else {
    if (type_slot < 0) return;

    int dsnlm_idx = 0 + type_slot * NBASIS * NUM_OF_ABC;
    double dnlm_dc[3] = {0.0};
    double s2[3] = {0.0};
    s2[0] = s[0] * s[0];
    s2[1] = s[1] * s[1];
    s2[2] = s[2] * s[2];
    double ds1s2 = 0.0;
    double ds1s2_c = 0.0;
    double d_tmp = 0.0;

      int dsnlm_i = dsnlm_idx + k * NUM_OF_ABC;
      dnlm_dc[0] = dsnlm_dc[dsnlm_i + 0];
      dnlm_dc[1] = dsnlm_dc[dsnlm_i + 1];
      dnlm_dc[2] = dsnlm_dc[dsnlm_i + 2];
      {
        double tmpr = 0.0;
        double dnlm_drij[3] = {0.0};
        dnlm_drij[0] = fnp * blm[0];
        dnlm_drij[1] = fnp * blm[1];
        dnlm_drij[2] = fnp * blm[2];
        double dnlm_drij_dc[3] = {0.0};
        dnlm_drij_dc[0] = (fnp12[k] * rij_Lsq - fn12[k] * rij_L2sq) * blm[0];
        dnlm_drij_dc[1] = (fnp12[k] * rij_Lsq - fn12[k] * rij_L2sq) * blm[1];
        dnlm_drij_dc[2] = (fnp12[k] * rij_Lsq - fn12[k] * rij_L2sq) * blm[2];
        tmpr += 4.0 * C5B[0] * (3.0 * s2[0] * dnlm_dc[0] * dnlm_drij[0] + s2[0] * s[0] * dnlm_drij_dc[0]);
        ds1s2 = s[1] * dnlm_drij[1] + s[2] * dnlm_drij[2];
        ds1s2_c = 2.0 * s[1] * dnlm_dc[1] + 2.0 * s[2] * dnlm_dc[2];
        d_tmp = dnlm_dc[1] * dnlm_drij[1] + s[1] * dnlm_drij_dc[1] + dnlm_dc[2] * dnlm_drij[2] + s[2] * dnlm_drij_dc[2];
        tmpr += 2.0 * C5B[1] * (
          dnlm_dc[0] * dnlm_drij[0] * (s2[1] + s2[2]) + s[0] * dnlm_drij_dc[0] * (s2[1] + s2[2]) +
          s[0] * dnlm_drij[0] * ds1s2_c + 2.0 * s[0] * dnlm_dc[0] * ds1s2 + s2[0] * d_tmp);
        tmpr += 4.0 * C5B[2] * (ds1s2_c * ds1s2 + (s2[1] + s2[2]) * d_tmp);
        sink.add(type_slot, n, k, Fp * scd_r12[0] * tmpr);
      }
      {
        double tmpx = 0.0;
        double dnlm_dxij[3] = {0.0};
        dnlm_dxij[0] = 0.0;
        dnlm_dxij[1] = fn;
        dnlm_dxij[2] = 0.0;
        double dnlm_dxij_dc[3] = {0.0};
        dnlm_dxij_dc[1] = fn12[k] * rij_Lsq;
        tmpx += 4.0 * C5B[0] * (3.0 * s2[0] * dnlm_dc[0] * dnlm_dxij[0] + s2[0] * s[0] * dnlm_dxij_dc[0]);
        ds1s2 = s[1] * dnlm_dxij[1] + s[2] * dnlm_dxij[2];
        ds1s2_c = 2.0 * s[1] * dnlm_dc[1] + 2.0 * s[2] * dnlm_dc[2];
        d_tmp = dnlm_dc[1] * dnlm_dxij[1] + s[1] * dnlm_dxij_dc[1] + dnlm_dc[2] * dnlm_dxij[2] + s[2] * dnlm_dxij_dc[2];
        tmpx += 2.0 * C5B[1] * (
          dnlm_dc[0] * dnlm_dxij[0] * (s2[1] + s2[2]) + s[0] * dnlm_dxij_dc[0] * (s2[1] + s2[2]) +
          s[0] * dnlm_dxij[0] * ds1s2_c + 2.0 * s[0] * dnlm_dc[0] * ds1s2 + s2[0] * d_tmp);
        tmpx += 4.0 * C5B[2] * (ds1s2_c * ds1s2 + (s2[1] + s2[2]) * d_tmp);
        sink.add(type_slot, n, k, Fp * scd_r12[1] * tmpx);
      }
      {
        double tmpy = 0.0;
        double dnlm_dyij[3] = {0.0};
        dnlm_dyij[0] = 0.0;
        dnlm_dyij[1] = 0.0;
        dnlm_dyij[2] = fn;
        double dnlm_dyij_dc[3] = {0.0};
        dnlm_dyij_dc[2] = fn12[k] * rij_Lsq;
        tmpy += 4.0 * C5B[0] * (3.0 * s2[0] * dnlm_dc[0] * dnlm_dyij[0] + s2[0] * s[0] * dnlm_dyij_dc[0]);
        ds1s2 = s[1] * dnlm_dyij[1] + s[2] * dnlm_dyij[2];
        ds1s2_c = 2.0 * s[1] * dnlm_dc[1] + 2.0 * s[2] * dnlm_dc[2];
        d_tmp = dnlm_dc[1] * dnlm_dyij[1] + s[1] * dnlm_dyij_dc[1] + dnlm_dc[2] * dnlm_dyij[2] + s[2] * dnlm_dyij_dc[2];
        tmpy += 2.0 * C5B[1] * (
          dnlm_dc[0] * dnlm_dyij[0] * (s2[1] + s2[2]) + s[0] * dnlm_dyij_dc[0] * (s2[1] + s2[2]) +
          s[0] * dnlm_dyij[0] * ds1s2_c + 2.0 * s[0] * dnlm_dc[0] * ds1s2 + s2[0] * d_tmp);
        tmpy += 4.0 * C5B[2] * (ds1s2_c * ds1s2 + (s2[1] + s2[2]) * d_tmp);
        sink.add(type_slot, n, k, Fp * scd_r12[2] * tmpy);
      }
      {
        double tmpz = 0.0;
        double dnlm_dzij[3] = {0.0};
        dnlm_dzij[0] = fn;
        dnlm_dzij[1] = 0.0;
        dnlm_dzij[2] = 0.0;
        double dnlm_dzij_dc[3] = {0.0};
        dnlm_dzij_dc[0] = fn12[k] * rij_Lsq;
        tmpz += 4.0 * C5B[0] * (3.0 * s2[0] * dnlm_dc[0] * dnlm_dzij[0] + s2[0] * s[0] * dnlm_dzij_dc[0]);
        ds1s2 = s[1] * dnlm_dzij[1] + s[2] * dnlm_dzij[2];
        ds1s2_c = 2.0 * s[1] * dnlm_dc[1] + 2.0 * s[2] * dnlm_dc[2];
        d_tmp = dnlm_dc[1] * dnlm_dzij[1] + s[1] * dnlm_dzij_dc[1] + dnlm_dc[2] * dnlm_dzij[2] + s[2] * dnlm_dzij_dc[2];
        tmpz += 2.0 * C5B[1] * (
          dnlm_dc[0] * dnlm_dzij[0] * (s2[1] + s2[2]) + s[0] * dnlm_dzij_dc[0] * (s2[1] + s2[2]) +
          s[0] * dnlm_dzij[0] * ds1s2_c + 2.0 * s[0] * dnlm_dc[0] * ds1s2 + s2[0] * d_tmp);
        tmpz += 4.0 * C5B[2] * (ds1s2_c * ds1s2 + (s2[1] + s2[2]) * d_tmp);
        sink.add(type_slot, n, k, Fp * scd_r12[3] * tmpz);
      }


  }
}

template<bool CROSS, int NMAX, int NBASIS, int TYPE_TILE>
__device__ __forceinline__ void accumulate_five_body(
  const double* fn12, const double* fnp12,
  const double* blm,
  const double* dblm_x, const double* dblm_y, const double* dblm_z, const double* dblm_r,
  const double* scd_r12, const double* dsnlm_dc, const double* s,
  double rij_Lsq, double rij_L2sq, double fn, double fnp, double Fp,
  int type_slot, int tile_count, int n, SharedTileSink<NMAX, NBASIS, TYPE_TILE> sink)
{
  const AngularContribution c{fn12, fnp12, blm,
      dblm_x, dblm_y, dblm_z, dblm_r, scd_r12, dsnlm_dc, s,
      rij_Lsq, rij_L2sq, fn, fnp, Fp, type_slot, n, sink.values};
  if constexpr (CROSS) {
    for (int uj = 0; uj < tile_count; ++uj) {
      if (type_slot == uj) continue;
      #pragma unroll
      for (int k = 0; k < NBASIS; ++k) {
        accumulate_five_body_basis<CROSS, NMAX, NBASIS, TYPE_TILE>(c, k, uj);
      }
    }
  } else {
    if (type_slot < 0) return;
    #pragma unroll
    for (int k = 0; k < NBASIS; ++k) {
      accumulate_five_body_basis<CROSS, NMAX, NBASIS, TYPE_TILE>(c, k, 0);
    }
  }
}


template<int L, int NMAX, int NBASIS, int LMAX3, bool HAS4, bool HAS5, int TYPE_TILE>
// The call owns all order-specific temporaries. Keeping this boundary, with
// only a context pointer as its argument, prevents register lifetimes from
// spanning angular orders without imposing an artificial register cap.
__device__ __forceinline__ void accumulate_angular_order(const NeighborContext<NBASIS>& w)
{
  const int n = w.n, type_slot = w.type_slot, tile_count = w.tile_count;
  const double fn = w.fn, fnp = w.fnp;
  const double rij_Lsq = w.rij_Lsq, rij_L2sq = w.rij_L2sq;
  const double* scd_r12 = w.scd_r12;
  const double* fn12 = w.fn12;
  const double* fnp12 = w.fnp12;
  const double* Fp = w.Fp;
  const double* dsnlm_dc = w.dsnlm_dc;
  const double* sum_fxyz = w.sum_fxyz;
  const SharedTileSink<NMAX, NBASIS, TYPE_TILE> sink{w.output};
  constexpr int offset = L * L - 1;
  const auto scratch = angular_workspace_view(w.angular, offset);
  const double* sums = sum_fxyz + n * NUM_OF_ABC + offset;
  const double* weighted_sums = w.weighted_sums + n * NUM_OF_ABC + offset;

  // For L=1/2, the higher-body and angular terms target the same addresses.
  // Combine them per (type slot, basis) while keeping the live range inside
  // this angular order; L=3/4 retain the original one-atomic-per-address path.
  if constexpr ((L == 1 && HAS5) || (L == 2 && HAS4)) {
    for (int output_slot = 0; output_slot < tile_count; ++output_slot) {
      const bool own_type = type_slot == output_slot;
      #pragma unroll
      for (int k = 0; k < NBASIS; ++k) {
        double basis_sum = 0.0;
        const AngularContribution body{fn12, fnp12, scratch.blm,
            scratch.dblm_x, scratch.dblm_y, scratch.dblm_z, scratch.dblm_r,
            scd_r12, dsnlm_dc, sums, rij_Lsq, rij_L2sq, fn, fnp,
            Fp[NMAX * LMAX3 + (L == 1 ? NMAX : 0) + n],
            type_slot, n, w.output, &basis_sum};
        if constexpr (L == 1) {
          if (own_type) {
            accumulate_five_body_basis<false, NMAX, NBASIS, TYPE_TILE>(body, k, 0);
          } else {
            accumulate_five_body_basis<true, NMAX, NBASIS, TYPE_TILE>(body, k, output_slot);
          }
        } else {
          if (own_type) {
            accumulate_four_body_basis<false, NMAX, NBASIS, TYPE_TILE>(body, k, 0);
          } else {
            accumulate_four_body_basis<true, NMAX, NBASIS, TYPE_TILE>(body, k, output_slot);
          }
        }

        const AngularContribution angular{fn12, fnp12, scratch.blm,
            scratch.dblm_x, scratch.dblm_y, scratch.dblm_z, scratch.dblm_r,
            scd_r12, dsnlm_dc, weighted_sums,
            rij_Lsq, rij_L2sq, fn, fnp, Fp[n * LMAX3 + L - 1],
            type_slot, n, w.output, &basis_sum};
        if (own_type) {
          accumulate_direct_basis<L, NMAX, NBASIS, TYPE_TILE>(angular, k, 0);
        } else {
          accumulate_cross_basis<L, NMAX, NBASIS, TYPE_TILE>(angular, k, output_slot);
        }
        atomicAdd(&w.output[(output_slot * NMAX + n) * NBASIS + k], basis_sum);
      }
    }
    return;
  }
  // Keep the legacy five-body, L=1, four-body, L=2, L=3, L=4 sequence.
  // The higher-body terms consume the unweighted sums of the same order.
  if constexpr (L == 1 && HAS5) {
    accumulate_five_body<false, NMAX, NBASIS, TYPE_TILE>(fn12, fnp12,
        scratch.blm, scratch.dblm_x, scratch.dblm_y, scratch.dblm_z, scratch.dblm_r,
        scd_r12, dsnlm_dc, sums,
        rij_Lsq, rij_L2sq, fn, fnp, Fp[NMAX * LMAX3 + NMAX + n], type_slot, tile_count, n, sink);
    accumulate_five_body<true, NMAX, NBASIS, TYPE_TILE>(fn12, fnp12,
        scratch.blm, scratch.dblm_x, scratch.dblm_y, scratch.dblm_z, scratch.dblm_r,
        scd_r12, dsnlm_dc, sums,
        rij_Lsq, rij_L2sq, fn, fnp, Fp[NMAX * LMAX3 + NMAX + n], type_slot, tile_count, n, sink);
  }
  if constexpr (L == 2 && HAS4) {
    accumulate_four_body<false, NMAX, NBASIS, TYPE_TILE>(fn12, fnp12,
        scratch.blm, scratch.dblm_x, scratch.dblm_y, scratch.dblm_z, scratch.dblm_r,
        scd_r12, dsnlm_dc, sums,
        rij_Lsq, rij_L2sq, fn, fnp, Fp[NMAX * LMAX3 + n], type_slot, tile_count, n, sink);
    accumulate_four_body<true, NMAX, NBASIS, TYPE_TILE>(fn12, fnp12,
        scratch.blm, scratch.dblm_x, scratch.dblm_y, scratch.dblm_z, scratch.dblm_r,
        scd_r12, dsnlm_dc, sums,
        rij_Lsq, rij_L2sq, fn, fnp, Fp[NMAX * LMAX3 + n], type_slot, tile_count, n, sink);
  }
  accumulate_direct<L, NMAX, NBASIS, TYPE_TILE>(fn12, fnp12,
        scratch.blm, scratch.dblm_x, scratch.dblm_y, scratch.dblm_z, scratch.dblm_r,
        scd_r12, dsnlm_dc, weighted_sums,
        rij_Lsq, rij_L2sq, fn, fnp, Fp[n * LMAX3 + L - 1], type_slot, n, sink);
  accumulate_cross<L, NMAX, NBASIS, TYPE_TILE>(fn12, fnp12,
        scratch.blm, scratch.dblm_x, scratch.dblm_y, scratch.dblm_z, scratch.dblm_r,
        scd_r12, dsnlm_dc, weighted_sums,
        rij_Lsq, rij_L2sq, fn, fnp, Fp[n * LMAX3 + L - 1], type_slot, tile_count, n, sink);
}

// The angular producer owns all polynomial arrays. This consumer retains only
// the radial recurrence while visiting the four orders in their original order.
template<int NMAX, int NBASIS, int LMAX3, bool HAS4, bool HAS5, int TYPE_TILE>
__device__ __forceinline__ void accumulate_neighbor(NeighborContext<NBASIS>& w)
{
  static_assert(LMAX3 == 4, "This numerical port supports the OMat24 angular orders");
  w.d12inv = 1.0 / w.d12;
  w.rij_Lsq = w.d12inv;
  w.rij_L2sq = w.d12inv * w.d12inv;
  w.fnp = w.fnp * w.d12inv - w.fn * w.d12inv * w.d12inv;
  w.fn = w.fn * w.d12inv;
  accumulate_angular_order<1, NMAX, NBASIS, LMAX3, HAS4, HAS5, TYPE_TILE>(w);
  w.fnp = w.fnp * w.d12inv - w.fn * w.d12inv * w.d12inv;
  w.fn = w.fn * w.d12inv;
  w.rij_Lsq = w.rij_L2sq;
  w.rij_L2sq = w.rij_L2sq * w.d12inv;
  accumulate_angular_order<2, NMAX, NBASIS, LMAX3, HAS4, HAS5, TYPE_TILE>(w);
  w.fnp = w.fnp * w.d12inv - w.fn * w.d12inv * w.d12inv;
  w.fn = w.fn * w.d12inv;
  w.rij_Lsq = w.rij_L2sq;
  w.rij_L2sq = w.rij_L2sq * w.d12inv;
  accumulate_angular_order<3, NMAX, NBASIS, LMAX3, HAS4, HAS5, TYPE_TILE>(w);
  w.fnp = w.fnp * w.d12inv - w.fn * w.d12inv * w.d12inv;
  w.fn = w.fn * w.d12inv;
  w.rij_Lsq = w.rij_L2sq;
  w.rij_L2sq = w.rij_L2sq * w.d12inv;
  accumulate_angular_order<4, NMAX, NBASIS, LMAX3, HAS4, HAS5, TYPE_TILE>(w);
}

} // namespace nep_mb_secondgrad_opt

template<int NMAX, int NBASIS, int TYPE_TILE>
__host__ __device__ constexpr size_t nep_mb_secondgrad_shared_bytes() {
  return nep_mb_secondgrad_opt::SharedLayout<NMAX, NBASIS, TYPE_TILE>::bytes;
}

template<int NMAX, int NBASIS, int CTA_THREADS>
__device__ __forceinline__ void nep_mb_precompute_body(NepMbSecondGradArgs a) {
  using namespace nep_mb_secondgrad_opt;
  const int row = blockIdx.x * CTA_THREADS + threadIdx.x;
  const int row_count = a.atom_count * a.max_neighbors;
  const int weighted_value_count = a.atom_count * NMAX * 24;
  const int grid_stride = gridDim.x * CTA_THREADS;
  for (int i = row; i < weighted_value_count; i += grid_stride) {
    const_cast<double*>(a.weighted_sums)[i] =
        a.sum_fxyz[i] * C3B[i % 24];
  }
  if (row >= row_count) return;
  const int64_t neighbor = a.neighbor_list[row];
  if (neighbor < 0) return;
  const double distance = a.d12[row * 4];
  if (distance > a.rcut) return;
  double* angular = const_cast<double*>(a.angular_workspace) + row * (120 + 2 * NBASIS);
  const double x = a.d12[row * 4 + 1];
  const double y = a.d12[row * 4 + 2];
  const double z = a.d12[row * 4 + 3];
  build_angular_scratch<1>(distance, x, y, z, angular_workspace_view(angular, 0));
  build_angular_scratch<2>(distance, x, y, z, angular_workspace_view(angular, 3));
  build_angular_scratch<3>(distance, x, y, z, angular_workspace_view(angular, 8));
  build_angular_scratch<4>(distance, x, y, z, angular_workspace_view(angular, 15));
  double fc12, fcp12;
  find_fc_and_fcp(a.rcut, a.rcut_inv, distance, fc12, fcp12);
  find_fn_and_fnp(NBASIS, a.rcut_inv, distance, fc12, fcp12,
                  angular + 120, angular + 120 + NBASIS);
}

template<int NMAX, int NBASIS, int LMAX3, bool HAS4, bool HAS5, int TYPE_TILE, int CTA_THREADS>
__device__ __forceinline__ void nep_mb_fused_body(NepMbSecondGradArgs a) {
  using namespace nep_mb_secondgrad_opt;
  static_assert(LMAX3 == 4 && (!HAS5 || HAS4), "Supported NEP angular channels");
  static_assert(CTA_THREADS == 128, "Reused HIP optimized CTA width");
  extern __shared__ __align__(8) unsigned char shared_raw[];
  SharedLayout<NMAX, NBASIS, TYPE_TILE> s(shared_raw);
  const int center = blockIdx.x;
  const int tid = threadIdx.x;
  const int center_type = static_cast<int>(a.atom_type[center]);
  for (int i = tid; i < 4; i += CTA_THREADS) s.type_bits[i] = 0;
  if (tid == 0) *s.local_type_count = 0;

  // The public argument points at the first many-body column. Its row stride
  // includes the preceding two-body columns, as in the legacy launcher.
  const int fp_start = center * (a.feat_2b_count + a.many_body_feat_count);
  constexpr int CHANNELS = LMAX3 + (HAS4 ? 1 : 0) + (HAS5 ? 1 : 0);
  for (int i = tid; i < NMAX * CHANNELS; i += CTA_THREADS) {
    const int column = i < NMAX * LMAX3
        ? (i % LMAX3) * NMAX + i / LMAX3 : i;
    s.fp[i] = a.de_dfeat[fp_start + column];
  }
  for (int i = tid; i < NMAX * 24; i += CTA_THREADS) {
    s.sum_fxyz[i] = a.sum_fxyz[center * NMAX * 24 + i];
  }
  __syncthreads();

  // Discover only types that can contribute to this center. The ascending list
  // is metadata; neither the atom storage nor neighbor order is changed.
  for (int j = tid; j < a.max_neighbors; j += CTA_THREADS) {
    const int row = center * a.max_neighbors + j;
    const int64_t neighbor = a.neighbor_list[row];
    if (neighbor >= 0 && a.d12[row * 4] <= a.rcut) {
      const int type = static_cast<int>(a.atom_type[neighbor]);
      atomicOr(&s.type_bits[type >> 5], 1u << (type & 31));
    }
  }
  __syncthreads();
  if (tid == 0) {
    for (int type = 0; type < a.atom_types; ++type) {
      if (s.type_bits[type >> 5] & (1u << (type & 31))) {
        s.local_types[(*s.local_type_count)++] = type;
      }
    }
  }
  __syncthreads();

  const SharedTileSink<NMAX, NBASIS, TYPE_TILE> sink{s.output_tile};
  for (int tile = 0; tile < *s.local_type_count; tile += TYPE_TILE) {
    const int remaining = *s.local_type_count - tile;
    const int tile_count = remaining < TYPE_TILE ? remaining : TYPE_TILE;
    for (int i = tid; i < TYPE_TILE * NMAX * NBASIS; i += CTA_THREADS) {
      s.output_tile[i] = 0.0;
    }
    for (int i = tid; i < TYPE_TILE * NBASIS * 24; i += CTA_THREADS) {
      s.dsnlm_tile[i] = 0.0;
    }
    __syncthreads();

    // dsnlm/dc depends only on geometry and the radial basis. Rebuild the
    // active type tile in shared memory instead of retaining the dense
    // [atom, all element types, basis, 24] tensor between autograd passes.
    // Each neighbor belongs to exactly one tile, so the total arithmetic over
    // all tiles is linear in the number of valid neighbors.
    for (int j = tid; j < a.max_neighbors; j += CTA_THREADS) {
      const int row = center * a.max_neighbors + j;
      const int64_t neighbor = a.neighbor_list[row];
      if (neighbor < 0) continue;
      const double distance = a.d12[row * 4];
      if (distance > a.rcut) continue;
      const int neighbor_type = static_cast<int>(a.atom_type[neighbor]);
      int type_slot = -1;
      for (int slot = 0; slot < tile_count; ++slot) {
        if (s.local_types[tile + slot] == neighbor_type) type_slot = slot;
      }
      if (type_slot < 0) continue;

      const double* fn12 = a.angular_workspace + row * (120 + 2 * NBASIS) + 120;
      double angular[24] = {0.0};
      accumulate_blm_rij(
          distance, a.d12[row * 4 + 1], a.d12[row * 4 + 2],
          a.d12[row * 4 + 3], angular);
      #pragma unroll
      for (int k = 0; k < NBASIS; ++k) {
        #pragma unroll
        for (int m = 0; m < 24; ++m) {
          atomicAdd(&s.dsnlm_tile[(type_slot * NBASIS + k) * 24 + m],
                    angular[m] * fn12[k]);
        }
      }
    }
    __syncthreads();

    // Expand each valid neighbor into NMAX independent tasks. OMat24 has only
    // a few angular neighbors per center, so assigning an entire neighbor to
    // one lane leaves most of the warp idle during the expensive angular
    // algebra. Distributing (neighbor, n) pairs fills those lanes while
    // preserving the arithmetic and accumulation order inside each pair.
    const int neighbor_n_tasks = a.max_neighbors * NMAX;
    for (int task = tid; task < neighbor_n_tasks; task += CTA_THREADS) {
      const int j = task / NMAX;
      const int n = task - j * NMAX;
      const int row = center * a.max_neighbors + j;
      const int64_t neighbor = a.neighbor_list[row];
      if (neighbor < 0) continue;
      const double distance = a.d12[row * 4];
      if (distance > a.rcut) continue;
      const int neighbor_type = static_cast<int>(a.atom_type[neighbor]);
      int type_slot = -1;
      for (int slot = 0; slot < tile_count; ++slot) {
        if (s.local_types[tile + slot] == neighbor_type) type_slot = slot;
      }
      NeighborContext<NBASIS> work;
      work.d12 = distance;
      work.scd_r12[0] = a.grad_second[row * 4];
      work.scd_r12[1] = a.grad_second[row * 4 + 1];
      work.scd_r12[2] = a.grad_second[row * 4 + 2];
      work.scd_r12[3] = a.grad_second[row * 4 + 3];
      work.Fp = s.fp;
      work.dsnlm_dc = s.dsnlm_tile;
      work.sum_fxyz = s.sum_fxyz;
      work.weighted_sums = a.weighted_sums + center * NMAX * 24;
      work.angular = a.angular_workspace + row * (120 + 2 * NBASIS);
      work.output = sink.values;
      work.n = n;
      work.type_slot = type_slot;
      work.tile_count = tile_count;
      work.fn12 = work.angular + 120;
      work.fnp12 = work.angular + 120 + NBASIS;
      const double* fn12 = work.fn12;
      const double* fnp12 = work.fnp12;

      const int coeff_start =
          ((center_type * a.atom_types + neighbor_type) * NMAX + n) * NBASIS;
      double gn12 = 0.0, gnp12 = 0.0;
      #pragma unroll
      for (int k = 0; k < NBASIS; ++k) {
        gn12 += fn12[k] * a.coeff3[coeff_start + k];
        gnp12 += fnp12[k] * a.coeff3[coeff_start + k];
      }
      work.fn = gn12;
      work.fnp = gnp12;
      accumulate_neighbor<NMAX, NBASIS, LMAX3, HAS4, HAS5, TYPE_TILE>(work);
    }
    __syncthreads();
    for (int i = tid; i < tile_count * NMAX * NBASIS; i += CTA_THREADS) {
      const int type = s.local_types[tile + i / (NMAX * NBASIS)];
      const int output = (center_type * a.atom_types + type) * NMAX * NBASIS
                          + i % (NMAX * NBASIS);
      atomicAdd(&a.gradsecond_c3[output], s.output_tile[i]);
    }
    __syncthreads();
  }
}
extern "C" __global__ void nep_mb_secondgrad_precompute_angular(NepMbSecondGradArgs a) {
 nep_mb_precompute_body<JIT_NN,JIT_KB,256>(a);
}
extern "C" __global__ void nep_mb_secondgrad_fused(NepMbSecondGradArgs a) {
 nep_mb_fused_body<JIT_NN,JIT_KB,4,JIT_HAS4,JIT_HAS5,4,128>(a);
}

)MATPLMBRTC";
