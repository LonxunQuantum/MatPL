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

#ifndef TYPESIN
#define TYPESIN 20
#endif
const int TYPES = TYPESIN;

// const int TYPES = 20;
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