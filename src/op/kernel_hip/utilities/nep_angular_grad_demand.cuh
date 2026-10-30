// 本文件的 C3B/C4B/C5B 使用与 nep_utilities.cuh 相同的双精度字面常数。
// 角向一阶反向按输出需求裁剪，移植自本树 CUDA 路径。
// helper 的局部系数梯度布局与 CUDA 保持一致，避免使用 HIP 旧 helper 的全局布局。
// 基础数学函数和常数来自 HIP nep_utilities.cuh；GPUMD 许可证同该文件。
#pragma once
namespace matpl_angular_demand {
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
  double* dfeat_c3_base,
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
    sum_fxyz[n * NUM_OF_ABC + 0] * (0.238732414637843),
    sum_fxyz[n * NUM_OF_ABC + 1] * (0.119366207318922),
    sum_fxyz[n * NUM_OF_ABC + 2] * (0.119366207318922)};
  get_f12_1(d12inv, fn, fnp, Fp[n*lmax_3], s1, r12, f12, f12d, n1, n2);

  // l = 2
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  double s2[5] = {
    sum_fxyz[n * NUM_OF_ABC + 3] * (0.099471839432435),
    sum_fxyz[n * NUM_OF_ABC + 4] * (0.596831036594608),
    sum_fxyz[n * NUM_OF_ABC + 5] * (0.596831036594608),
    sum_fxyz[n * NUM_OF_ABC + 6] * (0.149207759148652),
    sum_fxyz[n * NUM_OF_ABC + 7] * (0.149207759148652)};
  get_f12_2(d12, d12inv, fn, fnp, Fp[n*lmax_3+1], s2, r12, f12, f12d+4, n1, n2);
  // l = 3
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  double s3[7] = {
    sum_fxyz[n * NUM_OF_ABC + 8] * (0.139260575205408),
    sum_fxyz[n * NUM_OF_ABC + 9] * (0.104445431404056),
    sum_fxyz[n * NUM_OF_ABC + 10] * (0.104445431404056),
    sum_fxyz[n * NUM_OF_ABC + 11] * (1.044454314040563),
    sum_fxyz[n * NUM_OF_ABC + 12] * (1.044454314040563),
    sum_fxyz[n * NUM_OF_ABC + 13] * (0.174075719006761),
    sum_fxyz[n * NUM_OF_ABC + 14] * (0.174075719006761)};
  get_f12_3(d12, d12inv, fn, fnp, Fp[n*lmax_3+2], s3, r12, f12, f12d+8, n1, n2);
  // l = 4
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  double s4[9] = {
    sum_fxyz[n * NUM_OF_ABC + 15] * (0.011190581936149),
    sum_fxyz[n * NUM_OF_ABC + 16] * (0.223811638722978),
    sum_fxyz[n * NUM_OF_ABC + 17] * (0.223811638722978),
    sum_fxyz[n * NUM_OF_ABC + 18] * (0.111905819361489),
    sum_fxyz[n * NUM_OF_ABC + 19] * (0.111905819361489),
    sum_fxyz[n * NUM_OF_ABC + 20] * (1.566681471060845),
    sum_fxyz[n * NUM_OF_ABC + 21] * (1.566681471060845),
    sum_fxyz[n * NUM_OF_ABC + 22] * (0.195835183882606),
    sum_fxyz[n * NUM_OF_ABC + 23] * (0.195835183882606)};
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
    // int dc_id = dc_start_idx + type_j * n_max_angular * n_base_angular + n*n_base_angular + kk;
    dfeat_c3_base[kk] += tmp1;
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
  double* dfeat_c3_base,
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
    sum_fxyz[n * NUM_OF_ABC + 0] * (0.238732414637843),
    sum_fxyz[n * NUM_OF_ABC + 1] * (0.119366207318922),
    sum_fxyz[n * NUM_OF_ABC + 2] * (0.119366207318922)};
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
  s2[0] *= (0.099471839432435);
  s2[1] *= (0.596831036594608);
  s2[2] *= (0.596831036594608);
  s2[3] *= (0.149207759148652);
  s2[4] *= (0.149207759148652);
  get_f12_2(d12, d12inv, fn, fnp, Fp[n*lmax_3+1], s2, r12, f12, f12d + 4,  n1, n2);
  // l = 3
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  double s3[7] = {
    sum_fxyz[n * NUM_OF_ABC + 8] * (0.139260575205408),
    sum_fxyz[n * NUM_OF_ABC + 9] * (0.104445431404056),
    sum_fxyz[n * NUM_OF_ABC + 10] * (0.104445431404056),
    sum_fxyz[n * NUM_OF_ABC + 11] * (1.044454314040563),
    sum_fxyz[n * NUM_OF_ABC + 12] * (1.044454314040563),
    sum_fxyz[n * NUM_OF_ABC + 13] * (0.174075719006761),
    sum_fxyz[n * NUM_OF_ABC + 14] * (0.174075719006761)};
  get_f12_3(d12, d12inv, fn, fnp, Fp[n*lmax_3+2], s3, r12, f12, f12d + 8, n1, n2);
  // l = 4
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  double s4[9] = {
    sum_fxyz[n * NUM_OF_ABC + 15] * (0.011190581936149),
    sum_fxyz[n * NUM_OF_ABC + 16] * (0.223811638722978),
    sum_fxyz[n * NUM_OF_ABC + 17] * (0.223811638722978),
    sum_fxyz[n * NUM_OF_ABC + 18] * (0.111905819361489),
    sum_fxyz[n * NUM_OF_ABC + 19] * (0.111905819361489),
    sum_fxyz[n * NUM_OF_ABC + 20] * (1.566681471060845),
    sum_fxyz[n * NUM_OF_ABC + 21] * (1.566681471060845),
    sum_fxyz[n * NUM_OF_ABC + 22] * (0.195835183882606),
    sum_fxyz[n * NUM_OF_ABC + 23] * (0.195835183882606)};
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

    double tmp2_4b = 3.0 * (-0.007499480826664) * s24bsq[0] * ds0_r + 
              (-0.134990654879954) * ds0_r * (s24bsq[1] + s24bsq[2]) + (-0.134990654879954) * s24b[0] * (2.0 * s24b[1] * ds1_r + 2.0 * s24b[2] * ds2_r) +
              (0.067495327439977) * ds0_r * (s24bsq[3] + s24bsq[4]) + (0.067495327439977) * s24b[0] * (2.0 * s24b[3] * ds3_r + 2.0 * s24b[4] * ds4_r) +
              (0.404971964639861) * ds3_r * (s24bsq[2] - s24bsq[1]) + (0.404971964639861) * s24b[3] * (2.0 * s24b[2] * ds2_r - 2.0 * s24b[1] * ds1_r) +
              (-0.809943929279723) *(ds1_r * s24b[2] * s24b[4] + s24b[1] * ds2_r * s24b[4] + s24b[1] * s24b[2] * ds4_r);

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
    // int dc_id = dc_start_idx + type_j * n_max_angular * n_base_angular + n*n_base_angular + kk;
    dfeat_c3_base[kk] += tmp1;
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
  double* dfeat_c3_base,
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
  
  s1[0] *= (0.238732414637843);
  s1[1] *= (0.119366207318922);
  s1[2] *= (0.119366207318922);
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
  s2[0] *= (0.099471839432435);
  s2[1] *= (0.596831036594608);
  s2[2] *= (0.596831036594608);
  s2[3] *= (0.149207759148652);
  s2[4] *= (0.149207759148652);
  get_f12_2(d12, d12inv, fn, fnp, Fp[n*lmax_3+1], s2, r12, f12, f12d+4, n1, n2);

  // l = 3
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  double s3[7] = {
    sum_fxyz[n * NUM_OF_ABC + 8] * (0.139260575205408),
    sum_fxyz[n * NUM_OF_ABC + 9] * (0.104445431404056),
    sum_fxyz[n * NUM_OF_ABC + 10] * (0.104445431404056),
    sum_fxyz[n * NUM_OF_ABC + 11] * (1.044454314040563),
    sum_fxyz[n * NUM_OF_ABC + 12] * (1.044454314040563),
    sum_fxyz[n * NUM_OF_ABC + 13] * (0.174075719006761),
    sum_fxyz[n * NUM_OF_ABC + 14] * (0.174075719006761)};
  get_f12_3(d12, d12inv, fn, fnp, Fp[n*lmax_3+2], s3, r12, f12, f12d+8, n1, n2);

  // l = 4
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  double s4[9] = {
    sum_fxyz[n * NUM_OF_ABC + 15] * (0.011190581936149),
    sum_fxyz[n * NUM_OF_ABC + 16] * (0.223811638722978),
    sum_fxyz[n * NUM_OF_ABC + 17] * (0.223811638722978),
    sum_fxyz[n * NUM_OF_ABC + 18] * (0.111905819361489),
    sum_fxyz[n * NUM_OF_ABC + 19] * (0.111905819361489),
    sum_fxyz[n * NUM_OF_ABC + 20] * (1.566681471060845),
    sum_fxyz[n * NUM_OF_ABC + 21] * (1.566681471060845),
    sum_fxyz[n * NUM_OF_ABC + 22] * (0.195835183882606),
    sum_fxyz[n * NUM_OF_ABC + 23] * (0.195835183882606)};
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

    double tmp1_5b = 4.0 * (0.026596810706114) * s15bsq[0] * s15b[0] * ds5b0_r + 2.0 * (0.053193621412227) * s15b[0] * ds5b0_r * (s15bsq[1] + s15bsq[2]) + 
            (0.053193621412227) * s15bsq[0] * 2.0 * (s15b[1] * ds5b1_r + s15b[2] * ds5b2_r) +
            4.0 * (0.026596810706114) * (s15bsq[1] + s15bsq[2]) * (s15b[1] * ds5b1_r + s15b[2] * ds5b2_r);

    // l = 2 with 4b
    ds0_r = s_rij_blm[3] * fn12[kk];
    ds1_r = s_rij_blm[4] * fn12[kk];
    ds2_r = s_rij_blm[5] * fn12[kk];
    ds3_r = s_rij_blm[6] * fn12[kk];
    ds4_r = s_rij_blm[7] * fn12[kk];
    double tmp2_4b = 3.0 * (-0.007499480826664) * s24bsq[0] * ds0_r + 
              (-0.134990654879954) * ds0_r * (s24bsq[1] + s24bsq[2]) + (-0.134990654879954) * s24b[0] * (2.0 * s24b[1] * ds1_r + 2.0 * s24b[2] * ds2_r) +
              (0.067495327439977) * ds0_r * (s24bsq[3] + s24bsq[4]) + (0.067495327439977) * s24b[0] * (2.0 * s24b[3] * ds3_r + 2.0 * s24b[4] * ds4_r) +
              (0.404971964639861) * ds3_r * (s24bsq[2] - s24bsq[1]) + (0.404971964639861) * s24b[3] * (2.0 * s24b[2] * ds2_r - 2.0 * s24b[1] * ds1_r) +
              (-0.809943929279723) *(ds1_r * s24b[2] * s24b[4] + s24b[1] * ds2_r * s24b[4] + s24b[1] * s24b[2] * ds4_r);

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
    dfeat_c3_base[kk] += tmp1;
    // 
  }
}

static __global__ void find_angular_grad_output_pruned(
  const int N,
  const int num_types,
  const int num_types_sq,
  const int neigh_num,
  const int L_max3,
  const int L_max4,
  const int L_max5,
  const int feat_2b_nums,
  const int feat_3b_nums,
  const double rc_angular,
  const double rcinv_angular,
  const int n_max_angular,
  const int basis_size_angular,
  const int64_t* g_NL_radial,
  const double* g_d12_radial,
  const double * coeff3,
  const int64_t* g_type,
  const double * grad_output,
  const double* g_sum_fxyz,
  double* dsnlm_dc,
  double* grad_coeff3,
  double* dfeat_drij,
  double* grad_d12_angular
  )
{
  // 共享内存：每个线程块（一个中心原子）共享Fp和sum_fxyz
  const int Fp_size = MAX_DIM_ANGULAR;
  const int sum_fxyz_size = n_max_angular * NUM_OF_ABC;
  // 共享内存布局
  extern __shared__ char shared_mem_raw[];
  double* s_Fp = (double*)shared_mem_raw;
  double* s_sum_fxyz = s_Fp + Fp_size;

  int tid = threadIdx.x; // 线程ID
  int n1 = blockIdx.x;  // 每个block处理一个中心原子
  if (n1 >= N) return;

  int b3_nums = n_max_angular * L_max3;
  int g_sum_start = n1 * n_max_angular * NUM_OF_ABC;
  // 线程0加载共享数据（Fp和sum_fxyz对于同一中心原子的所有近邻是相同的）
  if (tid == 0) {
    int de_start = n1 * (feat_3b_nums + feat_2b_nums);
    int dd = 0;

    // 加载Fp到共享内存
    for (int nn = 0; nn < n_max_angular; ++nn) {
      for (int ll = 0; ll < L_max3; ++ll) {
        s_Fp[dd] = grad_output[de_start + feat_2b_nums + ll * n_max_angular + nn];
        dd++;
      }
    }

    if (L_max4 > 0) {
      for (int ll = 0; ll < n_max_angular; ++ll) {
        s_Fp[b3_nums + ll] = grad_output[de_start + feat_2b_nums + b3_nums + ll];
      }
    }

    if (L_max5 > 0) {
      for (int ll = 0; ll < n_max_angular; ++ll) {
        s_Fp[b3_nums + n_max_angular + ll] = grad_output[de_start + feat_2b_nums + b3_nums + n_max_angular + ll];
      }
    }
  }

  // 所有线程协作加载sum_fxyz
  for (int d = tid; d < sum_fxyz_size; d += blockDim.x) {
    s_sum_fxyz[d] = g_sum_fxyz[g_sum_start + d];
  }

  __syncthreads(); // 确保共享内存加载完成

  // A fixed-size block also handles neighbor lists larger than one CTA.
  for (int i1 = tid; i1 < neigh_num; i1 += blockDim.x) {
    int neigh_start_idx = n1 * neigh_num;
    int n2 = g_NL_radial[neigh_start_idx + i1];
    if (n2 < 0) continue;

    int t1 = g_type[n1];
    int t2 = g_type[n2];

    // 计算各个起始索引
    int r12_start_idx = n1 * neigh_num * 4;
    int rij_idx = r12_start_idx + i1 * 4;
    double d12 = g_d12_radial[rij_idx];
    int dc_start_idx = n1 * num_types * n_max_angular * basis_size_angular;
    if (d12 > rc_angular) continue;

    double r12[3] = {g_d12_radial[rij_idx+1], g_d12_radial[rij_idx+2], g_d12_radial[rij_idx+3]};
    double f12[4] = {0.0};

    // 计算径向函数
    double fc12, fcp12;
    find_fc_and_fcp(rc_angular, rcinv_angular, d12, fc12, fcp12);

    double fn12[MAX_NUM_N];
    double fnp12[MAX_NUM_N];
    find_fn_and_fnp(basis_size_angular, rcinv_angular, d12, fc12, fcp12, fn12, fnp12);

    // 计算球谐函数相关项
    double s[NUM_OF_ABC] = {0.0};
    if (grad_coeff3 != nullptr || dsnlm_dc != nullptr) {
      accumulate_blm_rij(d12, r12[0], r12[1], r12[2], s);
    }
    // The helpers use this bound only for the coefficient-gradient contraction.
    const int grad_basis_size = grad_coeff3 != nullptr ? basis_size_angular : 0;

    int c3_start_idx = t1 * num_types * n_max_angular * basis_size_angular;
    int c_I_J_idx = c3_start_idx + t2 * n_max_angular * basis_size_angular;
    int drij_idx = n1 * neigh_num * feat_3b_nums * 4 + i1 * feat_3b_nums * 4;

    // 遍历所有n
    for (int n = 0; n < n_max_angular; ++n) {
      double gn12 = 0.0;
      double gnp12 = 0.0;

      // 计算gn12和gnp12
      for (int k = 0; k < basis_size_angular; ++k) {
        int c_index = c_I_J_idx + n * basis_size_angular + k;
        gn12 += fn12[k] * coeff3[c_index];
        gnp12 += fnp12[k] * coeff3[c_index];
      }

      double f12d[MAX_LMAX * 4] = {0.0};
      double dfeat_c3_base[MAX_NUM_N] = {0.0}; //临时存储cfeat_c3 对应nbase下的梯度
      if (L_max5 > 0) {
        accumulate_f12_with_5body(
          n, d12, r12, gn12, gnp12, s_Fp, s_sum_fxyz,
          s, f12, f12d, dfeat_c3_base, fn12, fnp12,
          t2, num_types, L_max3,
          n_max_angular, grad_basis_size, dc_start_idx, n1, i1);
      } else if (L_max4 > 0) {
        accumulate_f12_with_4body(
          n, d12, r12, gn12, gnp12, s_Fp, s_sum_fxyz,
          s, f12, f12d, dfeat_c3_base, fn12, fnp12,
          t2, num_types, L_max3,
          n_max_angular, grad_basis_size, dc_start_idx, n1, i1);
      } else {
        accumulate_f12(
          n, d12, r12, gn12, gnp12, s_Fp, s_sum_fxyz,
          s, f12, f12d, dfeat_c3_base, fn12, fnp12,
          t2, num_types, L_max3,
          n_max_angular, grad_basis_size, dc_start_idx, n1, i1);
      }

      if (dfeat_drij != nullptr) {
        const int angular_channels = L_max3 + (L_max5 > 0 ? 2 : (L_max4 > 0 ? 1 : 0));
        for (int l_idx = 0; l_idx < angular_channels; ++l_idx) {
          const int out = drij_idx + l_idx * n_max_angular * 4 + n * 4;
          dfeat_drij[out + 0] = f12d[l_idx * 4 + 3];
          dfeat_drij[out + 1] = f12d[l_idx * 4 + 0];
          dfeat_drij[out + 2] = f12d[l_idx * 4 + 1];
          dfeat_drij[out + 3] = f12d[l_idx * 4 + 2];
        }
      }

      // Accumulate directly into [center_type, neighbor_type, n, basis].
      // The caller zero-initializes this output on the current CUDA stream.
      for (int baseid = 0; baseid < grad_basis_size; ++baseid) {
        const int dc_id = c_I_J_idx + n * basis_size_angular + baseid;
        atomicAdd(&grad_coeff3[dc_id], dfeat_c3_base[baseid]);
      }
      // 对dsnlm_dc进行累加（只需要在n==0时）
      if (dsnlm_dc != nullptr && n == 0) {
        int dsnlm_dc_idx = n1 * num_types * basis_size_angular * NUM_OF_ABC +
                          t2 * basis_size_angular * NUM_OF_ABC;

        // 使用原子操作确保线程安全（多个相同类型的近邻会更新同一位置）
        for (int kk = 0; kk < basis_size_angular; kk++) {
          int dsnlm_id = dsnlm_dc_idx + kk * NUM_OF_ABC;
          double fn_val = fn12[kk];

          // 使用原子加法
          atomicAdd(&dsnlm_dc[dsnlm_id + 0], s[0] * fn_val);
          atomicAdd(&dsnlm_dc[dsnlm_id + 1], s[1] * fn_val);
          atomicAdd(&dsnlm_dc[dsnlm_id + 2], s[2] * fn_val);
          atomicAdd(&dsnlm_dc[dsnlm_id + 3], s[3] * fn_val);
          atomicAdd(&dsnlm_dc[dsnlm_id + 4], s[4] * fn_val);
          atomicAdd(&dsnlm_dc[dsnlm_id + 5], s[5] * fn_val);
          atomicAdd(&dsnlm_dc[dsnlm_id + 6], s[6] * fn_val);
          atomicAdd(&dsnlm_dc[dsnlm_id + 7], s[7] * fn_val);
          atomicAdd(&dsnlm_dc[dsnlm_id + 8], s[8] * fn_val);
          atomicAdd(&dsnlm_dc[dsnlm_id + 9], s[9] * fn_val);
          atomicAdd(&dsnlm_dc[dsnlm_id + 10], s[10] * fn_val);
          atomicAdd(&dsnlm_dc[dsnlm_id + 11], s[11] * fn_val);
          atomicAdd(&dsnlm_dc[dsnlm_id + 12], s[12] * fn_val);
          atomicAdd(&dsnlm_dc[dsnlm_id + 13], s[13] * fn_val);
          atomicAdd(&dsnlm_dc[dsnlm_id + 14], s[14] * fn_val);
          atomicAdd(&dsnlm_dc[dsnlm_id + 15], s[15] * fn_val);
          atomicAdd(&dsnlm_dc[dsnlm_id + 16], s[16] * fn_val);
          atomicAdd(&dsnlm_dc[dsnlm_id + 17], s[17] * fn_val);
          atomicAdd(&dsnlm_dc[dsnlm_id + 18], s[18] * fn_val);
          atomicAdd(&dsnlm_dc[dsnlm_id + 19], s[19] * fn_val);
          atomicAdd(&dsnlm_dc[dsnlm_id + 20], s[20] * fn_val);
          atomicAdd(&dsnlm_dc[dsnlm_id + 21], s[21] * fn_val);
          atomicAdd(&dsnlm_dc[dsnlm_id + 22], s[22] * fn_val);
          atomicAdd(&dsnlm_dc[dsnlm_id + 23], s[23] * fn_val);
        }
      }
    }

    // Each neighbor slot has a unique writer, including across stride iterations.
    if (grad_d12_angular != nullptr) {
      grad_d12_angular[rij_idx]   = f12[3];
      grad_d12_angular[rij_idx+1] = f12[0];
      grad_d12_angular[rij_idx+2] = f12[1];
      grad_d12_angular[rij_idx+3] = f12[2];
    }
  }
}
}

// 少元素模型保留块内累加，避免全局 double 原子操作集中到少量系数。
// 输出需求语义与上面的 CUDA 布局一致，仅累加策略不同。
namespace matpl_angular_lds_demand {
static __device__ __forceinline__ void accumulate_f12_opt(
  const int n,
  const double d12,
  const double* r12,
  double fn,
  double fnp,
  const double* Fp,
  const double* sum_fxyz,
  const double* s_rij_blm,
  double* f12,
  double* dfeat_drij,
  const int drij_idx,
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
    sum_fxyz[n * NUM_OF_ABC + 0] * (0.238732414637843),
    sum_fxyz[n * NUM_OF_ABC + 1] * (0.119366207318922),
    sum_fxyz[n * NUM_OF_ABC + 2] * (0.119366207318922)};
  {
    double f12d_group[4] = {0.0};
    get_f12_1(
      d12inv, fn, fnp, Fp[n*lmax_3], s1, r12, f12, f12d_group, n1, n2);
    if (dfeat_drij != nullptr) store_f12d_group(
      dfeat_drij + drij_idx + n * 4, f12d_group);
  }

  // l = 2
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  double s2[5] = {
    sum_fxyz[n * NUM_OF_ABC + 3] * (0.099471839432435),
    sum_fxyz[n * NUM_OF_ABC + 4] * (0.596831036594608),
    sum_fxyz[n * NUM_OF_ABC + 5] * (0.596831036594608),
    sum_fxyz[n * NUM_OF_ABC + 6] * (0.149207759148652),
    sum_fxyz[n * NUM_OF_ABC + 7] * (0.149207759148652)};
  {
    double f12d_group[4] = {0.0};
    get_f12_2(
      d12, d12inv, fn, fnp, Fp[n*lmax_3+1], s2, r12, f12,
      f12d_group, n1, n2);
    if (dfeat_drij != nullptr) store_f12d_group(
      dfeat_drij + drij_idx + (n_max_angular + n) * 4, f12d_group);
  }
  // l = 3
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  double s3[7] = {
    sum_fxyz[n * NUM_OF_ABC + 8] * (0.139260575205408),
    sum_fxyz[n * NUM_OF_ABC + 9] * (0.104445431404056),
    sum_fxyz[n * NUM_OF_ABC + 10] * (0.104445431404056),
    sum_fxyz[n * NUM_OF_ABC + 11] * (1.044454314040563),
    sum_fxyz[n * NUM_OF_ABC + 12] * (1.044454314040563),
    sum_fxyz[n * NUM_OF_ABC + 13] * (0.174075719006761),
    sum_fxyz[n * NUM_OF_ABC + 14] * (0.174075719006761)};
  {
    double f12d_group[4] = {0.0};
    get_f12_3(
      d12, d12inv, fn, fnp, Fp[n*lmax_3+2], s3, r12, f12,
      f12d_group, n1, n2);
    if (dfeat_drij != nullptr) store_f12d_group(
      dfeat_drij + drij_idx + (2 * n_max_angular + n) * 4,
      f12d_group);
  }
  // l = 4
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  double s4[9] = {
    sum_fxyz[n * NUM_OF_ABC + 15] * (0.011190581936149),
    sum_fxyz[n * NUM_OF_ABC + 16] * (0.223811638722978),
    sum_fxyz[n * NUM_OF_ABC + 17] * (0.223811638722978),
    sum_fxyz[n * NUM_OF_ABC + 18] * (0.111905819361489),
    sum_fxyz[n * NUM_OF_ABC + 19] * (0.111905819361489),
    sum_fxyz[n * NUM_OF_ABC + 20] * (1.566681471060845),
    sum_fxyz[n * NUM_OF_ABC + 21] * (1.566681471060845),
    sum_fxyz[n * NUM_OF_ABC + 22] * (0.195835183882606),
    sum_fxyz[n * NUM_OF_ABC + 23] * (0.195835183882606)};
  {
    double f12d_group[4] = {0.0};
    get_f12_4(
      r12[0], r12[1], r12[2], d12, d12inv, fn, fnp,
      Fp[n*lmax_3+3], s4, f12, f12d_group, n1, n2);
    if (dfeat_drij != nullptr) store_f12d_group(
      dfeat_drij + drij_idx + (3 * n_max_angular + n) * 4,
      f12d_group);
  }
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
    atomicAdd(&dfeat_c3[dc_id], tmp1);
  }
}

static __device__ __forceinline__ void accumulate_f12_with_4body_opt(
  const int n,
  const double d12,
  const double* r12,
  double fn,
  double fnp,
  const double* Fp,
  const double* sum_fxyz,
  const double* s_rij_blm,
  double* f12,
  double* dfeat_drij,
  const int drij_idx,
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
    sum_fxyz[n * NUM_OF_ABC + 0] * (0.238732414637843),
    sum_fxyz[n * NUM_OF_ABC + 1] * (0.119366207318922),
    sum_fxyz[n * NUM_OF_ABC + 2] * (0.119366207318922)};
  {
    double f12d_group[4] = {0.0};
    get_f12_1(
      d12inv, fn, fnp, Fp[n*lmax_3], s1, r12, f12, f12d_group, n1, n2);
    if (dfeat_drij != nullptr) store_f12d_group(
      dfeat_drij + drij_idx + n * 4, f12d_group);
  }
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

  {
    double f12d_group[4] = {0.0};
    get_f12_4body(
      d12, d12inv, fn, fnp, Fp[n_max_angular * lmax_3 + n], s2,
      r12, f12, f12d_group, n1, n2);
    if (dfeat_drij != nullptr) store_f12d_group(
      dfeat_drij + drij_idx + (lmax_3 * n_max_angular + n) * 4,
      f12d_group);
  }
  s2[0] *= (0.099471839432435);
  s2[1] *= (0.596831036594608);
  s2[2] *= (0.596831036594608);
  s2[3] *= (0.149207759148652);
  s2[4] *= (0.149207759148652);
  {
    double f12d_group[4] = {0.0};
    get_f12_2(
      d12, d12inv, fn, fnp, Fp[n*lmax_3+1], s2, r12, f12,
      f12d_group, n1, n2);
    if (dfeat_drij != nullptr) store_f12d_group(
      dfeat_drij + drij_idx + (n_max_angular + n) * 4, f12d_group);
  }
  // l = 3
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  double s3[7] = {
    sum_fxyz[n * NUM_OF_ABC + 8] * (0.139260575205408),
    sum_fxyz[n * NUM_OF_ABC + 9] * (0.104445431404056),
    sum_fxyz[n * NUM_OF_ABC + 10] * (0.104445431404056),
    sum_fxyz[n * NUM_OF_ABC + 11] * (1.044454314040563),
    sum_fxyz[n * NUM_OF_ABC + 12] * (1.044454314040563),
    sum_fxyz[n * NUM_OF_ABC + 13] * (0.174075719006761),
    sum_fxyz[n * NUM_OF_ABC + 14] * (0.174075719006761)};
  {
    double f12d_group[4] = {0.0};
    get_f12_3(
      d12, d12inv, fn, fnp, Fp[n*lmax_3+2], s3, r12, f12,
      f12d_group, n1, n2);
    if (dfeat_drij != nullptr) store_f12d_group(
      dfeat_drij + drij_idx + (2 * n_max_angular + n) * 4,
      f12d_group);
  }
  // l = 4
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  double s4[9] = {
    sum_fxyz[n * NUM_OF_ABC + 15] * (0.011190581936149),
    sum_fxyz[n * NUM_OF_ABC + 16] * (0.223811638722978),
    sum_fxyz[n * NUM_OF_ABC + 17] * (0.223811638722978),
    sum_fxyz[n * NUM_OF_ABC + 18] * (0.111905819361489),
    sum_fxyz[n * NUM_OF_ABC + 19] * (0.111905819361489),
    sum_fxyz[n * NUM_OF_ABC + 20] * (1.566681471060845),
    sum_fxyz[n * NUM_OF_ABC + 21] * (1.566681471060845),
    sum_fxyz[n * NUM_OF_ABC + 22] * (0.195835183882606),
    sum_fxyz[n * NUM_OF_ABC + 23] * (0.195835183882606)};
  {
    double f12d_group[4] = {0.0};
    get_f12_4(
      r12[0], r12[1], r12[2], d12, d12inv, fn, fnp,
      Fp[n*lmax_3+3], s4, f12, f12d_group, n1, n2);
    if (dfeat_drij != nullptr) store_f12d_group(
      dfeat_drij + drij_idx + (3 * n_max_angular + n) * 4,
      f12d_group);
  }

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

    double tmp2_4b = 3.0 * (-0.007499480826664) * s24bsq[0] * ds0_r +
              (-0.134990654879954) * ds0_r * (s24bsq[1] + s24bsq[2]) + (-0.134990654879954) * s24b[0] * (2.0 * s24b[1] * ds1_r + 2.0 * s24b[2] * ds2_r) +
              (0.067495327439977) * ds0_r * (s24bsq[3] + s24bsq[4]) + (0.067495327439977) * s24b[0] * (2.0 * s24b[3] * ds3_r + 2.0 * s24b[4] * ds4_r) +
              (0.404971964639861) * ds3_r * (s24bsq[2] - s24bsq[1]) + (0.404971964639861) * s24b[3] * (2.0 * s24b[2] * ds2_r - 2.0 * s24b[1] * ds1_r) +
              (-0.809943929279723) *(ds1_r * s24b[2] * s24b[4] + s24b[1] * ds2_r * s24b[4] + s24b[1] * s24b[2] * ds4_r);

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
    atomicAdd(&dfeat_c3[dc_id], tmp1);
  }
}

static __device__ __forceinline__ void accumulate_f12_with_5body_opt(
  const int n,
  const double d12,
  const double* r12,
  double fn,
  double fnp,
  const double* Fp,
  const double* sum_fxyz,
  const double* s_rij_blm,
  double* f12,
  double* dfeat_drij,
  const int drij_idx,
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

  {
    double f12d_group[4] = {0.0};
    get_f12_5body(
      d12, d12inv, fn, fnp,
      Fp[n_max_angular * lmax_3 + n_max_angular + n], s1, r12, f12,
      f12d_group, n1, n2);
    if (dfeat_drij != nullptr) store_f12d_group(
      dfeat_drij + drij_idx + ((lmax_3 + 1) * n_max_angular + n) * 4,
      f12d_group);
  }

  s1[0] *= (0.238732414637843);
  s1[1] *= (0.119366207318922);
  s1[2] *= (0.119366207318922);
  {
    double f12d_group[4] = {0.0};
    get_f12_1(
      d12inv, fn, fnp, Fp[n*lmax_3], s1, r12, f12, f12d_group, n1, n2);
    if (dfeat_drij != nullptr) store_f12d_group(
      dfeat_drij + drij_idx + n * 4, f12d_group);
  }

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

  {
    double f12d_group[4] = {0.0};
    get_f12_4body(
      d12, d12inv, fn, fnp, Fp[n_max_angular * lmax_3 + n], s2,
      r12, f12, f12d_group, n1, n2);
    if (dfeat_drij != nullptr) store_f12d_group(
      dfeat_drij + drij_idx + (lmax_3 * n_max_angular + n) * 4,
      f12d_group);
  }
  s2[0] *= (0.099471839432435);
  s2[1] *= (0.596831036594608);
  s2[2] *= (0.596831036594608);
  s2[3] *= (0.149207759148652);
  s2[4] *= (0.149207759148652);
  {
    double f12d_group[4] = {0.0};
    get_f12_2(
      d12, d12inv, fn, fnp, Fp[n*lmax_3+1], s2, r12, f12,
      f12d_group, n1, n2);
    if (dfeat_drij != nullptr) store_f12d_group(
      dfeat_drij + drij_idx + (n_max_angular + n) * 4, f12d_group);
  }

  // l = 3
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  double s3[7] = {
    sum_fxyz[n * NUM_OF_ABC + 8] * (0.139260575205408),
    sum_fxyz[n * NUM_OF_ABC + 9] * (0.104445431404056),
    sum_fxyz[n * NUM_OF_ABC + 10] * (0.104445431404056),
    sum_fxyz[n * NUM_OF_ABC + 11] * (1.044454314040563),
    sum_fxyz[n * NUM_OF_ABC + 12] * (1.044454314040563),
    sum_fxyz[n * NUM_OF_ABC + 13] * (0.174075719006761),
    sum_fxyz[n * NUM_OF_ABC + 14] * (0.174075719006761)};
  {
    double f12d_group[4] = {0.0};
    get_f12_3(
      d12, d12inv, fn, fnp, Fp[n*lmax_3+2], s3, r12, f12,
      f12d_group, n1, n2);
    if (dfeat_drij != nullptr) store_f12d_group(
      dfeat_drij + drij_idx + (2 * n_max_angular + n) * 4,
      f12d_group);
  }

  // l = 4
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  double s4[9] = {
    sum_fxyz[n * NUM_OF_ABC + 15] * (0.011190581936149),
    sum_fxyz[n * NUM_OF_ABC + 16] * (0.223811638722978),
    sum_fxyz[n * NUM_OF_ABC + 17] * (0.223811638722978),
    sum_fxyz[n * NUM_OF_ABC + 18] * (0.111905819361489),
    sum_fxyz[n * NUM_OF_ABC + 19] * (0.111905819361489),
    sum_fxyz[n * NUM_OF_ABC + 20] * (1.566681471060845),
    sum_fxyz[n * NUM_OF_ABC + 21] * (1.566681471060845),
    sum_fxyz[n * NUM_OF_ABC + 22] * (0.195835183882606),
    sum_fxyz[n * NUM_OF_ABC + 23] * (0.195835183882606)};
  {
    double f12d_group[4] = {0.0};
    get_f12_4(
      r12[0], r12[1], r12[2], d12, d12inv, fn, fnp,
      Fp[n*lmax_3+3], s4, f12, f12d_group, n1, n2);
    if (dfeat_drij != nullptr) store_f12d_group(
      dfeat_drij + drij_idx + (3 * n_max_angular + n) * 4,
      f12d_group);
  }

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

    double tmp1_5b = 4.0 * (0.026596810706114) * s15bsq[0] * s15b[0] * ds5b0_r + 2.0 * (0.053193621412227) * s15b[0] * ds5b0_r * (s15bsq[1] + s15bsq[2]) +
            (0.053193621412227) * s15bsq[0] * 2.0 * (s15b[1] * ds5b1_r + s15b[2] * ds5b2_r) +
            4.0 * (0.026596810706114) * (s15bsq[1] + s15bsq[2]) * (s15b[1] * ds5b1_r + s15b[2] * ds5b2_r);

    // l = 2 with 4b
    ds0_r = s_rij_blm[3] * fn12[kk];
    ds1_r = s_rij_blm[4] * fn12[kk];
    ds2_r = s_rij_blm[5] * fn12[kk];
    ds3_r = s_rij_blm[6] * fn12[kk];
    ds4_r = s_rij_blm[7] * fn12[kk];
    double tmp2_4b = 3.0 * (-0.007499480826664) * s24bsq[0] * ds0_r +
              (-0.134990654879954) * ds0_r * (s24bsq[1] + s24bsq[2]) + (-0.134990654879954) * s24b[0] * (2.0 * s24b[1] * ds1_r + 2.0 * s24b[2] * ds2_r) +
              (0.067495327439977) * ds0_r * (s24bsq[3] + s24bsq[4]) + (0.067495327439977) * s24b[0] * (2.0 * s24b[3] * ds3_r + 2.0 * s24b[4] * ds4_r) +
              (0.404971964639861) * ds3_r * (s24bsq[2] - s24bsq[1]) + (0.404971964639861) * s24b[3] * (2.0 * s24b[2] * ds2_r - 2.0 * s24b[1] * ds1_r) +
              (-0.809943929279723) *(ds1_r * s24b[2] * s24b[4] + s24b[1] * ds2_r * s24b[4] + s24b[1] * s24b[2] * ds4_r);

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
    atomicAdd(&dfeat_c3[dc_id], tmp1);
  }
}

static __global__ void find_angular_grad_output_pruned_lds(
  const int N,
  const int num_types,
  const int num_types_sq,
  const int neigh_num,
  const int L_max3,
  const int L_max4,
  const int L_max5,
  const int feat_2b_nums,
  const int feat_3b_nums, // 3b + 4b + 5b
  const double rc_angular,
  const double rcinv_angular,
  const int n_max_angular,
  const int basis_size_angular,
  const int64_t* __restrict__ g_NL_radial,
  const double* __restrict__ g_d12_radial,
  const double * __restrict__ coeff3,
  const int64_t* __restrict__ g_type,
  const double * __restrict__ grad_output,
  const double* __restrict__ g_sum_fxyz,
  double* dsnlm_dc,
  double* grad_coeff3,
  double* dfeat_drij,//[batch*atom, neighbornum, 3b_feat_num, 4]
  double* grad_d12_angular
  )
{
  constexpr int BPA_BLOCK = 64;
  const int n1 = blockIdx.x; // one block per atom; gridDim.x == N
  const int tid = threadIdx.x;

  extern __shared__ double bpa_smem[];
  const int sum_len   = n_max_angular * NUM_OF_ABC;
  const int c3_len    = num_types * n_max_angular * basis_size_angular;
  const int coeff_acc_len = grad_coeff3 != nullptr ? c3_len : 0;
  const int grad_basis_size = grad_coeff3 != nullptr ? basis_size_angular : 0;
  const int dsnlm_len = dsnlm_dc != nullptr ? num_types * basis_size_angular * NUM_OF_ABC : 0;
  double* s_Fp    = bpa_smem;                    // [MAX_DIM_ANGULAR]
  double* s_sum   = s_Fp + MAX_DIM_ANGULAR;      // [sum_len]
  double* s_coef  = s_sum + sum_len;             // [c3_len]
  double* s_c3    = s_coef + c3_len;             // [c3_len] shared accumulator
  double* s_dsnlm = s_c3 + coeff_acc_len;               // [dsnlm_len] shared accumulator
  __shared__ int s_first_bad;

  // zero shared accumulators
  for (int i = tid; i < coeff_acc_len; i += BPA_BLOCK) s_c3[i] = 0.0;
  for (int i = tid; i < dsnlm_len; i += BPA_BLOCK) s_dsnlm[i] = 0.0;

  const int g_sum_start = n1 * sum_len;
  const int r12_start_idx = n1 * neigh_num * 4;
  const int dc_start_idx = n1 * c3_len;
  const int dsnlm_dc_start_idx = n1 * dsnlm_len;
  const int de_start = n1 * (feat_3b_nums + feat_2b_nums);// dE/dq
  const int neigh_start_idx = n1 * neigh_num;
  const int dfeat_dr_start = n1 * neigh_num * feat_3b_nums * 4;

  // stage Fp (all three groups, flattened; layout identical to the original)
  const int b3_nums = n_max_angular * L_max3;
  for (int e = tid; e < b3_nums; e += BPA_BLOCK) {
    const int nn = e / L_max3;
    const int ll = e % L_max3;
    s_Fp[e] = grad_output[de_start + feat_2b_nums + ll * n_max_angular + nn];
  }
  if (L_max4 > 0) {
    for (int e = tid; e < n_max_angular; e += BPA_BLOCK) {
      s_Fp[b3_nums + e] = grad_output[de_start + feat_2b_nums + b3_nums + e];
    }
  }
  if (L_max5 > 0) {
    for (int e = tid; e < n_max_angular; e += BPA_BLOCK) {
      s_Fp[b3_nums + n_max_angular + e] = grad_output[de_start + feat_2b_nums + b3_nums + n_max_angular + e];
    }
  }
  // stage sum_fxyz ([N, n_max, NUM_OF_ABC] row of this atom)
  for (int i = tid; i < sum_len; i += BPA_BLOCK) {
    s_sum[i] = g_sum_fxyz[g_sum_start + i];
  }
  // stage this atom's coeff3 row: coeff3[t1, t2, n, k] for all t2
  const int t1 = (int)g_type[n1];
  const int c3_stage_start = t1 * c3_len;
  for (int i = tid; i < c3_len; i += BPA_BLOCK) {
    s_coef[i] = coeff3[c3_stage_start + i];
  }
  if (tid == 0) s_first_bad = 0x7fffffff;
  __syncthreads(); // zeros + staged data + s_first_bad visible

  // exact break semantics: first neighbor index that is invalid (n2<0 or
  // d12 > rc) truncates the list; neighbors at/after it are not processed.
  for (int i = tid; i < neigh_num; i += BPA_BLOCK) {
    const int n2v = (int)g_NL_radial[neigh_start_idx + i];
    const double d12v = g_d12_radial[r12_start_idx + i * 4];
    if (n2v < 0 || d12v > rc_angular) atomicMin(&s_first_bad, i);
  }
  __syncthreads();
  const int n_active = min(s_first_bad, neigh_num);

  for (int i1 = tid; i1 < n_active; i1 += BPA_BLOCK) {
    const int n2 = (int)g_NL_radial[neigh_start_idx + i1];
    const int t2 = (int)g_type[n2];
    const int rij_idx = r12_start_idx + i1 * 4;
    const double d12 = g_d12_radial[rij_idx];
    const int drij_idx = dfeat_dr_start + i1 * feat_3b_nums * 4;
    double f12[4] = {0.0};

    double fc12, fcp12;
    find_fc_and_fcp(rc_angular, rcinv_angular, d12, fc12, fcp12);
    double fn12[MAX_NUM_N];
    double fnp12[MAX_NUM_N];
    find_fn_and_fnp(
      basis_size_angular, rcinv_angular, d12, fc12, fcp12, fn12, fnp12);

    const double r12[3] = {g_d12_radial[rij_idx+1], g_d12_radial[rij_idx+2], g_d12_radial[rij_idx+3]};
    double s[NUM_OF_ABC] = {0.0};
    if (grad_coeff3 != nullptr || dsnlm_dc != nullptr)
      accumulate_blm_rij(d12, r12[0], r12[1], r12[2], s);// blm * 1/(r_ij^L)
    for (int n = 0; n < n_max_angular; ++n) {
      double gn12 = 0.0;
      double gnp12 = 0.0;
      for (int k = 0; k < basis_size_angular; ++k) {
        const double c = s_coef[t2 * n_max_angular * basis_size_angular + n * basis_size_angular + k];
        gn12 += fn12[k] * c;
        gnp12 += fnp12[k] * c;
      }
      if (L_max5 > 0) {
        accumulate_f12_with_5body_opt(
          n, d12, r12, gn12, gnp12, s_Fp, s_sum,
            s, f12, dfeat_drij, drij_idx, s_c3, fn12, fnp12,
            t2, num_types, L_max3,
            n_max_angular, grad_basis_size, 0, n1, i1);
      } else if (L_max4 > 0) {
        accumulate_f12_with_4body_opt(
          n, d12, r12, gn12, gnp12, s_Fp, s_sum,
            s, f12, dfeat_drij, drij_idx, s_c3, fn12, fnp12,
            t2, num_types, L_max3,
            n_max_angular, grad_basis_size, 0, n1, i1);
      } else {
        accumulate_f12_opt(
          n, d12, r12, gn12, gnp12, s_Fp, s_sum,
            s, f12, dfeat_drij, drij_idx, s_c3, fn12, fnp12,
            t2, num_types, L_max3,
            n_max_angular, grad_basis_size, 0, n1, i1);
      }
      if (dsnlm_dc != nullptr && n == 0) {
        double* dslots = &s_dsnlm[t2 * basis_size_angular * NUM_OF_ABC];
        for(int kk = 0; kk < basis_size_angular; kk++){
          const double fk = fn12[kk];
          #pragma unroll
          for (int j = 0; j < NUM_OF_ABC; ++j) {
            atomicAdd(&dslots[kk * NUM_OF_ABC + j], s[j] * fk);
          }
        }
      }
    }

    // (atom, neighbor)-disjoint slot: single writer per block -> plain +=
    grad_d12_angular[rij_idx]   += f12[3];
    grad_d12_angular[rij_idx+1] += f12[0];
    grad_d12_angular[rij_idx+2] += f12[1];
    grad_d12_angular[rij_idx+3] += f12[2];
  }
  __syncthreads(); // shared accumulators final

  // write back per-atom shared accumulators (each slot one owner thread)
  for (int i = tid; i < coeff_acc_len; i += BPA_BLOCK) {
    atomicAdd(&grad_coeff3[t1 * c3_len + i], s_c3[i]);
  }
  if (dsnlm_dc != nullptr) {
    for (int i = tid; i < dsnlm_len; i += BPA_BLOCK) {
      dsnlm_dc[dsnlm_dc_start_idx + i] += s_dsnlm[i];
    }
  }
}
}

// 对输出需求和真实 n/basis/类型数做编译期特化，减少局部数组与存活变量。
namespace matpl_angular_demand {
template<int KB, int NN, int TT, int LL4, int LL5, int Outputs>
static __global__ void find_angular_grad_specialized(
  const int N,
  const int unused_num_types,
  const int unused_num_types_sq,
  const int neigh_num,
  const int unused_L_max3,
  const int unused_L_max4,
  const int unused_L_max5,
  const int feat_2b_nums,
  const int feat_3b_nums,
  const double rc_angular,
  const double rcinv_angular,
  const int unused_n_max_angular,
  const int unused_basis_size_angular,
  const int64_t* g_NL_radial,
  const double* g_d12_radial,
  const double * coeff3,
  const int64_t* g_type,
  const double * grad_output,
  const double* g_sum_fxyz,
  double* dsnlm_dc,
  double* grad_coeff3,
  double* dfeat_drij,
  double* grad_d12_angular
  )
{
  constexpr int num_types=TT, num_types_sq=TT*TT, L_max3=4, L_max4=LL4, L_max5=LL5;
  constexpr int n_max_angular=NN, basis_size_angular=KB;

  // 共享内存：每个线程块（一个中心原子）共享Fp和sum_fxyz
  const int Fp_size = MAX_DIM_ANGULAR;
  const int sum_fxyz_size = n_max_angular * NUM_OF_ABC;
  // 共享内存布局
  extern __shared__ char shared_mem_raw[];
  double* s_Fp = (double*)shared_mem_raw;
  double* s_sum_fxyz = s_Fp + Fp_size;

  int tid = threadIdx.x; // 线程ID
  int n1 = blockIdx.x;  // 每个block处理一个中心原子
  if (n1 >= N) return;

  int b3_nums = n_max_angular * L_max3;
  int g_sum_start = n1 * n_max_angular * NUM_OF_ABC;
  // 线程0加载共享数据（Fp和sum_fxyz对于同一中心原子的所有近邻是相同的）
  if (tid == 0) {
    int de_start = n1 * (feat_3b_nums + feat_2b_nums);
    int dd = 0;

    // 加载Fp到共享内存
    for (int nn = 0; nn < n_max_angular; ++nn) {
      for (int ll = 0; ll < L_max3; ++ll) {
        s_Fp[dd] = grad_output[de_start + feat_2b_nums + ll * n_max_angular + nn];
        dd++;
      }
    }

    if (L_max4 > 0) {
      for (int ll = 0; ll < n_max_angular; ++ll) {
        s_Fp[b3_nums + ll] = grad_output[de_start + feat_2b_nums + b3_nums + ll];
      }
    }

    if (L_max5 > 0) {
      for (int ll = 0; ll < n_max_angular; ++ll) {
        s_Fp[b3_nums + n_max_angular + ll] = grad_output[de_start + feat_2b_nums + b3_nums + n_max_angular + ll];
      }
    }
  }

  // 所有线程协作加载sum_fxyz
  for (int d = tid; d < sum_fxyz_size; d += blockDim.x) {
    s_sum_fxyz[d] = g_sum_fxyz[g_sum_start + d];
  }

  __syncthreads(); // 确保共享内存加载完成

  // A fixed-size block also handles neighbor lists larger than one CTA.
  for (int i1 = tid; i1 < neigh_num; i1 += blockDim.x) {
    int neigh_start_idx = n1 * neigh_num;
    int n2 = g_NL_radial[neigh_start_idx + i1];
    if (n2 < 0) continue;

    int t1 = g_type[n1];
    int t2 = g_type[n2];

    // 计算各个起始索引
    int r12_start_idx = n1 * neigh_num * 4;
    int rij_idx = r12_start_idx + i1 * 4;
    double d12 = g_d12_radial[rij_idx];
    int dc_start_idx = n1 * num_types * n_max_angular * basis_size_angular;
    if (d12 > rc_angular) continue;

    double r12[3] = {g_d12_radial[rij_idx+1], g_d12_radial[rij_idx+2], g_d12_radial[rij_idx+3]};
    double f12[4] = {0.0};

    // 计算径向函数
    double fc12, fcp12;
    find_fc_and_fcp(rc_angular, rcinv_angular, d12, fc12, fcp12);

    double fn12[KB];
    double fnp12[KB];
    find_fn_and_fnp(basis_size_angular, rcinv_angular, d12, fc12, fcp12, fn12, fnp12);

    // 计算球谐函数相关项
    double s[NUM_OF_ABC] = {0.0};
    if ((Outputs & 1) || (Outputs & 2)) {
      accumulate_blm_rij(d12, r12[0], r12[1], r12[2], s);
    }
    // The helpers use this bound only for the coefficient-gradient contraction.
    const int grad_basis_size = (Outputs & 1) ? basis_size_angular : 0;

    int c3_start_idx = t1 * num_types * n_max_angular * basis_size_angular;
    int c_I_J_idx = c3_start_idx + t2 * n_max_angular * basis_size_angular;
    int drij_idx = n1 * neigh_num * feat_3b_nums * 4 + i1 * feat_3b_nums * 4;

    // 遍历所有n
    #pragma unroll 1
    for (int n = 0; n < n_max_angular; ++n) {
      double gn12 = 0.0;
      double gnp12 = 0.0;

      // 计算gn12和gnp12
      for (int k = 0; k < basis_size_angular; ++k) {
        int c_index = c_I_J_idx + n * basis_size_angular + k;
        gn12 += fn12[k] * coeff3[c_index];
        gnp12 += fnp12[k] * coeff3[c_index];
      }

      double f12d[(4 + (LL4 > 0) + (LL5 > 0)) * 4] = {0.0};
      double dfeat_c3_base[KB] = {0.0}; //临时存储cfeat_c3 对应nbase下的梯度
      if (L_max5 > 0) {
        accumulate_f12_with_5body(
          n, d12, r12, gn12, gnp12, s_Fp, s_sum_fxyz,
          s, f12, f12d, dfeat_c3_base, fn12, fnp12,
          t2, num_types, L_max3,
          n_max_angular, grad_basis_size, dc_start_idx, n1, i1);
      } else if (L_max4 > 0) {
        accumulate_f12_with_4body(
          n, d12, r12, gn12, gnp12, s_Fp, s_sum_fxyz,
          s, f12, f12d, dfeat_c3_base, fn12, fnp12,
          t2, num_types, L_max3,
          n_max_angular, grad_basis_size, dc_start_idx, n1, i1);
      } else {
        accumulate_f12(
          n, d12, r12, gn12, gnp12, s_Fp, s_sum_fxyz,
          s, f12, f12d, dfeat_c3_base, fn12, fnp12,
          t2, num_types, L_max3,
          n_max_angular, grad_basis_size, dc_start_idx, n1, i1);
      }

      if ((Outputs & 4)) {
        const int angular_channels = L_max3 + (L_max5 > 0 ? 2 : (L_max4 > 0 ? 1 : 0));
        for (int l_idx = 0; l_idx < angular_channels; ++l_idx) {
          const int out = drij_idx + l_idx * n_max_angular * 4 + n * 4;
          dfeat_drij[out + 0] = f12d[l_idx * 4 + 3];
          dfeat_drij[out + 1] = f12d[l_idx * 4 + 0];
          dfeat_drij[out + 2] = f12d[l_idx * 4 + 1];
          dfeat_drij[out + 3] = f12d[l_idx * 4 + 2];
        }
      }

      // Accumulate directly into [center_type, neighbor_type, n, basis].
      // The caller zero-initializes this output on the current CUDA stream.
      for (int baseid = 0; baseid < grad_basis_size; ++baseid) {
        const int dc_id = c_I_J_idx + n * basis_size_angular + baseid;
        atomicAdd(&grad_coeff3[dc_id], dfeat_c3_base[baseid]);
      }
      // 对dsnlm_dc进行累加（只需要在n==0时）
      if ((Outputs & 2) && n == 0) {
        int dsnlm_dc_idx = n1 * num_types * basis_size_angular * NUM_OF_ABC +
                          t2 * basis_size_angular * NUM_OF_ABC;

        // 使用原子操作确保线程安全（多个相同类型的近邻会更新同一位置）
        for (int kk = 0; kk < basis_size_angular; kk++) {
          int dsnlm_id = dsnlm_dc_idx + kk * NUM_OF_ABC;
          double fn_val = fn12[kk];

          // 使用原子加法
          atomicAdd(&dsnlm_dc[dsnlm_id + 0], s[0] * fn_val);
          atomicAdd(&dsnlm_dc[dsnlm_id + 1], s[1] * fn_val);
          atomicAdd(&dsnlm_dc[dsnlm_id + 2], s[2] * fn_val);
          atomicAdd(&dsnlm_dc[dsnlm_id + 3], s[3] * fn_val);
          atomicAdd(&dsnlm_dc[dsnlm_id + 4], s[4] * fn_val);
          atomicAdd(&dsnlm_dc[dsnlm_id + 5], s[5] * fn_val);
          atomicAdd(&dsnlm_dc[dsnlm_id + 6], s[6] * fn_val);
          atomicAdd(&dsnlm_dc[dsnlm_id + 7], s[7] * fn_val);
          atomicAdd(&dsnlm_dc[dsnlm_id + 8], s[8] * fn_val);
          atomicAdd(&dsnlm_dc[dsnlm_id + 9], s[9] * fn_val);
          atomicAdd(&dsnlm_dc[dsnlm_id + 10], s[10] * fn_val);
          atomicAdd(&dsnlm_dc[dsnlm_id + 11], s[11] * fn_val);
          atomicAdd(&dsnlm_dc[dsnlm_id + 12], s[12] * fn_val);
          atomicAdd(&dsnlm_dc[dsnlm_id + 13], s[13] * fn_val);
          atomicAdd(&dsnlm_dc[dsnlm_id + 14], s[14] * fn_val);
          atomicAdd(&dsnlm_dc[dsnlm_id + 15], s[15] * fn_val);
          atomicAdd(&dsnlm_dc[dsnlm_id + 16], s[16] * fn_val);
          atomicAdd(&dsnlm_dc[dsnlm_id + 17], s[17] * fn_val);
          atomicAdd(&dsnlm_dc[dsnlm_id + 18], s[18] * fn_val);
          atomicAdd(&dsnlm_dc[dsnlm_id + 19], s[19] * fn_val);
          atomicAdd(&dsnlm_dc[dsnlm_id + 20], s[20] * fn_val);
          atomicAdd(&dsnlm_dc[dsnlm_id + 21], s[21] * fn_val);
          atomicAdd(&dsnlm_dc[dsnlm_id + 22], s[22] * fn_val);
          atomicAdd(&dsnlm_dc[dsnlm_id + 23], s[23] * fn_val);
        }
      }
    }

    // Each neighbor slot has a unique writer, including across stride iterations.
    if (grad_d12_angular != nullptr) {
      grad_d12_angular[rij_idx]   = f12[3];
      grad_d12_angular[rij_idx+1] = f12[0];
      grad_d12_angular[rij_idx+2] = f12[1];
      grad_d12_angular[rij_idx+3] = f12[2];
    }
  }
}
}

// 独立的上下文/系数收缩不携带位置导数临时量。
namespace matpl_angular_demand {
template<int KB, int NN, int TT, int LL4, int LL5>
static __global__ void find_angular_grad_coeff_split(
 const int N, const int neigh_num, const int feat2, const int feat3,
 const double rc, const double rcinv, const int64_t* NL,
 const double* rij, const int64_t* types, const double* grad,
 const double* sums, double* out) {
 constexpr int CH=4+(LL4>0)+(LL5>0);
 __shared__ double fp[CH], sm[24], acc[TT*KB];
 __shared__ int first_bad;
 const int a=blockIdx.x, n=blockIdx.y, tid=threadIdx.x;
 for(int j=tid;j<CH;j+=64) fp[j]=grad[a*(feat2+feat3)+j*NN+n];
 for(int j=tid;j<24;j+=64) sm[j]=sums[(a*NN+n)*24+j];
 for(int j=tid;j<TT*KB;j+=64) acc[j]=0.;
 if(tid==0) first_bad=neigh_num;
 __syncthreads();
 for(int i=tid;i<neigh_num;i+=64)
  if(NL[a*neigh_num+i]<0 || rij[(a*neigh_num+i)*4]>rc) atomicMin(&first_bad,i);
 __syncthreads();
 for(int i=tid;i<first_bad;i+=64) {
  const int idx=(a*neigh_num+i)*4, t=types[NL[a*neigh_num+i]];
  const double d=rij[idx], r[3]={rij[idx+1],rij[idx+2],rij[idx+3]};
  double blm[24]={}; accumulate_blm_rij(d,r[0],r[1],r[2],blm);
  double f[4]={}, fd[CH*4]={}, dc[1]={}, unit[1]={1.};
  // fn=0 使空间导数无效；其输出没有使用，编译器会删除相关计算。
  if constexpr(LL5>0)
   accumulate_f12_with_5body(0,d,r,0.,0.,fp,sm,blm,f,fd,dc,unit,unit,t,TT,4,1,1,0,a,i);
  else if constexpr(LL4>0)
   accumulate_f12_with_4body(0,d,r,0.,0.,fp,sm,blm,f,fd,dc,unit,unit,t,TT,4,1,1,0,a,i);
  else
   accumulate_f12(0,d,r,0.,0.,fp,sm,blm,f,fd,dc,unit,unit,t,TT,4,1,1,0,a,i);
  double fc; find_fc(rc,rcinv,d,fc);
  #pragma unroll
  for(int k=0;k<KB;k++) { double fn; find_fn(k,rcinv,d,fc,fn); atomicAdd(&acc[t*KB+k],dc[0]*fn); }
 }
 __syncthreads();
 const int t1=types[a];
 for(int j=tid;j<TT*KB;j+=64) if(acc[j]!=0.)
  atomicAdd(&out[((t1*TT+j/KB)*NN+n)*KB+j%KB],acc[j]);
}

template<int KB, int TT>
static __global__ void find_angular_grad_dsnlm_split(
 const int N,const int neigh_num,const double rc,const double rcinv,
 const int64_t* NL,const double* rij,const int64_t* types,double* out) {
 __shared__ double acc[TT*24];
 __shared__ int first_bad;
 const int a=blockIdx.x,k=blockIdx.y,tid=threadIdx.x;
 for(int j=tid;j<TT*24;j+=64) acc[j]=0.;
 if(tid==0) first_bad=neigh_num;
 __syncthreads();
 for(int i=tid;i<neigh_num;i+=64)
  if(NL[a*neigh_num+i]<0 || rij[(a*neigh_num+i)*4]>rc) atomicMin(&first_bad,i);
 __syncthreads();
 for(int i=tid;i<first_bad;i+=64) {
  const int idx=(a*neigh_num+i)*4,t=types[NL[a*neigh_num+i]];
  const double d=rij[idx]; double fc,fn,blm[24]={};
  find_fc(rc,rcinv,d,fc); find_fn(k,rcinv,d,fc,fn);
  accumulate_blm_rij(d,rij[idx+1],rij[idx+2],rij[idx+3],blm);
  #pragma unroll
  for(int j=0;j<24;j++) atomicAdd(&acc[t*24+j],blm[j]*fn);
 }
 __syncthreads();
 for(int j=tid;j<TT*24;j+=64) if(acc[j]!=0.) out[((a*TT+j/24)*KB+k)*24+j%24]+=acc[j];
}
}
namespace matpl_angular_demand {
template<int KB,int NN,int TT,int LL4,int LL5,bool Context>
static __global__ void find_angular_grad_spatial_split(
 const int neigh_num,const int feat2,const int feat3,const double rc,const double rcinv,
 const int64_t* NL,const double* rij,const double* coeff,const int64_t* types,
 const double* grad,const double* sums,double* context,double* out) {
 constexpr int CH=4+(LL4>0)+(LL5>0);
 __shared__ double fp[CH],sm[24];
 __shared__ int first_bad;
 const int a=blockIdx.x,n=blockIdx.y,tid=threadIdx.x;
 for(int j=tid;j<CH;j+=64) fp[j]=grad[a*(feat2+feat3)+j*NN+n];
 for(int j=tid;j<24;j+=64) sm[j]=sums[(a*NN+n)*24+j];
 if(tid==0) first_bad=neigh_num;
 __syncthreads();
 for(int i=tid;i<neigh_num;i+=64)
  if(NL[a*neigh_num+i]<0 || rij[(a*neigh_num+i)*4]>rc) atomicMin(&first_bad,i);
 __syncthreads();
 for(int i=tid;i<first_bad;i+=64) {
  const int idx=(a*neigh_num+i)*4;
  const int t1=types[a],t2=types[NL[a*neigh_num+i]];
  const double d=rij[idx],r[3]={rij[idx+1],rij[idx+2],rij[idx+3]};
  double gn=0.,gnp=0.;
  {
   double fc,fcp,fn[KB],fnp[KB];find_fc_and_fcp(rc,rcinv,d,fc,fcp);
   find_fn_and_fnp(KB,rcinv,d,fc,fcp,fn,fnp);
   #pragma unroll
   for(int k=0;k<KB;k++) {const double c=coeff[((t1*TT+t2)*NN+n)*KB+k];gn+=fn[k]*c;gnp+=fnp[k]*c;}
  }
  double f[4]={},fd[CH*4]={};
  // 无系数收缩，相关占位数组及 blm 全部由编译器删除。
  double dummy[1]={};
  if constexpr(LL5>0)
   accumulate_f12_with_5body(0,d,r,gn,gnp,fp,sm,dummy,f,fd,dummy,dummy,dummy,t2,TT,4,1,0,0,a,i);
  else if constexpr(LL4>0)
   accumulate_f12_with_4body(0,d,r,gn,gnp,fp,sm,dummy,f,fd,dummy,dummy,dummy,t2,TT,4,1,0,0,a,i);
  else
   accumulate_f12(0,d,r,gn,gnp,fp,sm,dummy,f,fd,dummy,dummy,dummy,t2,TT,4,1,0,0,a,i);
  if constexpr(Context) {
   #pragma unroll
   for(int l=0;l<CH;l++) {
    const int p=((a*neigh_num+i)*feat3+l*NN+n)*4;
    context[p]=fd[l*4+3];context[p+1]=fd[l*4];context[p+2]=fd[l*4+1];context[p+3]=fd[l*4+2];
   }
  }
  atomicAdd(&out[idx],f[3]);atomicAdd(&out[idx+1],f[0]);atomicAdd(&out[idx+2],f[1]);atomicAdd(&out[idx+3],f[2]);
 }
}
}

namespace matpl_angular_demand {
template<bool Context, int NN>
static __device__ __forceinline__ void angular_store_channel(double* context,int base,int channel,const double* fd) {
 if constexpr(Context) {
  const int p=base+channel*NN*4;
  context[p]=fd[3];context[p+1]=fd[0];context[p+2]=fd[1];context[p+3]=fd[2];
 }
}
template<bool Context,int NN,int LL4=2,int LL5=0>
static __device__ __forceinline__ void angular_spatial_scoped(
 int n,double d,const double* r,double fn,double fnp,const double* fp,const double* sm,
 double* f,double* context,int base) {
 const double inv=1./d;
 fnp=fnp*inv-fn*inv*inv;fn=fn*inv;
 {
  if constexpr(LL5>0) {
   double raw[3]={sm[n*24],sm[n*24+1],sm[n*24+2]},fd[4]={};
   get_f12_5body(d,inv,fn,fnp,fp[NN*5+n],raw,r,f,fd,0,0);
   angular_store_channel<Context,NN>(context,base,5,fd);
  }
  double s[3]={sm[n*24]*0.238732414637843,sm[n*24+1]*0.119366207318922,sm[n*24+2]*0.119366207318922};
  double fd[4]={};get_f12_1(inv,fn,fnp,fp[n*4],s,r,f,fd,0,0);
  angular_store_channel<Context,NN>(context,base,0,fd);
 }
 fnp=fnp*inv-fn*inv*inv;fn=fn*inv;
 {
  double s[5]={sm[n*24+3],sm[n*24+4],sm[n*24+5],sm[n*24+6],sm[n*24+7]};
  if constexpr(LL4>0) {
   double fd[4]={};get_f12_4body(d,inv,fn,fnp,fp[NN*4+n],s,r,f,fd,0,0);
   angular_store_channel<Context,NN>(context,base,4,fd);
  }
  s[0]*=0.099471839432435;s[1]*=0.596831036594608;s[2]*=0.596831036594608;s[3]*=0.149207759148652;s[4]*=0.149207759148652;
  double fd[4]={};get_f12_2(d,inv,fn,fnp,fp[n*4+1],s,r,f,fd,0,0);
  angular_store_channel<Context,NN>(context,base,1,fd);
 }
 fnp=fnp*inv-fn*inv*inv;fn=fn*inv;
 {
  double s[7]={sm[n*24+8]*0.139260575205408,sm[n*24+9]*0.104445431404056,sm[n*24+10]*0.104445431404056,sm[n*24+11]*1.044454314040563,sm[n*24+12]*1.044454314040563,sm[n*24+13]*0.174075719006761,sm[n*24+14]*0.174075719006761};
  double fd[4]={};get_f12_3(d,inv,fn,fnp,fp[n*4+2],s,r,f,fd,0,0);
  angular_store_channel<Context,NN>(context,base,2,fd);
 }
 fnp=fnp*inv-fn*inv*inv;fn=fn*inv;
 {
  double s[9]={sm[n*24+15]*0.011190581936149,sm[n*24+16]*0.223811638722978,sm[n*24+17]*0.223811638722978,sm[n*24+18]*0.111905819361489,sm[n*24+19]*0.111905819361489,sm[n*24+20]*1.566681471060845,sm[n*24+21]*1.566681471060845,sm[n*24+22]*0.195835183882606,sm[n*24+23]*0.195835183882606};
  double fd[4]={};get_f12_4(r[0],r[1],r[2],d,inv,fn,fnp,fp[n*4+3],s,f,fd,0,0);
  angular_store_channel<Context,NN>(context,base,3,fd);
 }
}
}
// 保留输出裁剪；每个中心原子一个 block，每个线程处理一个邻居。
// basis 和几何只算一次；短阶段之间使用显式 LDS 缓存，避免隐式 scratch。
namespace matpl_angular_demand {
template<int KB,int NN,int TT,int Outputs,int LL4=2,int LL5=0>
static __global__ void find_angular_grad_phase_fused(
 const int neigh_num,const int feat2,const int feat3,const double rc,const double rcinv,
 const int64_t* NL,const double* rij,const double* coeff,const int64_t* types,
 const double* grad,const double* sums,double* context,double* out,double* dc_out,double* ds_out) {
 constexpr int CH=4+(LL4>0)+(LL5>0);
 __shared__ double fp[NN*CH],sm[NN*24];
 __shared__ double contracted[((Outputs&2) && 24>2*NN)?24:2*NN][32];
 __shared__ double coeff_contracted[((Outputs&3)==3)?NN:1][32];
 __shared__ int head[(Outputs&3)?TT:1],next[(Outputs&3)?32:1];
 __shared__ double basis[((Outputs&3)?KB:1)][32];
 const int a=blockIdx.x,tid=threadIdx.x;
 for(int j=tid;j<NN*CH;j+=32) {
  const int n=j/CH,l=j%CH;
  fp[(l>=4?NN*l+n:n*4+l)]=grad[a*(feat2+feat3)+l*NN+n];
 }
 for(int j=tid;j<NN*24;j+=32) sm[j]=sums[a*NN*24+j];
 __syncthreads();
 for(int chunk=0;chunk<neigh_num;chunk+=32) {
  const int i=chunk+tid;
  if constexpr(Outputs&3) {for(int t=tid;t<TT;t+=32)head[t]=-1;__syncthreads();}
  const int neighbor=(i<neigh_num?NL[a*neigh_num+i]:-1);
  if(neighbor>=0 && rij[(a*neigh_num+i)*4]<=rc) {
  const int idx=(a*neigh_num+i)*4;
  const double d=rij[idx];
  const double r[3]={rij[idx+1],rij[idx+2],rij[idx+3]};
  const int t1=types[a],t2=types[neighbor],ci=(t1*TT+t2)*NN*KB;
  {
   double fc,fcp,fn[KB],fnp[KB];find_fc_and_fcp(rc,rcinv,d,fc,fcp);
   find_fn_and_fnp(KB,rcinv,d,fc,fcp,fn,fnp);
   if constexpr(Outputs&3) {
    #pragma unroll
    for(int k=0;k<KB;k++) basis[k][tid]=fn[k];
   }
   #pragma unroll 1
   for(int n=0;n<NN;n++) {
    double gn=0.,gnp=0.;
    #pragma unroll
    for(int k=0;k<KB;k++) {const double c=coeff[ci+n*KB+k];gn+=fn[k]*c;gnp+=fnp[k]*c;}
    contracted[n*2][tid]=gn;contracted[n*2+1][tid]=gnp;
   }
  }
  // 无球谐数组、basis 向量和系数梯度数组跨越位置导数阶段。
  {
   double f[4]={};
   #pragma unroll 1
   for(int n=0;n<NN;n++) {
    angular_spatial_scoped<(Outputs&4)!=0,NN,LL4,LL5>(n,d,r,contracted[n*2][tid],contracted[n*2+1][tid],fp,sm,f,context,((a*neigh_num+i)*feat3+n)*4);
   }
   out[idx]=f[3];out[idx+1]=f[0];out[idx+2]=f[1];out[idx+3]=f[2];
  }
  if constexpr(Outputs&3) {
   double blm[24]={};accumulate_blm_rij(d,r[0],r[1],r[2],blm);
   if constexpr(Outputs&1) {
    #pragma unroll 1
    for(int n=0;n<NN;n++) {
     double f[4]={},fd[CH*4]={},dc[1]={},unit[1]={1.};
     if constexpr(LL5>0) accumulate_f12_with_5body(n,d,r,0.,0.,fp,sm,blm,f,fd,dc,unit,unit,t2,TT,4,NN,1,0,a,i);
     else if constexpr(LL4>0) accumulate_f12_with_4body(n,d,r,0.,0.,fp,sm,blm,f,fd,dc,unit,unit,t2,TT,4,NN,1,0,a,i);
     else accumulate_f12(n,d,r,0.,0.,fp,sm,blm,f,fd,dc,unit,unit,t2,TT,4,NN,1,0,a,i);
     if constexpr((Outputs&3)==3) coeff_contracted[n][tid]=dc[0];
     else contracted[n][tid]=dc[0];
    }
   }
   if constexpr(Outputs&2) {
    #pragma unroll
    for(int j=0;j<24;j++) contracted[j][tid]=blm[j];

   }
   next[tid]=atomicExch(&head[t2],tid);
  }
  }
  if constexpr(Outputs&3) {
   __syncthreads();
   if constexpr(Outputs&1) {
    for(int j=tid;j<TT*NN*KB;j+=32) {
     const int t=j/(NN*KB),n=(j/KB)%NN,k=j%KB;
     if(head[t]>=0) {
      double value=0.;
      #pragma unroll 1
      for(int v=head[t];v>=0;v=next[v])value+=((Outputs&3)==3?coeff_contracted[n][v]:contracted[n][v])*basis[k][v];
      atomicAdd(&dc_out[types[a]*TT*NN*KB+j],value);
     }
    }
   }
   if constexpr(Outputs&2) {
   __syncthreads();
   for(int j=tid;j<TT*KB*24;j+=32) {
    const int t=j/(KB*24),k=(j/24)%KB,l=j%24;
    if(head[t]>=0) {
     double value=0.;
     #pragma unroll 1
     for(int v=head[t];v>=0;v=next[v])value+=contracted[l][v]*basis[k][v];
     ds_out[a*TT*KB*24+j]+=value;
    }
   }
   }
   __syncthreads();
  }
 }
}
}
