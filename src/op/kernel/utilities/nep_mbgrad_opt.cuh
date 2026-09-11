#pragma once

#include "nep_utilities_mb_secondc.cuh"
#include "../../include/nep_limits.h"
#include <cstddef>
#include <cstdint>

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
  __device__ __forceinline__ void add(int type_slot, int n, int basis, double value) const {
    atomicAdd(&values[(type_slot * NMAX + n) * NBASIS + basis], value);
  }
};

// One addressable per-neighbor context crosses the angular call boundary.
// Passing the FP64 state as a single reference avoids keeping a wide argument
// list live in caller registers while an angular specialization runs. This is
// thread-private storage; the cooperative shared-memory layout is unchanged.
template<int NBASIS>
struct NeighborContext {
  double d12;
  double r12[3];
  double scd_r12[4];
  double fn12[NBASIS];
  double fnp12[NBASIS];
  double fn, fnp, d12inv, rij_Lsq, rij_L2sq;
  const double* Fp;
  const double* dsnlm_dc;
  const double* sum_fxyz;
  double* output;
  int n, type_slot, tile_count;
};

// Each angular order owns only its 2L+1 FP64 components. In particular, do
// not rebuild the legacy six 24-element arrays in the neighbor scope.
template<int L>
struct AngularScratch {
  static_assert(L >= 1 && L <= 4, "Supported angular orders");
  double blm[2 * L + 1];
  double rij_blm[2 * L + 1];
  double dblm_x[2 * L + 1];
  double dblm_y[2 * L + 1];
  double dblm_z[2 * L + 1];
  double dblm_r[2 * L + 1];
};

template<int L>
__device__ __forceinline__ void build_angular_scratch(
    double d12, double x, double y, double z, AngularScratch<L>& scratch) {

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
  if constexpr (L == 1) {
    scratch.blm[0]     = z;                                             // Y10 blm
    scratch.dblm_r[0]  = 0.0;                                           // Y10 blm/dr
    scratch.dblm_x[0]  = 0.0;                                           // Y10 blm/dx
    scratch.dblm_y[0]  = 0.0;                                           // Y10 blm/dy
    scratch.dblm_z[0]  = 1.0;                                           // Y10 blm/dz
    scratch.rij_blm[0] = z12;                                           // Y10 sij_blm
    scratch.blm[1]     = x;                                             // Y11_real
    scratch.dblm_r[1]  = 0.0;
    scratch.dblm_x[1]  = 1.0;
    scratch.dblm_y[1]  = 0.0;
    scratch.dblm_z[1]  = 0.0;
    scratch.rij_blm[1] = x12;
    scratch.blm[2]     = y;                                             // Y11_imag
    scratch.dblm_r[2]  = 0.0;
    scratch.dblm_x[2]  = 0.0;
    scratch.dblm_y[2]  = 1.0;
    scratch.dblm_z[2]  = 0.0;
    scratch.rij_blm[2] = y12;
  }
  if constexpr (L == 2) {
    scratch.blm[0]     = 3.0 * z2- d12 * d12;                           // Y20
    scratch.dblm_r[0]  = -2.0 * d12;
    scratch.dblm_x[0]  = 0.0;
    scratch.dblm_y[0]  = 0.0;
    scratch.dblm_z[0]  = 6.0 * z;
    scratch.rij_blm[0] = 3.0 * z12sq - 1.0;
    scratch.blm[1]    = xz;                                             // Y21_real
    scratch.dblm_r[1] = 0.0;
    scratch.dblm_x[1] = z;
    scratch.dblm_y[1] = 0.0;
    scratch.dblm_z[1] = x;
    scratch.rij_blm[1]= x12 * z12;
    scratch.blm[2]    = yz;                                             // Y21_imag
    scratch.dblm_r[2] = 0.0;
    scratch.dblm_x[2] = 0.0;
    scratch.dblm_y[2] = z;
    scratch.dblm_z[2] = y;
    scratch.rij_blm[2]= y12 * z12;
    scratch.blm[3]    = x2 - y2;                                        // Y22_real
    scratch.dblm_r[3] = 0.0;
    scratch.dblm_x[3] = 2.0 * x;
    scratch.dblm_y[3] = -2.0 * y;
    scratch.dblm_z[3] = 0.0;
    scratch.rij_blm[3]= x12sq_minus_y12sq;
    scratch.blm[4]     = 2.0 * xy;                                      // Y22_imag
    scratch.dblm_r[4]  = 0.0;
    scratch.dblm_x[4]  = 2.0 * y;
    scratch.dblm_y[4]  = 2.0 * x;
    scratch.dblm_z[4]  = 0.0;
    scratch.rij_blm[4] = 2.0 * x12 * y12;
  }
  if constexpr (L == 3) {
    scratch.blm[0]     = (5.0 * z2 - 3.0 * r2) * z;                     // Y30
    scratch.dblm_r[0]  = -6.0 * z * d12;
    scratch.dblm_x[0]  = 0.0;
    scratch.dblm_y[0]  = 0.0;
    scratch.dblm_z[0]  = 15 * z2 - 3 * r2;
    scratch.rij_blm[0] = (5.0 * z12sq - 3.0) * z12;
    scratch.blm[1]     = (5.0 * z2 - r2) * x;                          // Y31_real
    scratch.dblm_r[1]  = -2.0 * x * d12;
    scratch.dblm_x[1]  = 5.0 * z2 - r2;
    scratch.dblm_y[1]  = 0.0;
    scratch.dblm_z[1]  = 10.0 * xz;
    scratch.rij_blm[1] = (5.0 * z12sq - 1.0) * x12;
    scratch.blm[2]    = (5.0 * z2 - r2) * y;                          // Y31_imag
    scratch.dblm_r[2] = -2.0 * y * d12;
    scratch.dblm_x[2] = 0.0;
    scratch.dblm_y[2] = 5.0 * z2 - r2;
    scratch.dblm_z[2] = 10.0 * yz;
    scratch.rij_blm[2]= (5.0 * z12sq - 1.0) * y12;
    scratch.blm[3]    = (x2 - y2) * z;                                // Y32_real
    scratch.dblm_r[3] = 0.0;
    scratch.dblm_x[3] = 2.0 * xz;
    scratch.dblm_y[3] = -2.0 * yz;
    scratch.dblm_z[3] = x2 - y2;
    scratch.rij_blm[3]= x12sq_minus_y12sq * z12;
    scratch.blm[4]     = 2.0 * xyz;                                // Y32_imag
    scratch.dblm_r[4]  = 0.0;
    scratch.dblm_x[4]  = 2.0 * yz;
    scratch.dblm_y[4]  = 2.0 * xz;
    scratch.dblm_z[4]  = 2.0 * xy;
    scratch.rij_blm[4] = 2.0 * x12 * y12 * z12;
    scratch.blm[5]    = (x2 - 3.0 * y2) * x;                           // Y33_real
    scratch.dblm_r[5] = 0.0;
    scratch.dblm_x[5] = 3.0 * (x2 - y2);
    scratch.dblm_y[5] = -6.0 * xy;
    scratch.dblm_z[5] = 0.0;
    scratch.rij_blm[5]= (x12 * x12 - 3.0 * y12 * y12) * x12;
    scratch.blm[6]    = (3.0 * x2 - y2) * y;                           // Y33_imag
    scratch.dblm_r[6] = 0.0;
    scratch.dblm_x[6] = 6.0 * xy;
    scratch.dblm_y[6] = 3.0 * (x2 - y2);
    scratch.dblm_z[6] = 0.0;
    scratch.rij_blm[6]= (3.0 * x12 * x12 - y12 * y12) * y12;
  }
  if constexpr (L == 4) {
    scratch.blm[0]    = (35.0 * z2 - 30.0 * r2) * z2 + 3.0 * r2 * r2;   // Y40
    scratch.dblm_r[0] = (-60.0) * z2 * d12 + 12.0 * r3;
    scratch.dblm_x[0] = 0.0;
    scratch.dblm_y[0] = 0.0;
    scratch.dblm_z[0] = 140.0 * z3 - 60.0 * z * r2;
    scratch.rij_blm[0]= ((35.0 * z12sq - 30.0) * z12sq + 3.0);
    scratch.blm[1]    = (7.0 * z2 - 3.0 * r2) * xz;                    // Y41_real
    scratch.dblm_r[1] = -6.0 * xz * d12;
    scratch.dblm_x[1] = 7.0 * z3 - 3.0 * z * r2;
    scratch.dblm_y[1] = 0.0;
    scratch.dblm_z[1]  = 21.0 * x * z2 - 3.0 * x * r2;
    scratch.rij_blm[1] = (7.0 * z12sq - 3.0) * x12 * z12;
    scratch.blm[2]    = (7.0 * z2 - 3.0 * r2) * yz;                    // Y41_iamg
    scratch.dblm_r[2] = -6.0 * yz * d12;
    scratch.dblm_x[2] = 0.0;
    scratch.dblm_y[2] = 7.0 * z3 - 3.0 * z * r2;
    scratch.dblm_z[2] = 21.0 * y * z2 - 3.0 * y * r2;
    scratch.rij_blm[2]= (7.0 * z12sq - 3.0) * y12 * z12;
    scratch.blm[3]    = (7.0 * z2 - r2) * x2my2;                       // Y42_real
    scratch.dblm_r[3] = 2.0 * d12 * (y2 - x2);
    scratch.dblm_x[3] = 14.0 * x * z2 - 2.0 * x * r2;
    scratch.dblm_y[3] = 2.0 * y *(r2 - 7.0 * z2);
    scratch.dblm_z[3] = 14.0 * x2 * z - 14.0 * y2 * z;
    scratch.rij_blm[3]= (7.0 * z12sq - 1.0) * x12sq_minus_y12sq;
    scratch.blm[4]    = (7.0 * z2 - r2) * 2.0 * xy;                    // Y42_imag
    scratch.dblm_r[4] = -4.0 * xy * d12;
    scratch.dblm_x[4] = 2.0 * y * (7.0 * z2 - r2);
    scratch.dblm_y[4] = 2.0 * x * (7.0 * z2 - r2);
    scratch.dblm_z[4] = 28.0 * xyz;
    scratch.rij_blm[4]= (7.0 * z12sq - 1.0) * x12 * y12 * 2.0;
    scratch.blm[5]    = (x2 - 3.0 * y2) * xz;                          // Y43_real
    scratch.dblm_r[5] = 0.0;
    scratch.dblm_x[5] = 3.0 * z * (x2 - y2);
    scratch.dblm_y[5] = -6.0 * xyz;
    scratch.dblm_z[5] = x3 - 3.0 * x * y2;
    scratch.rij_blm[5]= (x12sq - 3.0 * y12sq) * x12 * z12;
    scratch.blm[6]    = (3.0 * x2 - y2) * yz;                         // Y43_imag
    scratch.dblm_r[6] = 0.0;
    scratch.dblm_x[6] = 6.0 * xyz;
    scratch.dblm_y[6] = 3.0 * x2 * z - 3.0 * y2 * z;
    scratch.dblm_z[6] = 3.0 * y * x2 - y3;
    scratch.rij_blm[6]= (3.0 * x12sq - y12sq) * y12 * z12;
    scratch.blm[7]    = x2my2 * x2my2 - 4.0 * x2 * y2;                // Y44_real
    scratch.dblm_r[7] = 0.0;
    scratch.dblm_x[7] = 4.0 * x3 - 12.0 * x * y2;
    scratch.dblm_y[7] = 4.0 * y3 - 12.0 * x2 * y;
    scratch.dblm_z[7] = 0.0;
    scratch.rij_blm[7]= (x12sq_minus_y12sq * x12sq_minus_y12sq - 4.0 * x12sq * y12sq);
    scratch.blm[8]    = 4.0 * x2my2 * xy;                            // Y44_imag
    scratch.dblm_r[8] = 0.0;
    scratch.dblm_x[8] = 12.0 * x2 * y - 4.0 * y3;
    scratch.dblm_y[8] = 4.0 * x3 - 12.0 * x * y2;
    scratch.dblm_z[8] = 0.0;
    scratch.rij_blm[8]= (4.0 * x12 * y12 * x12sq_minus_y12sq);
  }
}

// Borrowed inputs for one basis contribution. Its arrays remain owned by the
// current angular order; no angular values escape into another L scope.
// The noinline basis helpers bound algebra temporaries to one k. Their callers
// keep fixed, unrolled k loops and issue the same r/x/y/z atomics in that order.
struct AngularContribution {
  const double *fn12, *fnp12, *blm, *rij_blm;
  const double *dblm_x, *dblm_y, *dblm_z, *dblm_r;
  const double *scd_r12, *dsnlm_dc, *s, *r12;
  double d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp;
  int type_slot, tile_count, n;
  double* output;
};

template<int L, int NMAX, int NBASIS, int TYPE_TILE>
__device__ __noinline__ void accumulate_direct_basis(const AngularContribution& c, int k, int uj)
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
  const SharedTileSink<NMAX, NBASIS, TYPE_TILE> sink{c.output};

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
  const double* blm, const double* rij_blm,
  const double* dblm_x, const double* dblm_y, const double* dblm_z, const double* dblm_r,
  const double* scd_r12, const double* dsnlm_dc, const double* s, const double* r12,
  double d12inv, double rij_Lsq, double rij_L2sq, double fn, double fnp, double Fp,
  int type_slot, int tile_count, int n, SharedTileSink<NMAX, NBASIS, TYPE_TILE> sink)
{
  const AngularContribution c{fn12, fnp12, blm, rij_blm,
      dblm_x, dblm_y, dblm_z, dblm_r, scd_r12, dsnlm_dc, s, r12,
      d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp, type_slot, tile_count, n, sink.values};
  if (type_slot < 0) return;
  #pragma unroll
  for (int k = 0; k < NBASIS; ++k) {
    accumulate_direct_basis<L, NMAX, NBASIS, TYPE_TILE>(c, k, 0);
  }
}


template<int L, int NMAX, int NBASIS, int TYPE_TILE>
__device__ __noinline__ void accumulate_cross_basis(const AngularContribution& c, int k, int uj)
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
  const SharedTileSink<NMAX, NBASIS, TYPE_TILE> sink{c.output};

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
  const double* blm, const double* rij_blm,
  const double* dblm_x, const double* dblm_y, const double* dblm_z, const double* dblm_r,
  const double* scd_r12, const double* dsnlm_dc, const double* s, const double* r12,
  double d12inv, double rij_Lsq, double rij_L2sq, double fn, double fnp, double Fp,
  int type_slot, int tile_count, int n, SharedTileSink<NMAX, NBASIS, TYPE_TILE> sink)
{
  const AngularContribution c{fn12, fnp12, blm, rij_blm,
      dblm_x, dblm_y, dblm_z, dblm_r, scd_r12, dsnlm_dc, s, r12,
      d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp, type_slot, tile_count, n, sink.values};
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
__device__ __noinline__ void accumulate_four_body_basis(const AngularContribution& c, int k, int uj)
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
  const SharedTileSink<NMAX, NBASIS, TYPE_TILE> sink{c.output};

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
  const double* blm, const double* rij_blm,
  const double* dblm_x, const double* dblm_y, const double* dblm_z, const double* dblm_r,
  const double* scd_r12, const double* dsnlm_dc, const double* s, const double* r12,
  double d12inv, double rij_Lsq, double rij_L2sq, double fn, double fnp, double Fp,
  int type_slot, int tile_count, int n, SharedTileSink<NMAX, NBASIS, TYPE_TILE> sink)
{
  const AngularContribution c{fn12, fnp12, blm, rij_blm,
      dblm_x, dblm_y, dblm_z, dblm_r, scd_r12, dsnlm_dc, s, r12,
      d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp, type_slot, tile_count, n, sink.values};
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
__device__ __noinline__ void accumulate_five_body_basis(const AngularContribution& c, int k, int uj)
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
  const SharedTileSink<NMAX, NBASIS, TYPE_TILE> sink{c.output};

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
  const double* blm, const double* rij_blm,
  const double* dblm_x, const double* dblm_y, const double* dblm_z, const double* dblm_r,
  const double* scd_r12, const double* dsnlm_dc, const double* s, const double* r12,
  double d12inv, double rij_Lsq, double rij_L2sq, double fn, double fnp, double Fp,
  int type_slot, int tile_count, int n, SharedTileSink<NMAX, NBASIS, TYPE_TILE> sink)
{
  const AngularContribution c{fn12, fnp12, blm, rij_blm,
      dblm_x, dblm_y, dblm_z, dblm_r, scd_r12, dsnlm_dc, s, r12,
      d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp, type_slot, tile_count, n, sink.values};
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
__device__ __noinline__ void accumulate_angular_order(const NeighborContext<NBASIS>& w)
{
  const int n = w.n, type_slot = w.type_slot, tile_count = w.tile_count;
  const double d12 = w.d12, fn = w.fn, fnp = w.fnp;
  const double d12inv = w.d12inv, rij_Lsq = w.rij_Lsq, rij_L2sq = w.rij_L2sq;
  const double* r12 = w.r12;
  const double* scd_r12 = w.scd_r12;
  const double* fn12 = w.fn12;
  const double* fnp12 = w.fnp12;
  const double* Fp = w.Fp;
  const double* dsnlm_dc = w.dsnlm_dc;
  const double* sum_fxyz = w.sum_fxyz;
  const SharedTileSink<NMAX, NBASIS, TYPE_TILE> sink{w.output};
  AngularScratch<L> scratch{};
  build_angular_scratch<L>(d12, r12[0], r12[1], r12[2], scratch);
  constexpr int offset = L * L - 1;
  double sums[2 * L + 1];
  #pragma unroll
  for (int m = 0; m < 2 * L + 1; ++m) {
    sums[m] = sum_fxyz[n * NUM_OF_ABC + offset + m];
  }
  // Keep the legacy five-body, L=1, four-body, L=2, L=3, L=4 sequence.
  // The higher-body terms consume the unweighted sums of the same order.
  if constexpr (L == 1 && HAS5) {
    accumulate_five_body<false, NMAX, NBASIS, TYPE_TILE>(fn12, fnp12,
        scratch.blm, scratch.rij_blm, scratch.dblm_x, scratch.dblm_y, scratch.dblm_z, scratch.dblm_r,
        scd_r12, dsnlm_dc, sums, r12,
        d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[NMAX * LMAX3 + NMAX + n], type_slot, tile_count, n, sink);
    accumulate_five_body<true, NMAX, NBASIS, TYPE_TILE>(fn12, fnp12,
        scratch.blm, scratch.rij_blm, scratch.dblm_x, scratch.dblm_y, scratch.dblm_z, scratch.dblm_r,
        scd_r12, dsnlm_dc, sums, r12,
        d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[NMAX * LMAX3 + NMAX + n], type_slot, tile_count, n, sink);
  }
  if constexpr (L == 2 && HAS4) {
    accumulate_four_body<false, NMAX, NBASIS, TYPE_TILE>(fn12, fnp12,
        scratch.blm, scratch.rij_blm, scratch.dblm_x, scratch.dblm_y, scratch.dblm_z, scratch.dblm_r,
        scd_r12, dsnlm_dc, sums, r12,
        d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[NMAX * LMAX3 + n], type_slot, tile_count, n, sink);
    accumulate_four_body<true, NMAX, NBASIS, TYPE_TILE>(fn12, fnp12,
        scratch.blm, scratch.rij_blm, scratch.dblm_x, scratch.dblm_y, scratch.dblm_z, scratch.dblm_r,
        scd_r12, dsnlm_dc, sums, r12,
        d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[NMAX * LMAX3 + n], type_slot, tile_count, n, sink);
  }
  #pragma unroll
  for (int m = 0; m < 2 * L + 1; ++m) {
    sums[m] *= C3B[offset + m];
  }
  accumulate_direct<L, NMAX, NBASIS, TYPE_TILE>(fn12, fnp12,
        scratch.blm, scratch.rij_blm, scratch.dblm_x, scratch.dblm_y, scratch.dblm_z, scratch.dblm_r,
        scd_r12, dsnlm_dc, sums, r12,
        d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n * LMAX3 + L - 1], type_slot, tile_count, n, sink);
  accumulate_cross<L, NMAX, NBASIS, TYPE_TILE>(fn12, fnp12,
        scratch.blm, scratch.rij_blm, scratch.dblm_x, scratch.dblm_y, scratch.dblm_z, scratch.dblm_r,
        scd_r12, dsnlm_dc, sums, r12,
        d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n * LMAX3 + L - 1], type_slot, tile_count, n, sink);
}

template<int NMAX, int NBASIS, int LMAX3, bool HAS4, bool HAS5, int TYPE_TILE>
__device__ __forceinline__ void accumulate_neighbor(NeighborContext<NBASIS>& w)
{
  static_assert(LMAX3 == 4, "This numerical port supports the OMat24 angular orders");
  w.d12inv = 1.0 / w.d12;
  w.rij_Lsq = w.d12inv;
  w.rij_L2sq = w.d12inv * w.d12inv;
  w.fnp = w.fnp * w.d12inv - w.fn * w.d12inv * w.d12inv;
  w.fn = w.fn * w.d12inv;
  {
    accumulate_angular_order<1, NMAX, NBASIS, LMAX3, HAS4, HAS5, TYPE_TILE>(w);
  }
  w.fnp = w.fnp * w.d12inv - w.fn * w.d12inv * w.d12inv;
  w.fn = w.fn * w.d12inv;
  w.rij_Lsq = w.rij_L2sq;
  w.rij_L2sq = w.rij_L2sq * w.d12inv;
  {
    accumulate_angular_order<2, NMAX, NBASIS, LMAX3, HAS4, HAS5, TYPE_TILE>(w);
  }
  w.fnp = w.fnp * w.d12inv - w.fn * w.d12inv * w.d12inv;
  w.fn = w.fn * w.d12inv;
  w.rij_Lsq = w.rij_L2sq;
  w.rij_L2sq = w.rij_L2sq * w.d12inv;
  {
    accumulate_angular_order<3, NMAX, NBASIS, LMAX3, HAS4, HAS5, TYPE_TILE>(w);
  }
  w.fnp = w.fnp * w.d12inv - w.fn * w.d12inv * w.d12inv;
  w.fn = w.fn * w.d12inv;
  w.rij_Lsq = w.rij_L2sq;
  w.rij_L2sq = w.rij_L2sq * w.d12inv;
  {
    accumulate_angular_order<4, NMAX, NBASIS, LMAX3, HAS4, HAS5, TYPE_TILE>(w);
  }
}

} // namespace nep_mb_secondgrad_opt

template<int NMAX, int NBASIS, int TYPE_TILE>
__host__ __device__ constexpr size_t nep_mb_secondgrad_shared_bytes() {
  return nep_mb_secondgrad_opt::SharedLayout<NMAX, NBASIS, TYPE_TILE>::bytes;
}

template<int NMAX, int NBASIS, int LMAX3, bool HAS4, bool HAS5, int TYPE_TILE, int CTA_THREADS>
__global__ void nep_mb_secondgrad_fused(NepMbSecondGradArgs a) {
  using namespace nep_mb_secondgrad_opt;
  static_assert(LMAX3 == 4 && HAS4 && HAS5, "Only OMat24 is dispatched");
  static_assert(CTA_THREADS == 64, "OMat24 uses a fixed CTA width");
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
  for (int i = tid; i < NMAX * 6; i += CTA_THREADS) {
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

      double fc12, fcp12;
      find_fc_and_fcp(a.rcut, a.rcut_inv, distance, fc12, fcp12);
      double fn12[NBASIS];
      find_fn(NBASIS, a.rcut_inv, distance, fc12, fn12);
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
      work.r12[0] = a.d12[row * 4 + 1];
      work.r12[1] = a.d12[row * 4 + 2];
      work.r12[2] = a.d12[row * 4 + 3];
      work.scd_r12[0] = a.grad_second[row * 4];
      work.scd_r12[1] = a.grad_second[row * 4 + 1];
      work.scd_r12[2] = a.grad_second[row * 4 + 2];
      work.scd_r12[3] = a.grad_second[row * 4 + 3];
      work.Fp = s.fp;
      work.dsnlm_dc = s.dsnlm_tile;
      work.sum_fxyz = s.sum_fxyz;
      work.output = sink.values;
      work.n = n;
      work.type_slot = type_slot;
      work.tile_count = tile_count;
      double (&fn12)[NBASIS] = work.fn12;
      double (&fnp12)[NBASIS] = work.fnp12;
      double fc12, fcp12;
      find_fc_and_fcp(a.rcut, a.rcut_inv, distance, fc12, fcp12);
      find_fn_and_fnp(NBASIS, a.rcut_inv, distance, fc12, fcp12, fn12, fnp12);

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
