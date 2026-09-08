#pragma once

#include "nep_utilities_mb_secondc.cuh"
#include "../../include/nep_limits.h"
#include <cstddef>
#include <cstdint>

// Correctness-first OMat24 path. Numerical expressions below are ported from
// nep_utilities_mb_secondc.cuh; only indexing and accumulation destinations differ.
struct NepMbSecondGradArgs {
  const double* grad_second;
  const double* d12;
  const int64_t* neighbor_list;
  const double* de_dfeat;
  const double* dsnlm_dc;
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

template<int L, int NMAX, int NBASIS, int TYPE_TILE>
__device__ __forceinline__ void accumulate_direct(
  const double* fn12, const double* fnp12,
  const double* blm, const double* rij_blm,
  const double* dblm_x, const double* dblm_y, const double* dblm_z, const double* dblm_r,
  const double* scd_r12, const double* dsnlm_dc, const double* s, const double* r12,
  double d12inv, double rij_Lsq, double rij_L2sq, double fn, double fnp, double Fp,
  int type_slot, int tile_count, int n, SharedTileSink<NMAX, NBASIS, TYPE_TILE> sink)
{
  if constexpr (L == 1) {
    if (type_slot < 0) return;

    double dfk = 0.0;
    int dsnlm_idx = 0 + type_slot * NBASIS * NUM_OF_ABC;
    for(int k=0; k < NBASIS; k++) {
      int dsnlm_i = dsnlm_idx + k * NUM_OF_ABC;
      double tmpr = 0.0, tmpx = 0.0, tmpy = 0.0, tmpz = 0.0;
      double rr0 = 0.0, rr1 = 0.0, rr2 = 0.0;
      double rrr0 = 0.0, rrr1 = 0.0, rrr2=0.0;
      rr0 =       C3B[0] * dsnlm_dc[dsnlm_i]   * fnp * blm[0];
      rr1 = 2.0 * C3B[1] * dsnlm_dc[dsnlm_i+1] * fnp * blm[1];
      rr2 = 2.0 * C3B[2] * dsnlm_dc[dsnlm_i+2] * fnp * blm[2];
      dfk = fnp12[k] * rij_Lsq - fn12[k] * rij_L2sq;
      rrr0 =       s[0] * dfk * blm[0];
      rrr1 = 2.0 * s[1] * dfk * blm[1];
      rrr2 = 2.0 * s[2] * dfk * blm[2];
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

  }
  if constexpr (L == 2) {
    if (type_slot < 0) return;

    int dsnlm_idx = 0 + type_slot * NBASIS * NUM_OF_ABC;
    for(int k=0; k < NBASIS; k++) {
      int dsnlm_i = dsnlm_idx + k * NUM_OF_ABC;
      double tmpr = 0.0, tmpx = 0.0, tmpy = 0.0, tmpz = 0.0;
      tmpr +=  C3B[3] * dsnlm_dc[dsnlm_i+3] * (fnp * blm[3] + fn * dblm_r[3]) +
                    2.0 * C3B[4] * dsnlm_dc[dsnlm_i+4] * fnp * blm[4] +
                    2.0 * C3B[5] * dsnlm_dc[dsnlm_i+5] * fnp * blm[5] +
                    2.0 * C3B[6] * dsnlm_dc[dsnlm_i+6] * fnp * blm[6] +
                    2.0 * C3B[7] * dsnlm_dc[dsnlm_i+7] * fnp * blm[7];
      tmpr += s[0] * ((fnp12[k] * rij_Lsq - 2.0 * fn12[k] * rij_L2sq) * blm[3] + fn12[k] * rij_Lsq * dblm_r[3]) +
              2.0 * s[1] * (fnp12[k] * rij_Lsq - 2.0 * fn12[k] * rij_L2sq) * blm[4] +
              2.0 * s[2] * (fnp12[k] * rij_Lsq - 2.0 * fn12[k] * rij_L2sq) * blm[5] +
              2.0 * s[3] * (fnp12[k] * rij_Lsq - 2.0 * fn12[k] * rij_L2sq) * blm[6] +
              2.0 * s[4] * (fnp12[k] * rij_Lsq - 2.0 * fn12[k] * rij_L2sq) * blm[7];
      sink.add(type_slot, n, k, 2.0 * Fp * scd_r12[0] * tmpr);
      tmpx +=
                    2.0 * C3B[4] * dsnlm_dc[dsnlm_i+4] * fn * dblm_x[4] +
                    2.0 * C3B[6] * dsnlm_dc[dsnlm_i+6] * fn * dblm_x[6] +
                    2.0 * C3B[7] * dsnlm_dc[dsnlm_i+7] * fn * dblm_x[7];
      tmpx +=
              2.0 * s[1] * fn12[k] * rij_Lsq * dblm_x[4] +
              2.0 * s[3] * fn12[k] * rij_Lsq * dblm_x[6] +
              2.0 * s[4] * fn12[k] * rij_Lsq * dblm_x[7];
      sink.add(type_slot, n, k, 2.0 * Fp * scd_r12[1] * tmpx);
      tmpy +=
                    2.0 * C3B[5] * dsnlm_dc[dsnlm_i+5] * fn * dblm_y[5] +
                    2.0 * C3B[6] * dsnlm_dc[dsnlm_i+6] * fn * dblm_y[6] +
                    2.0 * C3B[7] * dsnlm_dc[dsnlm_i+7] * fn * dblm_y[7];
      tmpy +=
              2.0 * s[2] * fn12[k] * rij_Lsq * dblm_y[5] +
              2.0 * s[3] * fn12[k] * rij_Lsq * dblm_y[6] +
              2.0 * s[4] * fn12[k] * rij_Lsq * dblm_y[7];
      sink.add(type_slot, n, k, 2.0 * Fp * scd_r12[2] * tmpy);
      tmpz +=  C3B[3] * dsnlm_dc[dsnlm_i+3] * fn * dblm_z[3] +
                    2.0 * C3B[4] * dsnlm_dc[dsnlm_i+4] * fn * dblm_z[4] +
                    2.0 * C3B[5] * dsnlm_dc[dsnlm_i+5] * fn * dblm_z[5];
      tmpz +=  s[0] * fn12[k] * rij_Lsq * dblm_z[3] +
                    2.0 * s[1] * fn12[k] * rij_Lsq * dblm_z[4] +
                    2.0 * s[2] * fn12[k] * rij_Lsq * dblm_z[5];
      sink.add(type_slot, n, k, 2.0 * Fp * scd_r12[3] * tmpz);
    }

  }
  if constexpr (L == 3) {
    if (type_slot < 0) return;

    int dsnlm_idx = 0 + type_slot * NBASIS * NUM_OF_ABC;
    for(int k=0; k < NBASIS; k++) {
      int dsnlm_i = dsnlm_idx + k * NUM_OF_ABC;
      double tmpr = 0.0, tmpx = 0.0, tmpy = 0.0, tmpz = 0.0;
      tmpr +=         C3B[8]  * dsnlm_dc[dsnlm_i+8] * (fnp * blm[8]  + fn * dblm_r[8]) +
                2.0 * C3B[9]  * dsnlm_dc[dsnlm_i+9] * (fnp * blm[9]  + fn * dblm_r[9]) +
                2.0 * C3B[10] * dsnlm_dc[dsnlm_i+10] * (fnp * blm[10] + fn * dblm_r[10])+
                2.0 * C3B[11] * dsnlm_dc[dsnlm_i+11] *  fnp * blm[11] +
                2.0 * C3B[12] * dsnlm_dc[dsnlm_i+12] *  fnp * blm[12] +
                2.0 * C3B[13] * dsnlm_dc[dsnlm_i+13] *  fnp * blm[13] +
                2.0 * C3B[14] * dsnlm_dc[dsnlm_i+14] *  fnp * blm[14];
      tmpr +=       s[0] * ((fnp12[k] * rij_Lsq - 3.0 * fn12[k] * rij_L2sq) * blm[8]  + fn12[k] * rij_Lsq * dblm_r[8]) +
              2.0 * s[1] * ((fnp12[k] * rij_Lsq - 3.0 * fn12[k] * rij_L2sq) * blm[9]  + fn12[k] * rij_Lsq * dblm_r[9]) +
              2.0 * s[2] * ((fnp12[k] * rij_Lsq - 3.0 * fn12[k] * rij_L2sq) * blm[10] + fn12[k] * rij_Lsq * dblm_r[10])+
              2.0 * s[3] *  (fnp12[k] * rij_Lsq - 3.0 * fn12[k] * rij_L2sq) * blm[11] +
              2.0 * s[4] *  (fnp12[k] * rij_Lsq - 3.0 * fn12[k] * rij_L2sq) * blm[12] +
              2.0 * s[5] *  (fnp12[k] * rij_Lsq - 3.0 * fn12[k] * rij_L2sq) * blm[13] +
              2.0 * s[6] *  (fnp12[k] * rij_Lsq - 3.0 * fn12[k] * rij_L2sq) * blm[14];
      sink.add(type_slot, n, k, 2.0 * Fp * scd_r12[0] * tmpr);
      tmpx +=
                2.0 * C3B[9]  * dsnlm_dc[dsnlm_i+9]  * fn * dblm_x[9] +
                2.0 * C3B[11] * dsnlm_dc[dsnlm_i+11] * fn * dblm_x[11] +
                2.0 * C3B[12] * dsnlm_dc[dsnlm_i+12] * fn * dblm_x[12] +
                2.0 * C3B[13] * dsnlm_dc[dsnlm_i+13] * fn * dblm_x[13] +
                2.0 * C3B[14] * dsnlm_dc[dsnlm_i+14] * fn * dblm_x[14];
      tmpx +=
                2.0 * s[1] * fn12[k] * rij_Lsq * dblm_x[9]  +
                2.0 * s[3] * fn12[k] * rij_Lsq * dblm_x[11] +
                2.0 * s[4] * fn12[k] * rij_Lsq * dblm_x[12] +
                2.0 * s[5] * fn12[k] * rij_Lsq * dblm_x[13] +
                2.0 * s[6] * fn12[k] * rij_Lsq * dblm_x[14];
      sink.add(type_slot, n, k, 2.0 * Fp * scd_r12[1] * tmpx);
      tmpy +=
                2.0 * C3B[10] * dsnlm_dc[dsnlm_i+10] * fn * dblm_y[10] +
                2.0 * C3B[11] * dsnlm_dc[dsnlm_i+11] * fn * dblm_y[11] +
                2.0 * C3B[12] * dsnlm_dc[dsnlm_i+12] * fn * dblm_y[12] +
                2.0 * C3B[13] * dsnlm_dc[dsnlm_i+13] * fn * dblm_y[13] +
                2.0 * C3B[14] * dsnlm_dc[dsnlm_i+14] * fn * dblm_y[14];
      tmpy +=
                2.0 * s[2] * fn12[k] * rij_Lsq * dblm_y[10] +
                2.0 * s[3] * fn12[k] * rij_Lsq * dblm_y[11] +
                2.0 * s[4] * fn12[k] * rij_Lsq * dblm_y[12] +
                2.0 * s[5] * fn12[k] * rij_Lsq * dblm_y[13] +
                2.0 * s[6] * fn12[k] * rij_Lsq * dblm_y[14];
      sink.add(type_slot, n, k, 2.0 * Fp * scd_r12[2] * tmpy);
      tmpz +=         C3B[8]  * dsnlm_dc[dsnlm_i+8]  * fn * dblm_z[8] +
                2.0 * C3B[9]  * dsnlm_dc[dsnlm_i+9]  * fn * dblm_z[9] +
                2.0 * C3B[10] * dsnlm_dc[dsnlm_i+10] * fn * dblm_z[10] +
                2.0 * C3B[11] * dsnlm_dc[dsnlm_i+11] * fn * dblm_z[11] +
                2.0 * C3B[12] * dsnlm_dc[dsnlm_i+12] * fn * dblm_z[12];
      tmpz +=         s[0] * fn12[k] * rij_Lsq * dblm_z[8] +
                2.0 * s[1] * fn12[k] * rij_Lsq * dblm_z[9] +
                2.0 * s[2] * fn12[k] * rij_Lsq * dblm_z[10]+
                2.0 * s[3] * fn12[k] * rij_Lsq * dblm_z[11]+
                2.0 * s[4] * fn12[k] * rij_Lsq * dblm_z[12];
      sink.add(type_slot, n, k, 2.0 * Fp * scd_r12[3] * tmpz);
    }

  }
  if constexpr (L == 4) {
    if (type_slot < 0) return;

    int dsnlm_idx = 0 + type_slot * NBASIS * NUM_OF_ABC;
    for(int k=0; k < NBASIS; k++) {
      int dsnlm_i = dsnlm_idx + k * NUM_OF_ABC;
      double tmpr = 0.0, tmpx = 0.0, tmpy = 0.0, tmpz = 0.0;
      tmpr +=       C3B[15] * dsnlm_dc[dsnlm_i+15] * (fnp * blm[15] + fn * dblm_r[15]) +
              2.0 * C3B[16] * dsnlm_dc[dsnlm_i+16] * (fnp * blm[16] + fn * dblm_r[16]) +
              2.0 * C3B[17] * dsnlm_dc[dsnlm_i+17] * (fnp * blm[17] + fn * dblm_r[17]) +
              2.0 * C3B[18] * dsnlm_dc[dsnlm_i+18] * (fnp * blm[18] + fn * dblm_r[18]) +
              2.0 * C3B[19] * dsnlm_dc[dsnlm_i+19] * (fnp * blm[19] + fn * dblm_r[19]) +
              2.0 * C3B[20] * dsnlm_dc[dsnlm_i+20] *  fnp * blm[20] +
              2.0 * C3B[21] * dsnlm_dc[dsnlm_i+21] *  fnp * blm[21] +
              2.0 * C3B[22] * dsnlm_dc[dsnlm_i+22] *  fnp * blm[22] +
              2.0 * C3B[23] * dsnlm_dc[dsnlm_i+23] *  fnp * blm[23];
      tmpr +=       s[0] * ((fnp12[k] * rij_Lsq - 4.0 * fn12[k] * rij_L2sq) * blm[15] + fn12[k]* rij_Lsq * dblm_r[15]) +
              2.0 * s[1] * ((fnp12[k] * rij_Lsq - 4.0 * fn12[k] * rij_L2sq) * blm[16] + fn12[k]* rij_Lsq * dblm_r[16]) +
              2.0 * s[2] * ((fnp12[k] * rij_Lsq - 4.0 * fn12[k] * rij_L2sq) * blm[17] + fn12[k]* rij_Lsq * dblm_r[17]) +
              2.0 * s[3] * ((fnp12[k] * rij_Lsq - 4.0 * fn12[k] * rij_L2sq) * blm[18] + fn12[k]* rij_Lsq * dblm_r[18]) +
              2.0 * s[4] * ((fnp12[k] * rij_Lsq - 4.0 * fn12[k] * rij_L2sq) * blm[19] + fn12[k]* rij_Lsq * dblm_r[19]) +
              2.0 * s[5] * ((fnp12[k] * rij_Lsq - 4.0 * fn12[k] * rij_L2sq) * blm[20]) +
              2.0 * s[6] * ((fnp12[k] * rij_Lsq - 4.0 * fn12[k] * rij_L2sq) * blm[21]) +
              2.0 * s[7] * ((fnp12[k] * rij_Lsq - 4.0 * fn12[k] * rij_L2sq) * blm[22]) +
              2.0 * s[8] * ((fnp12[k] * rij_Lsq - 4.0 * fn12[k] * rij_L2sq) * blm[23]);
      sink.add(type_slot, n, k, 2.0 * Fp * scd_r12[0] * tmpr);
      tmpx +=       C3B[15] * dsnlm_dc[dsnlm_i+15] * fn * dblm_x[15] +
              2.0 * C3B[16] * dsnlm_dc[dsnlm_i+16] * fn * dblm_x[16] +
              2.0 * C3B[17] * dsnlm_dc[dsnlm_i+17] * fn * dblm_x[17] +
              2.0 * C3B[18] * dsnlm_dc[dsnlm_i+18] * fn * dblm_x[18] +
              2.0 * C3B[19] * dsnlm_dc[dsnlm_i+19] * fn * dblm_x[19] +
              2.0 * C3B[20] * dsnlm_dc[dsnlm_i+20] * fn * dblm_x[20] +
              2.0 * C3B[21] * dsnlm_dc[dsnlm_i+21] * fn * dblm_x[21] +
              2.0 * C3B[22] * dsnlm_dc[dsnlm_i+22] * fn * dblm_x[22] +
              2.0 * C3B[23] * dsnlm_dc[dsnlm_i+23] * fn * dblm_x[23];
      tmpx +=       s[0] * fn12[k] * rij_Lsq * dblm_x[15] +
              2.0 * s[1] * fn12[k] * rij_Lsq * dblm_x[16] +
              2.0 * s[2] * fn12[k] * rij_Lsq * dblm_x[17] +
              2.0 * s[3] * fn12[k] * rij_Lsq * dblm_x[18] +
              2.0 * s[4] * fn12[k] * rij_Lsq * dblm_x[19] +
              2.0 * s[5] * fn12[k] * rij_Lsq * dblm_x[20] +
              2.0 * s[6] * fn12[k] * rij_Lsq * dblm_x[21] +
              2.0 * s[7] * fn12[k] * rij_Lsq * dblm_x[22] +
              2.0 * s[8] * fn12[k] * rij_Lsq * dblm_x[23];
      sink.add(type_slot, n, k, 2.0 * Fp * scd_r12[1] * tmpx);
      tmpy +=       C3B[15] * dsnlm_dc[dsnlm_i+15] * fn * dblm_y[15] +
              2.0 * C3B[16] * dsnlm_dc[dsnlm_i+16] * fn * dblm_y[16] +
              2.0 * C3B[17] * dsnlm_dc[dsnlm_i+17] * fn * dblm_y[17] +
              2.0 * C3B[18] * dsnlm_dc[dsnlm_i+18] * fn * dblm_y[18] +
              2.0 * C3B[19] * dsnlm_dc[dsnlm_i+19] * fn * dblm_y[19] +
              2.0 * C3B[20] * dsnlm_dc[dsnlm_i+20] * fn * dblm_y[20] +
              2.0 * C3B[21] * dsnlm_dc[dsnlm_i+21] * fn * dblm_y[21] +
              2.0 * C3B[22] * dsnlm_dc[dsnlm_i+22] * fn * dblm_y[22] +
              2.0 * C3B[23] * dsnlm_dc[dsnlm_i+23] * fn * dblm_y[23];
      tmpy +=       s[0] * fn12[k] * rij_Lsq * dblm_y[15] +
              2.0 * s[1] * fn12[k] * rij_Lsq * dblm_y[16] +
              2.0 * s[2] * fn12[k] * rij_Lsq * dblm_y[17] +
              2.0 * s[3] * fn12[k] * rij_Lsq * dblm_y[18] +
              2.0 * s[4] * fn12[k] * rij_Lsq * dblm_y[19] +
              2.0 * s[5] * fn12[k] * rij_Lsq * dblm_y[20] +
              2.0 * s[6] * fn12[k] * rij_Lsq * dblm_y[21] +
              2.0 * s[7] * fn12[k] * rij_Lsq * dblm_y[22] +
              2.0 * s[8] * fn12[k] * rij_Lsq * dblm_y[23];
      sink.add(type_slot, n, k, 2.0 * Fp * scd_r12[2] * tmpy);
      tmpz +=       C3B[15] * dsnlm_dc[dsnlm_i+15] * fn * dblm_z[15] +
              2.0 * C3B[16] * dsnlm_dc[dsnlm_i+16] * fn * dblm_z[16] +
              2.0 * C3B[17] * dsnlm_dc[dsnlm_i+17] * fn * dblm_z[17] +
              2.0 * C3B[18] * dsnlm_dc[dsnlm_i+18] * fn * dblm_z[18] +
              2.0 * C3B[19] * dsnlm_dc[dsnlm_i+19] * fn * dblm_z[19] +
              2.0 * C3B[20] * dsnlm_dc[dsnlm_i+20] * fn * dblm_z[20] +
              2.0 * C3B[21] * dsnlm_dc[dsnlm_i+21] * fn * dblm_z[21] +
              2.0 * C3B[22] * dsnlm_dc[dsnlm_i+22] * fn * dblm_z[22] +
              2.0 * C3B[23] * dsnlm_dc[dsnlm_i+23] * fn * dblm_z[23];
      tmpz +=       s[0] * fn12[k] * rij_Lsq * dblm_z[15] +
              2.0 * s[1] * fn12[k] * rij_Lsq * dblm_z[16] +
              2.0 * s[2] * fn12[k] * rij_Lsq * dblm_z[17] +
              2.0 * s[3] * fn12[k] * rij_Lsq * dblm_z[18] +
              2.0 * s[4] * fn12[k] * rij_Lsq * dblm_z[19] +
              2.0 * s[5] * fn12[k] * rij_Lsq * dblm_z[20] +
              2.0 * s[6] * fn12[k] * rij_Lsq * dblm_z[21] +
              2.0 * s[7] * fn12[k] * rij_Lsq * dblm_z[22] +
              2.0 * s[8] * fn12[k] * rij_Lsq * dblm_z[23];
      sink.add(type_slot, n, k, 2.0 * Fp * scd_r12[3] * tmpz);
    }

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
  if constexpr (L == 1) {
    for (int uj =0; uj < tile_count; uj++) {
      int j = uj;
      if (type_slot == j) continue;
      int dsnlm_idx = 0 + j * NBASIS * NUM_OF_ABC;

      for(int k=0; k < NBASIS; k++) {
        int dsnlm_i = dsnlm_idx + k * NUM_OF_ABC;
        double tmpr = 0.0, tmpx = 0.0, tmpy = 0.0, tmpz = 0.0;

        tmpr +=       C3B[0] * dsnlm_dc[dsnlm_i]   * fnp * blm[0];
        tmpr += 2.0 * C3B[1] * dsnlm_dc[dsnlm_i+1] * fnp * blm[1];
        tmpr += 2.0 * C3B[2] * dsnlm_dc[dsnlm_i+2] * fnp * blm[2];
        sink.add(uj, n, k, 2.0 * Fp * scd_r12[0] * tmpr);
        tmpx += 2.0 * C3B[1] * dsnlm_dc[dsnlm_i+1] * fn;
        sink.add(uj, n, k, 2.0 * Fp * scd_r12[1] * tmpx);
        tmpy += 2.0 * C3B[2] * dsnlm_dc[dsnlm_i+2] * fn;
        sink.add(uj, n, k, 2.0 * Fp * scd_r12[2] * tmpy);
        tmpz += C3B[0] * dsnlm_dc[dsnlm_i] * fn;
        sink.add(uj, n, k, 2.0 * Fp * scd_r12[3] * tmpz);
      }
    }

  }
  if constexpr (L == 2) {
    for (int uj =0; uj < tile_count; uj++) {
      int j = uj;
      if (type_slot == j) continue;
      int dsnlm_idx = 0 + j * NBASIS * NUM_OF_ABC;


      for(int k=0; k < NBASIS; k++) {
        int dsnlm_i = dsnlm_idx + k * NUM_OF_ABC;
        double tmpr = 0.0, tmpx = 0.0, tmpy = 0.0, tmpz = 0.0;

        tmpr +=  C3B[3] * dsnlm_dc[dsnlm_i+3] * (fnp * blm[3] + fn * dblm_r[3]) +
                      2.0 * C3B[4] * dsnlm_dc[dsnlm_i+4] * fnp * blm[4] +
                      2.0 * C3B[5] * dsnlm_dc[dsnlm_i+5] * fnp * blm[5] +
                      2.0 * C3B[6] * dsnlm_dc[dsnlm_i+6] * fnp * blm[6] +
                      2.0 * C3B[7] * dsnlm_dc[dsnlm_i+7] * fnp * blm[7];
        sink.add(uj, n, k, 2.0 * Fp * scd_r12[0] * tmpr);
        tmpx +=
                      2.0 * C3B[4] * dsnlm_dc[dsnlm_i+4] * fn * dblm_x[4] +
                      2.0 * C3B[6] * dsnlm_dc[dsnlm_i+6] * fn * dblm_x[6] +
                      2.0 * C3B[7] * dsnlm_dc[dsnlm_i+7] * fn * dblm_x[7];
        sink.add(uj, n, k, 2.0 * Fp * scd_r12[1] * tmpx);
        tmpy +=
                      2.0 * C3B[5] * dsnlm_dc[dsnlm_i+5] * fn * dblm_y[5] +
                      2.0 * C3B[6] * dsnlm_dc[dsnlm_i+6] * fn * dblm_y[6] +
                      2.0 * C3B[7] * dsnlm_dc[dsnlm_i+7] * fn * dblm_y[7];
        sink.add(uj, n, k, 2.0 * Fp * scd_r12[2] * tmpy);
        tmpz +=  C3B[3] * dsnlm_dc[dsnlm_i+3] * fn * dblm_z[3] +
                      2.0 * C3B[4] * dsnlm_dc[dsnlm_i+4] * fn * dblm_z[4] +
                      2.0 * C3B[5] * dsnlm_dc[dsnlm_i+5] * fn * dblm_z[5];
        sink.add(uj, n, k, 2.0 * Fp * scd_r12[3] * tmpz);
      }
    }

  }
  if constexpr (L == 3) {
    for (int uj =0; uj < tile_count; uj++) {
      int j = uj;
      if (type_slot == j) continue;
      int dsnlm_idx = 0 + j * NBASIS * NUM_OF_ABC;

      for(int k=0; k < NBASIS; k++) {
        int dsnlm_i = dsnlm_idx + k * NUM_OF_ABC;
        double tmpr = 0.0, tmpx = 0.0, tmpy = 0.0, tmpz = 0.0;

        tmpr +=         C3B[8]  * dsnlm_dc[dsnlm_i+8] * (fnp * blm[8]  + fn * dblm_r[8]) +
                  2.0 * C3B[9]  * dsnlm_dc[dsnlm_i+9] * (fnp * blm[9]  + fn * dblm_r[9]) +
                  2.0 * C3B[10] * dsnlm_dc[dsnlm_i+10] * (fnp * blm[10] + fn * dblm_r[10])+
                  2.0 * C3B[11] * dsnlm_dc[dsnlm_i+11] *  fnp * blm[11] +
                  2.0 * C3B[12] * dsnlm_dc[dsnlm_i+12] *  fnp * blm[12] +
                  2.0 * C3B[13] * dsnlm_dc[dsnlm_i+13] *  fnp * blm[13] +
                  2.0 * C3B[14] * dsnlm_dc[dsnlm_i+14] *  fnp * blm[14];
        sink.add(uj, n, k, 2.0 * Fp * scd_r12[0] * tmpr);
        tmpx +=
                  2.0 * C3B[9]  * dsnlm_dc[dsnlm_i+9]  * fn * dblm_x[9] +
                  2.0 * C3B[11] * dsnlm_dc[dsnlm_i+11] * fn * dblm_x[11] +
                  2.0 * C3B[12] * dsnlm_dc[dsnlm_i+12] * fn * dblm_x[12] +
                  2.0 * C3B[13] * dsnlm_dc[dsnlm_i+13] * fn * dblm_x[13] +
                  2.0 * C3B[14] * dsnlm_dc[dsnlm_i+14] * fn * dblm_x[14];
        sink.add(uj, n, k, 2.0 * Fp * scd_r12[1] * tmpx);
        tmpy +=
                  2.0 * C3B[10] * dsnlm_dc[dsnlm_i+10] * fn * dblm_y[10] +
                  2.0 * C3B[11] * dsnlm_dc[dsnlm_i+11] * fn * dblm_y[11] +
                  2.0 * C3B[12] * dsnlm_dc[dsnlm_i+12] * fn * dblm_y[12] +
                  2.0 * C3B[13] * dsnlm_dc[dsnlm_i+13] * fn * dblm_y[13] +
                  2.0 * C3B[14] * dsnlm_dc[dsnlm_i+14] * fn * dblm_y[14];
        sink.add(uj, n, k, 2.0 * Fp * scd_r12[2] * tmpy);
        tmpz +=         C3B[8]  * dsnlm_dc[dsnlm_i+8]  * fn * dblm_z[8] +
                  2.0 * C3B[9]  * dsnlm_dc[dsnlm_i+9]  * fn * dblm_z[9] +
                  2.0 * C3B[10] * dsnlm_dc[dsnlm_i+10] * fn * dblm_z[10] +
                  2.0 * C3B[11] * dsnlm_dc[dsnlm_i+11] * fn * dblm_z[11] +
                  2.0 * C3B[12] * dsnlm_dc[dsnlm_i+12] * fn * dblm_z[12];
        sink.add(uj, n, k, 2.0 * Fp * scd_r12[3] * tmpz);
      }
    }

  }
  if constexpr (L == 4) {
    for (int uj =0; uj < tile_count; uj++) {
      int j = uj;
      if (type_slot == j) continue;
      int dsnlm_idx = 0 + j * NBASIS * NUM_OF_ABC;

      for(int k=0; k < NBASIS; k++) {
        int dsnlm_i = dsnlm_idx + k * NUM_OF_ABC;
        double tmpr = 0.0, tmpx = 0.0, tmpy = 0.0, tmpz = 0.0;

        tmpr +=       C3B[15] * dsnlm_dc[dsnlm_i+15] * (fnp * blm[15] + fn * dblm_r[15]) +
                2.0 * C3B[16] * dsnlm_dc[dsnlm_i+16] * (fnp * blm[16] + fn * dblm_r[16]) +
                2.0 * C3B[17] * dsnlm_dc[dsnlm_i+17] * (fnp * blm[17] + fn * dblm_r[17]) +
                2.0 * C3B[18] * dsnlm_dc[dsnlm_i+18] * (fnp * blm[18] + fn * dblm_r[18]) +
                2.0 * C3B[19] * dsnlm_dc[dsnlm_i+19] * (fnp * blm[19] + fn * dblm_r[19]) +
                2.0 * C3B[20] * dsnlm_dc[dsnlm_i+20] *  fnp * blm[20] +
                2.0 * C3B[21] * dsnlm_dc[dsnlm_i+21] *  fnp * blm[21] +
                2.0 * C3B[22] * dsnlm_dc[dsnlm_i+22] *  fnp * blm[22] +
                2.0 * C3B[23] * dsnlm_dc[dsnlm_i+23] *  fnp * blm[23];
        sink.add(uj, n, k, 2.0 * Fp * scd_r12[0] * tmpr);
        tmpx +=       C3B[15] * dsnlm_dc[dsnlm_i+15] * fn * dblm_x[15] +
                2.0 * C3B[16] * dsnlm_dc[dsnlm_i+16] * fn * dblm_x[16] +
                2.0 * C3B[17] * dsnlm_dc[dsnlm_i+17] * fn * dblm_x[17] +
                2.0 * C3B[18] * dsnlm_dc[dsnlm_i+18] * fn * dblm_x[18] +
                2.0 * C3B[19] * dsnlm_dc[dsnlm_i+19] * fn * dblm_x[19] +
                2.0 * C3B[20] * dsnlm_dc[dsnlm_i+20] * fn * dblm_x[20] +
                2.0 * C3B[21] * dsnlm_dc[dsnlm_i+21] * fn * dblm_x[21] +
                2.0 * C3B[22] * dsnlm_dc[dsnlm_i+22] * fn * dblm_x[22] +
                2.0 * C3B[23] * dsnlm_dc[dsnlm_i+23] * fn * dblm_x[23];
        sink.add(uj, n, k, 2.0 * Fp * scd_r12[1] * tmpx);
        tmpy +=       C3B[15] * dsnlm_dc[dsnlm_i+15] * fn * dblm_y[15] +
                2.0 * C3B[16] * dsnlm_dc[dsnlm_i+16] * fn * dblm_y[16] +
                2.0 * C3B[17] * dsnlm_dc[dsnlm_i+17] * fn * dblm_y[17] +
                2.0 * C3B[18] * dsnlm_dc[dsnlm_i+18] * fn * dblm_y[18] +
                2.0 * C3B[19] * dsnlm_dc[dsnlm_i+19] * fn * dblm_y[19] +
                2.0 * C3B[20] * dsnlm_dc[dsnlm_i+20] * fn * dblm_y[20] +
                2.0 * C3B[21] * dsnlm_dc[dsnlm_i+21] * fn * dblm_y[21] +
                2.0 * C3B[22] * dsnlm_dc[dsnlm_i+22] * fn * dblm_y[22] +
                2.0 * C3B[23] * dsnlm_dc[dsnlm_i+23] * fn * dblm_y[23];
        sink.add(uj, n, k, 2.0 * Fp * scd_r12[2] * tmpy);
        tmpz +=       C3B[15] * dsnlm_dc[dsnlm_i+15] * fn * dblm_z[15] +
                2.0 * C3B[16] * dsnlm_dc[dsnlm_i+16] * fn * dblm_z[16] +
                2.0 * C3B[17] * dsnlm_dc[dsnlm_i+17] * fn * dblm_z[17] +
                2.0 * C3B[18] * dsnlm_dc[dsnlm_i+18] * fn * dblm_z[18] +
                2.0 * C3B[19] * dsnlm_dc[dsnlm_i+19] * fn * dblm_z[19] +
                2.0 * C3B[20] * dsnlm_dc[dsnlm_i+20] * fn * dblm_z[20] +
                2.0 * C3B[21] * dsnlm_dc[dsnlm_i+21] * fn * dblm_z[21] +
                2.0 * C3B[22] * dsnlm_dc[dsnlm_i+22] * fn * dblm_z[22] +
                2.0 * C3B[23] * dsnlm_dc[dsnlm_i+23] * fn * dblm_z[23];
        sink.add(uj, n, k, 2.0 * Fp * scd_r12[3] * tmpz);
      }
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
  if constexpr (CROSS) {
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

    for (int uj =0; uj < tile_count; uj++) {
      int j = uj;
      if (type_slot == j) continue;
      int dsnlm_idx = 0 + j * NBASIS * NUM_OF_ABC;

      for(int k=0; k < NBASIS; k++) {
        int dsnlm_i = dsnlm_idx + k * NUM_OF_ABC;
        double tmpr = 0.0, tmpx = 0.0, tmpy = 0.0, tmpz = 0.0;

        dnlm_dc[0] = dsnlm_dc[dsnlm_i + 3];
        dnlm_dc[1] = dsnlm_dc[dsnlm_i + 4];
        dnlm_dc[2] = dsnlm_dc[dsnlm_i + 5];
        dnlm_dc[3] = dsnlm_dc[dsnlm_i + 6];
        dnlm_dc[4] = dsnlm_dc[dsnlm_i + 7];
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
    }

  } else {
    if (type_slot < 0) return;

    int dsnlm_idx = 0 + type_slot * NBASIS * NUM_OF_ABC;
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
    for(int k=0; k < NBASIS; k++) {
      int dsnlm_i = dsnlm_idx + k * NUM_OF_ABC;
      double tmpr = 0.0, tmpx = 0.0, tmpy = 0.0, tmpz = 0.0;
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
__device__ __forceinline__ void accumulate_five_body(
  const double* fn12, const double* fnp12,
  const double* blm, const double* rij_blm,
  const double* dblm_x, const double* dblm_y, const double* dblm_z, const double* dblm_r,
  const double* scd_r12, const double* dsnlm_dc, const double* s, const double* r12,
  double d12inv, double rij_Lsq, double rij_L2sq, double fn, double fnp, double Fp,
  int type_slot, int tile_count, int n, SharedTileSink<NMAX, NBASIS, TYPE_TILE> sink)
{
  if constexpr (CROSS) {
    double dnlm_drij[3] = {0.0};
    dnlm_drij[0] = fnp * blm[0];
    dnlm_drij[1] = fnp * blm[1];
    dnlm_drij[2] = fnp * blm[2];
    double dnlm_dxij[3] = {0.0};
    dnlm_dxij[0] = 0.0;
    dnlm_dxij[1] = fn;
    dnlm_dxij[2] = 0.0;
    double dnlm_dyij[3] = {0.0};
    dnlm_dyij[0] = 0.0;
    dnlm_dyij[1] = 0.0;
    dnlm_dyij[2] = fn;
    double dnlm_dzij[3] = {0.0};
    dnlm_dzij[0] = fn;
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

    for (int uj =0; uj < tile_count; uj++) {
      int j = uj;
      if (type_slot == j) continue;
      int dsnlm_idx = 0 + j * NBASIS * NUM_OF_ABC;

      for(int k=0; k < NBASIS; k++) {
        int dsnlm_i = dsnlm_idx + k * NUM_OF_ABC;
        double tmpr = 0.0, tmpx = 0.0, tmpy = 0.0, tmpz = 0.0;

        dnlm_dc[0] = dsnlm_dc[dsnlm_i + 0];
        dnlm_dc[1] = dsnlm_dc[dsnlm_i + 1];
        dnlm_dc[2] = dsnlm_dc[dsnlm_i + 2];
        tmpr += 4.0 * C5B[0] * (3.0 * s2[0] * dnlm_dc[0] * dnlm_drij[0]);
        ds1s2 = s[1] * dnlm_drij[1] + s[2] * dnlm_drij[2];
        ds1s2_c = 2.0 * s[1] * dnlm_dc[1] + 2.0 * s[2] * dnlm_dc[2];
        d_tmp = dnlm_dc[1] * dnlm_drij[1] + dnlm_dc[2] * dnlm_drij[2];
        tmpr += 2.0 * C5B[1] * (
          dnlm_dc[0] * dnlm_drij[0] * (s2[1] + s2[2]) +
          s[0] * dnlm_drij[0] * ds1s2_c + 2.0 * s[0] * dnlm_dc[0] * ds1s2 + s2[0] * d_tmp);
        tmpr += 4.0 * C5B[2] * (ds1s2_c * ds1s2 + (s2[1] + s2[2]) * d_tmp);
        sink.add(uj, n, k, Fp * scd_r12[0] * tmpr);
        tmpx += 4.0 * C5B[0] * (3.0 * s2[0] * dnlm_dc[0] * dnlm_dxij[0]);
        ds1s2 = s[1] * dnlm_dxij[1] + s[2] * dnlm_dxij[2];
        ds1s2_c = 2.0 * s[1] * dnlm_dc[1] + 2.0 * s[2] * dnlm_dc[2];
        d_tmp = dnlm_dc[1] * dnlm_dxij[1] + dnlm_dc[2] * dnlm_dxij[2];
        tmpx += 2.0 * C5B[1] * (
          dnlm_dc[0] * dnlm_dxij[0] * (s2[1] + s2[2]) +
          s[0] * dnlm_dxij[0] * ds1s2_c + 2.0 * s[0] * dnlm_dc[0] * ds1s2 + s2[0] * d_tmp);
        tmpx += 4.0 * C5B[2] * (ds1s2_c * ds1s2 + (s2[1] + s2[2]) * d_tmp);
        sink.add(uj, n, k, Fp * scd_r12[1] * tmpx);
        tmpy += 4.0 * C5B[0] * (3.0 * s2[0] * dnlm_dc[0] * dnlm_dyij[0]);
        ds1s2 = s[1] * dnlm_dyij[1] + s[2] * dnlm_dyij[2];
        ds1s2_c = 2.0 * s[1] * dnlm_dc[1] + 2.0 * s[2] * dnlm_dc[2];
        d_tmp = dnlm_dc[1] * dnlm_dyij[1] + dnlm_dc[2] * dnlm_dyij[2];
        tmpy += 2.0 * C5B[1] * (
          dnlm_dc[0] * dnlm_dyij[0] * (s2[1] + s2[2]) +
          s[0] * dnlm_dyij[0] * ds1s2_c + 2.0 * s[0] * dnlm_dc[0] * ds1s2 + s2[0] * d_tmp);
        tmpy += 4.0 * C5B[2] * (ds1s2_c * ds1s2 + (s2[1] + s2[2]) * d_tmp);
        sink.add(uj, n, k, Fp * scd_r12[2] * tmpy);
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
    }

  } else {
    if (type_slot < 0) return;

    int dsnlm_idx = 0 + type_slot * NBASIS * NUM_OF_ABC;
    double dnlm_drij[3] = {0.0};
    dnlm_drij[0] = fnp * blm[0];
    dnlm_drij[1] = fnp * blm[1];
    dnlm_drij[2] = fnp * blm[2];
    double dnlm_dxij[3] = {0.0};
    dnlm_dxij[0] = 0.0;
    dnlm_dxij[1] = fn;
    dnlm_dxij[2] = 0.0;
    double dnlm_dyij[3] = {0.0};
    dnlm_dyij[0] = 0.0;
    dnlm_dyij[1] = 0.0;
    dnlm_dyij[2] = fn;
    double dnlm_dzij[3] = {0.0};
    dnlm_dzij[0] = fn;
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
    for(int k=0; k < NBASIS; k++) {
      int dsnlm_i = dsnlm_idx + k * NUM_OF_ABC;
      double tmpr = 0.0, tmpx = 0.0, tmpy = 0.0, tmpz = 0.0;
      dnlm_dc[0] = dsnlm_dc[dsnlm_i + 0];
      dnlm_dc[1] = dsnlm_dc[dsnlm_i + 1];
      dnlm_dc[2] = dsnlm_dc[dsnlm_i + 2];
      dnlm_drij_dc[0] = (fnp12[k] * rij_Lsq - fn12[k] * rij_L2sq) * blm[0];
      dnlm_drij_dc[1] = (fnp12[k] * rij_Lsq - fn12[k] * rij_L2sq) * blm[1];
      dnlm_drij_dc[2] = (fnp12[k] * rij_Lsq - fn12[k] * rij_L2sq) * blm[2];
      dnlm_dxij_dc[1] = fn12[k] * rij_Lsq;
      dnlm_dyij_dc[2] = fn12[k] * rij_Lsq;
      dnlm_dzij_dc[0] = fn12[k] * rij_Lsq;
      tmpr += 4.0 * C5B[0] * (3.0 * s2[0] * dnlm_dc[0] * dnlm_drij[0] + s2[0] * s[0] * dnlm_drij_dc[0]);
      ds1s2 = s[1] * dnlm_drij[1] + s[2] * dnlm_drij[2];
      ds1s2_c = 2.0 * s[1] * dnlm_dc[1] + 2.0 * s[2] * dnlm_dc[2];
      d_tmp = dnlm_dc[1] * dnlm_drij[1] + s[1] * dnlm_drij_dc[1] + dnlm_dc[2] * dnlm_drij[2] + s[2] * dnlm_drij_dc[2];
      tmpr += 2.0 * C5B[1] * (
        dnlm_dc[0] * dnlm_drij[0] * (s2[1] + s2[2]) + s[0] * dnlm_drij_dc[0] * (s2[1] + s2[2]) +
        s[0] * dnlm_drij[0] * ds1s2_c + 2.0 * s[0] * dnlm_dc[0] * ds1s2 + s2[0] * d_tmp);
      tmpr += 4.0 * C5B[2] * (ds1s2_c * ds1s2 + (s2[1] + s2[2]) * d_tmp);
      sink.add(type_slot, n, k, Fp * scd_r12[0] * tmpr);
      tmpx += 4.0 * C5B[0] * (3.0 * s2[0] * dnlm_dc[0] * dnlm_dxij[0] + s2[0] * s[0] * dnlm_dxij_dc[0]);
      ds1s2 = s[1] * dnlm_dxij[1] + s[2] * dnlm_dxij[2];
      ds1s2_c = 2.0 * s[1] * dnlm_dc[1] + 2.0 * s[2] * dnlm_dc[2];
      d_tmp = dnlm_dc[1] * dnlm_dxij[1] + s[1] * dnlm_dxij_dc[1] + dnlm_dc[2] * dnlm_dxij[2] + s[2] * dnlm_dxij_dc[2];
      tmpx += 2.0 * C5B[1] * (
        dnlm_dc[0] * dnlm_dxij[0] * (s2[1] + s2[2]) + s[0] * dnlm_dxij_dc[0] * (s2[1] + s2[2]) +
        s[0] * dnlm_dxij[0] * ds1s2_c + 2.0 * s[0] * dnlm_dc[0] * ds1s2 + s2[0] * d_tmp);
      tmpx += 4.0 * C5B[2] * (ds1s2_c * ds1s2 + (s2[1] + s2[2]) * d_tmp);
      sink.add(type_slot, n, k, Fp * scd_r12[1] * tmpx);
      tmpy += 4.0 * C5B[0] * (3.0 * s2[0] * dnlm_dc[0] * dnlm_dyij[0] + s2[0] * s[0] * dnlm_dyij_dc[0]);
      ds1s2 = s[1] * dnlm_dyij[1] + s[2] * dnlm_dyij[2];
      ds1s2_c = 2.0 * s[1] * dnlm_dc[1] + 2.0 * s[2] * dnlm_dc[2];
      d_tmp = dnlm_dc[1] * dnlm_dyij[1] + s[1] * dnlm_dyij_dc[1] + dnlm_dc[2] * dnlm_dyij[2] + s[2] * dnlm_dyij_dc[2];
      tmpy += 2.0 * C5B[1] * (
        dnlm_dc[0] * dnlm_dyij[0] * (s2[1] + s2[2]) + s[0] * dnlm_dyij_dc[0] * (s2[1] + s2[2]) +
        s[0] * dnlm_dyij[0] * ds1s2_c + 2.0 * s[0] * dnlm_dc[0] * ds1s2 + s2[0] * d_tmp);
      tmpy += 4.0 * C5B[2] * (ds1s2_c * ds1s2 + (s2[1] + s2[2]) * d_tmp);
      sink.add(type_slot, n, k, Fp * scd_r12[2] * tmpy);
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

template<int NMAX, int NBASIS, int LMAX3, bool HAS4, bool HAS5, int TYPE_TILE>
__device__ __forceinline__ void accumulate_neighbor(
  int n, double d12, const double* r12, double fn, double fnp,
  const double* Fp, const double* dsnlm_dc, const double* sum_fxyz,
  const double* blm, const double* rij_blm,
  const double* dblm_x, const double* dblm_y, const double* dblm_z, const double* dblm_r,
  const double* scd_r12, const double* fn12, const double* fnp12,
  int type_slot, int tile_count, SharedTileSink<NMAX, NBASIS, TYPE_TILE> sink)
{
  static_assert(LMAX3 == 4, "This numerical port supports the OMat24 angular orders");

  const double d12inv = 1.0 / d12;
  double rij_Lsq = d12inv;
  double rij_L2sq= d12inv  * d12inv;
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  double s1[3] = {
    sum_fxyz[n * NUM_OF_ABC + 0], sum_fxyz[n * NUM_OF_ABC + 1], sum_fxyz[n * NUM_OF_ABC + 2]};
  if constexpr (HAS5) { accumulate_five_body<false, NMAX, NBASIS, TYPE_TILE>(fn12, fnp12,
              blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
              scd_r12, dsnlm_dc, s1, r12,
              d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[NMAX * LMAX3 + NMAX + n], type_slot, tile_count, n, sink); }
  if constexpr (HAS5) { accumulate_five_body<true, NMAX, NBASIS, TYPE_TILE>(fn12, fnp12,
              blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
              scd_r12, dsnlm_dc, s1, r12,
              d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[NMAX * LMAX3 + NMAX + n], type_slot, tile_count, n, sink); }
  s1[0] *= C3B[0];
  s1[1] *= C3B[1];
  s1[2] *= C3B[2];
  accumulate_direct<1, NMAX, NBASIS, TYPE_TILE>(fn12, fnp12,
              blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
              scd_r12, dsnlm_dc, s1, r12,
              d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n*LMAX3], type_slot, tile_count, n, sink);
  accumulate_cross<1, NMAX, NBASIS, TYPE_TILE>(fn12, fnp12,
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s1, r12,
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n*LMAX3], type_slot, tile_count, n, sink);
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
  if constexpr (HAS4) { accumulate_four_body<false, NMAX, NBASIS, TYPE_TILE>(fn12, fnp12,
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s2, r12,
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[NMAX * LMAX3 + n], type_slot, tile_count, n, sink); }
  if constexpr (HAS4) { accumulate_four_body<true, NMAX, NBASIS, TYPE_TILE>(fn12, fnp12,
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s2, r12,
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[NMAX * LMAX3 + n], type_slot, tile_count, n, sink); }
  s2[0] *= C3B[3];
  s2[1] *= C3B[4];
  s2[2] *= C3B[5];
  s2[3] *= C3B[6];
  s2[4] *= C3B[7];
  accumulate_direct<2, NMAX, NBASIS, TYPE_TILE>(fn12, fnp12,
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s2, r12,
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n*LMAX3+1], type_slot, tile_count, n, sink);
  accumulate_cross<2, NMAX, NBASIS, TYPE_TILE>(fn12, fnp12,
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s2, r12,
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n*LMAX3+1], type_slot, tile_count, n, sink);
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
  accumulate_direct<3, NMAX, NBASIS, TYPE_TILE>(fn12, fnp12,
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s3, r12,
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n*LMAX3+2], type_slot, tile_count, n, sink);
  accumulate_cross<3, NMAX, NBASIS, TYPE_TILE>(fn12, fnp12,
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s3, r12,
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n*LMAX3+2], type_slot, tile_count, n, sink);
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
  accumulate_direct<4, NMAX, NBASIS, TYPE_TILE>(fn12, fnp12,
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s4, r12,
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n*LMAX3+3], type_slot, tile_count, n, sink);
  accumulate_cross<4, NMAX, NBASIS, TYPE_TILE>(fn12, fnp12,
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                scd_r12, dsnlm_dc, s4, r12,
                d12inv, rij_Lsq, rij_L2sq, fn, fnp, Fp[n*LMAX3+3], type_slot, tile_count, n, sink);

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
  static_assert(CTA_THREADS == 32 || CTA_THREADS == 64, "Supported CTA sizes");
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
    for (int i = tid; i < tile_count * NBASIS * 24; i += CTA_THREADS) {
      const int type = s.local_types[tile + i / (NBASIS * 24)];
      s.dsnlm_tile[i] = a.dsnlm_dc[(center * a.atom_types + type) * NBASIS * 24
                                  + i % (NBASIS * 24)];
    }
    __syncthreads();

    // Every thread reaches every tile barrier, including padding-only threads.
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
      const double r12[3] = {a.d12[row * 4 + 1], a.d12[row * 4 + 2], a.d12[row * 4 + 3]};
      const double scd_r12[4] = {a.grad_second[row * 4], a.grad_second[row * 4 + 1],
                                a.grad_second[row * 4 + 2], a.grad_second[row * 4 + 3]};
      double fn12[NBASIS], fnp12[NBASIS];
      double fc12, fcp12;
      find_fc_and_fcp(a.rcut, a.rcut_inv, distance, fc12, fcp12);
      find_fn_and_fnp(NBASIS, a.rcut_inv, distance, fc12, fcp12, fn12, fnp12);

      double blm[24] = {0.0}, rij_blm[24] = {0.0};
      double dblm_x[24] = {0.0}, dblm_y[24] = {0.0};
      double dblm_z[24] = {0.0}, dblm_r[24] = {0.0};
      scd_accumulate_blm_rij(distance, r12[0], r12[1], r12[2],
                             blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r);
      const int coeff_start = (center_type * a.atom_types + neighbor_type) * NMAX * NBASIS;
      for (int n = 0; n < NMAX; ++n) {
        double gn12 = 0.0, gnp12 = 0.0;
        for (int k = 0; k < NBASIS; ++k) {
          gn12 += fn12[k] * a.coeff3[coeff_start + n * NBASIS + k];
          gnp12 += fnp12[k] * a.coeff3[coeff_start + n * NBASIS + k];
        }
        accumulate_neighbor<NMAX, NBASIS, LMAX3, HAS4, HAS5, TYPE_TILE>(
            n, distance, r12, gn12, gnp12, s.fp, s.dsnlm_tile, s.sum_fxyz,
            blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
            scd_r12, fn12, fnp12, type_slot, tile_count, sink);
      }
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
