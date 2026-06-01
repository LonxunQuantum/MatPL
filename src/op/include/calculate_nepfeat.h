#pragma once
#include <cstdint>

// ============================================================================
// Phase B: All launchers are now template<typename T> with explicit
// instantiations for float and double in their respective .cu files.
// The wrapper layer (calculate_nepfeat.cpp etc.) uses AT_DISPATCH_FLOATING_TYPES
// to select the correct instantiation at runtime.
// ============================================================================

// --- 2b Feature ---
template <typename T>
void launch_calculate_nepfeat(
    const T * coeff2, const T * d12_radial,
    const int64_t* NL_radial, const int64_t* atom_map,
    const T rcut_radial,
    T * feat_2b, T * dfeat_c2, T * dfeat_2b, T * dfeat_2b_noc,
    const int natoms, const int neigh_num,
    const int n_max, const int n_base, const int num_types, const int device);

template <typename T>
void launch_calculate_nepfeat_grad(
    const T * grad_output, const T * dfeat_c2, const T * dfeat_2b,
    const int64_t* atom_map,
    T * grad_coeff2, T * grad_d12_radial,
    const int natoms, const int neigh_num,
    const int n_max_2b, const int n_base_2b, const int n_types,
    const int multi_feat_num, const int device);

template <typename T>
void launch_calculate_nepfeat_secondgradout(
    const T * grad_second, const T * dfeat_b, T * gradsecond_out,
    const int atom_nums, const int maxneighs, const int n_max, const int device);

template <typename T>
void launch_calculate_nepfeat_secondgradout_c2(
    const T * grad_second, const T * de_feat, const T * dfeat_2b_noc,
    const int64_t* atom_map, const int64_t* NL_radial,
    T * gradsecond_c2,
    const int atom_nums, const int maxneighs,
    const int n_max_2b, const int n_base_2b, const int atom_types,
    const int multi_feat_num, const int device);

// --- Force ---
template <typename T>
void launch_calculate_nepforce(
    const int64_t * nblist, const T * dE, const T * Ri_d,
    const int natoms, const int neigh_num, T * force, const int device);

template <typename T>
void launch_calculate_nepforce_grad(
    const int64_t * nblist, const T * Ri_d, const T * net_grad,
    const int natoms, const int neigh_num, T * grad, const int device);

// --- Virial ---
template <typename T>
void launch_calculate_nepvirial(
    const int64_t * nblist, const T * dE, const T * Rij, const T * Ri_d,
    const int64_t * num_atom, const int batch_num,
    const int natoms, const int neigh_num,
    T * virial, T * atom_virial, const int device);

template <typename T>
void launch_calculate_nepvirial_grad(
    const int64_t * nblist, const T * Rij, const T * Ri_d, const T * net_grad,
    const int natoms, const int neigh_num, T * grad, const int device);

// --- 3b (MbFeat) ---
template <typename T>
void launch_calculate_nepmbfeat(
    const T * coeff3, const T * d12,
    const int64_t * NL, const int64_t * atom_map,
    T * feat_3b, T * dfeat_c3, T * dfeat_3b, T * dfeat_3b_noc, T * sum_fxyz,
    const T rcut, const int natoms, const int neigh_num,
    const int n_max_3b, const int n_base_3b,
    const int lmax_3, const int lmax_4, const int lmax_5,
    const int num_types, const int device);

template <typename T>
void launch_calculate_nepmbfeat_grad(
    const T * grad_output, const T * coeff3, const T * r12,
    const int64_t * NL, const int64_t * atom_map,
    T * sum_fxyz, T * grad_coeff3, T * grad_d12_3b, T * dsnlm_dc, T * dfeat_drij,
    const T rcut_angular, const int atom_nums, const int neigh_num,
    const int feat_2b_num, const int n_max_3b, const int n_base_3b,
    const int lmax_3, const int lmax_4, const int lmax_5,
    const int n_types, const int device_id);

template <typename T>
void launch_calculate_nepmbfeat_secondgradout(
    const T * grad_second, const T * dfeat_b, T * gradsecond_gradout,
    const int atom_nums, const int maxneighs, const int feat_mu_nums, const int device);

template <typename T>
void launch_calculate_nepmbfeat_secondgradout_c3(
    const T * grad_second, const T * d12, const int64_t* NL,
    const T * de_dfeat, const T * dsnlm_dc, const T * sum_fxyz,
    const int64_t* atom_map, const T * coeff3, T * gradsecond_c3,
    const T rcut_angular, const int atom_nums, const int maxneighs,
    const int n_max_3b, const int n_base_3b, const int atom_types,
    const int lmax_3, const int lmax_4, const int lmax_5,
    const int feat_2b_num, const int multi_feat_num, const int device);
