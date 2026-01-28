#include <torch/extension.h>
#include "../include/calculate_nepneighbor.h"

void torch_launch_calculate_maxneigh(
    const torch::Tensor &num_atoms,
    const torch::Tensor &num_atoms_sum,
    const torch::Tensor &box,
    const torch::Tensor &box_orig, 
    const torch::Tensor &num_cell, 
    const torch::Tensor &position,
    const double cutoff_2b,
    const double cutoff_3b,
    const int64_t total_frames,
    const int64_t total_atoms,
    torch::Tensor &NN_radial, 
    torch::Tensor &NN_angular,
    const int64_t atom_type_num,
    const bool with_type,
    const torch::Tensor &atom_type_map

){
    launch_calculate_maxneigh(
        (const int64_t *) num_atoms.data_ptr(),
        (const int64_t *) num_atoms_sum.data_ptr(),
        (const double *) box.data_ptr(),
        (const double *) box_orig.data_ptr(), 
        (const int64_t *) num_cell.data_ptr(), 
        (const double *) position.data_ptr(),
        cutoff_2b,
        cutoff_3b,
        total_frames,
        total_atoms,
        (int64_t *) NN_radial.data_ptr(), 
        (int64_t *) NN_angular.data_ptr(),
        atom_type_num,
        with_type,
        (int64_t *) atom_type_map.data_ptr()
    );
}

void torch_launch_calculate_neighbor(
    const torch::Tensor &num_atoms,
    const torch::Tensor &num_atoms_sum, 
    const torch::Tensor &atom_type_map, 
    const torch::Tensor &atom_types,
    const torch::Tensor &box,
    const torch::Tensor &box_orig,
    const torch::Tensor &num_cell, 
    const torch::Tensor &position, 
    const double cutoff_2b,
    const double cutoff_3b,
    const int64_t max_NN_radial,
    const int64_t max_NN_angular,
    const int64_t total_frames,
    const int64_t total_atoms,
    torch::Tensor &NN_radial, 
    torch::Tensor &NL_radial,
    torch::Tensor &NN_angular, 
    torch::Tensor &NL_angular,
    torch::Tensor &Ri_radial, 
    torch::Tensor &Ri_angular
){
    int64_t shape = Ri_radial.sizes()[2];
    bool with_rij = false;
    if (shape > 3) with_rij = true;
    launch_calculate_neighbor(
        (const int64_t *) num_atoms.data_ptr(),
        (const int64_t *) num_atoms_sum.data_ptr(),
        (const int64_t *) atom_type_map.data_ptr(),
        (const int64_t *) atom_types.data_ptr(), 
        (const double  *) box.data_ptr(),
        (const double  *) box_orig.data_ptr(), 
        (const int64_t *) num_cell.data_ptr(), 
        (const double  *) position.data_ptr(),
        cutoff_2b,
        cutoff_3b,
        max_NN_radial,
        max_NN_angular,
        total_frames,
        total_atoms,
        (int64_t *) NN_radial.data_ptr(), 
        (int64_t *) NL_radial.data_ptr(),
        (int64_t *) NN_angular.data_ptr(), 
        (int64_t *) NL_angular.data_ptr(),
        (double  *) Ri_radial.data_ptr(), 
        (double  *) Ri_angular.data_ptr(),
        with_rij
    );
}

void torch_launch_calculate_descriptor(
    const torch::Tensor &coeff2,
    const torch::Tensor &coeff3,
    const torch::Tensor &d12,
    const torch::Tensor &NL,
    const torch::Tensor &atom_map,
    const double rcut_radial,
    const double rcut_angular,
    torch::Tensor &feats,
    int64_t total_atoms,
    int64_t neigh_num,
    int64_t n_max_2b,
    int64_t n_base_2b,
    int64_t n_max_3b,
    int64_t n_base_3b,
    int64_t lmax_3,
    int64_t lmax_4,
    int64_t lmax_5,
    int64_t n_types
){
    launch_calculate_descriptor(
        coeff2.data_ptr<double>(),
        coeff3.data_ptr<double>(),
        d12.data_ptr<double>(),
        NL.data_ptr<int64_t>(),
        atom_map.data_ptr<int64_t>(),
        rcut_radial,
        rcut_angular,
        feats.data_ptr<double>(),
        total_atoms,
        neigh_num,
        n_max_2b,
        n_base_2b,
        n_max_3b,
        n_base_3b,
        lmax_3,
        lmax_4,
        lmax_5,
        n_types
    );   
}