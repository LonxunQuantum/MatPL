#include <torch/torch.h>
#include <iostream>
#include "../include/CalcOps.h"
#include "../include/cpu_calculate_nepneighbor.h"

torch::autograd::variable_list calculateForce_cpu(
    at::Tensor list_neigh,
    at::Tensor dE,
    at::Tensor Ri_d,
    at::Tensor F,
    at::Tensor nghost_tensor) 
    {
        return {torch::autograd::Variable()};
    }

// the following is the code virial

torch::autograd::variable_list calculateVirial_cpu(
    at::Tensor list_neigh,
    at::Tensor dE,
    at::Tensor Rij,
    at::Tensor Ri_d,
    at::Tensor nghost_tensor) 
    {
        return {torch::autograd::Variable()};
    }
    
// the following is the code compress

torch::autograd::variable_list calculateCompress_cpu(
    at::Tensor f2,
    at::Tensor coefficient) 
    {
        return {torch::autograd::Variable()};
    }


std::vector<torch::Tensor> calculate_maxneigh_cpu(
    const torch::Tensor &num_atoms,
    const torch::Tensor &box,
    const torch::Tensor &box_orig, 
    const torch::Tensor &num_cell, 
    const torch::Tensor &position,
    const double cutoff_2b,
    const double cutoff_3b,
    const int64_t atom_type_num,
    const torch::Tensor &atom_type_map,
    const bool with_type
){
    auto dtype = position.dtype();
    auto index_dtype = num_atoms.dtype();
    auto device = position.device();
    int64_t total_frames = num_atoms.sizes()[0];
    int64_t total_atoms = position.sizes()[0];
    torch::Tensor atom_num_list_sum = torch::cumsum(num_atoms, 0, torch::kInt64);
    if (with_type == false) {
        torch::Tensor NN_radial = torch::empty({total_atoms}, torch::TensorOptions().dtype(index_dtype).device(device));
        torch::Tensor NN_angular = torch::empty({total_atoms}, torch::TensorOptions().dtype(index_dtype).device(device));
        
        launch_calculate_maxneigh_cpu(
            (const int64_t *) num_atoms.data_ptr(),
            (const int64_t *) atom_num_list_sum.data_ptr(),
            (const double  *) box.data_ptr(),
            (const double  *) box_orig.data_ptr(),
            (const int64_t *) num_cell.data_ptr(),
            (const double  *) position.data_ptr(),
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
        return {NN_radial, NN_angular};
    } else {
        torch::Tensor NN_radial = torch::empty({total_atoms, atom_type_num}, torch::TensorOptions().dtype(index_dtype).device(device));
        torch::Tensor NN_angular = torch::empty({total_atoms, atom_type_num}, torch::TensorOptions().dtype(index_dtype).device(device));
        
        launch_calculate_maxneigh_cpu(
            (const int64_t *) num_atoms.data_ptr(),
            (const int64_t *) atom_num_list_sum.data_ptr(),
            (const double  *) box.data_ptr(),
            (const double  *) box_orig.data_ptr(),
            (const int64_t *) num_cell.data_ptr(),
            (const double  *) position.data_ptr(),
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
        return {NN_radial, NN_angular};        
    }
    
}

std::vector<torch::Tensor> calculate_neighbor_cpu(
    const torch::Tensor &num_atoms, 
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
    const bool with_rij
){
    auto device = position.device();
    auto dtype = position.dtype();
    auto index_dtype = num_atoms.dtype();
    torch::Tensor atom_num_list_sum = torch::cumsum(num_atoms, 0, torch::kInt64);
    int64_t total_frames = num_atoms.sizes()[0];
    int64_t total_atoms = position.sizes()[0];
    
    torch::Tensor NN_radial  = torch::empty({total_atoms}, torch::TensorOptions().dtype(index_dtype).device(device));
    torch::Tensor NL_radial  = torch::full( {total_atoms, max_NN_radial}, -1, torch::TensorOptions().dtype(index_dtype).device(device));
    torch::Tensor NN_angular = torch::empty({total_atoms}, torch::TensorOptions().dtype(index_dtype).device(device));
    torch::Tensor NL_angular = torch::full( {total_atoms, max_NN_angular}, -1, torch::TensorOptions().dtype(index_dtype).device(device));
    if (!with_rij){
        torch::Tensor Ri_radial = torch::full(
            {total_atoms, max_NN_radial, 3}, 0.0, torch::TensorOptions().dtype(dtype).device(device));
        torch::Tensor Ri_angular = torch::full(
            {total_atoms, max_NN_angular,3}, 0.0, torch::TensorOptions().dtype(dtype).device(device));        
        launch_calculate_neighbor_cpu(
            (int64_t *) num_atoms.data_ptr(),
            (int64_t *) atom_num_list_sum.data_ptr(), 
            (int64_t *) atom_type_map.data_ptr(), 
            (int64_t *) atom_types.data_ptr(),
            (double  *) box.data_ptr(),
            (double  *) box_orig.data_ptr(),
            (int64_t *) num_cell.data_ptr(), 
            (double  *) position.data_ptr(), 
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
        return {NN_radial, NN_angular, NL_radial, NL_angular, Ri_radial, Ri_angular};
    } else {
        torch::Tensor Ri_radial = torch::full(
            {total_atoms, max_NN_radial, 4}, 0.0, torch::TensorOptions().dtype(dtype).device(device));
        torch::Tensor Ri_angular = torch::full(
            {total_atoms, max_NN_angular, 4}, 0.0, torch::TensorOptions().dtype(dtype).device(device));        
        launch_calculate_neighbor_cpu(
            (int64_t *) num_atoms.data_ptr(),
            (int64_t *) atom_num_list_sum.data_ptr(), 
            (int64_t *) atom_type_map.data_ptr(), 
            (int64_t *) atom_types.data_ptr(),
            (double  *) box.data_ptr(),
            (double  *) box_orig.data_ptr(),
            (int64_t *) num_cell.data_ptr(), 
            (double  *) position.data_ptr(), 
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
        return {NN_radial, NN_angular, NL_radial, NL_angular, Ri_radial, Ri_angular};
    }
}

std::vector<torch::Tensor> calculate_descriptor_cpu(
    const torch::Tensor &weight_radial,
    const torch::Tensor &weight_angular,
    const torch::Tensor &Ri_radial,
    const torch::Tensor &NL_radial,
    const torch::Tensor &atom_type_map,
    const double cutoff_radial,
    const double cutoff_angular,
    const int64_t max_NN_radial,
    const int64_t lmax_3,
    const int64_t lmax_4,
    const int64_t lmax_5
){
    auto device = Ri_radial.device();
    auto dtype = Ri_radial.dtype();
    auto dims = weight_radial.sizes();
    int64_t  num_types     = dims[0];
    int64_t  n_max_radial  = dims[2];
    int64_t  n_base_radial = dims[3];
    dims = weight_angular.sizes();
    int64_t  n_max_angular = dims[2];
    int64_t  n_base_angular= dims[3];

    int64_t total_atoms = Ri_radial.sizes()[0];
    int64_t dim_angular = 0;
        if (lmax_3 > 0) {
            dim_angular += n_max_angular * lmax_3;
        }
        if (lmax_4 > 0) {
            dim_angular += n_max_angular;
        } 
        if (lmax_5 > 0) {
            dim_angular += n_max_angular;
        }
    int64_t dim_radial = n_max_radial;
    torch::Tensor feats = torch::zeros({total_atoms, dim_radial+dim_angular}, torch::TensorOptions().dtype(dtype).device(device));
    launch_calculate_descriptor_cpu(
        (double  *) weight_radial.data_ptr(), 
        (double  *) weight_angular.data_ptr(), 
        (double  *) Ri_radial.data_ptr(),
        (int64_t *) NL_radial.data_ptr(), 
        (int64_t *) atom_type_map.data_ptr(), 
        cutoff_radial, 
        cutoff_angular, 
        (double  *) feats.data_ptr(), 
        total_atoms,
        max_NN_radial, 
        n_max_radial, 
        n_base_radial, 
        n_max_angular, 
        n_base_angular, 
        lmax_3, 
        lmax_4, 
        lmax_5, 
        num_types);
    return {feats};
}

