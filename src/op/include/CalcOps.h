#include <torch/torch.h>
#include <iostream>

class CalculateForceFuncs {
    public:
        static torch::autograd::variable_list forward(
            at::Tensor list_neigh,
            at::Tensor dE,
            at::Tensor Ri_d,
            at::Tensor F,
            at::Tensor nghost_tensor);

        static torch::autograd::variable_list backward(
            torch::autograd::variable_list grad_output,
            at::Tensor list_neigh,
            at::Tensor dE,
            at::Tensor Ri_d);
};

class CalculateForce : public torch::autograd::Function<CalculateForce> {
    public:
        static torch::autograd::variable_list forward(
            torch::autograd::AutogradContext *ctx,
            at::Tensor list_neigh,
            at::Tensor dE,
            at::Tensor Ri_d,
            at::Tensor F,
            at::Tensor nghost_tensor);

        static torch::autograd::variable_list backward(
            torch::autograd::AutogradContext *ctx,
            torch::autograd::variable_list grad_output);
};

torch::autograd::variable_list calculateForce(
    at::Tensor list_neigh,
    at::Tensor dE,
    at::Tensor Ri_d,
    at::Tensor F,
    at::Tensor nghost_tensor);

torch::autograd::variable_list calculateForce_cpu(
    at::Tensor list_neigh,
    at::Tensor dE,
    at::Tensor Ri_d,
    at::Tensor F,
    at::Tensor nghost_tensor);

// the following is the code virial
class CalculateVirialFuncs {
    public:
        static torch::autograd::variable_list forward(
            at::Tensor list_neigh,
            at::Tensor dE,
            at::Tensor Rij,
            at::Tensor Ri_d,
            at::Tensor nghost_tensor
        );

        static torch::autograd::variable_list backward(
            torch::autograd::variable_list grad_output,
            at::Tensor list_neigh,
            at::Tensor dE,
            at::Tensor Rij,
            at::Tensor Ri_d
        );
};

class CalculateVirial : public torch::autograd::Function<CalculateVirial> {
    public:
        static torch::autograd::variable_list forward(
            torch::autograd::AutogradContext *ctx,
            at::Tensor list_neigh,
            at::Tensor dE,
            at::Tensor Rij,
            at::Tensor Ri_d,
            at::Tensor nghost_tensor);

        static torch::autograd::variable_list backward(
            torch::autograd::AutogradContext *ctx,
            torch::autograd::variable_list grad_output);
};

torch::autograd::variable_list calculateVirial(
    at::Tensor list_neigh,
    at::Tensor dE,
    at::Tensor Rij,
    at::Tensor Ri_d,
    at::Tensor nghost_tensor);

torch::autograd::variable_list calculateVirial_cpu(
    at::Tensor list_neigh,
    at::Tensor dE,
    at::Tensor Rij,
    at::Tensor Ri_d,
    at::Tensor nghost_tensor);

// the following is the code compress
class CalculateCompressFuncs {
    public:
        static torch::autograd::variable_list forward(
            at::Tensor f2,
            at::Tensor coefficient);

        static torch::autograd::variable_list backward(
            torch::autograd::variable_list grad_output,
            at::Tensor f2,
            at::Tensor coefficient);
};

class CalculateCompress : public torch::autograd::Function<CalculateCompress> {
    public:
        static torch::autograd::variable_list forward(
            torch::autograd::AutogradContext *ctx,
            at::Tensor f2,
            at::Tensor coefficient);

        static torch::autograd::variable_list backward(
            torch::autograd::AutogradContext *ctx,
            torch::autograd::variable_list grad_output);
};

torch::autograd::variable_list calculateCompress(
    at::Tensor f2,
    at::Tensor coefficient);

torch::autograd::variable_list calculateCompress_cpu(
    at::Tensor f2,
    at::Tensor coefficient);

// the following is the code nep feature
class CalculateNepFeat : public torch::autograd::Function<CalculateNepFeat> {
    public:
        static torch::autograd::variable_list forward(
            torch::autograd::AutogradContext *ctx,
            at::Tensor coeff2,
            at::Tensor d12_radial,
            at::Tensor NL_radial,
            at::Tensor atom_map,
            at::Tensor feats,
            double rcut_radial,
            int64_t multi_feat_num,
            int64_t fix_cij);

        static torch::autograd::variable_list backward(
            torch::autograd::AutogradContext *ctx,
            torch::autograd::variable_list grad_output);
};

class CalculateNepFeatGrad : public torch::autograd::Function<CalculateNepFeatGrad> {
    public:
        static torch::autograd::variable_list forward(
            torch::autograd::AutogradContext *ctx,
            at::Tensor grad_input,
            at::Tensor coeff2,
            at::Tensor d12_radial,
            at::Tensor NL_radial,
            at::Tensor dfeat_c2,
            at::Tensor dfeat_2b,
            at::Tensor dfeat_2b_noc,
            at::Tensor atom_map,
            int64_t multi_feat_num,
            int64_t fix_cij);

        static torch::autograd::variable_list backward(
            torch::autograd::AutogradContext *ctx,
            torch::autograd::variable_list grad_output);
};

torch::autograd::variable_list calculateNepFeat(
    at::Tensor coeff2,
    at::Tensor d12_radial,
    at::Tensor NL_radial,
    at::Tensor atom_map,
    at::Tensor feats,
    double rcut_radial,
    int64_t feat_multi_nums,
    int64_t fix_cij);


// the following is the code nep multi feature
class CalculateNepMbFeat : public torch::autograd::Function<CalculateNepMbFeat> {
    public:
        static torch::autograd::variable_list forward(
            torch::autograd::AutogradContext *ctx,
            at::Tensor coeff3,
            at::Tensor d12,
            at::Tensor NL,
            at::Tensor atom_map,
            at::Tensor feats,
            int64_t feat_2b_num,
            int64_t lmax_3,
            int64_t lmax_4,
            int64_t lmax_5,
            double rcut_angluar,
            int64_t fix_cij);

        static torch::autograd::variable_list backward(
            torch::autograd::AutogradContext *ctx,
            torch::autograd::variable_list grad_output);
};

class CalculateNepMbFeatGrad : public torch::autograd::Function<CalculateNepMbFeatGrad> {
    public:
        static torch::autograd::variable_list forward(
            torch::autograd::AutogradContext *ctx,
            at::Tensor grad_input,
            at::Tensor coeff3,
            at::Tensor d12,
            at::Tensor NL,
            at::Tensor dfeat_c3,
            at::Tensor dfeat_3b,
            at::Tensor dfeat_3b_noc,
            at::Tensor sum_fxyz,
            at::Tensor atom_map,
            int64_t feat_2b_num,
            int64_t lmax_3,
            int64_t lmax_4,
            int64_t lmax_5,
            double rcut_angluar,
            int64_t fix_cij
            );

        static torch::autograd::variable_list backward(
            torch::autograd::AutogradContext *ctx,
            torch::autograd::variable_list grad_output);
};

torch::autograd::variable_list calculateNepMbFeat(
    at::Tensor coeff3,
    at::Tensor d12_angluar,
    at::Tensor NL_angluar,
    at::Tensor atom_map,
    at::Tensor feats,
    int64_t feat_2b_num,
    int64_t lmax_3,
    int64_t lmax_4,
    int64_t lmax_5,
    double rcut_angluar,
    int64_t fix_cij);

//nep force
class CalculateNepForce : public torch::autograd::Function<CalculateNepForce> {
    public:
        static torch::autograd::variable_list forward(
            torch::autograd::AutogradContext *ctx,
            at::Tensor list_neigh,
            at::Tensor dE,
            at::Tensor Ri_d,
            at::Tensor F);

        static torch::autograd::variable_list backward(
            torch::autograd::AutogradContext *ctx,
            torch::autograd::variable_list grad_output);
};

torch::autograd::variable_list calculateNepForce(
    at::Tensor list_neigh,
    at::Tensor dE,
    at::Tensor Ri_d,
    at::Tensor F);

//nep virial
class CalculateNepVirial : public torch::autograd::Function<CalculateNepVirial> {
    public:
        static torch::autograd::variable_list forward(
            torch::autograd::AutogradContext *ctx,
            at::Tensor list_neigh,
            at::Tensor dE,
            at::Tensor Rij,
            at::Tensor Ri_d,
            at::Tensor num_atom);

        static torch::autograd::variable_list backward(
            torch::autograd::AutogradContext *ctx,
            torch::autograd::variable_list grad_output);
};

torch::autograd::variable_list calculateNepVirial(
    at::Tensor list_neigh,
    at::Tensor dE,
    at::Tensor Rij,
    at::Tensor Ri_d,
    at::Tensor num_atom);

std::vector<torch::Tensor> calculate_maxneigh(
    const torch::Tensor &num_atoms,
    const torch::Tensor &box,
    const torch::Tensor &box_orig, 
    const torch::Tensor &num_cell, 
    const torch::Tensor &position,
    const double cutoff_2b,
    const double cutoff_3b,
    const int64_t atom_type_num,
    const torch::Tensor &atom_type_map,
    const bool with_type = false);

std::vector<torch::Tensor> calculate_neighbor(
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
    const bool with_rij=false);

std::vector<torch::Tensor> calculate_descriptor(
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
    const int64_t lmax_5);



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
    const bool with_type = false
    );

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
    const bool with_rij=false);

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
    const int64_t lmax_5);
