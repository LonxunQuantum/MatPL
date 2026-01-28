void launch_calculate_nepforce(
    const int64_t * nblist,
    const double * dE,
    const double * Ri_d,
    const int natoms,
    const int neigh_num,
    double * force,
    const int device
);

void launch_calculate_nepforce_grad(
    const int64_t * nblist,
    const double * Ri_d,
    const double * net_grad,
    const int natoms,
    const int neigh_num,
    double * grad,
    const int device
);

void launch_calculate_nepvirial(
    const int64_t * nblist,
    const double * dE,
    const double * Rij,
    const double * Ri_d,
    const int64_t * num_atom,
    const int batch_num,
    const int natoms,
    const int neigh_num,
    double * virial,
    double * atom_virial,
    const int device
);

void launch_calculate_nepvirial_grad(
    const int64_t * nblist,
    const double * Rij,
    const double * Ri_d,
    const double * net_grad,
    const int natoms,
    const int neigh_num,
    double * grad,
    const int device
);

void launch_calculate_nepfeat(
    const double * coeff2,
    const double * d12_radial,
    const int64_t* NL_radial,
    const int64_t* atom_map,
    const double rcut_radial,
    double * feat_2b,
    double * dfeat_c2,
    double * dfeat_2b,
    double * dfeat_2b_noc,
    const int natoms,
    const int neigh_num,
    const int n_max,
    const int n_base,
    const int num_types,
    const int device
);

void launch_calculate_nepfeat_grad(
    const double * grad_output,
    const double * dfeat_c2,
    const double * dfeat_2b,
    const int64_t* atom_map,
    double * grad_coeff2,
    double * grad_d12_radial,
    const int natoms, 
    const int neigh_num, 
    const int n_max_2b, 
    const int n_base_2b,
    const int n_types,
    const int multi_feat_num,
    const int device
);

void launch_calculate_nepfeat_secondgradout(
    const double * grad_second,
    const double * dfeat_b,
    double * gradsecond_out,
    const int atom_nums, 
    const int maxneighs, 
    const int n_max, 
    const int device
);

void launch_calculate_nepfeat_secondgradout_c2(
    const double * grad_second,
    const double * de_feat,
    const double * dfeat_2b_noc,
    const int64_t* atom_map,
    const int64_t* NL_radial,
    double * gradsecond_c2,
    const int atom_nums, 
    const int maxneighs, 
    const int n_max_2b, 
    const int n_base_2b, 
    const int atom_types, 
    const int multi_feat_num,
    const int device
);

void launch_calculate_nepmbfeat(
    const double * coeff3,
    const double * d12,
    const int64_t* NL,
    const int64_t* atom_map,
    double * feat_3b,
    double * dfeat_c3,
    double * dfeat_3b,
    double * dfeat_3b_noc,
    double * sum_fxyz,
    const double rcut,
    const int natoms,
    const int neigh_num,
    const int n_max_3b, 
    const int n_base_3b,
    const int lmax_3,
    const int lmax_4,
    const int lmax_5,
    const int num_types,
    const int device
);

void launch_calculate_nepmbfeat_grad(
            const double * grad_output,
            const double * coeff3, 
            const double * r12,
            const int64_t* NL, 
            const int64_t* atom_map, 
            double * sum_fxyz,
            double * grad_coeff3, 
            double * grad_d12_3b,
            double * dsnlm_dc,
            double * dfeat_drij,
            const int rcut_angular,
            const int atom_nums, 
            const int neigh_num, 
            const int feat_2b_num, 
            const int n_max_3b, 
            const int n_base_3b,
            const int lmax_3,
            const int lmax_4,
            const int lmax_5,
            const int n_types, 
            const int device_id
);

void launch_calculate_nepmbfeat_secondgradout(
    const double * grad_second,
    const double * dfeat_b,
    double * gradsecond_out,
    const int atom_nums, 
    const int maxneighs, 
    const int feat_mb_nums, 
    const int device
);

void launch_calculate_nepmbfeat_secondgradout_c3(
    const double * grad_second,
    const double * d12,
    const int64_t* NL,
    const double * de_dfeat,
    const double * dsnlm_dc,
    const double * sum_fxyz,
    const int64_t* atom_map,
    const double * coeff3,
    double * gradsecond_c3,
    const double rcut_angular,
    const int atom_nums, 
    const int maxneighs, 
    const int n_max_3b, 
    const int n_base_3b, 
    const int atom_types, 
    const int lmax_3,
    const int lmax_4,
    const int lmax_5,
    const int feat_2b_num,
    const int multi_feat_num,
    const int device
);
