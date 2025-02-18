
void launch_calculate_maxneigh_cpu(
    const int64_t * num_atoms,
    const int64_t * num_atoms_sum,
    const double  * box,
    const double  * box_orig, 
    const int64_t * num_cell, 
    const double  * position,
    const double   cutoff_2b,
    const double   cutoff_3b,
    const int64_t  total_frames,
    const int64_t  total_atoms,
    int64_t      * NN_radial, 
    int64_t      * NN_angular
);

void launch_calculate_neighbor_cpu(
    const int64_t * num_atoms,
    const int64_t * num_atoms_sum,
    const int64_t * atom_type_map,
    const int64_t * atom_types, 
    const double  * box,
    const double  * box_orig, 
    const int64_t * num_cell, 
    const double  * position,
    const double   cutoff_2b,
    const double   cutoff_3b,
    const int64_t  max_NN_radial,
    const int64_t  max_NN_angular,
    const int64_t  total_frames,
    const int64_t  total_atoms,
    int64_t      * NN_radial, 
    int64_t      * NL_radial,
    int64_t      * NN_angular, 
    int64_t      * NL_angular,
    double       * Ri_radial, 
    double       * Ri_angular,
    bool with_rij = false
);

void launch_calculate_descriptor_cpu(
    const double  * coeff2,
    const double  * coeff3,
    const double  * r12,
    const int64_t * NL,
    const int64_t * atom_map,
    const double rcut_radial,
    const double rcut_angular,
    double * feats,
    const int64_t total_atoms,
    const int64_t neigh_num,
    const int64_t n_max_2b,
    const int64_t n_base_2b,
    const int64_t n_max_3b,
    const int64_t n_base_3b,
    const int64_t lmax_3,
    const int64_t lmax_4,
    const int64_t lmax_5,
    const int64_t n_types
);