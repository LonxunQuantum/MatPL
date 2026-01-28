template<typename DType>
void launch_calculate_force(
    const int * nblist,
    const DType * dE,
    const DType * Ri_d,
    const int batch_size,
    const int natoms,
    const int neigh_num,
    DType * force,
    const int nghost,
    const int device
);

template<typename DType>
void launch_calculate_virial_force(
    const int * nblist,
    const DType * dE,
    const DType * Rij,
    const DType * Ri_d,
    const int batch_size,
    const int natoms,
    const int neigh_num,
    DType * virial,
    DType * atom_virial,
    const int nghost,
    const int device
);