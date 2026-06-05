/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.

In the compute() function, the following major additions and modifications have been made compared to GPUMD's compute_large_box(https://github.com/brucefan1983/GPUMD/blob/master/src/force/nep.cu):
 1. Modified the data structures to adapt to the neighbor list index format from LAMMPS KOKKOS.

 2. Added handling for ghost atoms, which are unique to LAMMPS.

 3. Split the descriptor calculation into two independent parts: two-body and many-body, computed separately.

 4. Reworked the kernels for two-body and many-body force calculations so that threads within a block collaborate on computing forces for a central atom.

 5. Optimized register spilling issues by utilizing shared memory.

In the read_neptxt function for parsing NEP.txt is adapted from the GPUMD source code.(https://github.com/brucefan1983/GPUMD/blob/master/src/force/nep.cu (NEP::NEP(const char* file_potential, const int num_atoms)). )
  It has been modified to align with LAMMPS's print output and extended to support parsing both nep4.txt from GPUMD and nep5.txt from MatPL.

  wuxingxing@pwmat.com and MatPL development team. 2026. Beijing Lonxun Quantum Co.,Ltd.

*/
#pragma once
#include "../utilities/common.cuh"
#include "../utilities/gpu_vector.cuh"
#include <tuple>
#include <utility> // for std::move
// #include <Kokkos_Core.hpp>


struct LMP_Data  {
  GPU_Vector<int> type;
  GPU_Vector<int> ilist;
  GPU_Vector<NEP_FLOAT> position; // 将double坐标转换为float，读取会更快
};

struct NEPKK_Data {
  GPU_Vector<NEP_FLOAT> f12x; // 3-body or manybody partial forces
  GPU_Vector<NEP_FLOAT> f12y; // 3-body or manybody partial forces
  GPU_Vector<NEP_FLOAT> f12z; // 3-body or manybody partial forces
  GPU_Vector<NEP_FLOAT> Fp;
  GPU_Vector<NEP_FLOAT> sum_fxyz;
  GPU_Vector<int> NN_radial;    // radial neighbor list
  GPU_Vector<int> NL_radial;    // radial neighbor list
  GPU_Vector<int> NN_angular;   // angular neighbor list
  GPU_Vector<int> NL_angular;   // angular neighbor list
  GPU_Vector<NEP_FLOAT> parameters; // parameters to be optimized
  GPU_Vector<NEP_FLOAT> param_c2;
  GPU_Vector<NEP_FLOAT> param_c3;
  GPU_Vector<double> potential_per_atom;
  GPU_Vector<double> potential_all;
  GPU_Vector<double> force_per_atom;
  GPU_Vector<double> virial_per_atom;
  GPU_Vector<double> total_virial;
  GPU_Vector<double> partial_virial;
};

class NEPKK
{
public:
  struct ParaMB {
    bool use_typewise_cutoff_zbl = false;
    NEP_FLOAT typewise_cutoff_zbl_factor = FLOAT_LIT(0.0);
    int version = 4; // NEP version
    int model_type = 0; // 0=potential, 1=dipole, 2=polarizability, 3=temperature-dependent free energy
    NEP_FLOAT rc_radial = FLOAT_LIT(0.0);     // radial cutoff
    NEP_FLOAT rc_angular = FLOAT_LIT(0.0);    // angular cutoff
    NEP_FLOAT rc_radial_square = FLOAT_LIT(0.0); // rc_radial * rc_radial
    NEP_FLOAT rc_angular_square = FLOAT_LIT(0.0); // rc_angular * rc_angular
    NEP_FLOAT rcinv_radial = FLOAT_LIT(0.0);  // inverse of the radial cutoff
    NEP_FLOAT rcinv_angular = FLOAT_LIT(0.0); // inverse of the angular cutoff
    int MN_radial = 200;
    int MN_angular = 100;
    int n_max_radial = 0;  // n_radial = 0, 1, 2, ..., n_max_radial
    int n_max_angular = 0; // n_angular = 0, 1, 2, ..., n_max_angular
    int n_max_radial_plus1 = 0;  // n_radial = 0, 1, 2, ..., n_max_radial
    int n_max_angular_plus1 = 0; // n_angular = 0, 1, 2, ..., n_max_angular
    int L_max = 0;         // l = 0, 1, 2, ..., L_max
    int dim_angular;
    int num_L;
    int basis_size_radial = 8;  // for nep3
    int basis_size_angular = 8; // for nep3
    int basis_size_radial_plus1 = 8;  // for nep3
    int basis_size_angular_plus1 = 8; // for nep3
    int num_types_sq = 0;       // for nep3
    int num_c_radial = 0;       // for nep3
    int num_types = 0;
    NEP_FLOAT q_scaler[140];
  };

  struct ANN {
    int dim = 0;                 // dimension of the descriptor
    int num_neurons1 = 0;        // number of neurons in the 1st hidden layer
    int num_para = 0;            // number of parameters
    int num_para_ann = 0;
    int num_c2 = 0;
    int num_c3 = 0;
    const NEP_FLOAT* w0[NUM_ELEMENTS]; // weight from the input layer to the hidden layer
    const NEP_FLOAT* b0[NUM_ELEMENTS]; // bias for the hidden layer
    const NEP_FLOAT* w1[NUM_ELEMENTS]; // weight from the hidden layer to the output layer
    const NEP_FLOAT* b1;             // bias for the output layer
    NEP_FLOAT* c;
  };

  struct ZBL {
    bool enabled = false;
    bool flexibled = false;
    NEP_FLOAT rc_inner = FLOAT_LIT(1.0);
    NEP_FLOAT rc_outer = FLOAT_LIT(2.0);
    std::vector<NEP_FLOAT> cpu_para;
    std::vector<int> cpu_atomic_numbers; // typewise zbl, as index, should be int
    GPU_Vector<NEP_FLOAT> para;
    GPU_Vector<int> atomic_numbers; // typewise zbl, as index, should be int
    int num_types;
  };

  NEPKK();
  void read_neptxt(const char* file_potential, const bool is_rank_0, const int rank_id, const int device_id, const int ff_id);
  void set_atom_type_map(int type_nums, const int* type_list);
  void convert_C(NEP_FLOAT* c, int NtypeI, int Nmax, int Nbase);// 调整C的维度
  ~NEPKK(void);

  bool USE_SHAREMEM_C2 = false;
  bool USE_SHAREMEM_C3 = false;

  ParaMB paramb;
  ANN annmb;
  ZBL zbl;
  NEPKK_Data nep_data;
  LMP_Data lmp_data;
  std::vector<int> cpu_element_atomic_number_list;
  GPU_Vector<int> atom_type_map; // 用于结构和力场中的元素顺序不一致时做映射

  int max_inum = 0;
  int max_nlocal = 0;
  int max_nall = 0;
  int allocate_once = 0;
  void update_potential(NEP_FLOAT* parameters, ANN& ann);
  void reset_nep_data(int inum, int n_local, int n_all, int vflag_either);
  void free_nep_data();
  void checkMemoryUsage(int sgin=0);

  bool getGPUMemoryStats(size_t& total_memory, size_t& used_memory, size_t& free_memory);
  int calculateMaxAtoms(int MN_radial, int MN_angular, int lmp_num_neigh, int ANNNB_DIM, int nlocal, int nprocs_total);
  void compute(
    int eflag_global,       // energy_total flag
    int eflag_atom,         // ei peratom flag
    int vflag_either,       // = bool(vfalg_atom or vflag_global)
    int vflag_global,       // virial_global flag
    int vflag_atom,         // virial peratom flag
    int cvflag_atom,        // virial-9 peratom flag
    int nall,               // nlocal + nghost
    int inum, 
    int nlocal,
    int max_neighbors,          // row nums of firstneigh
    int num_neighbors,          // col nums of firstneigh -> j = firstneigh[i + num_neighs * jj] & NEIGHMASK
    int* itype,                 // atoms' type,the len is [n_all]
    int* ilist,                 // atom i list
    int* numneigh,              // the neighbor nums of each i, [inum]
    int* firstneigh,            // the neighbor list of each i, [inum * NM]
    double* position,           // postion of atoms x, [n_all * 3]
    double* potential_per_atom_lmp,  // the output of ei
    double* potential_per_atom_copy, // the output of ei
    double* force_per_atom_lmp,      // the output of force
    double* force_per_atom_copy,
    double* virial_per_atom,
    double* cvirial_per_atom,
    double* h_etot_virial_global // len=7: etot + 6 virials
    );

  bool free_neigh = false;
  bool is_gpumd_nep = false;
  bool rank_0 = false;
  int device;
  int rank;
  int ff_index;
  double* cv_per_atom;
};