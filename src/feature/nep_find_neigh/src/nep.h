/*
List of modified records by Wu Xingxing (email stars_sparkling@163.com)
1. Added network structure support for NEP4 model independent bias
    Modified force field reading;
    Modified the applyann_one_layer method;
2. Added handling of inconsistency between the atomic order of the input structure of LAMMPS and the atomic order in the force field
3. In order to adapt to multiple model biases, the function has been added with computefor_lamps() and the int model_index parameter has been added  
4. Support GPUMD NEP shared bias and PWMLFF NEP independent bias forcefield

We have made the following improvements based on NEP4
http://doc.lonxun.com/MatPL/models/nep/
*/

/*
the open source code from https://github.com/brucefan1983/NEP_CPU
the licnese of NEP_CPU is as follows:
    Copyright 2022 Zheyong Fan, Junjie Wang, Eric Lindgren
    This file is part of NEP_CPU.
    NEP_CPU is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    NEP_CPU is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with NEP_CPU.  If not, see <http://www.gnu.org/licenses/>.
*/

/*----------------------------------------------------------------------------80
A CPU implementation of the neuroevolution potential (NEP)
Ref: Zheyong Fan et al., Neuroevolution machine learning potentials:
Combining high accuracy and low cost in atomistic simulations and application to
heat transport, Phys. Rev. B. 104, 104309 (2021).
------------------------------------------------------------------------------*/

#pragma once
#include <string>
#include <vector>

class NEP_CPU
{
public:
  struct ParaMB {
    bool use_typewise_cutoff_zbl = false;
    double typewise_cutoff_zbl_factor = 0.0;
    int model_type = 0; // 0=potential, 1=dipole, 2=polarizability
    int charge_mode = 0;
    int version = 4;
    double rc_radial = 0.0;
    double rc_angular = 0.0;
    double rcinv_radial = 0.0;
    double rcinv_angular = 0.0;
    int n_max_radial = 0;
    int n_max_angular = 0;
    int L_max = 0;
    int dim_angular;
    int num_L;
    int basis_size_radial = 8;
    int basis_size_angular = 8;
    int num_types_sq = 0;
    int num_c_radial = 0;
    int num_types = 0;
    double q_scaler[140];
    int atomic_numbers[94];
  };

  struct ANN {
    int dim = 0;
    int num_neurons1 = 0;
    int num_para = 0;
    int num_para_ann = 0;
    int num_c2 = 0;
    int num_c3 = 0;
    const double* w0[103];
    const double* b0[103];
    const double* w1[103];
    const double* sqrt_epsilon_inf;
    const double* b1;
    const double* c;
    // for the scalar part of polarizability
    const double* w0_pol[103];
    const double* b0_pol[103];
    const double* w1_pol[103];
    const double* b1_pol;
  };

  struct Charge_Para {
    int num_kpoints_max = 50000;
    double alpha = 0.0;
    double alpha_factor = 0.0;
  };

  struct ZBL {
    bool enabled = false;
    bool flexibled = false;
    int num_types;
    double rc_inner = 1.0;
    double rc_outer = 2.0;
    double atomic_numbers[103];
    double para[550];
  };

  NEP_CPU();
  NEP_CPU(const std::string& potential_filename);

  void init_from_file(const std::string& potential_filename, const bool is_rank_0);

  void find_neigh(
    const double rc_radial,
    const double rc_angular,
    const int MN, // max neighs of config ,which will be num_types * max_neigh set in json file
    const std::vector<int>& atom_type_map,
    const std::vector<double>& box,
    const std::vector<double>& position);

  // type[num_atoms] should be integers 0, 1, ..., mapping to the atom types in nep.txt in order
  // box[9] is ordered as ax, bx, cx, ay, by, cy, az, bz, cz
  // position[num_atoms * 3] is ordered as x[num_atoms], y[num_atoms], z[num_atoms]
  // potential[num_atoms]
  // force[num_atoms * 3] is ordered as fx[num_atoms], fy[num_atoms], fz[num_atoms]
  // virial[num_atoms * 9] is ordered as v_xx[num_atoms], v_xy[num_atoms], v_xz[num_atoms],
  // v_yx[num_atoms], v_yy[num_atoms], v_yz[num_atoms], v_zx[num_atoms], v_zy[num_atoms],
  // v_zz[num_atoms]
  // descriptor[num_atoms * dim] is ordered as d0[num_atoms], d1[num_atoms], ...

  void compute(
    const std::vector<int>& type,
    const std::vector<double>& box,
    const std::vector<double>& position,
    std::vector<double>& potential,
    std::vector<double>& force,
    std::vector<double>& virial,
    std::vector<double>&  total_virial);

  ParaMB paramb;
  ANN annmb;
  ZBL zbl;
  Charge_Para charge_para;

  int num_atoms = 0;
  int num_cells[3];
  double ebox[18];
  // NN nums of neighbors; NL neighbor lists; NLT neigbors' type
  std::vector<int> NN_radial, NL_radial, NLT_radial, NN_angular, NL_angular, NLT_angular;
  std::vector<double> r12_radial;
  std::vector<double> r12_angular;
  std::vector<double> r12;
  std::vector<double> Fp;
  std::vector<double> charge;
  std::vector<double> charge_derivative;
  std::vector<double> bec;
  std::vector<double> D_real;
  std::vector<int> num_kpoints;
  std::vector<double> kx;
  std::vector<double> ky;
  std::vector<double> kz;
  std::vector<double> G;
  std::vector<double> S_real;
  std::vector<double> S_imag;
  std::vector<double> sum_fxyz;
  std::vector<double> parameters;
  std::vector<std::string> element_list;
  std::vector<int> element_atomic_number_list;

  std::vector<int> map_atom_types;     //pair_coeff       * * 72 8
  std::vector<int> map_atom_type_idx; // the nep.txt order is [8, 72], so the idx is [1, 0]
            
  void update_potential(double* parameters, ANN& ann);
  void allocate_memory(const int N);
};
