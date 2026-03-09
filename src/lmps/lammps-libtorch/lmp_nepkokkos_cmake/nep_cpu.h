/*
List of modified records by Wu Xingxing (email stars_sparkling@163.com)
1. Added network structure support for NEP4 model independent bias
    Modified force field reading;
    Modified the applyann_one_layer method;
2. Added handling of inconsistency between the atomic order of the input structure of LAMMPS and the atomic order in the force field
3. In order to adapt to multiple model biases, the function has been added with computefor_lamps() and the int model_index parameter has been added  
4. Support GPUMD NEP shared bias and MATPL NEP independent bias forcefield

We have made the following improvements based on NEP4
http://doc.lonxun.com/PWMLFF/models/nep/NEP%20model/
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

// #define USE_TABLE_FOR_RADIAL_FUNCTIONS

class NEP3_CPU
{
public:
  struct ParaMB {
    int model_type = 0; // 0=potential, 1=dipole, 2=polarizability
    int version = 2;
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
    const double* b1;
    const double* c;
    // for the scalar part of polarizability
    const double* w0_pol[103];
    const double* b0_pol[103];
    const double* w1_pol[103];
    const double* b1_pol;
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

  struct DFTD3 {
    double s6 = 0.0;
    double s8 = 0.0;
    double a1 = 0.0;
    double a2 = 0.0;
    double rc_radial = 20.0;
    double rc_angular = 10.0;
    int atomic_number[94]; // H to Pu
    std::vector<double> cn;
    std::vector<double> dc6_sum;
    std::vector<double> dc8_sum;
  };

  NEP3_CPU();
  NEP3_CPU(const std::string& potential_filename);

  void read_neptxt(const std::string& potential_filename, const bool is_rank_0);
  void compute_for_lammps(
    int nlocal,              // list->nlocal
    int inum,                // list->inum
    int* ilist,              // list->ilist
    int* numneigh,           // list->numneigh
    int** firstneigh,        // list->firstneigh
    int* type,               // atom->type
    double** x,              // atom->x
    double& total_potential, // total potential energy for the current processor
    double total_virial[6],  // total virial for the current processor
    double* potential,       // eatom or nullptr
    double** f,              // atom->f
    double** virial,          // cvatom or nullptr
    int model_index          // for multimodels' deviation
  );

  int num_atoms = 0;
  int num_cells[3];
  double ebox[18];
  ParaMB paramb;
  ANN annmb;
  ZBL zbl;
  std::vector<int> NN_radial, NL_radial, NN_angular, NL_angular;
  std::vector<double> r12;
  std::vector<double> Fp;
  std::vector<double> sum_fxyz;
  std::vector<double> parameters;
  std::vector<std::string> element_list;
  std::vector<int> element_atomic_number_list;

  std::vector<int> map_atom_types;     //pair_coeff       * * 72 8
  std::vector<int> map_atom_type_idx; // the nep.txt order is [8, 72], so the idx is [1, 0]
            
  void update_potential(double* parameters, ANN& ann);
};
