/*
This code is developed based on the GPUMD source code and added ghost atom processing in LAMMPS. 
  Support multi GPUs.
  Support GPUMD NEP shared bias and PWMLFF NEP independent bias forcefield.

We have made the following improvements based on NEP4
http://doc.lonxun.com/MatPL/models/nep/
*/

/*
    the open source code from https://github.com/brucefan1983/GPUMD
    the licnese of NEP_CPU is as follows:

    Copyright 2017 Zheyong Fan, Ville Vierimaa, Mikko Ervasti, and Ari Harju
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
*/

/*----------------------------------------------------------------------------80
The neuroevolution potential (NEP)
Ref: Zheyong Fan et al., Neuroevolution machine learning potentials:
Combining high accuracy and low cost in atomistic simulations and application to
heat transport, Phys. Rev. B. 104, 104309 (2021).
------------------------------------------------------------------------------*/

#include "nep.cuh"
#include "nep_functions.cuh"
#include "ewald.cuh"
#include "pppm.cuh"
#include "../utilities/common.cuh"
#include "../utilities/error.cuh"
#include "../utilities/nep_utilities.cuh"
#include "math.h"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

const std::string ELEMENTS[NUM_ELEMENTS] = {
  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si", "P",
  "S",  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
  "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh",
  "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
  "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re",
  "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
  "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"};


int countNonEmptyLines(const char* filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "open file error in coutline function: " << filename << std::endl;
        exit(1);
    }
    std::string line;
    int nonEmptyLineCount = 0;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            nonEmptyLineCount++;
        }
    }
    file.close();
    return nonEmptyLineCount;
}

static void get_expanded_box(const double rc, const Box& box, NEP::ExpandedBox& ebox)
{
  double volume = box.get_volume();

  // printf("======gpu volume %f ======\n", volume);
  double thickness_x = volume / box.get_area(0);
  double thickness_y = volume / box.get_area(1);
  double thickness_z = volume / box.get_area(2);
  ebox.num_cells[0] = int(ceil(2.0 * rc / thickness_x));
  ebox.num_cells[1] = int(ceil(2.0 * rc / thickness_y));
  ebox.num_cells[2] = int(ceil(2.0 * rc / thickness_z));

  ebox.h[0] = box.cpu_h[0] * ebox.num_cells[0];
  ebox.h[3] = box.cpu_h[3] * ebox.num_cells[0];
  ebox.h[6] = box.cpu_h[6] * ebox.num_cells[0];

  ebox.h[1] = box.cpu_h[1] * ebox.num_cells[1];
  ebox.h[4] = box.cpu_h[4] * ebox.num_cells[1];
  ebox.h[7] = box.cpu_h[7] * ebox.num_cells[1];

  ebox.h[2] = box.cpu_h[2] * ebox.num_cells[2];
  ebox.h[5] = box.cpu_h[5] * ebox.num_cells[2];
  ebox.h[8] = box.cpu_h[8] * ebox.num_cells[2];

  ebox.h[9]  = ebox.h[4] * ebox.h[8] - ebox.h[5] * ebox.h[7];
  ebox.h[10] = ebox.h[2] * ebox.h[7] - ebox.h[1] * ebox.h[8];
  ebox.h[11] = ebox.h[1] * ebox.h[5] - ebox.h[2] * ebox.h[4];
  ebox.h[12] = ebox.h[5] * ebox.h[6] - ebox.h[3] * ebox.h[8];
  ebox.h[13] = ebox.h[0] * ebox.h[8] - ebox.h[2] * ebox.h[6];
  ebox.h[14] = ebox.h[2] * ebox.h[3] - ebox.h[0] * ebox.h[5];
  ebox.h[15] = ebox.h[3] * ebox.h[7] - ebox.h[4] * ebox.h[6];
  ebox.h[16] = ebox.h[1] * ebox.h[6] - ebox.h[0] * ebox.h[7];
  ebox.h[17] = ebox.h[0] * ebox.h[4] - ebox.h[1] * ebox.h[3];

  double det = ebox.h[0] * (ebox.h[4] * ebox.h[8] - ebox.h[5] * ebox.h[7]) +
               ebox.h[1] * (ebox.h[5] * ebox.h[6] - ebox.h[3] * ebox.h[8]) +
               ebox.h[2] * (ebox.h[3] * ebox.h[7] - ebox.h[4] * ebox.h[6]);
  for (int n = 9; n < 18; n++) {
    ebox.h[n] /= det;
  }
  // printf("===ebox[0-8]  %f %f %f %f %f %f %f %f %f  ===\n", ebox.h[0], ebox.h[1], ebox.h[2], ebox.h[3], ebox.h[4], ebox.h[5], ebox.h[6], ebox.h[7], ebox.h[8]);
  // printf("===ebox[9-17] %f %f %f %f %f %f %f %f %f  ===\n", ebox.h[9], ebox.h[10], ebox.h[11], ebox.h[12], ebox.h[13], ebox.h[14], ebox.h[15], ebox.h[16], ebox.h[17]);
}

NEP::NEP() {}

void NEP::init_from_file(const char* file_potential, const bool is_rank_0, const int in_device_id)
{
  int neplinenums = countNonEmptyLines(file_potential);

  rank_0 = is_rank_0;
  device_id = in_device_id;
  if (device_id == 0) {
    print_potential_info = true;
  }
  atom_nums = 0;
  std::ifstream input(file_potential);
  if (!input.is_open()) {
    std::cout << "Failed to open " << file_potential << std::endl;
    exit(1);
  }

  // nep3 1 C
  std::vector<std::string> tokens = get_tokens(input);
  if (tokens.size() < 3) {
    std::cout << "The first line of nep.txt should have at least 3 items." << std::endl;
    exit(1);
  }
  if (tokens[0] == "nep4") {
    paramb.version = 4;
    zbl.enabled = false;
  } else if (tokens[0] == "nep4_zbl") {
    paramb.version = 4;
    zbl.enabled = true;
  } else if (tokens[0] == "nep4_charge2") {
    paramb.version = 4;
    paramb.charge_mode = 2;
    zbl.enabled = false;
  } else if (tokens[0] == "nep4_zbl_charge2") {
    paramb.version = 4;
    paramb.charge_mode = 2;
    zbl.enabled = true;
  } else if (tokens[0] == "nep5_charge2") {
    paramb.version = 5;
    paramb.charge_mode = 2;
    zbl.enabled = false;
  } else if (tokens[0] == "nep5_zbl_charge2") {
    paramb.version = 5;
    paramb.charge_mode = 2;
    zbl.enabled = true;
  } else if (tokens[0] == "nep5") {
    paramb.model_type = 0;
    paramb.version = 5;
    zbl.enabled = false;
  } else if (tokens[0] == "nep5_zbl") {
    paramb.model_type = 0;
    paramb.version = 5;
    zbl.enabled = true;
  }
  paramb.num_types = get_int_from_token(tokens[1], __FILE__, __LINE__);
  if (tokens.size() != 2 + paramb.num_types) {
    std::cout << "The first line of nep.txt should have " << paramb.num_types << " atom symbols."
              << std::endl;
    exit(1);
  }
  if (print_potential_info) {
    if (paramb.num_types == 1) {
      printf("Use the NEP%d potential with %d atom type.\n", paramb.version, paramb.num_types);
    } else {
      printf("Use the NEP%d potential with %d atom types.\n", paramb.version, paramb.num_types);
    }
  }
  element_atomic_number_list.resize(paramb.num_types);
  for (int n = 0; n < paramb.num_types; ++n) {
    int atomic_number = 0;
    for (int m = 0; m < NUM_ELEMENTS; ++m) {
      if (tokens[2 + n] == ELEMENTS[m]) {
        atomic_number = m + 1;
        break;
      }
    }
    element_atomic_number_list[n] = atomic_number;
    zbl.atomic_numbers[n] = atomic_number;
    if (print_potential_info) {
      printf("    type %d (%s).\n", n, tokens[2 + n].c_str());
    }
  }

// zbl 0.7 1.4
  if (zbl.enabled) {
    tokens = get_tokens(input);
    if (tokens.size() != 3 && tokens.size() != 4) {
      std::cout << "This line should be zbl rc_inner rc_outer [zbl_factor]." << std::endl;
      exit(1);
    }
    zbl.rc_inner = get_double_from_token(tokens[1], __FILE__, __LINE__);
    zbl.rc_outer = get_double_from_token(tokens[2], __FILE__, __LINE__);
    if (zbl.rc_inner == 0 && zbl.rc_outer == 0) {
      zbl.flexibled = true;
      printf("    has the flexible ZBL potential\n");
    } else {
      if (tokens.size() == 4) {
        paramb.typewise_cutoff_zbl_factor = get_double_from_token(tokens[3], __FILE__, __LINE__);
        paramb.use_typewise_cutoff_zbl = true;
        printf("    has the universal ZBL with typewise cutoff with a factor of %g.\n",
          paramb.typewise_cutoff_zbl_factor);
      } else {
        printf(
          "    has the universal ZBL with inner cutoff %g A and outer cutoff %g A.\n",
          zbl.rc_inner,
          zbl.rc_outer);
      }
    }
  }

  // cutoff 4.2 3.7 80 47
  tokens = get_tokens(input);
  if (tokens.size() != 3 && tokens.size() != 5) {
    std::cout << "This line should be cutoff rc_radial rc_angular [MN_radial] [MN_angular].\n";
    exit(1);
  }
  paramb.rc_radial = get_float_from_token(tokens[1], __FILE__, __LINE__);
  paramb.rc_angular = get_float_from_token(tokens[2], __FILE__, __LINE__);
  if (print_potential_info) {
    printf("    radial cutoff = %g A.\n", paramb.rc_radial);
    printf("    angular cutoff = %g A.\n", paramb.rc_angular);
  }
  if (paramb.rc_radial > paramb.rc_angular) {
    paramb.MN_radial = 500;
    paramb.MN_angular = 200;
  } else {
    paramb.MN_radial = 500;
    paramb.MN_angular = 500;
  }
  if (tokens.size() == 5) {
    int MN_radial = get_int_from_token(tokens[3], __FILE__, __LINE__);
    int MN_angular = get_int_from_token(tokens[4], __FILE__, __LINE__);
    if (print_potential_info) {
      printf("    MN_radial = %d.\n", MN_radial);
      printf("    MN_angular = %d.\n", MN_angular);
    }
    paramb.MN_radial = int(ceil(MN_radial * 1.25));
    paramb.MN_angular = int(ceil(MN_angular * 1.25));
    if (print_potential_info) {
      printf("    enlarged MN_radial = %d.\n", paramb.MN_radial);
      printf("    enlarged MN_angular = %d.\n", paramb.MN_angular);
    }
  }

  if (paramb.charge_mode == 2) {
    charge_para.alpha = float(PI) / paramb.rc_radial;
    charge_para.alpha_factor = 0.25f / (charge_para.alpha * charge_para.alpha);
  }

  // n_max 10 8
  tokens = get_tokens(input);
  if (tokens.size() != 3) {
    std::cout << "This line should be n_max n_max_radial n_max_angular." << std::endl;
    exit(1);
  }
  paramb.n_max_radial = get_int_from_token(tokens[1], __FILE__, __LINE__);
  paramb.n_max_angular = get_int_from_token(tokens[2], __FILE__, __LINE__);
  if (print_potential_info) {
    printf("    n_max_radial = %d.\n", paramb.n_max_radial);
    printf("    n_max_angular = %d.\n", paramb.n_max_angular);
  }
  // basis_size 10 8
  if (paramb.version >= 3) {
    tokens = get_tokens(input);
    if (tokens.size() != 3) {
      std::cout << "This line should be basis_size basis_size_radial basis_size_angular."
                << std::endl;
      exit(1);
    }
    paramb.basis_size_radial = get_int_from_token(tokens[1], __FILE__, __LINE__);
    paramb.basis_size_angular = get_int_from_token(tokens[2], __FILE__, __LINE__);
    if (print_potential_info) {
      printf("    basis_size_radial = %d.\n", paramb.basis_size_radial);
      printf("    basis_size_angular = %d.\n", paramb.basis_size_angular);
    }
  }

  // l_max
  tokens = get_tokens(input);
  if (paramb.version == 2) {
    if (tokens.size() != 2) {
      std::cout << "This line should be l_max l_max_3body." << std::endl;
      exit(1);
    }
  } else {
    if (tokens.size() != 4) {
      std::cout << "This line should be l_max l_max_3body l_max_4body l_max_5body." << std::endl;
      exit(1);
    }
  }

  paramb.L_max = get_int_from_token(tokens[1], __FILE__, __LINE__);
  if (print_potential_info) {
    printf("    l_max_3body = %d.\n", paramb.L_max);
  }
  paramb.num_L = paramb.L_max;

  if (paramb.version >= 3) {
    int L_max_4body = get_int_from_token(tokens[2], __FILE__, __LINE__);
    int L_max_5body = get_int_from_token(tokens[3], __FILE__, __LINE__);
    if (print_potential_info) {
      printf("    l_max_4body = %d.\n", L_max_4body);
      printf("    l_max_5body = %d.\n", L_max_5body);
    }
    if (L_max_4body == 2) {
      paramb.num_L += 1;
    }
    if (L_max_5body == 1) {
      paramb.num_L += 1;
    }
  }

  paramb.dim_angular = (paramb.n_max_angular + 1) * paramb.num_L;

  // ANN
  tokens = get_tokens(input);
  if (tokens.size() != 3) {
    std::cout << "This line should be ANN num_neurons 0." << std::endl;
    exit(1);
  }
  annmb.num_neurons1 = get_int_from_token(tokens[1], __FILE__, __LINE__);
  annmb.dim = (paramb.n_max_radial + 1) + paramb.dim_angular;
  if (paramb.model_type == 3) {
    annmb.dim += 1;
  }
  if (print_potential_info) {
    printf("    ANN = %d-%d-1.\n", annmb.dim, annmb.num_neurons1);
  }
  // calculated parameters:
  paramb.rcinv_radial = 1.0f / paramb.rc_radial;
  paramb.rcinv_angular = 1.0f / paramb.rc_angular;
  paramb.num_types_sq = paramb.num_types * paramb.num_types;

  annmb.num_c2   = paramb.num_types_sq * (paramb.n_max_radial + 1) * (paramb.basis_size_radial + 1);
  annmb.num_c3   = paramb.num_types_sq * (paramb.n_max_angular + 1) * (paramb.basis_size_angular + 1);
  
  if (paramb.charge_mode == 2) {
    annmb.num_para_ann = (annmb.dim + 3) * annmb.num_neurons1 * paramb.num_types + 2;
    if (paramb.version == 5) {
      annmb.num_para_ann += paramb.num_types;
    }
  } else if (paramb.version == 4) {
    annmb.num_para_ann = (annmb.dim + 2) * annmb.num_neurons1 * paramb.num_types;
  } else {
    annmb.num_para_ann = ((annmb.dim + 2) * annmb.num_neurons1 + 1) * paramb.num_types + 1;
  }
  int tmp = 0;
  tmp = annmb.num_para_ann + annmb.num_c2 + annmb.num_c3 + 6 + annmb.dim;

  int num_type_zbl = 0;
  if (zbl.enabled && zbl.flexibled) {
    num_type_zbl = (paramb.num_types * (paramb.num_types + 1)) / 2;
    neplinenums -= (1 + 10*num_type_zbl);// zbl 0 0; fixed zbl
  } else if (zbl.enabled) {
    neplinenums  -= 1; // zbl a b
  }

  if (paramb.charge_mode == 2) {
    is_gpumd_nep = (paramb.version == 4);
  } else if (paramb.num_types == 1) {
    is_gpumd_nep = false;
  } else if (paramb.version == 4) {
    if (neplinenums  == (tmp + 1)) {
      is_gpumd_nep = true;
      printf("    the input nep4 potential file is from GPUMD.\n");
    } else if (neplinenums  == (tmp + paramb.num_types)) {
          printf("    the input nep4 potential file is from MatPL.\n");
    } else {
    printf("    parameter parsing error, the number of nep parameters [MatPL %d, GPUMD %d] does not match the text lines %d.\n", tmp+paramb.num_types, (tmp+1), neplinenums);
    exit(1);
    }
  }

  if (paramb.charge_mode == 2) {
    annmb.num_para = annmb.num_para_ann;
  } else if (paramb.version == 4 ){
    annmb.num_para = annmb.num_para_ann + paramb.num_types;
  } else {
    annmb.num_para = annmb.num_para_ann;
  }
  
  if (print_potential_info) {
    if (paramb.charge_mode == 2) {
      printf("    number of neural network parameters = %d.\n", annmb.num_para);
    } else {
      printf("    number of neural network parameters = %d.\n", is_gpumd_nep == false ? annmb.num_para : annmb.num_para-paramb.num_types+1);
    }
  }
  int num_para_descriptor =annmb.num_c2 + annmb.num_c3;
    // paramb.num_types_sq * ((paramb.n_max_radial + 1) * (paramb.basis_size_radial + 1) +
    //                        (paramb.n_max_angular + 1) * (paramb.basis_size_angular + 1));
  if (print_potential_info) {
    printf("    number of descriptor parameters = %d.\n", num_para_descriptor);
  }
  annmb.num_para += num_para_descriptor;
  if (print_potential_info) {
    if (paramb.charge_mode == 2) {
      printf("    total number of parameters = %d.\n", annmb.num_para);
    } else {
      printf("    total number of parameters = %d.\n", is_gpumd_nep == false ? annmb.num_para : annmb.num_para-paramb.num_types+1);
    }
  }
  paramb.num_c_radial =
    paramb.num_types_sq * (paramb.n_max_radial + 1) * (paramb.basis_size_radial + 1);

  // NN and descriptor parameters
  std::vector<float> parameters(annmb.num_para);
  for (int n = 0; n < annmb.num_para; ++n) {
    if (paramb.charge_mode == 0 && is_gpumd_nep == true && (n >= annmb.num_para_ann + 1) && (n < annmb.num_para_ann + paramb.num_types)) {
      parameters[n] = parameters[annmb.num_para_ann];
      if (print_potential_info) {
        printf("    copy the last bias parameters[%d]=%f to parameters[%d]=%f \n", annmb.num_para_ann, parameters[annmb.num_para_ann], n, parameters[n]);
      }
    } else {
      tokens = get_tokens(input);
      parameters[n] = (float) get_double_from_token(tokens[0], __FILE__, __LINE__);
    }
  }
  nep_data.parameters.resize(annmb.num_para);
  nep_data.parameters.copy_from_host(parameters.data());
  update_potential(nep_data.parameters.data(), annmb);


  for (int d = 0; d < annmb.dim; ++d) {
    tokens = get_tokens(input);
    paramb.q_scaler[d] = (float) get_double_from_token(tokens[0], __FILE__, __LINE__);
    // std::cout<<"q_scaler " << d << " " << paramb.q_scaler[d] << std::endl;
  }

  // flexible zbl potential parameters
  if (zbl.flexibled) {
    int num_type_zbl = (paramb.num_types * (paramb.num_types + 1)) / 2;
    for (int d = 0; d < 10 * num_type_zbl; ++d) {
      tokens = get_tokens(input);
      zbl.para[d] = (float) get_double_from_token(tokens[0], __FILE__, __LINE__);
    }
    zbl.num_types = paramb.num_types;
  }
}

NEP::~NEP(void)
{
  pppm_destroy(pppm_data);
}


void NEP::checkMemoryUsage(int sgin) {
  // if (rank_0) {
    size_t free_mem, total_mem;
    cudaError_t error = cudaMemGetInfo(&free_mem, &total_mem);
    if (error != cudaSuccess) {
        std::cerr << "cudaMemGetInfo failed: " << cudaGetErrorString(error) << std::endl;
        return;
    }
    std::cout << device_id << " Free memory: "  << sgin << " " << free_mem / (1024.0 * 1024.0) << " MB" << std::endl;
    // std::cout << device_id << " Total memory: " << sgin << " " << total_mem / (1024.0 * 1024.0) << " MB" << std::endl;
  // }
}

void NEP::rest_nep_data(int input_atom_num) {
  if (atom_nums != input_atom_num) {
    atom_nums = input_atom_num;
    nep_data.NN_radial.resize(atom_nums);
    nep_data.NL_radial.resize(atom_nums * paramb.MN_radial);
    nep_data.NN_angular.resize(atom_nums);
    nep_data.NL_angular.resize(atom_nums * paramb.MN_angular);
    nep_data.potential_per_atom.resize(atom_nums);
    // lmp_data.ilist.resize(atom_nums);
    // lmp_data.numneigh.resize(atom_nums);
    // lmp_data.firstneigh.resize(atom_nums*max_neighbor);
    nep_data.r12.resize(atom_nums * paramb.MN_radial*6);
    nep_data.f12x.resize(atom_nums * paramb.MN_angular);
    nep_data.f12y.resize(atom_nums * paramb.MN_angular);
    nep_data.f12z.resize(atom_nums * paramb.MN_angular);

    nep_data.Fp.resize(atom_nums * annmb.dim);
    nep_data.sum_fxyz.resize(atom_nums * (paramb.n_max_angular + 1) * NUM_OF_ABC);
  
    nep_data.force_per_atom.resize(atom_nums * 3);
    nep_data.virial_per_atom.resize(atom_nums * 9);
    nep_data.total_virial.resize(9);
    lmp_data.type.resize(atom_nums);
    lmp_data.position.resize(atom_nums*3);

    nep_data.cpu_potential_per_atom.resize(atom_nums);
    nep_data.cpu_force_per_atom.resize(atom_nums * 3);
    nep_data.cpu_virial_per_atom.resize(atom_nums * 9);
    nep_data.cpu_total_virial.resize(9);
    if (paramb.charge_mode == 2) {
      nep_data.charge.resize(atom_nums);
      nep_data.charge_derivative.resize(atom_nums * annmb.dim);
      nep_data.bec.resize(atom_nums * 9);
      nep_data.D_real.resize(atom_nums);
      nep_data.num_kpoints.resize(1);
      nep_data.kx.resize(charge_para.num_kpoints_max);
      nep_data.ky.resize(charge_para.num_kpoints_max);
      nep_data.kz.resize(charge_para.num_kpoints_max);
      nep_data.G.resize(charge_para.num_kpoints_max);
      nep_data.S_real.resize(charge_para.num_kpoints_max);
      nep_data.S_imag.resize(charge_para.num_kpoints_max);
      nep_data.cpu_charge.resize(atom_nums);
      nep_data.cpu_bec.resize(atom_nums * 9);
    }
  }
  nep_data.r12.fill(0.0);
  nep_data.potential_per_atom.fill(0.0);
  nep_data.force_per_atom.fill(0.0);
  nep_data.virial_per_atom.fill(0.0);
  nep_data.total_virial.fill(0.0);
  if (paramb.charge_mode == 2) {
    nep_data.charge.fill(0.0f);
    nep_data.charge_derivative.fill(0.0f);
    nep_data.bec.fill(0.0f);
    nep_data.D_real.fill(0.0f);
    nep_data.num_kpoints.fill(0);
    nep_data.S_real.fill(0.0f);
    nep_data.S_imag.fill(0.0f);
  }
}

void NEP::update_potential(float* parameters, ANN& ann)
{
  float* pointer = parameters;
  if (paramb.charge_mode == 2) {
    const int num_outputs = 2;
    for (int t = 0; t < paramb.num_types; ++t) {
      ann.w0[t] = pointer;
      pointer += ann.num_neurons1 * ann.dim;
      ann.b0[t] = pointer;
      pointer += ann.num_neurons1;
      ann.w1[t] = pointer;
      pointer += ann.num_neurons1 * num_outputs;
      if (paramb.version == 5) {
        pointer += 1;
      }
    }
    ann.sqrt_epsilon_inf = pointer;
    pointer += 1;
    ann.b1 = pointer;
    pointer += 1;
    ann.c = pointer;
    return;
  }
  for (int t = 0; t < paramb.num_types; ++t) {
    ann.w0[t] = pointer;
    pointer += ann.num_neurons1 * ann.dim;
    ann.b0[t] = pointer;
    pointer += ann.num_neurons1;
    ann.w1[t] = pointer;
    pointer += ann.num_neurons1;
    if (paramb.version == 5) {
      pointer += 1; // one extra bias for NEP5 stored in ann.w1[t]
    }
  }
  ann.b1 = pointer;
  // pointer += 1;
  pointer += (paramb.version == 4 ? paramb.num_types : 1);
  // if is gpumd nep, copy the last bais as multi biases
  ann.c = pointer;
}

void NEP::inference(
  int N, //atom nums
  int* itype_cpu, //atoms' type,the len is [n_all]
  double* box_cpu, // [xx, yx, zx, xy, yy, zy, xz, yz, zz]
  double* position_cpu, // postion of atoms x, [n_all * 3]
  const char* kspace_method,
  double total_charge
  ) {
  int BLOCK_SIZE = 64;
  int grid_size = (N- 1) / BLOCK_SIZE + 1;

  rest_nep_data(N);

  lmp_data.type.copy_from_host(itype_cpu);
  lmp_data.position.copy_from_host(position_cpu);

  // std::vector<double> tmp_posiion(N*3);
  // lmp_data.position.copy_to_host(tmp_posiion.data());
  // for (int i = 0; i < N; i++) {
  //   printf("p[%d]= %f %f %f \n",i, tmp_posiion[i], tmp_posiion[N+i], tmp_posiion[2*N+i]);
  // }
  box.cpu_h[0] = box_cpu[0]; 
  box.cpu_h[1] = box_cpu[1]; 
  box.cpu_h[2] = box_cpu[2]; 
  box.cpu_h[3] = box_cpu[3]; 
  box.cpu_h[4] = box_cpu[4]; 
  box.cpu_h[5] = box_cpu[5]; 
  box.cpu_h[6] = box_cpu[6]; 
  box.cpu_h[7] = box_cpu[7]; 
  box.cpu_h[8] = box_cpu[8]; 
  box.triclinic = (box.cpu_h[1] != 0.0 || box.cpu_h[2] != 0.0 || box.cpu_h[3] != 0.0 ||
                   box.cpu_h[5] != 0.0 || box.cpu_h[6] != 0.0 || box.cpu_h[7] != 0.0) ? 1 : 0;
  box.get_inverse();
  
  get_expanded_box(paramb.rc_radial, box, ebox);
  int size_x12 = atom_nums * paramb.MN_radial;
  int N1 = 0;
  find_neighbor<<<grid_size, BLOCK_SIZE>>>(
    paramb,
    N,
    N1,
    box,
    ebox,
    lmp_data.position.data(),
    lmp_data.position.data() + N,
    lmp_data.position.data() + N * 2,
    nep_data.NN_radial.data(),
    nep_data.NL_radial.data(),
    nep_data.NN_angular.data(),
    nep_data.NL_angular.data(),
    nep_data.r12.data(),
    nep_data.r12.data() + size_x12,
    nep_data.r12.data() + size_x12 * 2,
    nep_data.r12.data() + size_x12 * 3,
    nep_data.r12.data() + size_x12 * 4,
    nep_data.r12.data() + size_x12 * 5
    );
  CUDA_CHECK_KERNEL

  // gpu_sort_neighbor_list<<<N, paramb.MN_radial, paramb.MN_radial * sizeof(int)>>>(
  //   N, nep_data.NN_radial.data(), nep_data.NL_radial.data());
  // CUDA_CHECK_KERNEL

  // gpu_sort_neighbor_list<<<N, paramb.MN_angular, paramb.MN_angular * sizeof(int)>>>(
  //   N, nep_data.NN_angular.data(), nep_data.NL_angular.data());
  // CUDA_CHECK_KERNEL

  // print the neighbor list
  // std::vector<int> tmp_NN_radial(N);
  // std::vector<int> tmp_NL_radial(N*paramb.MN_radial);
  // nep_data.NN_radial.copy_to_host(tmp_NN_radial.data());
  // nep_data.NL_radial.copy_to_host(tmp_NL_radial.data());
  // for (int i=0; i < N; i++){
  //   printf("atom %d neighbors %d, neighbor list is: \n", i, tmp_NN_radial[i]);
  //   for (int j=0; j < tmp_NN_radial[i]; j++){
  //     printf("%d ",tmp_NL_radial[j*N+i]);
  //   }
  //   printf("\n");
  // }

  if (paramb.charge_mode == 2) {
    find_descriptor_charge2<<<grid_size, BLOCK_SIZE>>>(
      paramb,
      annmb,
      N,
      N1,
      nep_data.NN_radial.data(),
      nep_data.NL_radial.data(),
      nep_data.NN_angular.data(),
      nep_data.NL_angular.data(),
      lmp_data.type.data(),
      nep_data.r12.data(),
      nep_data.r12.data() + size_x12,
      nep_data.r12.data() + size_x12 * 2,
      nep_data.r12.data() + size_x12 * 3,
      nep_data.r12.data() + size_x12 * 4,
      nep_data.r12.data() + size_x12 * 5,
      nep_data.potential_per_atom.data(),
      nep_data.Fp.data(),
      nep_data.charge.data(),
      nep_data.charge_derivative.data(),
      nep_data.sum_fxyz.data());
    CUDA_CHECK_KERNEL

    shift_total_charge<<<1, 1024>>>(N, nep_data.charge.data(), static_cast<float>(total_charge));
    CUDA_CHECK_KERNEL

    find_bec_diagonal<<<grid_size, BLOCK_SIZE>>>(
      N,
      nep_data.charge.data(),
      nep_data.bec.data());
    CUDA_CHECK_KERNEL

    find_bec_radial<<<grid_size, BLOCK_SIZE>>>(
      paramb,
      annmb,
      N,
      N1,
      nep_data.NN_radial.data(),
      nep_data.NL_radial.data(),
      lmp_data.type.data(),
      nep_data.r12.data(),
      nep_data.r12.data() + size_x12,
      nep_data.r12.data() + size_x12 * 2,
      nep_data.charge_derivative.data(),
      nep_data.bec.data());
    CUDA_CHECK_KERNEL

    find_bec_angular<<<grid_size, BLOCK_SIZE>>>(
      paramb,
      annmb,
      N,
      N1,
      nep_data.NN_angular.data(),
      nep_data.NL_angular.data(),
      lmp_data.type.data(),
      nep_data.r12.data() + size_x12 * 3,
      nep_data.r12.data() + size_x12 * 4,
      nep_data.r12.data() + size_x12 * 5,
      nep_data.charge_derivative.data(),
      nep_data.sum_fxyz.data(),
      nep_data.bec.data());
    CUDA_CHECK_KERNEL

    scale_bec<<<grid_size, BLOCK_SIZE>>>(
      N,
      annmb.sqrt_epsilon_inf,
      nep_data.bec.data());
    CUDA_CHECK_KERNEL

    const std::string kspace = (kspace_method == nullptr) ? "ewald" : std::string(kspace_method);
    if (kspace == "pppm") {
      pppm_find_force_charge2(
        pppm_data,
        N,
        N1,
        N,
        charge_para.alpha,
        charge_para.alpha_factor,
        box,
        nep_data.charge,
        lmp_data.position,
        nep_data.D_real,
        nep_data.force_per_atom,
        nep_data.virial_per_atom,
        nep_data.total_virial,
        nep_data.potential_per_atom);
    } else if (kspace == "ewald") {
      ewald_find_force_charge2(
        N,
        BLOCK_SIZE,
        grid_size,
        charge_para.num_kpoints_max,
        charge_para.alpha,
        charge_para.alpha_factor,
        box,
        nep_data.charge,
        lmp_data.position,
        nep_data.num_kpoints,
        nep_data.kx,
        nep_data.ky,
        nep_data.kz,
        nep_data.G,
        nep_data.S_real,
        nep_data.S_imag,
        nep_data.D_real,
        nep_data.force_per_atom,
        nep_data.virial_per_atom,
        nep_data.total_virial,
        nep_data.potential_per_atom);
    } else {
      std::cout << "kspace_method must be ewald or pppm, got " << kspace << std::endl;
      exit(1);
    }

    zero_mean_D_real_charge2<<<1, 1024>>>(N, nep_data.D_real.data());
    CUDA_CHECK_KERNEL
  } else {
    find_descriptor<<<grid_size, BLOCK_SIZE>>>(
      paramb,
      annmb,
      box,
      ebox,
      N,
      N1,
      nep_data.NN_radial.data(),
      nep_data.NL_radial.data(),
      nep_data.NN_angular.data(),
      nep_data.NL_angular.data(),
      lmp_data.type.data(),
      nep_data.r12.data(),
      nep_data.r12.data() + size_x12,
      nep_data.r12.data() + size_x12 * 2,
      nep_data.r12.data() + size_x12 * 3,
      nep_data.r12.data() + size_x12 * 4,
      nep_data.r12.data() + size_x12 * 5,
      nep_data.potential_per_atom.data(),
      nep_data.Fp.data(),
      nep_data.virial_per_atom.data(),
      nep_data.sum_fxyz.data()
    );
    CUDA_CHECK_KERNEL
  }
  // cudaDeviceSynchronize();
  // nep_data.potential_per_atom.copy_to_host(cpu_potential_per_atom);
  // for (int ii = 0; ii < N; ii++) {
  //   printf("before zbl ei[%d]=%f\n", ii, cpu_potential_per_atom[ii]);
  // }
  
  // // bool is_dipole = paramb.model_type == 1;
  find_force_radial<<<grid_size, BLOCK_SIZE>>>(
    paramb,
    annmb,
    N,
    N1, 
    nep_data.NN_radial.data(),
    nep_data.NL_radial.data(),
    lmp_data.type.data(),
    nep_data.r12.data(),
    nep_data.r12.data() + size_x12,
    nep_data.r12.data() + size_x12 * 2,
    nep_data.Fp.data(),
    nep_data.charge_derivative.data(),
    nep_data.D_real.data(),
    nep_data.force_per_atom.data(),
    nep_data.force_per_atom.data() + N,
    nep_data.force_per_atom.data() + N * 2,
    nep_data.virial_per_atom.data(),
    nep_data.total_virial.data()
  );
  CUDA_CHECK_KERNEL
  // cudaDeviceSynchronize();
  // nep_data.potential_per_atom.copy_to_host(cpu_potential_per_atom);
  // std::vector<double> cpu_force_per_atom(N*3);
  // nep_data.force_per_atom.copy_to_host(cpu_force_per_atom.data());
  // for (int ii = 0; ii < N; ii++) {
  //   printf("radial force[%d]=%f %f %f\n", ii, 
  //     cpu_force_per_atom[ii], cpu_force_per_atom[ii+ N], cpu_force_per_atom[ii+ N*2]);
  // }
  // std::vector<double> tmp_viral(N * 9);
  // nep_data.virial_per_atom.copy_to_host(tmp_viral.data());
  // for (int ii = 0; ii < N; ii++) {
  //     printf("radial force virial[%d] = [%f %f %f]\n", 
  //     ii, tmp_viral[ii], tmp_viral[ii + N], tmp_viral[ii + N * 2]);
  // }

  find_force_angular<<<grid_size, BLOCK_SIZE>>>(
    paramb,
    annmb,
    N,
    N1,
    nep_data.NN_angular.data(),
    nep_data.NL_angular.data(),
    lmp_data.type.data(),
    nep_data.r12.data() + size_x12 * 3,
    nep_data.r12.data() + size_x12 * 4,
    nep_data.r12.data() + size_x12 * 5,
    nep_data.Fp.data(),
    nep_data.charge_derivative.data(),
    nep_data.D_real.data(),
    nep_data.sum_fxyz.data(),
    nep_data.force_per_atom.data(),
    nep_data.force_per_atom.data() + N,
    nep_data.force_per_atom.data() + N * 2,
    nep_data.virial_per_atom.data(),
    nep_data.total_virial.data());
  CUDA_CHECK_KERNEL

  if (zbl.enabled) {
    find_force_ZBL<<<grid_size, BLOCK_SIZE>>>(
      paramb,
      zbl,
      N,
      N1,
      nep_data.NN_angular.data(),
      nep_data.NL_angular.data(),
      lmp_data.type.data(),
      nep_data.r12.data() + size_x12 * 3,
      nep_data.r12.data() + size_x12 * 4,
      nep_data.r12.data() + size_x12 * 5,
      nep_data.force_per_atom.data(),
      nep_data.force_per_atom.data() + N,
      nep_data.force_per_atom.data() + N * 2, 
      nep_data.virial_per_atom.data(),
      nep_data.total_virial.data(), 
      nep_data.potential_per_atom.data());
    CUDA_CHECK_KERNEL
  }

  // checkMemoryUsage(4);
  // cudaDeviceSynchronize();
  // nep_data.force_per_atom.copy_to_host(cpu_force_per_atom);
  // for (int ii = 0; ii < N; ii++) {
  //   printf("zbl force[%d]=%f %f %f\n", 
  //   ii, cpu_force_per_atom[ii], cpu_force_per_atom[ii+N], cpu_force_per_atom[ii+N*2]);
  // }

  // grid_size = (N - 1) / BLOCK_SIZE + 1;
  // calculate_total_virial<<<grid_size, BLOCK_SIZE>>>(
  //                           nep_data.virial_per_atom.data(), 
  //                           nep_data.total_virial.data(), 
  //                           N);

  // CUDA_CHECK_KERNEL
  // cudaDeviceSynchronize();

  nep_data.total_virial.copy_to_host(nep_data.cpu_total_virial.data());
  double temp[9] = {nep_data.cpu_total_virial[0], nep_data.cpu_total_virial[3], nep_data.cpu_total_virial[4],
                  nep_data.cpu_total_virial[3], nep_data.cpu_total_virial[1], nep_data.cpu_total_virial[5],
                  nep_data.cpu_total_virial[4], nep_data.cpu_total_virial[5], nep_data.cpu_total_virial[2]};
  for (int i = 0; i < 9; ++i) {
      nep_data.cpu_total_virial[i] = temp[i];
  }

  nep_data.potential_per_atom.copy_to_host(nep_data.cpu_potential_per_atom.data());
  nep_data.force_per_atom.copy_to_host(nep_data.cpu_force_per_atom.data());
  if (paramb.charge_mode == 2) {
    nep_data.charge.copy_to_host(nep_data.cpu_charge.data());
    nep_data.bec.copy_to_host(nep_data.cpu_bec.data());
  }

  // for (int ii = 0; ii < N; ii++) {
  //   if (1) {
  //     printf("end ei[%d]=%f\n", ii, nep_data.cpu_potential_per_atom[ii]);
  //   }
  // }

  // for (int ii = 0; ii < N; ii++) {
  //   if (1) {
  //     printf("m_cpu_force[%d] = [%f %f %f] m_cpu_virial[%d] = [%f %f %f]\n", 
  //     ii, cpu_force_per_atom[ii], cpu_force_per_atom[ii + N], cpu_force_per_atom[ii + N * 2],
  //     ii, tmp_viral[ii], tmp_viral[ii + N], tmp_viral[ii + N * 2]);
  //   }
  // }

  // for (int ii = 0; ii < 6; ii++) {
  //   printf("cpu_total_virial[%d]=%f\n", ii, cpu_total_virial[ii]);
  // }

  // for (int ii = 0; ii < N; ii++) {
  //   if (ii % 1 == 0) {
  //     printf("cpu_ei[%d]=%f cpu_force[%d] = [%f %f %f]\n", ii, cpu_potential_per_atom[ii], ii, cpu_force_per_atom[ii], cpu_force_per_atom[ii + N], cpu_force_per_atom[ii + n_all * 2]);
  //   }
  // }
}
