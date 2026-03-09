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

#include "nepkk.cuh"
#include "nep_kernal_function.cuh"
#include "../utilities/common.cuh"
#include "../utilities/error.cuh"
#include "../utilities/nep_utilities.cuh"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cub/cub.cuh>

const size_t SHAREMEM_32 = 32768; //32KB
// const size_t SHAREMEM_48 = 49152;
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

NEPKK::NEPKK()
{
}

void NEPKK::read_neptxt(const char* file_potential, const bool is_rank_0, const int rank_id, const int device_id, const int ff_id)
{
  int neplinenums = countNonEmptyLines(file_potential);
  rank_0 = is_rank_0;
  rank   = rank_id;
  device = device_id;
  ff_index = ff_id;
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
  if (rank_0) {
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
    if (rank_0) {
      printf("    type %d (%s).\n", n, tokens[2 + n].c_str());
    }
  }

// zbl 0.7 1.4
  if (zbl.enabled) {
    tokens = get_tokens(input);
    if (tokens.size() != 3) {
      std::cout << "This line should be zbl rc_inner rc_outer." << std::endl;
      exit(1);
    }
    zbl.rc_inner = get_float_from_token(tokens[1], __FILE__, __LINE__);
    zbl.rc_outer = get_float_from_token(tokens[2], __FILE__, __LINE__);
    if (zbl.rc_inner == 0 && zbl.rc_outer == 0) {
      zbl.flexibled = true;
      if (rank_0) printf("    has the flexible ZBL potential\n");
    } else {
      if (rank_0) printf(\
        "    has the universal ZBL with inner cutoff %g A and outer cutoff %g A.\n",\
        zbl.rc_inner, zbl.rc_outer);
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
  if (rank_0) {
    printf("    radial cutoff = %g A.\n", paramb.rc_radial);
    printf("    angular cutoff = %g A.\n", paramb.rc_angular);
  }
  paramb.rc_radial_square = paramb.rc_radial * paramb.rc_radial;
  paramb.rc_angular_square = paramb.rc_angular * paramb.rc_angular;
  
  if (tokens.size() == 5) {
    int MN_radial = get_int_from_token(tokens[3], __FILE__, __LINE__);
    int MN_angular = get_int_from_token(tokens[4], __FILE__, __LINE__);
    if (rank_0) {
      printf("    MN_radial = %d.\n", MN_radial);
      printf("    MN_angular = %d.\n", MN_angular);
    }
    paramb.MN_radial = int(ceil(MN_radial * 1.25));
    paramb.MN_angular = int(ceil(MN_angular * 1.25));
    if (rank_0) {
      printf("    enlarged MN_radial = %d.\n", paramb.MN_radial);
      printf("    enlarged MN_angular = %d.\n", paramb.MN_angular);
    }
  } else {
    if (paramb.rc_radial > paramb.rc_angular) {
      paramb.MN_radial = 500;
      paramb.MN_angular = 200;
    } else {
      paramb.MN_radial = 500;
      paramb.MN_angular = 500;
    }
      printf("    default MN_radial = %d.\n", paramb.MN_radial);
      printf("    default MN_angular = %d.\n", paramb.MN_angular);
  }

  // n_max 10 8
  tokens = get_tokens(input);
  if (tokens.size() != 3) {
    std::cout << "This line should be n_max n_max_radial n_max_angular." << std::endl;
    exit(1);
  }
  paramb.n_max_radial = get_int_from_token(tokens[1], __FILE__, __LINE__);
  paramb.n_max_angular = get_int_from_token(tokens[2], __FILE__, __LINE__);
  if (rank_0) { 
    printf("    n_max_radial = %d.\n", paramb.n_max_radial);
    printf("    n_max_angular = %d.\n", paramb.n_max_angular);
  }
  paramb.n_max_radial_plus1 = paramb.n_max_radial + 1;
  paramb.n_max_angular_plus1 = paramb.n_max_angular + 1;

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
    if (rank_0) {
      printf("    basis_size_radial = %d.\n", paramb.basis_size_radial);
      printf("    basis_size_angular = %d.\n", paramb.basis_size_angular);
    }
  paramb.basis_size_radial_plus1 = paramb.basis_size_radial + 1;
  paramb.basis_size_angular_plus1 = paramb.basis_size_angular + 1;
  }

  // l_max
  tokens = get_tokens(input);
  if (tokens.size() != 4) {
    std::cout << "This line should be Lmax 4 2 1 or 4 2 0 or 4 0 0, representing l_max_3body, l_max_4body, and l_max_5body respectively." << std::endl;
    exit(1);
  }

  paramb.L_max = get_int_from_token(tokens[1], __FILE__, __LINE__);
  if (rank_0) {
    if (paramb.L_max == 4){
      printf("    l_max_3body = %d.\n", paramb.L_max);
    } else {
      std::cout << "This value l_max_3body should be 4." << std::endl;
      exit(1);
    }
  }
  paramb.num_L = paramb.L_max;

  int L_max_4body = get_int_from_token(tokens[2], __FILE__, __LINE__);
  int L_max_5body = get_int_from_token(tokens[3], __FILE__, __LINE__);
  if (rank_0) {
    if (L_max_4body != 2 && L_max_4body != 0) {
      std::cout << "This value L_max_4body should be 2 or 0" << std::endl;
      exit(1);    
    }
    if (L_max_5body != 1 && L_max_5body != 0) {
      std::cout << "This value L_max_5body should be 1 or 0" << std::endl;
      exit(1);    
    }
    printf("    l_max_4body = %d.\n", L_max_4body);
    printf("    l_max_5body = %d.\n", L_max_5body);
  }
  if (L_max_4body == 2) {
    paramb.num_L += 1;
  }
  if (L_max_5body == 1) {
    paramb.num_L += 1;
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
  if (rank_0) {
    printf("    ANN = %d-%d-1.\n", annmb.dim, annmb.num_neurons1);
  }
  // calculated parameters:
  paramb.rcinv_radial = 1.0f / paramb.rc_radial;
  paramb.rcinv_angular = 1.0f / paramb.rc_angular;
  paramb.num_types_sq = paramb.num_types * paramb.num_types;

  annmb.num_c2   = paramb.num_types_sq * (paramb.n_max_radial + 1) * (paramb.basis_size_radial + 1);
  annmb.num_c3   = paramb.num_types_sq * (paramb.n_max_angular + 1) * (paramb.basis_size_angular + 1);
  
  if (paramb.version == 4) {
    annmb.num_para_ann = (annmb.dim + 2) * annmb.num_neurons1 * paramb.num_types;
  } else{//5
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

  if (paramb.num_types == 1) {
    is_gpumd_nep = false;
  } else if (paramb.version == 4) {
    if (neplinenums  == (tmp + 1)) {
      is_gpumd_nep = true;
      if (rank_0) printf("   the input nep4 potential file is from GPUMD.\n");
    } else if (neplinenums  == (tmp + paramb.num_types)) {
      if (rank_0) printf("   the input nep4 potential file is from MatPL.\n");
    } else {
      printf("    parameter parsing error, the number of nep parameters [MatPL %d, GPUMD %d] does not match the text lines %d.\n", tmp+paramb.num_types, (tmp+1), neplinenums);
    exit(1);
    }
  }

  if (paramb.version == 4 ){
    annmb.num_para = annmb.num_para_ann + paramb.num_types;
  } else {
    annmb.num_para = annmb.num_para_ann;
  }
  
  if (rank_0) {
    printf("    number of neural network parameters = %d.\n", is_gpumd_nep == false ? annmb.num_para : annmb.num_para-paramb.num_types+1);
  }
  int num_para_descriptor =annmb.num_c2 + annmb.num_c3;
    // paramb.num_types_sq * ((paramb.n_max_radial + 1) * (paramb.basis_size_radial + 1) +
    //                        (paramb.n_max_angular + 1) * (paramb.basis_size_angular + 1));
  if (rank_0) {
    printf("    number of descriptor parameters = %d.\n", num_para_descriptor);
  }
  annmb.num_para += num_para_descriptor;
  if (rank_0) {
    printf("    total number of parameters = %d.\n", is_gpumd_nep == false ? annmb.num_para : annmb.num_para-paramb.num_types+1);
  }
  paramb.num_c_radial =
    paramb.num_types_sq * (paramb.n_max_radial + 1) * (paramb.basis_size_radial + 1);

  // NN and descriptor parameters
  std::vector<float> parameters(annmb.num_para);
  for (int n = 0; n < annmb.num_para; ++n) {
    if (is_gpumd_nep == true && (n >= annmb.num_para_ann + 1) && (n < annmb.num_para_ann + paramb.num_types)) {
      parameters[n] = parameters[annmb.num_para_ann];
      if (rank_0) {
        printf("    copy the last bias parameters[%d]=%f to parameters[%d]=%f \n", annmb.num_para_ann, parameters[annmb.num_para_ann], n, parameters[n]);
      }
    } else {
      tokens = get_tokens(input);
      parameters[n] = get_float_from_token(tokens[0], __FILE__, __LINE__);
    }
  }
  nep_data.parameters.resize(annmb.num_para);
  nep_data.parameters.copy_from_host(parameters.data());
  update_potential(nep_data.parameters.data(), annmb);

  for (int d = 0; d < annmb.dim; ++d) {
    tokens = get_tokens(input);
    paramb.q_scaler[d] = get_float_from_token(tokens[0], __FILE__, __LINE__);
    // std::cout<<"q_scaler " << d << " " << paramb.q_scaler[d] << std::endl;
  }

  cudaMemcpyToSymbol(Q_SCALER, 
          paramb.q_scaler,                          // 设备上的 c 数组指针（或从 host 拷贝）
          annmb.dim * sizeof(float),
          0,                            // offset = 0
          cudaMemcpyHostToDevice);

  // flexible zbl potential parameters
  if (zbl.flexibled) {
    int num_type_zbl = (paramb.num_types * (paramb.num_types + 1)) / 2;
    for (int d = 0; d < 10 * num_type_zbl; ++d) {
      tokens = get_tokens(input);
      zbl.para[d] = get_float_from_token(tokens[0], __FILE__, __LINE__);
    }
    zbl.num_types = paramb.num_types;
  }

  USE_SHAREMEM_C2 = SHAREMEM_32 > annmb.num_c2 * 4; //在c数组的容量小于32KB时开启
  USE_SHAREMEM_C3 = SHAREMEM_32 > annmb.num_c3 * 4;
  printf("========= USE_SHAREMEM_C2 %d USE_SHAREMEM_C3 %d ==========\n", USE_SHAREMEM_C2, USE_SHAREMEM_C3);
}

NEPKK::~NEPKK(void)
{
  // nothing
}

void NEPKK::checkMemoryUsage(int sgin) {
  // if (rank_0) {
    size_t free_mem, total_mem;
    cudaError_t error = cudaMemGetInfo(&free_mem, &total_mem);
    if (error != cudaSuccess) {
        std::cerr << "cudaMemGetInfo failed: " << cudaGetErrorString(error) << std::endl;
        return;
    }
    std::cout << device << " Free memory: "  << sgin << " " << free_mem / (1024.0 * 1024.0) << " MB" << std::endl;
    // std::cout << device << " Total memory: " << sgin << " " << total_mem / (1024.0 * 1024.0) << " MB" << std::endl;
  // }
}

// 计算支持的最大原子数
int NEPKK::calculateMaxAtoms(int MN_radial, int MN_angular, int lmp_num_neigh, int ANNNB_DIM, int nlocal, int nprocs_total) {
    // 数据类型大小（字节）
    const int SIZE_INT = 4;
    const int SIZE_FLOAT = 4;
    const int SIZE_DOUBLE = 8;
    const int NUM_OF_ABC = 24;
    const double SAFETY_FACTOR = 1.0;
    
    // 获取GPU显存
    cudaDeviceProp prop;
    int device_count = 0;
    
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        return -1;
    }
    
    err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        return -1;
    }
    
    size_t total_gpu_memory = prop.totalGlobalMem;
    
    // 计算单个原子所需的显存
    size_t memory_per_atom = 0;
    size_t tmp_nep_neigh = 0; 
    // LMP_Data 数组
    memory_per_atom += SIZE_INT;  // nep_ilist
    memory_per_atom += SIZE_INT;  // nep_type
    
    // NEPKK_Data 数组    
    tmp_nep_neigh += SIZE_INT;  // nep_NN_radial
    tmp_nep_neigh += MN_radial * SIZE_INT;  // nep_NL_radial
    tmp_nep_neigh += SIZE_INT;  // nep_NN_angular
    tmp_nep_neigh += MN_angular * SIZE_INT;  // nep_NL_angular
    memory_per_atom += tmp_nep_neigh;
    memory_per_atom += MN_angular * SIZE_FLOAT;  // nep_f12x
    memory_per_atom += MN_angular * SIZE_FLOAT;  // nep_f12y  
    memory_per_atom += MN_angular * SIZE_FLOAT;  // nep_f12z
    
    memory_per_atom += SIZE_DOUBLE;  // nep_potential_per_atom
    
    memory_per_atom += ANNNB_DIM * SIZE_FLOAT;  // nep_Fp
    memory_per_atom += (4 + 1) * NUM_OF_ABC * SIZE_FLOAT;  // nep_sum_fxyz
    
    // 考虑max_nall = 1.2 * max_nlocal
    memory_per_atom += 1.2 * 3 * SIZE_DOUBLE;  // nep_force_per_atom
    memory_per_atom += 1.2 * 6 * SIZE_DOUBLE;  // nep_virial_per_atom
    
    // lammps部分 (max_lmp_nlocal = max_lmp_nall = 1.2 * max_nlocal)
    memory_per_atom += 1.2 * SIZE_DOUBLE;  // lmp_potential_per_atom
    memory_per_atom += 1.2 * 3 * SIZE_DOUBLE;  // lmp_force_per_atom
    memory_per_atom += 1.2 * 6 * SIZE_DOUBLE;  // lmp_virial_per_atom
    if (nprocs_total > 1) {
      memory_per_atom += nprocs_total * 1.2 * SIZE_DOUBLE;  // glob_lmp_potential_per_atom
      memory_per_atom += nprocs_total * 1.2 * 3 * SIZE_DOUBLE;  // glob_lmp_force_per_atom
      memory_per_atom += nprocs_total * 1.2 * 6 * SIZE_DOUBLE;  // glob_lmp_virial_per_atom
    }
    // 邻居列表相关 (lmp_max_neigh = max_nlocal)
    memory_per_atom += SIZE_INT;  // lmp_num_neigh
    memory_per_atom += lmp_num_neigh * SIZE_INT;  // lmp_max_neigh
    memory_per_atom += SIZE_INT;  // lmp_ilist
    memory_per_atom += SIZE_INT;  // lmp_type
    // 多核多卡并行下的kokkos构建近邻表缓存
    if (nprocs_total > 1) {
      memory_per_atom += SIZE_INT;  // lmp_num_neigh
      memory_per_atom += lmp_num_neigh * SIZE_INT;  // lmp_max_neigh
      memory_per_atom += SIZE_INT;  // lmp_ilist
      memory_per_atom += SIZE_INT;  // lmp_type
    }

    // 计算最大原子数
    size_t available_memory = total_gpu_memory * SAFETY_FACTOR;
    int max_atoms = static_cast<int>(available_memory / memory_per_atom);
    if (nprocs_total == 1) {
      if (nlocal > max_atoms) {
        printf("Warning! Nlocal(%d, nums of local atoms) > Nmax(%d, the max number of atoms supported in single GPU). There will be a risk of memory overflow, it is recommended to prioritize adding GPU devices or scaling the system.\n", nlocal, max_atoms);
      } else {
        printf("The max number of atoms supported in single GPU ≈ %d\n", max_atoms);
      }
    } else {
      if (nlocal > max_atoms) {
        printf("Warning! Nlocal(%d, nums of local atoms) > Nmax(%d, the max number of atoms supported in single GPU in parallel with multiple GPUs). ", nlocal, max_atoms);
        memory_per_atom -= tmp_nep_neigh;
        int max_atoms_large = static_cast<int>(available_memory / memory_per_atom);
        printf("The nep 2b and 3b neighborlist free operation will be enabled to save memory(Nmax from %d to %d), but will result in a performance degradation of about 7\\% . However, there is still a risk of memory overflow, it is recommended to prioritize adding GPU devices or scaling the system.\n", max_atoms, max_atoms_large);
        max_atoms = max_atoms_large;
        free_neigh = true;
      } else {
        printf("The max number of atoms supported in single GPU ≈ %d\n", max_atoms);
      }
    }
    return max_atoms;
}

void NEPKK::free_nep_data() {
    nep_data.NN_radial.free();
    nep_data.NL_radial.free();
    nep_data.NN_angular.free();
    nep_data.NL_angular.free();
}

void NEPKK::reset_nep_data(int inum, int nlocal, int nall, int vflag_either) {
  // printf(" ====rand %d device %d rest data maxinum %d maxnlocal %d maxnall %d curinum %d curlocal %d curnall %d device %d rank %d=====\n", \
        rank, device, max_inum, max_nlocal, max_nall, inum, nlocal, nall, device, rank);
  if (allocate_once==0) {
    nep_data.potential_all.resize(1);
    nep_data.total_virial.resize(6);
    allocate_once = 1;
  }

  if (max_inum < inum) {
    max_inum = inum;
  }

  if (max_nlocal < nlocal) { 
    max_nlocal = nlocal;
    lmp_data.ilist.resize(max_nlocal);
    nep_data.f12x.resize(max_nlocal * paramb.MN_angular);
    nep_data.f12y.resize(max_nlocal * paramb.MN_angular);
    nep_data.f12z.resize(max_nlocal * paramb.MN_angular);

    nep_data.potential_per_atom.resize(max_nlocal);
    
    nep_data.Fp.resize(max_nlocal * annmb.dim);//复用，存储特征值，之后存储能量对特征值导数
    nep_data.sum_fxyz.resize(max_nlocal * (paramb.n_max_angular + 1) * NUM_OF_ABC);

    // nep_data.NN_radial.resize(max_nlocal, 0);
    // nep_data.NL_radial.resize(max_nlocal * paramb.MN_radial, 0);
    nep_data.NN_angular.resize(max_nlocal, 0);
    nep_data.NL_angular.resize(max_nlocal * paramb.MN_angular, 0);
  }

  if (max_nall < nall) {
    max_nall = nall;
    lmp_data.position.resize(max_nall*3);
    lmp_data.type.resize(max_nall);
    // nep_data.force_per_atom.resize(max_nall * 3);
    // if (vflag_either) nep_data.virial_per_atom.resize(max_nall * 6);
  }

  // nep_data.NN_radial.fill(0);
  // nep_data.NL_radial.fill(0);
  lmp_data.ilist.fill(-1);
  lmp_data.type.fill(-1);
  nep_data.NN_angular.fill(0);
  nep_data.NL_angular.fill(0);

  // nep_data.f12x.fill(0);
  // nep_data.f12y.fill(0);
  // nep_data.f12z.fill(0);
  nep_data.Fp.fill(0); //需要临时存放特征值
  nep_data.sum_fxyz.fill(0); // 直接保存，需要置零

  nep_data.potential_per_atom.fill(0.0);
  // nep_data.force_per_atom.fill(0.0); 
  if (vflag_either) {
    // nep_data.virial_per_atom.fill(0.0);
    nep_data.total_virial.fill(0.0);
  }
  nep_data.potential_all.fill(0.0);
}

void NEPKK::update_potential(float* parameters, ANN& ann)
{
  float* pointer = parameters;
  for (int t = 0; t < paramb.num_types; ++t) {
    if (t > 0 && paramb.version == 3) { // Use the same set of NN parameters for NEP2 and NEPKK_CPU
      pointer -= (ann.dim + 2) * ann.num_neurons1;
    }
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
  convert_C(ann.c, paramb.num_types, paramb.n_max_radial, paramb.basis_size_radial);
  convert_C(ann.c+ann.num_c2, paramb.num_types, paramb.n_max_angular, paramb.basis_size_angular);
  // for (int t = 0; t < paramb.num_types; ++t) {
  //   ann.c_2[t]
  // }
}

void NEPKK::convert_C(float* d_c, int NtypeI, int Nmax, int Nbase){
  //c的维度[Nmax+1,Nbase+1,Ntype,Ntype] to [Ntype,Ntype,Nmax+1,Nbase+1]
  int total_elements = (Nmax + 1) * (Nbase + 1) * NtypeI * NtypeI;
  GPU_Vector<float> copy_c(total_elements);
  int threads_per_block = 256;
  int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;
  convert_c_dim<<<blocks_per_grid, threads_per_block>>>(
        d_c, copy_c.data(), NtypeI, Nmax, Nbase);
  copy_c.copy_to_device(d_c, total_elements);
}

void NEPKK::set_atom_type_map(int type_nums, const int* type_list){
  atom_type_map.resize(type_nums);
  atom_type_map.copy_from_host(type_list);
}

// small box possibly used for active learning:
void NEPKK::compute(
    int eflag_global,       // energy_total flag
    int eflag_atom,         // ei peratom flag
    int vflag_either,       // = bool(vfalg_atom or vflag_global)
    int vflag_global,       // virial_global flag
    int vflag_atom,         // virial peratom flag
    int nall,               // nlocal + nghost
    int inum, 
    int nlocal,
    int max_J_neigh,          // row nums of firstneigh
    int max_I_neigh,          // col nums of firstneigh -> j = firstneigh[i + num_neighs * jj] & NEIGHMASK
    int* itype,                 //atoms' type,the len is [n_all]
    int* ilist,                 // atom i list
    int* numneigh,              // the neighbor nums of each i, [inum]
    int* firstneigh,            // the neighbor list of each i, [inum * NM]
    double* position,           // postion of atoms x, [n_all * 3]
    double* potential_per_atom_lmp,  // the output of ei
    double* potential_per_atom_copy, // the output of ei
    double* force_per_atom_lmp,      // the output of force
    double* force_per_atom_copy,
    double* virial_per_atom,
    double* h_etot_virial_global // len=7: etot + 6 virials
) {
  int BLOCK_SIZE256 = 256;
  // int BLOCK_SIZE128 = 128;
  int BLOCK_SIZE64 = 64;
  int BLOCK_SIZE32 = 32;
  
  double* force_per_atom;
  if (ff_index == 0) {
    force_per_atom = force_per_atom_lmp; // after calc, copy to f_copy
  } else {
    force_per_atom = force_per_atom_copy; 
  }

  reset_nep_data(inum, nlocal, nall, vflag_either);
  doubleTofloat<<<(nall*3 + BLOCK_SIZE256 - 1) / BLOCK_SIZE256, BLOCK_SIZE256>>>(
    lmp_data.position.data(), 
    position, 
    nall*3);
  CUDA_CHECK_KERNEL

  convert_atom_types<<<(nall + BLOCK_SIZE256 - 1) / BLOCK_SIZE256, BLOCK_SIZE256>>>(
    nall,
    inum,
    nlocal,
    ilist,
    itype,
    atom_type_map.data(),
    lmp_data.ilist.data(),
    lmp_data.type.data()
    );
  CUDA_CHECK_KERNEL

  // 不能对lammps的原始近邻表排序，可能对一些功能产生错误。
  //增大到64后，占据多了，66.6%，但是速度变慢了一倍。因为驻留线程多了之后造成了更高的内存带宽压力
  if (USE_SHAREMEM_C2) {//把两体项系数C全部load入共享内存 Ntype*Ntype*(Nmax+1)*(Nbase+1) * 4-float
    size_t shared_mem_size = annmb.num_c2 * sizeof(float);
    calc_2b_descriptor_sharemem<<<(inum - 1) / BLOCK_SIZE32 + 1, BLOCK_SIZE32, shared_mem_size>>>(
      paramb,
      annmb.c,
      inum,
      nlocal,
      device,
      max_I_neigh,
      numneigh,
      firstneigh,
      nep_data.NN_angular.data(),
      nep_data.NL_angular.data(),
      ilist,
      lmp_data.type.data(),
      lmp_data.position.data(),
      nep_data.Fp.data()); //临时存储特征值
    CUDA_CHECK_KERNEL
  } else {
    calc_2b_descriptor<<<(inum - 1) / BLOCK_SIZE32 + 1, BLOCK_SIZE32>>>(
      paramb,
      annmb,
      inum,
      nlocal,
      device,
      max_I_neigh,
      numneigh,
      firstneigh,
      nep_data.NN_angular.data(),
      nep_data.NL_angular.data(),
      ilist,
      lmp_data.type.data(),
      lmp_data.position.data(),
      nep_data.Fp.data()); //临时存储特征值
    CUDA_CHECK_KERNEL    
  }

  // 奇偶排序，行存储，适合于lammps近邻的下标部分有序
  // sort_neighbor_simple_fast_kernel<<<nlocal, 128, paramb.MN_angular * sizeof(int)>>>(
  //   nlocal, paramb.MN_angular, nep_data.NN_angular.data(), nep_data.NL_angular.data());
  // CUDA_CHECK_KERNEL

  gpu_sort_neighbor_list<<<nlocal, paramb.MN_angular, paramb.MN_angular * sizeof(int)>>>(
    nlocal, nep_data.NN_angular.data(), nep_data.NL_angular.data());
  CUDA_CHECK_KERNEL

  if (USE_SHAREMEM_C3) {// 把多体系数项C全部load到shared memory中
    size_t shared_mem_size = annmb.num_c3 * sizeof(float);
    calc_3b_descriptor_sharemem<<<(inum - 1) / BLOCK_SIZE64 + 1, BLOCK_SIZE64, shared_mem_size>>>(
      paramb,
      annmb,
      annmb.c+annmb.num_c2,
      inum,
      nlocal,
      device,
      annmb.dim,
      nep_data.NN_angular.data(),
      nep_data.NL_angular.data(),
      ilist,
      lmp_data.type.data(),
      lmp_data.position.data(),
      nep_data.Fp.data(),
      nep_data.potential_per_atom.data(),
      nep_data.sum_fxyz.data());
    CUDA_CHECK_KERNEL
  } else {
    calc_3b_descriptor<<<(inum - 1) / BLOCK_SIZE32 + 1, BLOCK_SIZE32>>>(
      paramb,
      annmb,
      inum,
      nlocal,
      device,
      annmb.dim,
      nep_data.NN_angular.data(),
      nep_data.NL_angular.data(),
      ilist,
      lmp_data.type.data(),
      lmp_data.position.data(),
      nep_data.Fp.data(),
      nep_data.potential_per_atom.data(),
      nep_data.sum_fxyz.data());
    CUDA_CHECK_KERNEL
  }

  size_t smem_bytes = 3 * sizeof(float) * BLOCK_SIZE64;  // 力分量 fx,fy,fz
  if (vflag_either) {
      smem_bytes += 6 * sizeof(float) * BLOCK_SIZE64;    // 维里 6 个分量
  }
  smem_bytes += paramb.n_max_radial_plus1 * sizeof(float);// n_max_radial_plus1 一定是小于线程块大小的
  // backward_force_2b_perneigh 核函数相比于backward_force_2b 在3090上能获得接近一半的时间减少，但是在4090上反而性能下降
  // 将中心原子的Fp放入共享内存性能几乎没有提升，块内线程处理每个近邻，导致取Fp地址缺乏连续性（在4090由于L2cache 更大，影响更明显）
  // 这部分优化思路：需要把calc3bfeature这里的粒度拆分，一个块处理一个中心原子，然后写Fp可以按照行优先存储（一个行对应一个中心原子的Fp)\
  // 此时不再存在写Fp的地址不连续问题,并且后续的backward force 可以获得收益 (wuxingxing.2026.2.28)
  if (smem_bytes < SHAREMEM_32) {
    backward_force_2b_perneigh<<<inum, BLOCK_SIZE64, smem_bytes>>>(
        vflag_either,
        paramb,
        annmb,
        nall,
        inum,
        nlocal,
        max_I_neigh,
        numneigh,
        firstneigh,
        ilist,
        lmp_data.type.data(),
        lmp_data.position.data(),
        nep_data.Fp.data(),
        force_per_atom,
        virial_per_atom
    );  
  } else {
  // 32 或 64 没什么提升空间-3090
  backward_force_2b<<<(inum - 1) / BLOCK_SIZE64 + 1, BLOCK_SIZE64>>>( 
    vflag_either,
    paramb,
    annmb,
    nall,
    inum,
    nlocal,
    max_I_neigh,
    numneigh,
    firstneigh,
    ilist,
    lmp_data.type.data(),
    lmp_data.position.data(),
    nep_data.Fp.data(),
    force_per_atom,
    virial_per_atom
    );
  }
  int shm_float_count = 3 + paramb.dim_angular + paramb.n_max_angular_plus1 * NUM_OF_ABC;
  shm_float_count += BLOCK_SIZE32 * MAX_NUM_N * 2;// 168个寄存器使用, block 64 会导致共享内存翻倍，驻留block减少
  size_t shared_bytes = shm_float_count * sizeof(float);
  if (shared_bytes < SHAREMEM_32) {
    dim3 grid(inum); // 中心原子数
    dim3 block(BLOCK_SIZE32);
    backward_force_3b_per_atom_sharemem<<<grid, block, shared_bytes>>>(
        paramb,
        annmb,
        inum,
        nlocal,
        nep_data.NN_angular.data(),
        nep_data.NL_angular.data(),
        ilist,
        lmp_data.type.data(),
        lmp_data.position.data(),
        nep_data.Fp.data(),
        nep_data.sum_fxyz.data(),
        nep_data.f12x.data(),
        nep_data.f12y.data(),
        nep_data.f12z.data()
    );
  } else {
    backward_force_3b_dqnl<<<(inum - 1) / BLOCK_SIZE32 + 1, BLOCK_SIZE32>>>(
      paramb,
      annmb,
      inum,
      nlocal,
      nep_data.NN_angular.data(),
      nep_data.NL_angular.data(),
      ilist,
      lmp_data.type.data(),
      lmp_data.position.data(),
      nep_data.Fp.data(),
      nep_data.sum_fxyz.data(),
      nep_data.f12x.data(),
      nep_data.f12y.data(),
      nep_data.f12z.data()
      );
  }
  CUDA_CHECK_KERNEL
  cudaDeviceSynchronize();

  backward_force_3b_merge<<<(inum - 1) / BLOCK_SIZE64 + 1, BLOCK_SIZE64>>>(
    vflag_either,
    nall,
    inum,
    nlocal,
    nep_data.NN_angular.data(),
    nep_data.NL_angular.data(),
    nep_data.f12x.data(),
    nep_data.f12y.data(),
    nep_data.f12z.data(),
    ilist,
    lmp_data.position.data(),
    force_per_atom,
    virial_per_atom);
  CUDA_CHECK_KERNEL
  cudaDeviceSynchronize(); 

  if (zbl.enabled) {
    backward_force_ZBL<<<(inum - 1) / BLOCK_SIZE64 + 1, BLOCK_SIZE64>>>(
      vflag_either,
      zbl,
      nall,
      inum,
      nlocal,
      paramb.MN_angular,
      nep_data.NN_angular.data(),
      nep_data.NL_angular.data(),
      ilist,
      lmp_data.type.data(),
      lmp_data.position.data(),
      force_per_atom,
      virial_per_atom,
      nep_data.potential_per_atom.data());
    CUDA_CHECK_KERNEL
  }

  // calculate virial global
  if (vflag_global) {
    calculate_total_virial<<<(nall - 1) / BLOCK_SIZE64 + 1, BLOCK_SIZE64>>>(
        virial_per_atom, 
        nep_data.total_virial.data(), 
        nall); 
    CUDA_CHECK_KERNEL
    cudaDeviceSynchronize();
    nep_data.total_virial.copy_to_host(h_etot_virial_global+1, 6);
  }

  if (eflag_global) {
    const int threads = 512;  // block 大小
     const int blocks = (nlocal + threads - 1) / threads;
     size_t shared_size = threads * sizeof(double);
     GPU_Vector<double> d_partial(blocks, 0.0);

     block_reduce<<<blocks, threads, shared_size>>>(nep_data.potential_per_atom.data(), d_partial.data(), nlocal);
     cudaDeviceSynchronize();
     global_reduce<<<1, threads, shared_size>>>(d_partial.data(), nep_data.potential_all.data(), blocks);  
     cudaDeviceSynchronize();
    nep_data.potential_all.copy_to_host(h_etot_virial_global, 1);
  }

  // copy energy 
  if (eflag_atom && ff_index == 0) {
    copyArrayKernel<<<(nlocal + BLOCK_SIZE256 - 1) / BLOCK_SIZE256, BLOCK_SIZE256>>>(potential_per_atom_lmp, nep_data.potential_per_atom.data(), nlocal);
    CUDA_CHECK_KERNEL
    cudaDeviceSynchronize();
  }
  if (potential_per_atom_copy) {
    copyArrayKernel<<<(nlocal + BLOCK_SIZE256 - 1) / BLOCK_SIZE256, BLOCK_SIZE256>>>(potential_per_atom_copy, nep_data.potential_per_atom.data(), nlocal);
    CUDA_CHECK_KERNEL
    cudaDeviceSynchronize();    
  }
  
  if (ff_index == 0 && force_per_atom_copy) {
    // copy force for deviation_calculate
    // if ff_index > 0, the force_per_atom is force_per_atom_copy
    copyArrayKernel<<<(nall*3 + BLOCK_SIZE256 - 1) / BLOCK_SIZE256, BLOCK_SIZE256>>>(force_per_atom_copy, force_per_atom, nall*3);
    CUDA_CHECK_KERNEL
    cudaDeviceSynchronize();
  }

  // if(device == 0) {
  //   std::vector<double> tmp_f(nall * 3);
  //   std::vector<double> tmp_e(nlocal);
  //   cudaMemcpy(tmp_f.data(), force_per_atom, sizeof(double) * nall * 3, cudaMemcpyDeviceToHost);
  //   cudaMemcpy(tmp_e.data(), potential_per_atom_copy, sizeof(double) * nlocal, cudaMemcpyDeviceToHost);
  //   for (int ii = 0; ii < inum ; ii++) {
  //     printf(" e[%d][%d] = %f f[%d][%d] = %f %f %f\n", ff_index, ii, tmp_e[ii], ff_index, ii, tmp_f[ii*3], tmp_f[ii*3+1], tmp_f[ii*3+2]);
  //   }
  // }

  // copy virial peratom
  // if (vflag_atom) {
  //   copyArrayKernel<<<(nall*6 + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(virial_per_atom, nep_data.virial_per_atom.data(), nall*6);
  //   CUDA_CHECK_KERNEL
  //   cudaDeviceSynchronize();
  // }

  // printf("======out etot %.15f virial %.15f %.15f %.15f %.15f %.15f %.15f rank %d device %d=======\n", \
    h_etot_virial_global[0], h_etot_virial_global[1], h_etot_virial_global[2], h_etot_virial_global[3], h_etot_virial_global[4], h_etot_virial_global[5], h_etot_virial_global[6], rank, device);
}
