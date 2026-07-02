// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */
/* The nep KOKKOS interface enables neighboring build operations to be moved to the device side, accelerating the overall speed of MD. Author wuxingxing@pwmat.com and MatPL development team. 2026. Beijing Lonxun Quantum Co.,Ltd.*/
#include "pair_nep_kokkos.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "kokkos.h"
#include "memory_kokkos.h"
#include "neighbor_kokkos.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "suffix.h"
#include "update.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <cuda_runtime.h>

using namespace LAMMPS_NS;

namespace {
void nepkk_mpi_allreduce_sum_double(const double* sendbuf, double* recvbuf, int count, void* context)
{
  MPI_Comm comm = *static_cast<MPI_Comm*>(context);
  MPI_Allreduce(sendbuf, recvbuf, count, MPI_DOUBLE, MPI_SUM, comm);
}
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairNEPKokkos<DeviceType>::PairNEPKokkos(LAMMPS *lmp) : PairNEP(lmp)
{
  centroidstressflag = CENTROID_AVAIL;
  local_maxeatom = 0;
  local_maxvatom = 0;
  local_maxcvatom = 0;
  respa_enable = 0;

  restartinfo = 0;
  manybody_flag = 1;
  single_enable = 0;
  one_coeff = 1;

  suffix_flag |= Suffix::KOKKOS;
  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = X_MASK | F_MASK | TAG_MASK | TYPE_MASK | ENERGY_MASK | VIRIAL_MASK;
  datamask_modify = F_MASK | ENERGY_MASK | VIRIAL_MASK;

  me = comm->me;
  h_etot_virial_global.resize(10);
  // This can write directly to log.lammps.
  // if (comm->me == 0) {
  //   utils::logmesg(lmp, "=== PairNEPKokkos Constructor Finished ===\n");
  // }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairNEPKokkos<DeviceType>::~PairNEPKokkos()
{
  // printf("=====rank %d device %d doing ~pairnep start copymode %d =====\n", rank, device_id, copymode);
  if (copymode) return;

  if (allocated) {
    memoryKK->destroy_kokkos(k_eatom, eatom);
    memoryKK->destroy_kokkos(k_vatom, vatom);
    memoryKK->destroy_kokkos(k_cvatom, cvatom);
  }
  memory->destroy(setflag);
  memory->destroy(cutsq);
  h_etot_virial_global = decltype(h_etot_virial_global)();
  if (is_rank_0 && explrError_fp != nullptr) {
        fclose(explrError_fp);
        explrError_fp = nullptr;
  }

  // printf("=====rank %d device %d doing ~pairnep end =====\n", rank, device_id);
  // NEP model cleanup if needed
  // nep_gpu_model.cleanup(); // Assuming NEP has a cleanup method
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

template<class DeviceType>
void PairNEPKokkos<DeviceType>::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag, n+1, n+1, "pair:setflag");
  memory->create(cutsq, n+1, n+1, "pair:cutsq");

  // for (int i = 1; i <= n; i++) {
  //   for (int j = i; j <= n; j++) {
  //     setflag[i][j] = 0;
  //   }
  // }
}
/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

template<class DeviceType>
void PairNEPKokkos<DeviceType>::settings(int narg, char **arg)
{
  is_rank_0 = (comm->me == 0);

  if (narg < 1) error->all(FLERR, "Illegal pair_style command");
  int iarg = 0;  // index of first forcefield file
  // Parse potential file
  num_ff = 0;
  while(iarg < narg) { // nep1.txt nep2.txt ... nepn.txt out_freq 10
    std::string arg_str(arg[iarg]);
    if (arg_str.find(".txt") != std::string::npos) {
      potential_files.push_back(arg_str);
      num_ff ++;
      iarg++;
    } else {
      break; // for out_freq
    }
  }
  while (iarg < narg) {
    if (strcmp(arg[iarg], "out_freq") == 0) {
        out_freq = utils::inumeric(FLERR, arg[++iarg], false, lmp);
    } else if (strcmp(arg[iarg], "out_file") == 0) {
        explrError_fname = arg[++iarg];
    } else if (strcmp(arg[iarg], "kspace") == 0 || strcmp(arg[iarg], "kspace_method") == 0) {
        kspace_method = arg[++iarg];
        if (kspace_method != "ewald" && kspace_method != "pppm") {
          error->all(FLERR, "pair_style matpl/nep/kk kspace must be ewald or pppm");
        }
    } else if (strcmp(arg[iarg], "pppm_spacing") == 0) {
        if (iarg + 1 >= narg) {
          error->all(FLERR, "pair_style matpl/nep/kk pppm_spacing requires one value");
        }
        pppm_mesh_spacing = utils::numeric(FLERR, arg[++iarg], false, lmp);
        if (pppm_mesh_spacing <= 0.0) {
          error->all(FLERR, "pair_style matpl/nep/kk pppm_spacing must be positive");
        }
    } else if (strcmp(arg[iarg], "pppm_mesh") == 0) {
        if (iarg + 1 >= narg) {
          error->all(FLERR, "pair_style matpl/nep/kk pppm_mesh requires friendly, power2, or 3 integer values");
        }
        std::string mesh_arg(arg[iarg + 1]);
        if (mesh_arg == "friendly") {
          pppm_mesh_mode = 1;
          for (int d = 0; d < 3; ++d) pppm_mesh[d] = 0;
          ++iarg;
        } else if (mesh_arg == "power2") {
          pppm_mesh_mode = 0;
          for (int d = 0; d < 3; ++d) pppm_mesh[d] = 0;
          ++iarg;
        } else {
          if (iarg + 3 >= narg) {
            error->all(FLERR, "pair_style matpl/nep/kk pppm_mesh requires friendly, power2, or 3 integer values");
          }
          pppm_mesh_mode = 2;
          for (int d = 0; d < 3; ++d) {
            pppm_mesh[d] = utils::inumeric(FLERR, arg[++iarg], false, lmp);
            if (pppm_mesh[d] <= 0) {
              error->all(FLERR, "pair_style matpl/nep/kk pppm_mesh values must be positive");
            }
          }
        }
    } 
    iarg++;    
  }

  if (is_rank_0 and num_ff > 1) {
      explrError_fp = fopen(&explrError_fname[0], "w");
      fprintf(explrError_fp, "%9s %16s %16s %16s %16s %16s %16s\n", 
      "#    step", "avg_devi_f", "min_devi_f", "max_devi_f", 
      "avg_devi_e", "min_devi_e", "max_devi_e");
      fflush(explrError_fp);
  }
    
  nprocs_total = comm->nprocs;
  // Do not hard-code device_id=0 or call cudaSetDevice manually.
  // Kokkos sets the current device during LAMMPS initialization.
  // Querying the current CUDA device is enough for logging/bookkeeping.
  
  if (std::is_same<DeviceType, LMPDeviceType>::value) {
    // Optional: query the current device for diagnostics.
    cudaGetDevice(&device_id);
    // printf("Rank %d: Using CUDA device %d (Kokkos-managed)\n", comm->me, device_id);
    rank = comm->me;
    // if (is_rank_0) {
    //   printf("Rank %d: Using CUDA device %d (Kokkos-managed)\n", comm->me, device_id);
    // }
    // No cudaSetDevice(device_id); Kokkos already selected the device.
  }
  nep_gpu_models.resize(num_ff);
  for (int i=0; i < num_ff; i++) {
    std::string model_file = potential_files[i];
    nep_gpu_models[i].read_neptxt(model_file.c_str(), is_rank_0, comm->me, device_id, i);
    nep_gpu_models[i].paramb.pppm_mesh_spacing = static_cast<NEP_FLOAT>(pppm_mesh_spacing);
    nep_gpu_models[i].paramb.pppm_mesh_mode = pppm_mesh_mode;
    for (int d = 0; d < 3; ++d) nep_gpu_models[i].paramb.pppm_mesh[d] = pppm_mesh[d];
    if (is_rank_0 && nep_gpu_models[i].paramb.charge_mode == 2) {
      utils::logmesg(lmp, "NEP Kokkos qNEP kspace method: " + kspace_method + "\n");
      if (kspace_method == "pppm") {
        utils::logmesg(lmp, "NEP Kokkos PPPM target mesh spacing: " + std::to_string(pppm_mesh_spacing) + "\n");
        if (pppm_mesh_mode == 1) {
          utils::logmesg(lmp, "NEP Kokkos PPPM mesh mode: friendly (2/3/5/7 FFT-friendly mesh)\n");
        } else if (pppm_mesh_mode == 2) {
          utils::logmesg(lmp, "NEP Kokkos PPPM mesh mode: explicit user mesh " + std::to_string(pppm_mesh[0]) + " " + std::to_string(pppm_mesh[1]) + " " + std::to_string(pppm_mesh[2]) + "\n");
        } else {
          utils::logmesg(lmp, "NEP Kokkos PPPM mesh mode: power2 (default, GPUMD-compatible). For large systems, use pppm_mesh friendly or explicit pppm_mesh Nx Ny Nz to reduce GPU memory.\n");
        }
      }
    }
    if (i == 0) {
      cutoff = nep_gpu_models[i].paramb.rc_radial;
      cutoffsq = cutoff * cutoff;
    }
    if (is_rank_0) {
      utils::logmesg(lmp, "NEP Kokkos potential " + model_file + " loaded successfully\n");
    }
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

template<class DeviceType>
void PairNEPKokkos<DeviceType>::coeff(int narg, char **arg)
{
  if (!allocated) allocate();
  for (int f1 = 0; f1 < num_ff; f1++) {
    std::vector<int> atom_type_module = nep_gpu_models[f1].cpu_element_atomic_number_list;
    std::vector<int> atom_types;
    for (int ii = 2; ii < narg; ++ii) {
      std::string element = utils::strdup(arg[ii]);
      int temp = find_atomic_number(element);
      // int temp = std::stoi(arg[ii]);
      auto iter = std::find(atom_type_module.begin(), atom_type_module.end(), temp);   
      if (iter != atom_type_module.end() || arg[ii] == 0)
      {
          int index = std::distance(atom_type_module.begin(), iter);
          // model_atom_type_idx.push_back(index); 
          // atom_types.push_back(temp);
          atom_types.push_back(index);
          // printf("=== rank %d device_id %d coeff the config atom type %d index in ff is %d\n",\
            rank, device_id, temp, index);
      }
      else
      {
          error->all(FLERR, "This element is not included in the nep file: " + potential_files[f1]);
      }
    }
    nep_gpu_models[f1].set_atom_type_map(narg-2, atom_types.data());
  }
  int ntypes = atom->ntypes;
  // For NEP, all types are handled by the model; set all pairs
  for (int i = 1; i <= ntypes; i++) {
    for (int j = i; j <= ntypes; j++) {
      setflag[i][j] = 1;
      cutsq[i][j] = cutoffsq;
    }
  }
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

template<class DeviceType>
void PairNEPKokkos<DeviceType>::init_style()
{
  if (strstr(update->integrate_style,"respa"))
    error->all(FLERR,"Pair NEP does not support rRESPA");

  neighflag = lmp->kokkos->neighflag;

  auto request = neighbor->add_request(this, NeighConst::REQ_FULL);
  request->set_kokkos_host(std::is_same<DeviceType,LMPHostType>::value &&
                           !std::is_same<DeviceType,LMPDeviceType>::value);
  request->set_kokkos_device(std::is_same<DeviceType,LMPDeviceType>::value);

  if (is_rank_0) printf("======== in init_style: neighflag = %d =========\n", neighflag);
  if (force->newton_pair == 0)
    error->all(FLERR, "Pair style matpl/nep/kk requires newton pair on");
  newton_pair = force->newton_pair;
  // printf("===== DEBUG: newton_pair = %d =====\n", newton_pair);
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

template<class DeviceType>
double PairNEPKokkos<DeviceType>::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR,"All pair coeffs are not set");

  cutsq[i][j] = cutoffsq;
  return cutoff;
}

/* ----------------------------------------------------------------------
   find atomic number for element key
------------------------------------------------------------------------- */

template<class DeviceType>
int PairNEPKokkos<DeviceType>::find_atomic_number(std::string& key)
{
  // Implement element to atomic number mapping if needed
  // For now, assume types are directly mapped
  std::transform(key.begin(), key.end(), key.begin(), ::tolower);
  if (key.length() == 1) { key += " "; }
  key.resize(2);

  std::vector<std::string> element_table = {
      "h ","he",
      "li","be","b ","c ","n ","o ","f ","ne",
      "na","mg","al","si","p ","s ","cl","ar",
      "k ","ca","sc","ti","v ","cr","mn","fe","co","ni","cu",
      "zn","ga","ge","as","se","br","kr",
      "rb","sr","y ","zr","nb","mo","tc","ru","rh","pd","ag",
      "cd","in","sn","sb","te","i ","xe",
      "cs","ba","la","ce","pr","nd","pm","sm","eu","gd","tb","dy",
      "ho","er","tm","yb","lu","hf","ta","w ","re","os","ir","pt",
      "au","hg","tl","pb","bi","po","at","rn",
      "fr","ra","ac","th","pa","u ","np","pu"
  };

  std::vector<std::string> element_table2 = {
      "1 ","2 ",
      "3 ","4 ","5 ","6 ","7 ","8 ","9 ","10",
      "11","12","13","14","15","16","17","18",
      "19","20","21","22","23","24","25","26","27","28","29",
      "30","31","32","33","34","35","36",
      "37","38","39","40","41","42","43","44","45","46","47",
      "48","49","50","51","52","53","54",
      "55","56","57","58","59","60","61","62","63","64","65","66",
      "67","68","69","70","71","72","73","74","75","76","77","78",
      "79","80","81","82","83","84","85","86",
      "87","88","89","90","91","92","93","94",
  };
  
  for (size_t i = 0; i < element_table.size(); ++i) {
      if (element_table[i] == key) {
          int atomic_number = i + 1;
          return atomic_number;
      }else if(element_table2[i] == key) {
          int atomic_number = i + 1;
          return atomic_number;
      }
  }

  // if not the case
  return -1;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairNEPKokkos<DeviceType>::compute(int eflag_in, int vflag_in)
{
  eflag = eflag_in;
  vflag = vflag_in;
  // if (neighflag == FULL)  
  no_virial_fdotr_compute = 1;
  ev_init(eflag, vflag, 0);
  // size_t total, used, free;
  bigint ntimestep = update->ntimestep;
  bool is_devi_step = (num_ff > 1) && (ntimestep % out_freq == 0);

  atomKK->sync(execution_space, X_MASK | F_MASK | TYPE_MASK);
  if (eflag || vflag) atomKK->modified(execution_space, datamask_modify);
  else atomKK->modified(execution_space,F_MASK);
  x = atomKK->k_x.template view<DeviceType>();
  f = atomKK->k_f.template view<DeviceType>();
  type = atomKK->k_type.template view<DeviceType>();

  nlocal = atom->nlocal;
  nall = atom->nlocal + atom->nghost;
  inum = list->inum;

  for (const auto &model : nep_gpu_models) {
    if (nlocal > model.max_nlocal) {
      const std::int64_t radial_neighbor_count =
        static_cast<std::int64_t>(nlocal) *
        static_cast<std::int64_t>(model.paramb.MN_radial);
      if (radial_neighbor_count > std::numeric_limits<int>::max()) {
        error->one(
          FLERR,
          "The atom count on a single GPU is too large (current local atom count: {}) and "
          "exceeds the supported memory capacity. Reduce the number of atoms per GPU or use "
          "MatPL-pro.",
          nlocal);
      }
    }
  }

  // printf("DEBUG rank=%d device_id=%d step=%d eflag=%d, vflag=%d, vflag_fdotr=%d eflag_atom=%d, vflag_atom=%d cvflag_atom=%d vflag_global=%d vflag_either %d centroidstressflag=%d atom->nmax %d maxeatom %d->%d maxvatom %d->%d maxcvatom %d->%d nlocal %d inum %d nall %d neighbor->includegroup %d no_virial_fdotr_compute %d lmp->kokkos->neighflag %d \n", \
        rank, device_id, ntimestep, eflag, vflag, vflag_fdotr, eflag_atom, vflag_atom, cvflag_atom, vflag_global, vflag_either, centroidstressflag, atom->nmax, maxeatom, local_maxeatom, maxvatom, local_maxvatom, maxcvatom, local_maxcvatom, nlocal, inum, nall, neighbor->includegroup, no_virial_fdotr_compute, lmp->kokkos->neighflag);
  // printf("======compute before k_eatom.extent(0) %d k_vatom.extent(0) %d k_cvatom.extent(0) %d ======\n",k_eatom.extent(0), k_vatom.extent(0), k_cvatom.extent(0));
  
  cur_atom_max = max(cur_atom_max, atom->nmax);
  double *virial_per_atom_ptr = nullptr;
  double *cvirial_per_atom_ptr = nullptr;

  if (eflag_atom) {
    if (local_maxeatom < cur_atom_max) local_maxeatom = cur_atom_max;
    memoryKK->destroy_kokkos(k_eatom, eatom);
    memoryKK->create_kokkos(k_eatom, eatom, local_maxeatom, "pair:eatom");
    d_eatom = k_eatom.view<DeviceType>();
  }

  if (cvflag_atom) {
    if (local_maxcvatom < cur_atom_max) local_maxcvatom = cur_atom_max;
    memoryKK->destroy_kokkos(k_cvatom, cvatom);
    memoryKK->create_kokkos(k_cvatom, cvatom, local_maxcvatom, "pair:cvatom");
    d_cvatom = k_cvatom.view<DeviceType>();
    Kokkos::deep_copy(d_cvatom, 0.0);
    cvirial_per_atom_ptr = d_cvatom.data();
  } else if (vflag_atom) {
    if (local_maxvatom < cur_atom_max) local_maxvatom = cur_atom_max;
    memoryKK->destroy_kokkos(k_vatom, vatom);
    memoryKK->create_kokkos(k_vatom, vatom, local_maxvatom, "pair:vatom");
    d_vatom = k_vatom.view<DeviceType>();
    Kokkos::deep_copy(d_vatom, 0.0);
    virial_per_atom_ptr = d_vatom.data();
  } else if (vflag_global) {
    if (local_maxvatom_work < cur_atom_max) {
      local_maxvatom_work = cur_atom_max;
      k_vatom_work = DAT::tdual_virial_array("pair:vatom_work", local_maxvatom_work);
    }
    d_vatom_work = k_vatom_work.view<DeviceType>();
    Kokkos::deep_copy(d_vatom_work, 0.0);
    virial_per_atom_ptr = d_vatom_work.data();
  }
  
  if (num_ff > 1) {
    if (global_nall < nall) {
      global_nall = nall;
      k_full_f = DAT::tdual_ffloat_1d("pair:full_f", num_ff * global_nall * 3);
    } else {
      Kokkos::deep_copy(k_full_f.d_view, 0.0);
    }
    h_full_f = k_full_f.h_view;

    if (global_nlocal < nlocal) {
      global_nlocal = nlocal;
      k_full_e = DAT::tdual_ffloat_1d("pair:full_e", num_ff * global_nlocal);
    } else {
      Kokkos::deep_copy(k_full_e.d_view, 0.0);
    }
    h_full_e = k_full_e.h_view;
  }

  // const int ignum = inum + list->gnum;
  NeighListKokkos<DeviceType>* k_list = static_cast<NeighListKokkos<DeviceType>*>(list);
  d_numneigh = k_list->d_numneigh;
  d_neighbors = k_list->d_neighbors;
  d_ilist = k_list->d_ilist;
  int max_neighs = d_neighbors.extent(1);
  int num_neighs = d_neighbors.extent(0);
  // printf("ntimestep %d rank=%d device_id=%d nranks=%d | (e,v,ee,eg,ea,ve,vg,va)=(%d,%d,%d,%d,%d,%d,%d,%d) | inum=%d, nall=%d | maxeatom=%d, maxvatom=%d | lmpneigh=%dX%d | xsize=%d X %d | fsize=%d X %d | vsize=%d X %d paramb-2b %d 3b %d\n", \
       ntimestep, rank, device_id, nprocs_total, eflag, vflag, eflag_either, eflag_global, eflag_atom, vflag_either, vflag_global, vflag_atom, inum, nall, maxeatom, maxvatom, num_neighs, max_neighs, x.extent(0), x.extent(1), f.extent(0), f.extent(1), d_vatom.extent(0), d_vatom.extent(1), nep_gpu_models[0].paramb.n_max_radial, nep_gpu_models[0].paramb.n_max_angular);
  // (e,v,ee,eg,ea,ve,vg,va)=(1,2,1,1,0,0,0,0) 
  // maxeatom is nonzero when compute pe/atom is enabled.
  // (e,v,ee,eg,ea,ve,vg,va)=(3,2,3,1,2,0,0,0)

  need_dup = lmp->kokkos->need_dup<DeviceType>();
  if (need_dup) {
    dup_f = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterDuplicated>(f);
    if (eflag_atom)
      dup_eatom = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterDuplicated>(d_eatom);
    if (vflag_atom)
      dup_vatom = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterDuplicated>(d_vatom);
  } else {
    ndup_f = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterNonDuplicated>(f);
    if (eflag_atom)
      ndup_eatom = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterNonDuplicated>(d_eatom);
    if (vflag_atom)
      ndup_vatom = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterNonDuplicated>(d_vatom);
  }
  copymode = 1;
  // EV_FLOAT ev;
  // EV_FLOAT ev_all;

  // nep_gpu_model.getGPUMemoryStats(total, used, free);
  // printf("MB-before-1 rank=%d device_id=%d step=%d Total=%f Used=%f Free=%f (GB)\n", rank, device_id, ntimestep, total / (1024.0 * 1024.0 * 1024.0), used / (1024.0 * 1024.0 * 1024.0), free / (1024.0 * 1024.0 * 1024.0));
  // if (nep_gpu_model.allocate_once==0 and is_rank_0) {
  //   int result = nep_gpu_model.calculateMaxAtoms(nep_gpu_model.paramb.MN_radial, nep_gpu_model.paramb.MN_angular, max_neighs, nep_gpu_model.annmb.dim, nlocal, nprocs_total);
  // }

  // Invoke NEP computation
  // printf("=========nep start compute force ==========\n");
  double box_h[9] = {
    domain->xprd, domain->xy,   domain->xz,
    0.0,          domain->yprd, domain->yz,
    0.0,          0.0,          domain->zprd
  };
  MPI_Comm nepkk_mpi_comm = world;
  if (!is_devi_step) {
    nep_gpu_models[0].compute(
      eflag_global,
      eflag_atom, 
      vflag_either,
      vflag_global,
      vflag_atom,
      cvflag_atom,
      nall,                    // nall = nlocal + nghost
      inum,                    // inum = number of local atoms
      nlocal,                  // nlocal
      max_neighs,
      num_neighs,
      const_cast<int*>(reinterpret_cast<const int*>(type.data())),
      const_cast<int*>(reinterpret_cast<const int*>(d_ilist.data())),        // ilist
      const_cast<int*>(reinterpret_cast<const int*>(d_numneigh.data())),     // numneigh
      const_cast<int*>(reinterpret_cast<const int*>(d_neighbors.data())),    // neighbors
      const_cast<double*>(reinterpret_cast<const double*>(x.data())),
      // const_cast<double*>(reinterpret_cast<const double*>(d_eatom.data())),
      // d_eatom.data(),
      // const_cast<double*>(reinterpret_cast<double*>(d_eatom.data())),
      d_eatom.data(),
      nullptr,
      f.data(),                                       // force output
      nullptr,
      virial_per_atom_ptr,                           // 6-component per-atom virial or internal work buffer
      cvirial_per_atom_ptr,                          // 9-component centroid virial
      box_h,
      kspace_method.c_str(),
      static_cast<long long>(atom->natoms),
      comm->nprocs,
      nepkk_mpi_allreduce_sum_double,
      &nepkk_mpi_comm,
      h_etot_virial_global.data()
    );
    // printf("=========end start compute force  etot %f ==========\n", h_etot_virial_global[0]);
    // Wait for NEP computation to complete
    Kokkos::fence();    
  } else {
    for (int ff_idx = num_ff - 1; ff_idx >= 0; --ff_idx) {
      auto d_full_e = k_full_e.view<DeviceType>();
      double* full_e_ptr = d_full_e.data() + ff_idx * nlocal;

      auto d_full_f = k_full_f.view<DeviceType>();
      double* full_f_ptr = d_full_f.data() + ff_idx * nall * 3;
      nep_gpu_models[ff_idx].compute(
        eflag_global,
        eflag_atom, 
        vflag_either,
        vflag_global,
        vflag_atom,
        cvflag_atom,
        nall,                    // nall = nlocal + nghost
        inum,                    // inum = number of local atoms
        nlocal,                  // nlocal
        max_neighs,
        num_neighs,
        const_cast<int*>(reinterpret_cast<const int*>(type.data())),
        const_cast<int*>(reinterpret_cast<const int*>(d_ilist.data())),        // ilist
        const_cast<int*>(reinterpret_cast<const int*>(d_numneigh.data())),     // numneigh
        const_cast<int*>(reinterpret_cast<const int*>(d_neighbors.data())),    // neighbors
        const_cast<double*>(reinterpret_cast<const double*>(x.data())),
        // const_cast<double*>(reinterpret_cast<const double*>(d_eatom.data())),
        // d_eatom.data(),
        // const_cast<double*>(reinterpret_cast<double*>(d_eatom.data())),
        d_eatom.data(),
        full_e_ptr,
        f.data(),                                       // force output
        full_f_ptr,
        virial_per_atom_ptr,                           // 6-component per-atom virial or internal work buffer
        cvirial_per_atom_ptr,                          // 9-component centroid virial
        box_h,
        kspace_method.c_str(),
        static_cast<long long>(atom->natoms),
        comm->nprocs,
        nepkk_mpi_allreduce_sum_double,
        &nepkk_mpi_comm,
        h_etot_virial_global.data()
        // neighbor_rebuilt
      );
      // printf("=========end start compute force  etot %f ==========\n", h_etot_virial_global[0]);
      // Wait for NEP computation to complete
      Kokkos::fence();
    }
    Kokkos::fence();
  }
  // nep_gpu_model.getGPUMemoryStats(total, used, free);
  // printf("MB-after-0 rank=%d device_id=%d step=%d Total=%f Used=%f Free=%f (GB)\n", rank, device_id, ntimestep, total / (1024.0 * 1024.0 * 1024.0), used / (1024.0 * 1024.0 * 1024.0), free / (1024.0 * 1024.0 * 1024.0));
  if (is_devi_step) {
    reverse_comm_mode = REVERSE_DEVIATION_FORCE;
    comm_reverse = 3 * num_ff;
    auto d_full_f = k_full_f.view<DeviceType>();
    auto d_full_e = k_full_e.view<DeviceType>();
    auto h_temp = Kokkos::create_mirror_view(d_full_f);
    Kokkos::deep_copy(h_temp, d_full_f);
    // printf("===== DEBUG: temp host BEFORE reverse_comm =====\n");
    // for (int ff = 0; ff < num_ff; ++ff) {
    //   printf("force temp host [ff %d]:\n", ff);
    //   for (int i = 0; i < std::min(12, inum); ++i) {
    //     int base_idx = ff * nall * 3 + i * 3;
    //     printf("[%d] = %f %f %f\n", i,
    //            h_temp(base_idx), h_temp(base_idx+1), h_temp(base_idx+2));
    //   }
    // }
    // reverse_comm calls pack/unpack below.
    comm->reverse_comm(this);  // 璋冪敤pack/unpack
    comm_reverse = 0;                          // 閲嶇疆
    reverse_comm_mode = REVERSE_NONE;

    auto h_final_f = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_full_f);
    auto h_final_e = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_full_e);
    atomKK->sync(Host, datamask_read | datamask_modify);
    auto h_x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x);
    auto h_f = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), f);
    auto h_eatom = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_eatom);
    auto h_ilist = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_ilist);

    double glb_avg_f_err, glb_max_f_err, glb_min_f_err;
    double glb_avg_ei_err, glb_max_ei_err, glb_min_ei_err;
    double avg_f_err, max_f_err, min_f_err, avg_ei_err, max_ei_err, min_ei_err;

    std::tuple<double, double, double, double, double, double> result = calc_max_error(
      h_ilist.data(),
      h_final_f.data(),
      h_final_e.data(),
      num_ff,
      inum,
      nlocal,
      nall,
      rank);

    avg_f_err = std::get<0>(result);
    max_f_err = std::get<1>(result);
    min_f_err = std::get<2>(result);
    avg_ei_err = std::get<3>(result);
    max_ei_err = std::get<4>(result);
    min_ei_err = std::get<5>(result);


    MPI_Allreduce(&max_f_err, &glb_max_f_err, 1, MPI_DOUBLE, MPI_MAX, world);
    MPI_Allreduce(&max_ei_err, &glb_max_ei_err, 1, MPI_DOUBLE, MPI_MAX, world);

    MPI_Allreduce(&min_f_err, &glb_min_f_err, 1, MPI_DOUBLE, MPI_MIN, world);
    MPI_Allreduce(&min_ei_err, &glb_min_ei_err, 1, MPI_DOUBLE, MPI_MIN, world);

    MPI_Allreduce(&avg_f_err, &glb_avg_f_err, 1, MPI_DOUBLE, MPI_SUM, world);
    MPI_Allreduce(&avg_ei_err, &glb_avg_ei_err, 1, MPI_DOUBLE, MPI_SUM, world);

    if (atom->natoms > 0) {
        glb_avg_f_err /= static_cast<double>(atom->natoms);
        glb_avg_ei_err /= static_cast<double>(atom->natoms);
    } else {
        glb_avg_f_err = 0.0;
        glb_avg_ei_err = 0.0;
    }
    if (is_rank_0) {
      fprintf(explrError_fp, "%9lld %16.9f %16.9f %16.9f %16.9f %16.9f %16.9f\n",
              update->ntimestep,
              glb_avg_f_err, glb_min_f_err, glb_max_f_err,
              glb_avg_ei_err, glb_min_ei_err, glb_max_ei_err);
      fflush(explrError_fp);
    }
  }

  if (cvflag_atom) {
    k_cvatom.template modify<DeviceType>();
    k_cvatom.sync<LMPHostType>();

    // auto h_cv = k_cvatom.h_view;
    // for (int i = 0; i < maxcvatom; ++i) {
    //   cvatom[i] = h_cv.data() + i * 9;
    // }

    // for (int i = 0; i < nall; ++i) {
    //   cvatom[i][0] = h_cv.data()[i + 0 * nall];  // xx
    //   cvatom[i][1] = h_cv.data()[i + 1 * nall];  // yy
    //   cvatom[i][2] = h_cv.data()[i + 2 * nall];  // zz
    //   cvatom[i][3] = h_cv.data()[i + 3 * nall];  // xy
    //   cvatom[i][4] = h_cv.data()[i + 4 * nall];  // xz
    //   cvatom[i][5] = h_cv.data()[i + 5 * nall];  // yz
    //   cvatom[i][6] = h_cv.data()[i + 6 * nall];  // yx
    //   cvatom[i][7] = h_cv.data()[i + 7 * nall];  // zx
    //   cvatom[i][8] = h_cv.data()[i + 8 * nall];  // zy
    // }

    // for (int i = 0; i < std::min(10, nlocal); ++i) {
    //   printf(" local cvatom[%6d]: %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f\n",
    //          i, cvatom[i][0], cvatom[i][1], cvatom[i][2], cvatom[i][3], cvatom[i][4], cvatom[i][5], cvatom[i][6], cvatom[i][7], cvatom[i][8]);
    // }
    // for (int i = nall-11; i < nall; ++i) {
    //   printf(" ghost cvatom[%6d]: %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f\n",
    //          i, cvatom[i][0], cvatom[i][1], cvatom[i][2], cvatom[i][3], cvatom[i][4], cvatom[i][5], cvatom[i][6], cvatom[i][7], cvatom[i][8]);
    // }

  }

  if (need_dup)
    Kokkos::Experimental::contribute(f, dup_f);

  if (eflag_global) eng_vdwl += h_etot_virial_global[0];
  if (vflag_global) {
    virial[0] += h_etot_virial_global[1];
    virial[1] += h_etot_virial_global[2];
    virial[2] += h_etot_virial_global[3];
    virial[3] += h_etot_virial_global[4];
    virial[4] += h_etot_virial_global[5];
    virial[5] += h_etot_virial_global[6];    
    // printf("virial[0-5]=%f %f %f %f %f %f\n", virial[0], virial[1], virial[2], virial[3], virial[4], virial[5]);
  }

  if (eflag_atom) {
    if (need_dup)
      Kokkos::Experimental::contribute(d_eatom, dup_eatom);
    k_eatom.template modify<DeviceType>();
    k_eatom.template sync<LMPHostType>();
  }

  if (vflag_atom) {
    if (need_dup)
      Kokkos::Experimental::contribute(d_vatom, dup_vatom);
    k_vatom.template modify<DeviceType>();
    k_vatom.template sync<LMPHostType>();
  }

  // if (vflag_fdotr) pair_virial_fdotr_compute(this); error in many body force.

  copymode = 0;
  // free duplicated memory
  if (need_dup) {
    dup_f     = decltype(dup_f)();
    dup_eatom = decltype(dup_eatom)();
    dup_vatom = decltype(dup_vatom)();
  }  
  // Synchronize MPI ranks after computation.
  MPI_Barrier(world);
}

template<class DeviceType>
void PairNEPKokkos<DeviceType>::qnep_bec_atom(double **array, int nmax)
{
  if (nep_gpu_models.empty() || !nep_gpu_models[0].has_charge_bec()) {
    error->all(FLERR, "compute qnep/bec/atom requires a NEP-charge pair_style matpl/nep/kk potential");
  }
  if (nall <= 0 || nlocal < 0 || inum <= 0) {
    error->all(FLERR, "compute qnep/bec/atom was invoked before pair_style matpl/nep/kk computed this step");
  }
  if (nmax < nlocal) {
    error->all(FLERR, "compute qnep/bec/atom internal array is smaller than atom->nlocal");
  }

  nep_gpu_models[0].compute_bec(
    inum,
    nlocal,
    nall,
    d_neighbors.extent(0),
    const_cast<int*>(reinterpret_cast<const int*>(d_ilist.data())),
    const_cast<int*>(reinterpret_cast<const int*>(d_numneigh.data())),
    const_cast<int*>(reinterpret_cast<const int*>(d_neighbors.data())));
  Kokkos::fence();

  h_bec_nep_float.resize(nall * 9);
  h_bec_reverse.resize(nall * 9);
  nep_gpu_models[0].copy_bec_to_host(h_bec_nep_float.data(), nall);
  for (int i = 0; i < nall * 9; ++i) {
    h_bec_reverse[i] = static_cast<double>(h_bec_nep_float[i]);
  }

  reverse_comm_mode = REVERSE_BEC;
  comm_reverse = 9;
  comm->reverse_comm(this);
  comm_reverse = 0;
  reverse_comm_mode = REVERSE_NONE;

  for (int i = 0; i < nlocal; ++i) {
    for (int d = 0; d < 9; ++d) {
      array[i][d] = h_bec_reverse[i + nall * d];
    }
  }
}

// pack_reverse_comm: pack either BEC values or model-deviation forces.
template<class DeviceType>
int PairNEPKokkos<DeviceType>::pack_reverse_comm(int n, int first, double *buf)
{
  if (reverse_comm_mode == REVERSE_BEC) {
    int m = 0;
    const int last = first + n;
    for (int i = first; i < last; ++i) {
      for (int d = 0; d < 9; ++d) {
        buf[m++] = h_bec_reverse[i + nall * d];
      }
    }
    return m;
  }

  auto d_full_f = k_full_f.view<DeviceType>();

  auto h_temp = Kokkos::create_mirror_view(d_full_f);
  Kokkos::deep_copy(h_temp, d_full_f);

  int m = 0;
  int last = first + n;
  for (int i = first; i < last; ++i) {
    for (int ff = 0; ff < num_ff; ++ff) {
      int base_idx = ff * nall * 3 + i * 3;
      buf[m++] = h_temp(base_idx + 0);
      buf[m++] = h_temp(base_idx + 1);
      buf[m++] = h_temp(base_idx + 2);
    }
  }

  return m;
}

// unpack_reverse_comm: accumulate data received from ghost owners.
template<class DeviceType>
void PairNEPKokkos<DeviceType>::unpack_reverse_comm(int n, int *list, double *buf)
{
  if (reverse_comm_mode == REVERSE_BEC) {
    int m = 0;
    for (int i = 0; i < n; ++i) {
      const int j = list[i];
      for (int d = 0; d < 9; ++d) {
        h_bec_reverse[j + nall * d] += buf[m++];
      }
    }
    return;
  }

  auto d_full_f = k_full_f.view<DeviceType>();

  auto h_temp = Kokkos::create_mirror_view(d_full_f);
  Kokkos::deep_copy(h_temp, d_full_f);

  int m = 0;
  for (int i = 0; i < n; ++i) {
    int j = list[i];
    for (int ff = 0; ff < num_ff; ++ff) {
      int base_idx = ff * nall * 3 + j * 3;
      h_temp(base_idx + 0) += buf[m++];
      h_temp(base_idx + 1) += buf[m++];
      h_temp(base_idx + 2) += buf[m++];
    }
  }

  Kokkos::deep_copy(d_full_f, h_temp);
}

template<class DeviceType>
std::tuple<double, double, double, double, double, double>
PairNEPKokkos<DeviceType>::calc_max_error(const int* ilist, const double* h_full_f, const double* h_full_e, const int num_ff, const int inum, const int nlocal, const int nall, const int rank)
{
    double num_ff_inv = 1.0 / num_ff;
    std::vector<double> f_ave(inum * 3, 0.0);
    std::vector<double> ei_ave(inum, 0.0);
    int i_index = 0;
    for (int ff = 0; ff < num_ff; ++ff) {
      for (int i = 0; i < inum; ++i) {
        i_index = ilist[i];
        int base_idx_f = ff * nall * 3 + i_index * 3;
        int base_idx_e = ff * nlocal + i_index;

        f_ave[i * 3 + 0] += h_full_f[base_idx_f + 0];
        f_ave[i * 3 + 1] += h_full_f[base_idx_f + 1];
        f_ave[i * 3 + 2] += h_full_f[base_idx_f + 2];

        ei_ave[i] += h_full_e[base_idx_e];
        // printf(" e[%d][%d] = %f f[%d][%d] = %f %f %f\n", ff, i, h_full_e[base_idx_e], ff, i, h_full_f[base_idx_f + 0], h_full_f[base_idx_f + 1], h_full_f[base_idx_f + 2]);
      }
    }

    for (int i = 0; i < inum * 3; ++i) {
        f_ave[i] *= num_ff_inv;
    }
    for (int i = 0; i < inum; ++i) {
        ei_ave[i] *= num_ff_inv;
    }

    std::vector<double> f_err_sq_sum(inum, 0.0);
    std::vector<double> ei_err_sq_sum(inum, 0.0);

    double avg_f_err = 0.0;
    double avg_ei_err = 0.0;

    double min_f_err = 1e9, max_f_err = 0.0;
    double min_ei_err = 1e9, max_ei_err = 0.0;
    double _tmp_f = 0.0;
    double _tmp_ei = 0.0;

    for (int ff = 0; ff < num_ff; ++ff) {
      for (int i = 0; i < inum; ++i) {
        i_index = ilist[i];
        int base_idx_f = ff * nall * 3 + i_index * 3;
        int base_idx_e = ff * nlocal + i_index;

        double dfx = h_full_f[base_idx_f + 0] - f_ave[i * 3 + 0];
        double dfy = h_full_f[base_idx_f + 1] - f_ave[i * 3 + 1];
        double dfz = h_full_f[base_idx_f + 2] - f_ave[i * 3 + 2];
        f_err_sq_sum[i]  += dfx*dfx + dfy*dfy + dfz*dfz;

        ei_err_sq_sum[i] += (h_full_e[base_idx_e] - ei_ave[i])*(h_full_e[base_idx_e] - ei_ave[i]);

      }
    }

    for (int i = 0; i < inum; ++i) {
      _tmp_f = sqrt(f_err_sq_sum[i] / num_ff);
      _tmp_ei  = sqrt(ei_err_sq_sum[i]/ num_ff);
      min_f_err = std::min(min_f_err, _tmp_f);
      max_f_err = std::max(max_f_err, _tmp_f);
      min_ei_err = std::min(min_ei_err, _tmp_ei);
      max_ei_err = std::max(max_ei_err, _tmp_ei);
      avg_f_err += _tmp_f;
      avg_ei_err += _tmp_ei;
    }

    return std::make_tuple(
        avg_f_err, 
        max_f_err, 
        min_f_err, 
        avg_ei_err,
        max_ei_err, 
        min_ei_err 
    );
}
/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class PairNEPKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairNEPKokkos<LMPHostType>;
#endif
}
