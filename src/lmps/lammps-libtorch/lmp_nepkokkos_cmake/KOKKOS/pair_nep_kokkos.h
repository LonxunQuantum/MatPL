/* -*- c++ -*- ----------------------------------------------------------
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

#ifdef PAIR_CLASS
// clang-format off
PairStyle(matpl/nep/kk, PairNEPKokkos<LMPDeviceType>);
PairStyle(matpl/nep/kk/device, PairNEPKokkos<LMPDeviceType>);
PairStyle(matpl/nep/kk/host, PairNEPKokkos<LMPHostType>);
// clang-format on
#else

#ifndef LMP_PAIR_NEP_KOKKOS_H
#define LMP_PAIR_NEP_KOKKOS_H

#include "pair_kokkos.h"
#include "pair_nep.h"
#include "neigh_list_kokkos.h"
#include "nepkk.cuh"
#include <utility>

namespace LAMMPS_NS {

template<class DeviceType>
class PairNEPKokkos : public PairNEP {
 public:
  enum {EnabledNeighFlags=HALF|HALFTHREAD|FULL};
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typedef EV_FLOAT value_type;

  PairNEPKokkos(class LAMMPS *);
  ~PairNEPKokkos() override;
  
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  double init_one(int, int) override;
  void init_style() override;
  void allocate();
  int find_atomic_number(std::string& key);
  int pack_reverse_comm(int, int, double *);
  void unpack_reverse_comm(int, int *, double *);
  std::tuple<double, double, double, double, double, double> calc_max_error(const int* ilist, const double* h_full_f, const double* h_full_e, const int num_ff, const int inum, const int nlocal, const int nall, const int rank);
 protected:
  
  // 坐标和力的一维视图包装器
   Kokkos::View<const double*, DeviceType> get_position_view() {
   return Kokkos::View<const double*, DeviceType>(reinterpret_cast<const double*>(x.data()), nall * 3);
   }

  // 邻居列表的一维扁平化视图
   Kokkos::View<const int*, DeviceType> get_neighbors_flat_view() {
   int max_neighbors = d_neighbors.extent(1);
   int row_num = d_neighbors.extent(0);
   return Kokkos::View<const int*, DeviceType>(
      reinterpret_cast<const int*>(d_neighbors.data()), 
      row_num * max_neighbors
   );
  }
  
  int me;
  int num_ff;
  bool is_rank_0;
  int nprocs_total;
  int global_nall = 0;
  int global_nlocal = 0;
  bool reverse_force = false;
  std::vector<NEPKK> nep_gpu_models;  // NEP model instance
  std::vector<std::string> potential_files;

  std::string explrError_fname = "explr.error";
  std::FILE *explrError_fp = nullptr;
  int out_freq = 1;
  double cutoff, cutoffsq;
  
  // Kokkos data
  int inum, nlocal, nall;
  typename AT::t_x_array_randomread x;
  typename AT::t_f_array f;
  typename AT::t_int_1d_randomread type;

  DAT::tdual_ffloat_1d k_full_f;  // DualView for full force [nall][3]
  HAT::t_ffloat_1d h_full_f;
  DAT::tdual_ffloat_1d k_full_e;  // DualView for full force [nall][3]
  HAT::t_ffloat_1d h_full_e;
  
  DAT::tdual_efloat_1d k_eatom;
  DAT::tdual_virial_array k_vatom;
//   DAT::tdual_ffloat_2d k_cutsq;
  typename AT::t_efloat_1d d_eatom;
  typename AT::t_virial_array d_vatom; //device [nall][6]
  
  typename AT::t_int_1d_randomread d_ilist;
  typename AT::t_int_1d_randomread d_numneigh;
  typename AT::t_neighbors_2d d_neighbors;
  
  int need_dup;
  int neighflag, newton_pair;
  int eflag, vflag;
  int rank, device_id;
  std::vector<double> h_etot_virial_global;
  // NEP data structures - Kokkos Views for raw pointers

  using KKDeviceType = typename KKDevice<DeviceType>::value;

  template<typename DataType, typename Layout>
  using DupScatterView = KKScatterView<DataType, Layout, KKDeviceType, KKScatterSum, KKScatterDuplicated>;

  template<typename DataType, typename Layout>
  using NonDupScatterView = KKScatterView<DataType, Layout, KKDeviceType, KKScatterSum, KKScatterNonDuplicated>;

  DupScatterView<F_FLOAT*[3], typename DAT::t_f_array::array_layout> dup_f;
  DupScatterView<E_FLOAT*, typename DAT::t_efloat_1d::array_layout> dup_eatom;
  DupScatterView<F_FLOAT*[6], typename DAT::t_virial_array::array_layout> dup_vatom;

  NonDupScatterView<F_FLOAT*[3], typename DAT::t_f_array::array_layout> ndup_f;
  NonDupScatterView<E_FLOAT*, typename DAT::t_efloat_1d::array_layout> ndup_eatom;
  NonDupScatterView<F_FLOAT*[6], typename DAT::t_virial_array::array_layout> ndup_vatom;

  friend void pair_virial_fdotr_compute<PairNEPKokkos>(PairNEPKokkos*);
};

}

#endif
#endif