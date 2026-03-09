/* -*- c++ -*- ----------------------------------------------------------
     PWmat-MLFF to LAMMPS
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(matpl, PairMATPL);
// clang-format on
#else



#ifndef LMP_PAIR_MLFF_H
#define LMP_PAIR_MLFF_H
#include "nep_cpu.h"
#ifdef USE_CUDA
#include "NEP_GPU/force/nep3.cuh"
#endif
#include "pair.h"
#include <iostream>
#include <torch/script.h>
#include <torch/torch.h>
namespace LAMMPS_NS {
    class PairMATPL : public Pair {
        public:
            PairMATPL(class LAMMPS *);
            ~PairMATPL() override;

            int nmax;
            double*** f_n;
            double** e_atom_n;

            std::tuple<std::vector<int>, std::vector<int>, std::vector<double>> generate_neighdata();
            std::tuple<std::vector<int>, std::vector<int>, std::vector<int>, std::vector<double>> generate_neighdata_nep();
            std::tuple<std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<float>> generate_neighdata_nep_gpu();
            std::tuple<std::vector<int>, std::vector<int>, std::vector<double>> convert_dim(bool is_build_neighbor=false);

            void compute(int, int) override;
            void settings(int, char **) override;
            void coeff(int, char **) override;
            double init_one(int, int) override;
            void init_style() override;
            int pack_reverse_comm(int, int, double* ) override;
            void unpack_reverse_comm(int, int*, double* ) override;
            void grow_memory(int nall);
            std::tuple<double, double, double, double, double, double> calc_max_error(double***, double**);

        protected:
            virtual void allocate();
            int find_atomic_number(std::string&);        
        private:
            int me;
            int num_ff;
            int p_ff_idx;
            unsigned seed;

            bool use_nep_gpu;
            
            // NEP3 nep_gpu_model;
            #ifdef USE_CUDA
            std::vector<NEP3> nep_gpu_models;
            #endif
            // NEP3_CPU nep_cpu_model;
            std::vector<NEP3_CPU> nep_cpu_models;

            torch::jit::script::Module module;//dp and nep jit model
            std::vector<torch::jit::script::Module> modules;

            std::vector<double> max_err_list;
            std::vector<double> max_err_ei_list;
            std::vector<double> min_err_list;
            std::vector<double> min_err_ei_list;
            std::vector<double> max_mean_err_list;
            std::vector<double> max_mean_err_ei_list;
            std::string explrError_fname = "explr.error";
            std::FILE *explrError_fp = nullptr;
            int out_freq = 1;

            torch::Device device = torch::kCPU;
            torch::Dtype dtype = torch::kFloat32;
            std::vector<int> atom_types;           // use for jit models
            std::vector<int> model_atom_type_idx;  // use for jit models 
            int model_ntypes;
            int model_type; // 0 for jitmodel(dp or nep) 1 for nep_cpu 2 for nep_gpu
            // DP params
            double cutoff;
            double cutoffsq;
            // NEP params
            double cutoff_radial;
            double cutoff_angular;
            //common params
            int max_neighbor;
            int nep_gpu_nm = 500; //maxneighbor of nep_gpu, fixed
            std::string model_name;

            // std::vector<int> imagetype, imagetype_map, neighbor_list;
            // std::vector<int> use_type;
            std::vector<int> imagetype_map; // for nep with jit
            std::vector<int> itype_convert_map, neighbor_list, neigbor_num_list, neighbor_angular_list, neigbor_angular_num_list; // for nep gpu, the raidal also for nep jit
            std::vector<int> neighbor_type_list; // for nep find neigh and forward
            std::vector<double> dR_neigh;   // for dp with jit, nep with jit
            std::vector<float> rij_nep_gpu; // for nep gpu

            std::vector<double> position_cpu; // for nep gpu optim
            std::vector<int>   firstneighbor_cpu; // for nep gpu optim

            std::vector<int> pre_itype; // the previous step atom types (local and ghost)
            int pre_nall = 0; // the previous step atom nums (local + ghost)
            int pre_nlocal = 0;
            int pre_nghost = 0;
            bool change_neighbor = false;
    };
}
#endif
#endif
