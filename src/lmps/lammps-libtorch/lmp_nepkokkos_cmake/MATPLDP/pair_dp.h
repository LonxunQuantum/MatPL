/* -*- c++ -*- ----------------------------------------------------------
     PWmat-MLFF to LAMMPS
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(matpl/dp, PairDP);
// clang-format on
#else

#ifndef LMP_PAIR_MATPLDP_H
#define LMP_PAIR_MATPLDP_H
#include "pair.h"
#include <iostream>
#include <torch/script.h>
#include <torch/torch.h>
namespace LAMMPS_NS {
    class PairDP : public Pair {
        public:
            PairDP(class LAMMPS *);
            ~PairDP() override;

            int nmax;
            double*** f_n;
            double** e_atom_n;

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

            std::tuple<std::vector<int>, std::vector<int>, std::vector<double>> generate_neighdata();
            std::tuple<std::vector<int>, std::vector<int>, std::vector<double>> convert_dim(bool is_build_neighbor=false);

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
            std::vector<int> imagetype_map, neighbor_list;        
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
            std::string model_name;

            // std::vector<int> imagetype, imagetype_map, neighbor_list;
            // std::vector<int> use_type;
            std::vector<double> dR_neigh;   // for dp with jit, nep with jit

            std::vector<int> pre_itype; // the previous step atom types (local and ghost)
            int pre_nall = 0; // the previous step atom nums (local + ghost)
            int pre_nlocal = 0;
            int pre_nghost = 0;
            bool change_neighbor = false;
    };
}
#endif
#endif
