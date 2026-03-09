#include <iostream>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include "pair_nep.h"
#include "nep_cpu.h"
#include "atom.h"
#include "comm.h"
#include "error.h"
#include "fix.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "update.h"
#include "domain.h"
#include <dlfcn.h>
#include <algorithm>
using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairNEP::PairNEP(LAMMPS *lmp) : Pair(lmp)
{
    me = comm->me;
	writedata = 1;
    comm_reverse = 3;

    restartinfo = 0;//  set to 0 if your pair style does not store data in restart files
    manybody_flag = 1; //set to 1 if your pair style is not pair-wise additive
    single_enable = 0; 
    // copymode = 0;
    // allocated = 0;

}

PairNEP::~PairNEP()
{
    // if (copymode)
    //     return;

    if (allocated) 
    {
        memory->destroy(setflag);
        memory->destroy(cutsq);
        // memory->destroy(f_n);
        // memory->destroy(e_atom_n);
    }

    if (me == 0 && explrError_fp != nullptr) {
        fclose(explrError_fp);
        explrError_fp = nullptr;
    }

}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairNEP::allocate()
{
    allocated = 1;
    int np1 = atom->ntypes ;
    memory->create(setflag, np1 + 1, np1 + 1, "pair:setflag");
    for (int i = 1; i <= np1; i++)
        for (int j = i; j <= np1; j++) setflag[i][j] = 0;
    memory->create(cutsq, np1 + 1, np1 + 1, "pair:cutsq");

}

/* ----------------------------------------------------------------------
   global settings pair_style 
------------------------------------------------------------------------- */

void PairNEP::settings(int narg, char** arg)
{
    int ff_idx;
    int iarg = 0;  // index of first forcefield file
    int rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (narg <= 0) error->all(FLERR, "Illegal pair_style command"); // numbers of args after 'pair_style matpl'
    std::vector<std::string> models;

    num_ff = 0;
    while (iarg < narg) {
        std::string arg_str(arg[iarg]);
        if (arg_str.find(".txt") != std::string::npos) {
            models.push_back(arg_str);
            num_ff++;
            iarg++;
        } else {
            break;
        }
    }
    while (iarg < narg) {
        if (strcmp(arg[iarg], "out_freq") == 0) {
            out_freq = utils::inumeric(FLERR, arg[++iarg], false, lmp);
        } else if (strcmp(arg[iarg], "out_file") == 0) {
            explrError_fname = arg[++iarg];
        } 
        iarg++;
    }

    if (me == 0 and num_ff > 1) {
        explrError_fp = fopen(&explrError_fname[0], "w");
        fprintf(explrError_fp, "%9s %16s %16s %16s %16s %16s %16s\n", 
        "#    step", "avg_devi_f", "min_devi_f", "max_devi_f", 
        "avg_devi_e", "min_devi_e", "max_devi_e");
        fflush(explrError_fp);
    }

    if (me == 0) utils::logmesg(this -> lmp, "<---- Loading model ---->");
    
    nep_cpu_models.resize(num_ff);

    for (ff_idx = 0; ff_idx < num_ff; ff_idx++) {
        std::string model_file = models[ff_idx];
        // load txt format model file, it should be nep
        bool is_rank_0 = (comm->me == 0);
        // if the gpu nums > 0 and libnep.so is exsits, use gpu model
        if (ff_idx > 0) {
            is_rank_0 = false; //for print
        }
        nep_cpu_models[ff_idx].read_neptxt(model_file, is_rank_0);
        if (me == 0) printf("\nLoading txt model file:   %s\n", model_file.c_str());
    }
    cutoff = nep_cpu_models[0].paramb.rc_radial;
    // memory->create(f_n, num_ff, nmax, 3, "pair_matpl:f_n");
    // memory->create(e_atom_n, num_ff, nmax, "pair_matpl:e_atom_n");
} 

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs pair_coeff 
------------------------------------------------------------------------- */
int PairNEP::find_atomic_number(std::string& key) {
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

void PairNEP::coeff(int narg, char** arg)
{
    if (!allocated) { allocate(); }

    // pair_coeff * * 
    int ilo, ihi, jlo, jhi;
    utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error); // arg[0] = *
    utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error); // arg[1] = *

    int count = 0;
    for(int i = ilo; i <= ihi; i++) {
        for(int j = MAX(jlo,i); j <= jhi; j++) 
        {
            setflag[i][j] = 1;
            count++;
        }
    }

    std::vector<int> atom_type_module = nep_cpu_models[0].element_atomic_number_list;
    model_ntypes = atom_type_module.size();
    // if (ntypes > model_ntypes || ntypes != narg - 2)  // type numbers in strucutre file and in pair_coeff should be the same
    // {
    //     error->all(FLERR, "Element mapping is not correct, ntypes = " + std::to_string(ntypes));
    // }
    for (int ii = 2; ii < narg; ++ii) {
        std::string element = utils::strdup(arg[ii]);  // LAMMPS提供的安全转换
        int temp = find_atomic_number(element);
        // int temp = std::stoi(arg[ii]);
        auto iter = std::find(atom_type_module.begin(), atom_type_module.end(), temp);   
        if (iter != atom_type_module.end() || arg[ii] == 0)
        {
            int index = std::distance(atom_type_module.begin(), iter);
            model_atom_type_idx.push_back(index); 
            atom_types.push_back(temp);
            for(int jj=0; jj < num_ff; ++jj){
                nep_cpu_models[jj].map_atom_types.push_back(temp);
                nep_cpu_models[jj].map_atom_type_idx.push_back(index);
            }
            // std::cout<<"=== the config atom type "<< temp << " index in ff is "  << index << std::endl;
        } else {
            error->all(FLERR, "This element is not included in the machine learning force field");
        }
    }
   if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairNEP::init_one(int i, int j)
{
    //if (setflag[i][j] == 0) { error->all(FLERR, "All pair coeffs are not set"); 

    return cutoff;
}


void PairNEP::init_style()
{
    if (force->newton_pair == 0) error->all(FLERR, "Pair style MATPL requires newton pair on");
    // Using a nearest neighbor table of type full
    neighbor->add_request(this, NeighConst::REQ_FULL);

    cutoffsq = cutoff * cutoff;
    int n = atom->ntypes;
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= n; j++)
        cutsq[i][j] = cutoffsq;
}
/* ---------------------------------------------------------------------- */

std::tuple<double, double, double, double, double, double> PairNEP::calc_max_error(double ***f_n, double **e_atom_n)
{
    int i, j;
    int ff_idx;
    double num_ff_inv;
    int nlocal = atom->nlocal;
    // int *tag = atom->tag;
    num_ff_inv = 1.0 / num_ff;

    for (ff_idx = 0; ff_idx < num_ff; ff_idx++) {
        p_ff_idx = ff_idx;
        comm->reverse_comm(this);
    }

    std::vector<double> f_ave;
    std::vector<double> f_err[num_ff];
    std::vector<double> f_max_meanff;
    std::vector<double> ei_ave;
    std::vector<double> ei_err[num_ff];
    std::vector<double> ei_max_meanff;

    f_ave.resize(nlocal * 3);
    ei_ave.resize(nlocal);
    f_max_meanff.resize(nlocal);
    ei_max_meanff.resize(nlocal);
    for (ff_idx = 0; ff_idx < num_ff; ff_idx++) {
        f_err[ff_idx].resize(nlocal * 3);
        ei_err[ff_idx].resize(nlocal);
    }

    // sum over all models
    for (ff_idx = 0; ff_idx < num_ff; ff_idx++) {
        for (i = 0; i < nlocal; i++) {
            // std::cout << "f_n[" << ff_idx << "][" << i << "][0] = " << tag[i] << " " << f_n[ff_idx][i][0] << std::endl;
            f_ave[i * 3 + 0] += f_n[ff_idx][i][0];
            f_ave[i * 3 + 1] += f_n[ff_idx][i][1];
            f_ave[i * 3 + 2] += f_n[ff_idx][i][2];
            ei_ave[i] += e_atom_n[ff_idx][i];
            // std::cout<< "ff " << ff_idx << " i " << i << " ei " <<  e_atom_n[ff_idx][i] << " force " << f_n[ff_idx][i][0] << " "  << f_n[ff_idx][i][1] << " "  << f_n[ff_idx][i][2] << std::endl;
        }
    }

    // calc ensemble average
    for (i = 0; i < 3 * nlocal; i++) {
        f_ave[i] *= num_ff_inv;
    }
    for (i = 0; i < nlocal; i++) {
        ei_ave[i] *= num_ff_inv;
    }

    // calc error
    for (ff_idx = 0; ff_idx < num_ff; ff_idx++) {
        for (i = 0; i < nlocal; i++) {
            f_err[ff_idx][i * 3 + 0] = f_n[ff_idx][i][0] - f_ave[i * 3 + 0];
            f_err[ff_idx][i * 3 + 1] = f_n[ff_idx][i][1] - f_ave[i * 3 + 1];
            f_err[ff_idx][i * 3 + 2] = f_n[ff_idx][i][2] - f_ave[i * 3 + 2];
            ei_err[ff_idx][i] = e_atom_n[ff_idx][i] - ei_ave[i];
        }
    }

    // find max error
    for (ff_idx = 0; ff_idx < num_ff; ff_idx++) {
        for (j = 0; j < nlocal * 3; j += 3) {
            f_max_meanff[j / 3] += f_err[ff_idx][j] * f_err[ff_idx][j] + f_err[ff_idx][j + 1] * f_err[ff_idx][j + 1] + f_err[ff_idx][j + 2] * f_err[ff_idx][j + 2];
        }
        for (j = 0; j < nlocal; j++) {
            ei_max_meanff[j] += ei_err[ff_idx][j] * ei_err[ff_idx][j];
        }
    }

    double min_f_err, max_f_err, avg_f_err, min_ei_err, max_ei_err, avg_ei_err;
    min_f_err = 10000;
    max_f_err = 0.0;
    avg_f_err = 0.0;
    min_ei_err = 10000;
    max_ei_err = 0.0;
    avg_ei_err = 0.0;

    double _tmp_f = 0.0;
    double _tmp_ei = 0.0;
    // find max_mean error
    for (j = 0; j < nlocal; j++) {
        _tmp_f = sqrt(f_max_meanff[j] / num_ff);
        _tmp_ei  = sqrt(ei_max_meanff[j]/ num_ff);
        if (min_f_err > _tmp_f) min_f_err = _tmp_f;
        if (max_f_err < _tmp_f) max_f_err = _tmp_f;
        if (min_ei_err > _tmp_ei) min_ei_err = _tmp_ei;
        if (max_ei_err < _tmp_ei) max_ei_err = _tmp_ei;
        avg_f_err  += _tmp_f;
        avg_ei_err += _tmp_ei;
    }

    return std::make_tuple(avg_f_err, max_f_err, min_f_err, avg_ei_err, max_ei_err, min_ei_err);
}

int PairNEP::pack_reverse_comm(int n, int first, double* buf) {
    int i, m, last;

    m = 0;
    last = first + n;
    for (i = first; i < last; i++) {
        buf[m++] = f_n[p_ff_idx][i][0];
        buf[m++] = f_n[p_ff_idx][i][1];
        buf[m++] = f_n[p_ff_idx][i][2];
    }
    return m;
}

void PairNEP::unpack_reverse_comm(int n, int* list, double* buf) {
    int i, j, m;

    m = 0;
    for (i = 0; i < n; i++) {
        j = list[i];
        f_n[p_ff_idx][j][0] += buf[m++];
        f_n[p_ff_idx][j][1] += buf[m++];
        f_n[p_ff_idx][j][2] += buf[m++];
    }

}

void PairNEP::grow_memory(int nall)
{
  if (nmax == 0) {
    nmax = nall;
    memory->create(f_n, num_ff, nmax, 3, "pair_matpl:f_n");
    memory->create(e_atom_n, num_ff, nmax, "pair_matpl:e_atom_n");
  } else if (nmax < nall) {
    nmax = nall;
    memory->grow(f_n, num_ff, nmax, 3, "pair_matpl:f_n");
    memory->grow(e_atom_n, num_ff, nmax, "pair_matpl:e_atom_n");
  }
}

void PairNEP::compute(int eflag, int vflag)
{
    if (eflag || vflag) ev_setup(eflag, vflag);

    // int newton_pair = force->newton_pair;
    int ff_idx=0;
    int nlocal = atom->nlocal;
    int current_timestep = update->ntimestep;
    // int total_timestep = update->laststep;
    int nghost = atom->nghost;
    int n_all = nlocal + nghost;
    // int inum, jnum, itype, jtype;

    double* per_atom_potential = nullptr;
    double** per_atom_virial = nullptr;
    double *virial = force->pair->virial;
    double **f = atom->f;
    // for dp and nep model from jitscript
    if (num_ff > 1) {
        grow_memory(n_all);
    }

    if (num_ff == 1) {
        double total_potential = 0.0;
        double total_virial[6] = {0.0};
        if (cvflag_atom) {
            per_atom_virial = cvatom;
        }
        if (eflag_atom) {
            per_atom_potential = eatom;
        }
        nep_cpu_models[0].compute_for_lammps(
        atom->nlocal, list->inum, list->ilist, list->numneigh, list->firstneigh, atom->type, atom->x,
        total_potential, total_virial, per_atom_potential, atom->f, per_atom_virial, ff_idx);
        if (eflag) {
            eng_vdwl += total_potential;
        }
        if (vflag) {
            for (int component = 0; component < 6; ++component) {
            virial[component] += total_virial[component];
            }
        }
    }
    else {
        // printf("namx %d atom->nmax %d nall %d\n", nmax, atom->nmax, n_all);
        if (cvflag_atom) {
            per_atom_virial = cvatom;
        }
        if (eflag_atom) {
            per_atom_potential = eatom;
        }

        for (ff_idx = 0; ff_idx < num_ff; ff_idx++) {
            if (ff_idx > 0 && (current_timestep % out_freq != 0)) continue;
            double total_potential = 0.0;
            double total_virial[6] = {0.0};
            // for multi models, the output step, should calculate deviation
            for (int i = 0; i < n_all; i++) {
                // printf(" ==== ff_idx %d atom %d num_ff %d n_all %d====\n", ff_idx, i, num_ff, n_all);
                f_n[ff_idx][i][0] = 0.0;
                f_n[ff_idx][i][1] = 0.0;
                f_n[ff_idx][i][2] = 0.0;
            }
            for (int i = 0; i < atom->nlocal; i++) {
                e_atom_n[ff_idx][i] = 0.0;
            }
            nep_cpu_models[ff_idx].compute_for_lammps(
                atom->nlocal, list->inum, list->ilist, list->numneigh, list->firstneigh,atom->type, atom->x,
                total_potential, total_virial, e_atom_n[ff_idx], f_n[ff_idx], per_atom_virial, ff_idx);
            if (ff_idx == 0) {
                for (int i = 0; i < n_all; i++) {
                    f[i][0] = f_n[0][i][0];
                    f[i][1] = f_n[0][i][1];
                    f[i][2] = f_n[0][i][2];
                }
                if (eflag_atom) {
                    for (int i = 0; i < atom->nlocal; i++) {
                        per_atom_potential[i] = e_atom_n[0][i];
                    }
                }
                if (eflag) {
                    eng_vdwl = total_potential;
                }
                if (vflag) {
                    for (int component = 0; component < 6; ++component) {
                        virial[component] += total_virial[component];
                    }
                }
            } // else multi models out steps
        }   // for ff_idx      
    } // model_type == 1: nep_cpu version
    //   exploration mode.
    //   calculate the error of the force

    // for deviation of multi models
    if (num_ff > 1 && (current_timestep % out_freq == 0)) {
        // calculate model deviation with Force
        std::tuple<double, double, double, double, double, double> result = calc_max_error(f_n, e_atom_n);

        double avg_f_err, max_f_err, min_f_err, avg_ei_err, max_ei_err, min_ei_err;
        double glb_avg_f_err, glb_max_f_err, glb_min_f_err, glb_avg_ei_err, glb_max_ei_err, glb_min_ei_err;

        avg_f_err = std::get<0>(result);
        max_f_err = std::get<1>(result);
        min_f_err = std::get<2>(result);
        avg_ei_err = std::get<3>(result);
        max_ei_err = std::get<4>(result);
        min_ei_err = std::get<5>(result);

        // max_err = result.first;
        // max_err_ei = result.second;

        MPI_Allreduce(&max_f_err, &glb_max_f_err, 1, MPI_DOUBLE, MPI_MAX, world);
        MPI_Allreduce(&min_f_err, &glb_min_f_err, 1, MPI_DOUBLE, MPI_MIN, world);
        MPI_Allreduce(&avg_f_err, &glb_avg_f_err, 1, MPI_DOUBLE, MPI_SUM, world);

        MPI_Allreduce(&max_ei_err, &glb_max_ei_err, 1, MPI_DOUBLE, MPI_MAX, world);
        MPI_Allreduce(&min_ei_err, &glb_min_ei_err, 1, MPI_DOUBLE, MPI_MIN, world);
        MPI_Allreduce(&avg_ei_err, &glb_avg_ei_err, 1, MPI_DOUBLE, MPI_SUM, world);

        if (atom->natoms > 0) {
            glb_avg_f_err /= double(atom->natoms);
            glb_avg_ei_err /= double(atom->natoms);
        }

        max_err_list.push_back(glb_max_f_err);
        max_err_ei_list.push_back(glb_max_ei_err);

        if (current_timestep % out_freq == 0) {
            if (me == 0) {
                // fprintf(explrError_fp, "%9d %16.9f %16.9f\n", (max_err_list.size()-1)*out_freq, global_max_err, global_max_err_ei);
                fprintf(explrError_fp, "%9d %16.9f %16.9f %16.9f %16.9f %16.9f %16.9f\n", 
                            current_timestep, glb_avg_f_err, glb_min_f_err, glb_max_f_err, 
                                glb_avg_ei_err, glb_min_ei_err, glb_max_ei_err);
                fflush(explrError_fp);
            } 
        }
    }
    
    // std::cout << "t4 " << (t5 - t4).count() * 0.000001 << "\tms" << std::endl;
    // std::cout << "t5 " << (t6 - t5).count() * 0.000001 << "\tms" << std::endl;
    // std::cout << "t6 " << (t7 - t6).count() * 0.000001 << "\tms" << std::endl;
}
