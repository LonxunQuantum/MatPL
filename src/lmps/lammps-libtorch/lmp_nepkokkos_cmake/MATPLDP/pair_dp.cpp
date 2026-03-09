#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include "pair_dp.h"
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
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif
using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairDP::PairDP(LAMMPS *lmp) : Pair(lmp)
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

PairDP::~PairDP()
{
    // if (copymode)
    //     return;

    if (allocated) 
    {
        memory->destroy(setflag);
        memory->destroy(cutsq);
        memory->destroy(f_n);
        memory->destroy(e_atom_n);
    }
    if (me == 0 && explrError_fp != nullptr) {
        fclose(explrError_fp);
        explrError_fp = nullptr;
    }
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairDP::allocate()
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

void PairDP::settings(int narg, char** arg)
{
    int ff_idx;
    int iarg = 0;  // index of first forcefield file
    int rank;
    int num_devices;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    num_devices = torch::cuda::device_count();

    if (narg <= 0) error->all(FLERR, "Illegal pair_style command"); // numbers of args after 'pair_style matpl'
    std::vector<std::string> models;

    num_ff = 0;
    while (iarg < narg) {
        std::string arg_str(arg[iarg]);
        if (arg_str.find(".pt") != std::string::npos || arg_str.find(".txt") != std::string::npos) {
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

    // device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    torch::DeviceType device_type = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    device = device_type == torch::kCUDA ?  torch::Device(device_type, rank % num_devices) : torch::Device(device_type);
    dtype = torch::kFloat64;

    // std::cout<<"the numbor of gpu is " <<  num_devices << std::endl;
    // std::cout<<"the mpi process is " << comm->nprocs << std::endl;
    
    if (me == 0) utils::logmesg(this -> lmp, "<---- Loading model ---->\n");
    
    for (ff_idx = 0; ff_idx < num_ff; ff_idx++) {
        std::string model_file = models[ff_idx];
        try
        {   // load jit model, they are DP
            // module = torch::jit::load(model_file, c10::Device(device)); 
            module = torch::jit::load(model_file, device);
            module.to(dtype);
            // module.eval();
            modules.push_back(module);
            if (me == 0) printf("\nLoading jitscript model file:   %s\n", model_file.c_str());
            model_type = 0;
        }
        catch (const c10::Error e)
        {
           std::cerr << "Failed to load model :" << e.msg() << std::endl;
        }
    }
    if (model_type == 0) {
        if ((device_type == torch::kCUDA) && (rank == 0)) {
            if (num_devices < comm->nprocs) {
            std::cout << "----------------------------------------------------------------------------------" << std::endl;
            std::cout << " Warning: There are " << num_devices << " GPUs available " << std::endl;
            std::cout << " But have " << comm->nprocs << " MPI processes, may result in poor performance!!!" << std::endl;
            std::cout << "----------------------------------------------------------------------------------" << std::endl;
            }
        }
        try {
            torch::jit::IValue model_type_value = module.attr("model_type");
            model_name = model_type_value.toString()->string();
        }
        catch (const torch::jit::ObjectAttributeError& e) {
            model_name = "DP";
        }
        if (model_name == "DP") {
            cutoff = module.attr("Rmax").toDouble();
        } else {  
            std::cout << "ERROR: the model_type of input model " << model_name << " is not supported! Please check the input model! " << std::endl;
            error->universe_all(FLERR, "ERROR: the model_type of input model is not supported! Please check the input model!");
        }

        //common params of DP
        max_neighbor = module.attr("maxNeighborNum").toInt();
        
        // print information
        if (me == 0) {
        utils::logmesg(this -> lmp, "<---- Load model successful!!! ---->");
        printf("\nDevice:       %s", device == torch::kCPU ? "CPU" : "GPU");
        printf("\nModel type:   %5d",5);
        printf("\nModel nums:   %5d",num_ff);
        printf("\ncutoff :      %12.6f",cutoff);
        printf("\nmax_neighbor: %5d\n", max_neighbor);
        }
    } 
    // since we need num_ff, so well allocate memory here
    // but not in allocate()
    nmax = num_ff;
    memory->create(f_n, num_ff, nmax, 3, "pair_dp:f_n");
    memory->create(e_atom_n, num_ff, nmax, "pair_dp:e_atom_n");
} 

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs pair_coeff 
------------------------------------------------------------------------- */
int PairDP::find_atomic_number(std::string& key) {
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

void PairDP::coeff(int narg, char** arg)
{
    // int ntypes = atom->ntypes;
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

    if (model_type == 0) {
        auto atom_type_module = module.attr("atom_type").toList();
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
            }
            else
            {
                error->all(FLERR, "This element is not included in the machine learning force field");
            }
        }
    }
   if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairDP::init_one(int i, int j)
{
    //if (setflag[i][j] == 0) { error->all(FLERR, "All pair coeffs are not set"); 

    return cutoff;
}


void PairDP::init_style()
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

std::tuple<double, double, double, double, double, double> PairDP::calc_max_error(double ***f_n, double **e_atom_n)
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

int PairDP::pack_reverse_comm(int n, int first, double* buf) {
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

void PairDP::unpack_reverse_comm(int n, int* list, double* buf) {
    int i, j, m;

    m = 0;
    for (i = 0; i < n; i++) {
        j = list[i];
        f_n[p_ff_idx][j][0] += buf[m++];
        f_n[p_ff_idx][j][1] += buf[m++];
        f_n[p_ff_idx][j][2] += buf[m++];
    }

}

void PairDP::grow_memory(int nall)
{
  if (nmax < nall) {
    // printf("allocate new %7d %7d %7d\n", update->ntimestep, nmax, nall);
    nmax = nall;
    memory->grow(f_n, num_ff, nmax, 3, "pair_matpl:f_n");
    memory->grow(e_atom_n, num_ff, nmax, "pair_matpl:e_atom_n");
  }
}

std::tuple<std::vector<int>, std::vector<int>, std::vector<double>> PairDP::generate_neighdata()
{   
    int i, j, ii, jj, inum, jnum, itype, jtype;
    double delx, dely, delz, rsq, rij;
    int *ilist, *jlist, *numneigh, **firstneigh;
    int etnum;

    double **x = atom->x;
    int *type = atom->type;
    int nlocal = atom->nlocal;
    int nghost = atom->nghost;
    // int ntypes = atom->ntypes;
    int ntypes = model_ntypes;
    int n_all = nlocal + nghost;
    double rc2 = cutoff * cutoff;

    double min_dR = 1000;
    double min_dR_all;

    inum = list->inum;
    ilist = list->ilist;
    numneigh = list->numneigh;
    firstneigh = list->firstneigh;

    std::vector<std::vector<int>> num_neigh(inum, std::vector<int>(ntypes));
    // imagetype.resize(inum);
    imagetype_map.resize(inum);
    neighbor_list.resize(inum * ntypes * max_neighbor);
    dR_neigh.resize(inum * ntypes * max_neighbor * 4);
    // use_type.resize(n_all);
    std::vector<int> type_to_model(n_all);


    for (int ii = 0; ii < n_all; ii++)
    {
        // use_type[ii] = atom_types[type[ii] - 1];
        type_to_model[ii] = model_atom_type_idx[type[ii] - 1] + 1;
        // type[0], type[1], type[2], type[3], type[4], : 2, 2, 1, 2, 2, ...
        // atom_types[0], atom_types[1] : 6, 1
        // use_type[0], use_type[1], use_type[2], use_type[3], use_type[4] : 1, 1, 6, 1, 1, ...
    }
    for (i = 0; i < nlocal; i++) {
        for (j = 0; j < ntypes; j++) {
            num_neigh[i][j] = 0;
        }
    }
    for (ii = 0; ii < inum; ii++)               // local atoms: 5, CH4
    {    
        i = ilist[ii];                          // 0, 1, 2, 3, 4
        // itype = type[i];                        // 2, 2, 1, 2, 2
        itype = type_to_model[i];                   // 1, 1, 3, 1, 1
        jlist = firstneigh[i];
        jnum = numneigh[i];                     // 4, 4, 4, 4, 4
        imagetype_map[ii] = itype - 1;          // 1, 1, 0, 1, 1        python index from 0
        // imagetype[ii] = use_type[i];            // 1, 1, 6, 1, 1

        for (jj = 0; jj < jnum; jj++)
        {
            j = jlist[jj];                      // 1, 2, 3, 4;   0, 2, 3, 4;   0, 1, 3, 4;   0, 1, 2, 4;   0, 1, 2, 3
            delx = x[j][0] - x[i][0];
            dely = x[j][1] - x[i][1];
            delz = x[j][2] - x[i][2];
            rsq = delx * delx + dely * dely + delz * delz;
            // jtype = type[j];                    // 2, 1, 2, 2;   2, 1, 2, 2;   2, 2, 2, 2;   2, 2, 1, 2;   2, 2, 1, 2
            jtype = type_to_model[j];               // 1, 3, 1, 1;   1, 3, 1, 1;   1, 1, 1, 1;   1, 1, 3, 1;   1, 1, 3, 1
            if (rsq <= rc2) 
            {
                etnum = num_neigh[i][jtype - 1];
                rij = sqrt(rsq);
                int index = i * ntypes * max_neighbor + (jtype - 1) * max_neighbor + etnum;
                dR_neigh[index * 4 + 0] = rij;
                dR_neigh[index * 4 + 1] = delx;
                dR_neigh[index * 4 + 2] = dely;
                dR_neigh[index * 4 + 3] = delz;
                neighbor_list[index] = j + 1;
                num_neigh[i][jtype - 1] += 1;
                // std::cout << "num_neigh[" << i << "][" << jtype - 1 << "] = " << num_neigh[i][jtype - 1] << std::endl;
                if (rsq < min_dR) min_dR = rsq;
                // std::cout<< "nlist index " << index << " j " << j+1 << std::endl;
            }
        }
    }

    MPI_Allreduce(&min_dR, &min_dR_all, 1, MPI_DOUBLE, MPI_MIN, world);

    if (min_dR_all < 0.61) {
        if (me == 0) {
            std::cout << "ERROR: there are two atoms too close, min_dR_all = " << min_dR_all << std::endl;
        }
        error->universe_all(FLERR, "there are two atoms too close");
    }
    return std::make_tuple(std::move(imagetype_map), std::move(neighbor_list), std::move(dR_neigh));
    // return std::make_tuple(imagetype, imagetype_map, neighbor_list, dR_neigh);
}

void PairDP::compute(int eflag, int vflag)
{
    if (eflag || vflag) ev_setup(eflag, vflag);
    
    // int newton_pair = force->newton_pair;
    int ff_idx;
    int nlocal = atom->nlocal;
    int current_timestep = update->ntimestep;
    // int total_timestep = update->laststep;
    // bool calc_virial_from_mlff = false;
    // bool calc_egroup_from_mlff = false;
    int ntypes = atom->ntypes;
    int nghost = atom->nghost;
    int n_all = nlocal + nghost;
    // int inum, jnum, itype, jtype;
    if (num_ff > 1) {
        grow_memory(n_all);
    }
    // bool is_build_neighbor = false;

    double *virial = force->pair->virial;
    double **f = atom->f;
    // for dp  from jitscript
    if (model_type == 0) {

        int inum = list->inum;
        // double **x = atom->x;
        // int *type = atom->type;
        // auto t4 = std::chrono::high_resolution_clock::now();
        std::vector<int> imagetype_map, neighbor_list, neighbor_type_list;
        std::vector<double> dR_neigh;

        if (model_name == "DP") {
            std::tie(imagetype_map, neighbor_list, dR_neigh) = generate_neighdata();
        }
        if (inum == 0) return;
        // auto t5 = std::chrono::high_resolution_clock::now();
        auto int_tensor_options = torch::TensorOptions().dtype(torch::kInt);
        auto float_tensor_options = torch::TensorOptions().dtype(torch::kFloat64);
        torch::Tensor imagetype_map_tensor = torch::from_blob(imagetype_map.data(), {inum}, int_tensor_options).to(device);
        torch::Tensor neighbor_list_tensor = torch::from_blob(neighbor_list.data(), {1, inum, max_neighbor * model_ntypes}, int_tensor_options).to(device);
        torch::Tensor dR_neigh_tensor = torch::from_blob(dR_neigh.data(), {1, inum, max_neighbor * model_ntypes, 4}, float_tensor_options).to(device,dtype);
        torch::Tensor atom_type_tensor = torch::from_blob(atom_types.data(), {ntypes}, int_tensor_options).to(device);
        
        torch::Tensor neighbor_type_list_tensor;
        // auto t6 = std::chrono::high_resolution_clock::now();
        /*
        do forward for 4 models
        only 1 is used for MD
        1, 2, 3, 4 all for the deviation
        */
        for (ff_idx = 0; ff_idx < num_ff; ff_idx++) {
            if (ff_idx > 0 && (current_timestep % out_freq != 0)) continue;
            torch::Tensor Force;
            torch::Tensor Ei;
            torch::Tensor Etot;
            torch::Tensor Virial;
            if (model_name == "DP") {
                auto output = modules[ff_idx].forward({neighbor_list_tensor, imagetype_map_tensor, atom_type_tensor, dR_neigh_tensor, nghost}).toTuple();
                Force = output->elements()[2].toTensor().to(torch::kCPU);
                Ei = output->elements()[1].toTensor().to(torch::kCPU);
                if (ff_idx == 0) {
                    Etot = output->elements()[0].toTensor().to(torch::kCPU);
                    Virial = output->elements()[4].toTensor().to(torch::kCPU);            
                }
            }
            auto F_ptr = Force.accessor<double, 3>();
            auto Ei_ptr = Ei.accessor<double, 2>();

            if (num_ff > 1 && (current_timestep % out_freq == 0)) {
                for (int i = 0; i < inum + nghost; i++)
                {
                    // f_n[ff_idx][i][0] = Force[0][i][0].item<double>();
                    // f_n[ff_idx][i][1] = Force[0][i][1].item<double>();
                    // f_n[ff_idx][i][2] = Force[0][i][2].item<double>();
                    f_n[ff_idx][i][0] = F_ptr[0][i][0];
                    f_n[ff_idx][i][1] = F_ptr[0][i][1];
                    f_n[ff_idx][i][2] = F_ptr[0][i][2];
                }
                for (int ii = 0; ii < inum; ii++) {
                    // e_atom_n[ff_idx][ii] = Ei[0][ii].item<double>();
                    e_atom_n[ff_idx][ii] = Ei_ptr[0][ii];
                }
            }

            if (ff_idx == 0) {
                // Etot = output->elements()[0].toTensor().to(torch::kCPU);
                // Virial = output->elements()[4].toTensor().to(torch::kCPU);
                // if (output->elements()[4].isTensor()) {
                //     calc_virial_from_mlff = true;
                //     torch::Tensor Virial = output->elements()[4].toTensor().to(torch::kCPU);
                // } else
                //     auto Virial = output->elements()[4];
                // get force

                // auto F_ptr = Force.accessor<double, 3>();
                // auto Ei_ptr = Ei.accessor<double, 2>();
                auto V_ptr = Virial.accessor<double, 2>();

                for (int i = 0; i < inum + nghost; i++)
                {
                    f[i][0] = F_ptr[0][i][0];
                    f[i][1] = F_ptr[0][i][1];
                    f[i][2] = F_ptr[0][i][2];
                }

                virial[0] = V_ptr[0][0];    // xx
                virial[1] = V_ptr[0][4];    // yy
                virial[2] = V_ptr[0][8];    // zz
                virial[3] = V_ptr[0][1];    // xy
                virial[4] = V_ptr[0][2];    // xz
                virial[5] = V_ptr[0][5];    // yz

                // get energy
                if (eflag) eng_vdwl = Etot[0][0].item<double>();

                if (eflag_atom)
                {
                    for (int ii = 0; ii < inum; ii++) {
                        eatom[ii] = Ei_ptr[0][ii];
                    }
                }
                // If virial needed calculate via F dot r.
                // if (vflag_fdotr) virial_fdotr_compute();
            }
        }
    } // if model_type == 0
    
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
