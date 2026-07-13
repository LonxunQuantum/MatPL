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

#include "nep_cpu.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <complex>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace
{
const int MAX_NEURON = 200; // maximum number of neurons in the hidden layer
const int NUM_OF_ABC = 24;  // 3 + 5 + 7 + 9 for L_max = 4
const int MAX_NUM_N = 20;   // n_max+1 = 19+1
const int MAX_DIM = MAX_NUM_N * 7;
const int MAX_DIM_ANGULAR = MAX_NUM_N * 6;
const double C3B[NUM_OF_ABC] = {
  0.238732414637843, 0.119366207318922, 0.119366207318922, 0.099471839432435, 0.596831036594608,
  0.596831036594608, 0.149207759148652, 0.149207759148652, 0.139260575205408, 0.104445431404056,
  0.104445431404056, 1.044454314040563, 1.044454314040563, 0.174075719006761, 0.174075719006761,
  0.011190581936149, 0.223811638722978, 0.223811638722978, 0.111905819361489, 0.111905819361489,
  1.566681471060845, 1.566681471060845, 0.195835183882606, 0.195835183882606};
const double C4B[5] = {
  -0.007499480826664, -0.134990654879954, 0.067495327439977, 0.404971964639861, -0.809943929279723};
const double C5B[3] = {0.026596810706114, 0.053193621412227, 0.026596810706114};

const double COVALENT_RADIUS[94] = {
  0.426667, 0.613333, 1.6,     1.25333, 1.02667, 1.0,     0.946667, 0.84,
  0.853333, 0.893333, 1.86667, 1.66667, 1.50667, 1.38667, 1.46667, 1.36,
  1.32,     1.28,     2.34667, 2.05333, 1.77333, 1.62667, 1.61333, 1.46667,
  1.42667,  1.38667,  1.33333, 1.32,    1.34667, 1.45333, 1.49333, 1.45333,
  1.53333,  1.46667,  1.52,    1.56,    2.52,     2.22667, 1.96,     1.85333,
  1.76,     1.65333,  1.53333, 1.50667, 1.50667, 1.44,    1.53333,  1.64,
  1.70667,  1.68,     1.68,     1.64,     1.76,    1.74667, 2.78667, 2.34667,
  2.16,     1.96,     2.10667,  2.09333,  2.08,     2.06667, 2.01333, 2.02667,
  2.01333,  2.0,      1.98667,  1.98667, 1.97333,  2.04,    1.94667, 1.82667,
  1.74667,  1.64,     1.57333,  1.54667, 1.48,     1.49333, 1.50667, 1.76,
  1.73333,  1.73333,  1.81333,  1.74667, 1.84,    1.89333, 2.68,     2.41333,
  2.22667,  2.10667,  2.02667,  2.04,     2.05333, 2.06667
};


const double K_C_SP = 14.399645; // 1/(4*PI*epsilon_0)
const double PI = 3.141592653589793;
const double PI_HALF = 1.570796326794897;
const int NUM_ELEMENTS = 103;
const std::string ELEMENTS[NUM_ELEMENTS] = {
  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si", "P",
  "S",  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
  "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh",
  "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
  "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re",
  "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
  "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"};

int countNonEmptyLines(const std::string& filename) {
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

void apply_ann_one_layer(
  const int dim,
  const int num_neurons1,
  const double* w0,
  const double* b0,
  const double* w1,
  const double* b1,
  double* q,
  double& energy,
  double* energy_derivative,
  double* latent_space,
  const int atom_type_i,
  const int model_version)
{
  for (int n = 0; n < num_neurons1; ++n) {
    double w0_times_q = 0.0;
    for (int d = 0; d < dim; ++d) {
      w0_times_q += w0[n * dim + d] * q[d];
    }
    double x1 = tanh(w0_times_q - b0[n]);
    latent_space[n] = w1[n] * x1; // also try x1
    energy += w1[n] * x1;
    for (int d = 0; d < dim; ++d) {
      double y1 = (1.0 - x1 * x1) * w0[n * dim + d];
      energy_derivative[d] += w1[n] * y1;
    }
  }
  energy -= (model_version == 4 ? b1[atom_type_i] : b1[0]);
  // std::cout << "ann last bias " <<  b1[atom_type_i] <<" type " << atom_type_i << " b1[0] " << b1[0] << std::endl;
}


void apply_ann_one_layer_nep5(
  const int dim,
  const int num_neurons1,
  const double* w0,
  const double* b0,
  const double* w1,
  const double* b1,
  double* q,
  double& energy,
  double* energy_derivative,
  double* latent_space,
  const int atom_type_i,
  const int model_version)
{
  for (int n = 0; n < num_neurons1; ++n) {
    double w0_times_q = 0.0;
    for (int d = 0; d < dim; ++d) {
      w0_times_q += w0[n * dim + d] * q[d];
    }
    double x1 = tanh(w0_times_q - b0[n]);
    latent_space[n] = w1[n] * x1; // also try x1
    energy += w1[n] * x1;
    for (int d = 0; d < dim; ++d) {
      double y1 = (1.0 - x1 * x1) * w0[n * dim + d];
      energy_derivative[d] += w1[n] * y1;
    }
  }
  energy -= w1[num_neurons1] + b1[0]; // typewise bias + common bias
  // std::cout << "ann last bias " <<  b1[atom_type_i] <<" type " << atom_type_i << " b1[0] " << b1[0] << std::endl;
}

void apply_ann_one_layer_charge(
  const int dim,
  const int num_neurons1,
  const double* w0,
  const double* b0,
  const double* w1,
  const double* b1,
  double* q,
  double& energy,
  double* energy_derivative,
  double& charge,
  double* charge_derivative)
{
  for (int n = 0; n < num_neurons1; ++n) {
    double w0_times_q = 0.0;
    for (int d = 0; d < dim; ++d) {
      w0_times_q += w0[n * dim + d] * q[d];
    }
    double x1 = tanh(w0_times_q - b0[n]);
    double tanh_derivative = 1.0 - x1 * x1;
    energy += w1[n] * x1;
    charge += w1[n + num_neurons1] * x1;
    for (int d = 0; d < dim; ++d) {
      double y1 = tanh_derivative * w0[n * dim + d];
      energy_derivative[d] += w1[n] * y1;
      charge_derivative[d] += w1[n + num_neurons1] * y1;
    }
  }
  energy -= b1[0];
}

void find_fc(double rc, double rcinv, double d12, double& fc)
{
  if (d12 < rc) {
    double x = d12 * rcinv;
    fc = 0.5 * cos(PI * x) + 0.5;
  } else {
    fc = 0.0;
  }
}

void find_fc_and_fcp(double rc, double rcinv, double d12, double& fc, double& fcp)
{
  if (d12 < rc) {
    double x = d12 * rcinv;
    fc = 0.5 * cos(PI * x) + 0.5;
    fcp = -PI_HALF * sin(PI * x);
    fcp *= rcinv;
  } else {
    fc = 0.0;
    fcp = 0.0;
  }
}

void find_fc_and_fcp_zbl(double r1, double r2, double d12, double& fc, double& fcp)
{
  if (d12 < r1) {
    fc = 1.0;
    fcp = 0.0;
  } else if (d12 < r2) {
    double pi_factor = PI / (r2 - r1);
    fc = cos(pi_factor * (d12 - r1)) * 0.5 + 0.5;
    fcp = -sin(pi_factor * (d12 - r1)) * pi_factor * 0.5;
  } else {
    fc = 0.0;
    fcp = 0.0;
  }
}

void find_phi_and_phip_zbl(double a, double b, double x, double& phi, double& phip)
{
  double tmp = a * exp(-b * x);
  phi += tmp;
  phip -= b * tmp;
}

void find_f_and_fp_zbl(
  double zizj,
  double a_inv,
  double rc_inner,
  double rc_outer,
  double d12,
  double d12inv,
  double& f,
  double& fp)
{
  double x = d12 * a_inv;
  f = fp = 0.0;
  double Zbl_para[8] = {0.18175, 3.1998, 0.50986, 0.94229, 0.28022, 0.4029, 0.02817, 0.20162};
  find_phi_and_phip_zbl(Zbl_para[0], Zbl_para[1], x, f, fp);
  find_phi_and_phip_zbl(Zbl_para[2], Zbl_para[3], x, f, fp);
  find_phi_and_phip_zbl(Zbl_para[4], Zbl_para[5], x, f, fp);
  find_phi_and_phip_zbl(Zbl_para[6], Zbl_para[7], x, f, fp);
  f *= zizj;
  fp *= zizj * a_inv;
  fp = fp * d12inv - f * d12inv * d12inv;
  f *= d12inv;
  double fc, fcp;
  find_fc_and_fcp_zbl(rc_inner, rc_outer, d12, fc, fcp);
  fp = fp * fc + f * fcp;
  f *= fc;
}

void find_f_and_fp_zbl(
  double* zbl_para,
  double zizj,
  double a_inv,
  double d12,
  double d12inv,
  double& f,
  double& fp)
{
  double x = d12 * a_inv;
  f = fp = 0.0f;
  find_phi_and_phip_zbl(zbl_para[2], zbl_para[3], x, f, fp);
  find_phi_and_phip_zbl(zbl_para[4], zbl_para[5], x, f, fp);
  find_phi_and_phip_zbl(zbl_para[6], zbl_para[7], x, f, fp);
  find_phi_and_phip_zbl(zbl_para[8], zbl_para[9], x, f, fp);
  f *= zizj;
  fp *= zizj * a_inv;
  fp = fp * d12inv - f * d12inv * d12inv;
  f *= d12inv;
  double fc, fcp;
  find_fc_and_fcp_zbl(zbl_para[0], zbl_para[1], d12, fc, fcp);
  fp = fp * fc + f * fcp;
  f *= fc;
}

void find_fn(const int n, const double rcinv, const double d12, const double fc12, double& fn)
{
  if (n == 0) {
    fn = fc12;
  } else if (n == 1) {
    double x = 2.0 * (d12 * rcinv - 1.0) * (d12 * rcinv - 1.0) - 1.0;
    fn = (x + 1.0) * 0.5 * fc12;
  } else {
    double x = 2.0 * (d12 * rcinv - 1.0) * (d12 * rcinv - 1.0) - 1.0;
    double t0 = 1.0;
    double t1 = x;
    double t2;
    for (int m = 2; m <= n; ++m) {
      t2 = 2.0 * x * t1 - t0;
      t0 = t1;
      t1 = t2;
    }
    fn = (t2 + 1.0) * 0.5 * fc12;
  }
}

void find_fn_and_fnp(
  const int n,
  const double rcinv,
  const double d12,
  const double fc12,
  const double fcp12,
  double& fn,
  double& fnp)
{
  if (n == 0) {
    fn = fc12;
    fnp = fcp12;
  } else if (n == 1) {
    double x = 2.0 * (d12 * rcinv - 1.0) * (d12 * rcinv - 1.0) - 1.0;
    fn = (x + 1.0) * 0.5;
    fnp = 2.0 * (d12 * rcinv - 1.0) * rcinv * fc12 + fn * fcp12;
    fn *= fc12;
  } else {
    double x = 2.0 * (d12 * rcinv - 1.0) * (d12 * rcinv - 1.0) - 1.0;
    double t0 = 1.0;
    double t1 = x;
    double t2;
    double u0 = 1.0;
    double u1 = 2.0 * x;
    double u2;
    for (int m = 2; m <= n; ++m) {
      t2 = 2.0 * x * t1 - t0;
      t0 = t1;
      t1 = t2;
      u2 = 2.0 * x * u1 - u0;
      u0 = u1;
      u1 = u2;
    }
    fn = (t2 + 1.0) * 0.5;
    fnp = n * u0 * 2.0 * (d12 * rcinv - 1.0) * rcinv;
    fnp = fnp * fc12 + fn * fcp12;
    fn *= fc12;
  }
}

void find_fn(const int n_max, const double rcinv, const double d12, const double fc12, double* fn)
{
  double x = 2.0 * (d12 * rcinv - 1.0) * (d12 * rcinv - 1.0) - 1.0;
  fn[0] = 1.0;
  fn[1] = x;
  for (int m = 2; m <= n_max; ++m) {
    fn[m] = 2.0 * x * fn[m - 1] - fn[m - 2];
  }
  for (int m = 0; m <= n_max; ++m) {
    fn[m] = (fn[m] + 1.0) * 0.5 * fc12;
  }
}

void find_fn_and_fnp(
  const int n_max,
  const double rcinv,
  const double d12,
  const double fc12,
  const double fcp12,
  double* fn,
  double* fnp)
{
  double x = 2.0 * (d12 * rcinv - 1.0) * (d12 * rcinv - 1.0) - 1.0;
  fn[0] = 1.0;
  fnp[0] = 0.0;
  fn[1] = x;
  fnp[1] = 1.0;
  double u0 = 1.0;
  double u1 = 2.0 * x;
  double u2;
  for (int m = 2; m <= n_max; ++m) {
    fn[m] = 2.0 * x * fn[m - 1] - fn[m - 2];
    fnp[m] = m * u1;
    u2 = 2.0 * x * u1 - u0;
    u0 = u1;
    u1 = u2;
  }
  for (int m = 0; m <= n_max; ++m) {
    fn[m] = (fn[m] + 1.0) * 0.5;
    fnp[m] *= 2.0 * (d12 * rcinv - 1.0) * rcinv;
    fnp[m] = fnp[m] * fc12 + fn[m] * fcp12;
    fn[m] *= fc12;
  }
}

void get_f12_1(
  const double d12inv,
  const double fn,
  const double fnp,
  const double Fp,
  const double* s,
  const double* r12,
  double* f12)
{
  double tmp = s[1] * r12[0];
  tmp += s[2] * r12[1];
  tmp *= 2.0;
  tmp += s[0] * r12[2];
  tmp *= Fp * fnp * d12inv * 2.0;
  for (int d = 0; d < 3; ++d) {
    f12[d] += tmp * r12[d];
  }
  tmp = Fp * fn * 2.0;
  f12[0] += tmp * 2.0 * s[1];
  f12[1] += tmp * 2.0 * s[2];
  f12[2] += tmp * s[0];
}

void get_f12_2(
  const double d12,
  const double d12inv,
  const double fn,
  const double fnp,
  const double Fp,
  const double* s,
  const double* r12,
  double* f12)
{
  double tmp = s[1] * r12[0] * r12[2];               // Re[Y21]
  tmp += s[2] * r12[1] * r12[2];                     // Im[Y21]
  tmp += s[3] * (r12[0] * r12[0] - r12[1] * r12[1]); // Re[Y22]
  tmp += s[4] * 2.0 * r12[0] * r12[1];               // Im[Y22]
  tmp *= 2.0;
  tmp += s[0] * (3.0 * r12[2] * r12[2] - d12 * d12); // Y20
  tmp *= Fp * fnp * d12inv * 2.0;
  for (int d = 0; d < 3; ++d) {
    f12[d] += tmp * r12[d];
  }
  tmp = Fp * fn * 4.0;
  f12[0] += tmp * (-s[0] * r12[0] + s[1] * r12[2] + 2.0 * s[3] * r12[0] + 2.0 * s[4] * r12[1]);
  f12[1] += tmp * (-s[0] * r12[1] + s[2] * r12[2] - 2.0 * s[3] * r12[1] + 2.0 * s[4] * r12[0]);
  f12[2] += tmp * (2.0 * s[0] * r12[2] + s[1] * r12[0] + s[2] * r12[1]);
}

void get_f12_4body(
  const double d12,
  const double d12inv,
  const double fn,
  const double fnp,
  const double Fp,
  const double* s,
  const double* r12,
  double* f12)
{
  double fn_factor = Fp * fn;
  double fnp_factor = Fp * fnp * d12inv;
  double y20 = (3.0 * r12[2] * r12[2] - d12 * d12);

  // derivative wrt s[0]
  double tmp0 = C4B[0] * 3.0 * s[0] * s[0] + C4B[1] * (s[1] * s[1] + s[2] * s[2]) +
                C4B[2] * (s[3] * s[3] + s[4] * s[4]);
  double tmp1 = tmp0 * y20 * fnp_factor;
  double tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] - tmp2 * 2.0 * r12[0];
  f12[1] += tmp1 * r12[1] - tmp2 * 2.0 * r12[1];
  f12[2] += tmp1 * r12[2] + tmp2 * 4.0 * r12[2];

  // derivative wrt s[1]
  tmp0 = C4B[1] * s[0] * s[1] * 2.0 - C4B[3] * s[3] * s[1] * 2.0 + C4B[4] * s[2] * s[4];
  tmp1 = tmp0 * r12[0] * r12[2] * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] + tmp2 * r12[2];
  f12[1] += tmp1 * r12[1];
  f12[2] += tmp1 * r12[2] + tmp2 * r12[0];

  // derivative wrt s[2]
  tmp0 = C4B[1] * s[0] * s[2] * 2.0 + C4B[3] * s[3] * s[2] * 2.0 + C4B[4] * s[1] * s[4];
  tmp1 = tmp0 * r12[1] * r12[2] * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0];
  f12[1] += tmp1 * r12[1] + tmp2 * r12[2];
  f12[2] += tmp1 * r12[2] + tmp2 * r12[1];

  // derivative wrt s[3]
  tmp0 = C4B[2] * s[0] * s[3] * 2.0 + C4B[3] * (s[2] * s[2] - s[1] * s[1]);
  tmp1 = tmp0 * (r12[0] * r12[0] - r12[1] * r12[1]) * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] + tmp2 * 2.0 * r12[0];
  f12[1] += tmp1 * r12[1] - tmp2 * 2.0 * r12[1];
  f12[2] += tmp1 * r12[2];

  // derivative wrt s[4]
  tmp0 = C4B[2] * s[0] * s[4] * 2.0 + C4B[4] * s[1] * s[2];
  tmp1 = tmp0 * (2.0 * r12[0] * r12[1]) * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] + tmp2 * 2.0 * r12[1];
  f12[1] += tmp1 * r12[1] + tmp2 * 2.0 * r12[0];
  f12[2] += tmp1 * r12[2];
}

void get_f12_5body(
  const double d12,
  const double d12inv,
  const double fn,
  const double fnp,
  const double Fp,
  const double* s,
  const double* r12,
  double* f12)
{
  double fn_factor = Fp * fn;
  double fnp_factor = Fp * fnp * d12inv;
  double s1_sq_plus_s2_sq = s[1] * s[1] + s[2] * s[2];

  // derivative wrt s[0]
  double tmp0 = C5B[0] * 4.0 * s[0] * s[0] * s[0] + C5B[1] * s1_sq_plus_s2_sq * 2.0 * s[0];
  double tmp1 = tmp0 * r12[2] * fnp_factor;
  double tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0];
  f12[1] += tmp1 * r12[1];
  f12[2] += tmp1 * r12[2] + tmp2;

  // derivative wrt s[1]
  tmp0 = C5B[1] * s[0] * s[0] * s[1] * 2.0 + C5B[2] * s1_sq_plus_s2_sq * s[1] * 4.0;
  tmp1 = tmp0 * r12[0] * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] + tmp2;
  f12[1] += tmp1 * r12[1];
  f12[2] += tmp1 * r12[2];

  // derivative wrt s[2]
  tmp0 = C5B[1] * s[0] * s[0] * s[2] * 2.0 + C5B[2] * s1_sq_plus_s2_sq * s[2] * 4.0;
  tmp1 = tmp0 * r12[1] * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0];
  f12[1] += tmp1 * r12[1] + tmp2;
  f12[2] += tmp1 * r12[2];
}

void get_f12_3(
  const double d12,
  const double d12inv,
  const double fn,
  const double fnp,
  const double Fp,
  const double* s,
  const double* r12,
  double* f12)
{
  double d12sq = d12 * d12;
  double x2 = r12[0] * r12[0];
  double y2 = r12[1] * r12[1];
  double z2 = r12[2] * r12[2];
  double xy = r12[0] * r12[1];
  double xz = r12[0] * r12[2];
  double yz = r12[1] * r12[2];

  double tmp = s[1] * (5.0 * z2 - d12sq) * r12[0];
  tmp += s[2] * (5.0 * z2 - d12sq) * r12[1];
  tmp += s[3] * (x2 - y2) * r12[2];
  tmp += s[4] * 2.0 * xy * r12[2];
  tmp += s[5] * r12[0] * (x2 - 3.0 * y2);
  tmp += s[6] * r12[1] * (3.0 * x2 - y2);
  tmp *= 2.0;
  tmp += s[0] * (5.0 * z2 - 3.0 * d12sq) * r12[2];
  tmp *= Fp * fnp * d12inv * 2.0;
  for (int d = 0; d < 3; ++d) {
    f12[d] += tmp * r12[d];
  }

  // x
  tmp = s[1] * (4.0 * z2 - 3.0 * x2 - y2);
  tmp += s[2] * (-2.0 * xy);
  tmp += s[3] * 2.0 * xz;
  tmp += s[4] * (2.0 * yz);
  tmp += s[5] * (3.0 * (x2 - y2));
  tmp += s[6] * (6.0 * xy);
  tmp *= 2.0;
  tmp += s[0] * (-6.0 * xz);
  f12[0] += tmp * Fp * fn * 2.0;
  // y
  tmp = s[1] * (-2.0 * xy);
  tmp += s[2] * (4.0 * z2 - 3.0 * y2 - x2);
  tmp += s[3] * (-2.0 * yz);
  tmp += s[4] * (2.0 * xz);
  tmp += s[5] * (-6.0 * xy);
  tmp += s[6] * (3.0 * (x2 - y2));
  tmp *= 2.0;
  tmp += s[0] * (-6.0 * yz);
  f12[1] += tmp * Fp * fn * 2.0;
  // z
  tmp = s[1] * (8.0 * xz);
  tmp += s[2] * (8.0 * yz);
  tmp += s[3] * (x2 - y2);
  tmp += s[4] * (2.0 * xy);
  tmp *= 2.0;
  tmp += s[0] * (9.0 * z2 - 3.0 * d12sq);
  f12[2] += tmp * Fp * fn * 2.0;
}

void get_f12_4(
  const double x,
  const double y,
  const double z,
  const double r,
  const double rinv,
  const double fn,
  const double fnp,
  const double Fp,
  const double* s,
  double* f12)
{
  const double r2 = r * r;
  const double x2 = x * x;
  const double y2 = y * y;
  const double z2 = z * z;
  const double xy = x * y;
  const double xz = x * z;
  const double yz = y * z;
  const double xyz = x * yz;
  const double x2my2 = x2 - y2;

  double tmp = s[1] * (7.0 * z2 - 3.0 * r2) * xz; // Y41_real
  tmp += s[2] * (7.0 * z2 - 3.0 * r2) * yz;       // Y41_imag
  tmp += s[3] * (7.0 * z2 - r2) * x2my2;          // Y42_real
  tmp += s[4] * (7.0 * z2 - r2) * 2.0 * xy;       // Y42_imag
  tmp += s[5] * (x2 - 3.0 * y2) * xz;             // Y43_real
  tmp += s[6] * (3.0 * x2 - y2) * yz;             // Y43_imag
  tmp += s[7] * (x2my2 * x2my2 - 4.0 * x2 * y2);  // Y44_real
  tmp += s[8] * (4.0 * xy * x2my2);               // Y44_imag
  tmp *= 2.0;
  tmp += s[0] * ((35.0 * z2 - 30.0 * r2) * z2 + 3.0 * r2 * r2); // Y40
  tmp *= Fp * fnp * rinv * 2.0;
  f12[0] += tmp * x;
  f12[1] += tmp * y;
  f12[2] += tmp * z;

  // x
  tmp = s[1] * z * (7.0 * z2 - 3.0 * r2 - 6.0 * x2);  // Y41_real
  tmp += s[2] * (-6.0 * xyz);                         // Y41_imag
  tmp += s[3] * 4.0 * x * (3.0 * z2 - x2);            // Y42_real
  tmp += s[4] * 2.0 * y * (7.0 * z2 - r2 - 2.0 * x2); // Y42_imag
  tmp += s[5] * 3.0 * z * x2my2;                      // Y43_real
  tmp += s[6] * 6.0 * xyz;                            // Y43_imag
  tmp += s[7] * 4.0 * x * (x2 - 3.0 * y2);            // Y44_real
  tmp += s[8] * 4.0 * y * (3.0 * x2 - y2);            // Y44_imag
  tmp *= 2.0;
  tmp += s[0] * 12.0 * x * (r2 - 5.0 * z2); // Y40
  f12[0] += tmp * Fp * fn * 2.0;
  // y
  tmp = s[1] * (-6.0 * xyz);                          // Y41_real
  tmp += s[2] * z * (7.0 * z2 - 3.0 * r2 - 6.0 * y2); // Y41_imag
  tmp += s[3] * 4.0 * y * (y2 - 3.0 * z2);            // Y42_real
  tmp += s[4] * 2.0 * x * (7.0 * z2 - r2 - 2.0 * y2); // Y42_imag
  tmp += s[5] * (-6.0 * xyz);                         // Y43_real
  tmp += s[6] * 3.0 * z * x2my2;                      // Y43_imag
  tmp += s[7] * 4.0 * y * (y2 - 3.0 * x2);            // Y44_real
  tmp += s[8] * 4.0 * x * (x2 - 3.0 * y2);            // Y44_imag
  tmp *= 2.0;
  tmp += s[0] * 12.0 * y * (r2 - 5.0 * z2); // Y40
  f12[1] += tmp * Fp * fn * 2.0;
  // z
  tmp = s[1] * 3.0 * x * (5.0 * z2 - r2);  // Y41_real
  tmp += s[2] * 3.0 * y * (5.0 * z2 - r2); // Y41_imag
  tmp += s[3] * 12.0 * z * x2my2;          // Y42_real
  tmp += s[4] * 24.0 * xyz;                // Y42_imag
  tmp += s[5] * x * (x2 - 3.0 * y2);       // Y43_real
  tmp += s[6] * y * (3.0 * x2 - y2);       // Y43_imag
  tmp *= 2.0;
  tmp += s[0] * 16.0 * z * (5.0 * z2 - 3.0 * r2); // Y40
  f12[2] += tmp * Fp * fn * 2.0;
}

void accumulate_f12(
  const int n,
  const int n_max_angular_plus_1,
  const double d12,
  const double* r12,
  double fn,
  double fnp,
  const double* Fp,
  const double* sum_fxyz,
  double* f12)
{
  const double d12inv = 1.0 / d12;
  // l = 1
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  double s1[3] = {
    sum_fxyz[n * NUM_OF_ABC + 0] * C3B[0], sum_fxyz[n * NUM_OF_ABC + 1] * C3B[1],
    sum_fxyz[n * NUM_OF_ABC + 2] * C3B[2]};
  get_f12_1(d12inv, fn, fnp, Fp[n], s1, r12, f12);
  // l = 2
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  double s2[5] = {
    sum_fxyz[n * NUM_OF_ABC + 3] * C3B[3], sum_fxyz[n * NUM_OF_ABC + 4] * C3B[4],
    sum_fxyz[n * NUM_OF_ABC + 5] * C3B[5], sum_fxyz[n * NUM_OF_ABC + 6] * C3B[6],
    sum_fxyz[n * NUM_OF_ABC + 7] * C3B[7]};
  get_f12_2(d12, d12inv, fn, fnp, Fp[n_max_angular_plus_1 + n], s2, r12, f12);
  // l = 3
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  double s3[7] = {sum_fxyz[n * NUM_OF_ABC + 8] * C3B[8],   sum_fxyz[n * NUM_OF_ABC + 9] * C3B[9],
                  sum_fxyz[n * NUM_OF_ABC + 10] * C3B[10], sum_fxyz[n * NUM_OF_ABC + 11] * C3B[11],
                  sum_fxyz[n * NUM_OF_ABC + 12] * C3B[12], sum_fxyz[n * NUM_OF_ABC + 13] * C3B[13],
                  sum_fxyz[n * NUM_OF_ABC + 14] * C3B[14]};
  get_f12_3(d12, d12inv, fn, fnp, Fp[2 * n_max_angular_plus_1 + n], s3, r12, f12);
  // l = 4
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  double s4[9] = {sum_fxyz[n * NUM_OF_ABC + 15] * C3B[15], sum_fxyz[n * NUM_OF_ABC + 16] * C3B[16],
                  sum_fxyz[n * NUM_OF_ABC + 17] * C3B[17], sum_fxyz[n * NUM_OF_ABC + 18] * C3B[18],
                  sum_fxyz[n * NUM_OF_ABC + 19] * C3B[19], sum_fxyz[n * NUM_OF_ABC + 20] * C3B[20],
                  sum_fxyz[n * NUM_OF_ABC + 21] * C3B[21], sum_fxyz[n * NUM_OF_ABC + 22] * C3B[22],
                  sum_fxyz[n * NUM_OF_ABC + 23] * C3B[23]};
  get_f12_4(
    r12[0], r12[1], r12[2], d12, d12inv, fn, fnp, Fp[3 * n_max_angular_plus_1 + n], s4, f12);
}

void accumulate_f12_with_4body(
  const int n,
  const int n_max_angular_plus_1,
  const double d12,
  const double* r12,
  double fn,
  double fnp,
  const double* Fp,
  const double* sum_fxyz,
  double* f12)
{
  const double d12inv = 1.0 / d12;
  // l = 1
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  double s1[3] = {
    sum_fxyz[n * NUM_OF_ABC + 0] * C3B[0], sum_fxyz[n * NUM_OF_ABC + 1] * C3B[1],
    sum_fxyz[n * NUM_OF_ABC + 2] * C3B[2]};
  get_f12_1(d12inv, fn, fnp, Fp[n], s1, r12, f12);
  // l = 2
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  double s2[5] = {
    sum_fxyz[n * NUM_OF_ABC + 3], sum_fxyz[n * NUM_OF_ABC + 4], sum_fxyz[n * NUM_OF_ABC + 5],
    sum_fxyz[n * NUM_OF_ABC + 6], sum_fxyz[n * NUM_OF_ABC + 7]};
  get_f12_4body(d12, d12inv, fn, fnp, Fp[4 * n_max_angular_plus_1 + n], s2, r12, f12);
  s2[0] *= C3B[3];
  s2[1] *= C3B[4];
  s2[2] *= C3B[5];
  s2[3] *= C3B[6];
  s2[4] *= C3B[7];
  get_f12_2(d12, d12inv, fn, fnp, Fp[n_max_angular_plus_1 + n], s2, r12, f12);
  // l = 3
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  double s3[7] = {sum_fxyz[n * NUM_OF_ABC + 8] * C3B[8],   sum_fxyz[n * NUM_OF_ABC + 9] * C3B[9],
                  sum_fxyz[n * NUM_OF_ABC + 10] * C3B[10], sum_fxyz[n * NUM_OF_ABC + 11] * C3B[11],
                  sum_fxyz[n * NUM_OF_ABC + 12] * C3B[12], sum_fxyz[n * NUM_OF_ABC + 13] * C3B[13],
                  sum_fxyz[n * NUM_OF_ABC + 14] * C3B[14]};
  get_f12_3(d12, d12inv, fn, fnp, Fp[2 * n_max_angular_plus_1 + n], s3, r12, f12);
  // l = 4
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  double s4[9] = {sum_fxyz[n * NUM_OF_ABC + 15] * C3B[15], sum_fxyz[n * NUM_OF_ABC + 16] * C3B[16],
                  sum_fxyz[n * NUM_OF_ABC + 17] * C3B[17], sum_fxyz[n * NUM_OF_ABC + 18] * C3B[18],
                  sum_fxyz[n * NUM_OF_ABC + 19] * C3B[19], sum_fxyz[n * NUM_OF_ABC + 20] * C3B[20],
                  sum_fxyz[n * NUM_OF_ABC + 21] * C3B[21], sum_fxyz[n * NUM_OF_ABC + 22] * C3B[22],
                  sum_fxyz[n * NUM_OF_ABC + 23] * C3B[23]};
  get_f12_4(
    r12[0], r12[1], r12[2], d12, d12inv, fn, fnp, Fp[3 * n_max_angular_plus_1 + n], s4, f12);
}

void accumulate_f12_with_5body(
  const int n,
  const int n_max_angular_plus_1,
  const double d12,
  const double* r12,
  double fn,
  double fnp,
  const double* Fp,
  const double* sum_fxyz,
  double* f12)
{
  const double d12inv = 1.0 / d12;
  // l = 1
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  double s1[3] = {
    sum_fxyz[n * NUM_OF_ABC + 0], sum_fxyz[n * NUM_OF_ABC + 1], sum_fxyz[n * NUM_OF_ABC + 2]};
  get_f12_5body(d12, d12inv, fn, fnp, Fp[5 * n_max_angular_plus_1 + n], s1, r12, f12);
  s1[0] *= C3B[0];
  s1[1] *= C3B[1];
  s1[2] *= C3B[2];
  get_f12_1(d12inv, fn, fnp, Fp[n], s1, r12, f12);
  // l = 2
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  double s2[5] = {
    sum_fxyz[n * NUM_OF_ABC + 3], sum_fxyz[n * NUM_OF_ABC + 4], sum_fxyz[n * NUM_OF_ABC + 5],
    sum_fxyz[n * NUM_OF_ABC + 6], sum_fxyz[n * NUM_OF_ABC + 7]};
  get_f12_4body(d12, d12inv, fn, fnp, Fp[4 * n_max_angular_plus_1 + n], s2, r12, f12);
  s2[0] *= C3B[3];
  s2[1] *= C3B[4];
  s2[2] *= C3B[5];
  s2[3] *= C3B[6];
  s2[4] *= C3B[7];
  get_f12_2(d12, d12inv, fn, fnp, Fp[n_max_angular_plus_1 + n], s2, r12, f12);
  // l = 3
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  double s3[7] = {sum_fxyz[n * NUM_OF_ABC + 8] * C3B[8],   sum_fxyz[n * NUM_OF_ABC + 9] * C3B[9],
                  sum_fxyz[n * NUM_OF_ABC + 10] * C3B[10], sum_fxyz[n * NUM_OF_ABC + 11] * C3B[11],
                  sum_fxyz[n * NUM_OF_ABC + 12] * C3B[12], sum_fxyz[n * NUM_OF_ABC + 13] * C3B[13],
                  sum_fxyz[n * NUM_OF_ABC + 14] * C3B[14]};
  get_f12_3(d12, d12inv, fn, fnp, Fp[2 * n_max_angular_plus_1 + n], s3, r12, f12);
  // l = 4
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  double s4[9] = {sum_fxyz[n * NUM_OF_ABC + 15] * C3B[15], sum_fxyz[n * NUM_OF_ABC + 16] * C3B[16],
                  sum_fxyz[n * NUM_OF_ABC + 17] * C3B[17], sum_fxyz[n * NUM_OF_ABC + 18] * C3B[18],
                  sum_fxyz[n * NUM_OF_ABC + 19] * C3B[19], sum_fxyz[n * NUM_OF_ABC + 20] * C3B[20],
                  sum_fxyz[n * NUM_OF_ABC + 21] * C3B[21], sum_fxyz[n * NUM_OF_ABC + 22] * C3B[22],
                  sum_fxyz[n * NUM_OF_ABC + 23] * C3B[23]};
  get_f12_4(
    r12[0], r12[1], r12[2], d12, d12inv, fn, fnp, Fp[3 * n_max_angular_plus_1 + n], s4, f12);
}

void accumulate_s(const double d12, double x12, double y12, double z12, const double fn, double* s)
{
  double d12inv = 1.0 / d12;
  x12 *= d12inv;
  y12 *= d12inv;
  z12 *= d12inv;
  double x12sq = x12 * x12;
  double y12sq = y12 * y12;
  double z12sq = z12 * z12;
  double x12sq_minus_y12sq = x12sq - y12sq;
  s[0] += z12 * fn;                                                            // Y10
  s[1] += x12 * fn;                                                            // Y11_real
  s[2] += y12 * fn;                                                            // Y11_imag
  s[3] += (3.0 * z12sq - 1.0) * fn;                                            // Y20
  s[4] += x12 * z12 * fn;                                                      // Y21_real
  s[5] += y12 * z12 * fn;                                                      // Y21_imag
  s[6] += x12sq_minus_y12sq * fn;                                              // Y22_real
  s[7] += 2.0 * x12 * y12 * fn;                                                // Y22_imag
  s[8] += (5.0 * z12sq - 3.0) * z12 * fn;                                      // Y30
  s[9] += (5.0 * z12sq - 1.0) * x12 * fn;                                      // Y31_real
  s[10] += (5.0 * z12sq - 1.0) * y12 * fn;                                     // Y31_imag
  s[11] += x12sq_minus_y12sq * z12 * fn;                                       // Y32_real
  s[12] += 2.0 * x12 * y12 * z12 * fn;                                         // Y32_imag
  s[13] += (x12 * x12 - 3.0 * y12 * y12) * x12 * fn;                           // Y33_real
  s[14] += (3.0 * x12 * x12 - y12 * y12) * y12 * fn;                           // Y33_imag
  s[15] += ((35.0 * z12sq - 30.0) * z12sq + 3.0) * fn;                         // Y40
  s[16] += (7.0 * z12sq - 3.0) * x12 * z12 * fn;                               // Y41_real
  s[17] += (7.0 * z12sq - 3.0) * y12 * z12 * fn;                               // Y41_iamg
  s[18] += (7.0 * z12sq - 1.0) * x12sq_minus_y12sq * fn;                       // Y42_real
  s[19] += (7.0 * z12sq - 1.0) * x12 * y12 * 2.0 * fn;                         // Y42_imag
  s[20] += (x12sq - 3.0 * y12sq) * x12 * z12 * fn;                             // Y43_real
  s[21] += (3.0 * x12sq - y12sq) * y12 * z12 * fn;                             // Y43_imag
  s[22] += (x12sq_minus_y12sq * x12sq_minus_y12sq - 4.0 * x12sq * y12sq) * fn; // Y44_real
  s[23] += (4.0 * x12 * y12 * x12sq_minus_y12sq) * fn;                         // Y44_imag
}

void find_q(const int n_max_angular_plus_1, const int n, const double* s, double* q)
{
  q[n] = C3B[0] * s[0] * s[0] + 2.0 * (C3B[1] * s[1] * s[1] + C3B[2] * s[2] * s[2]);
  q[n_max_angular_plus_1 + n] =
    C3B[3] * s[3] * s[3] + 2.0 * (C3B[4] * s[4] * s[4] + C3B[5] * s[5] * s[5] +
                                  C3B[6] * s[6] * s[6] + C3B[7] * s[7] * s[7]);
  q[2 * n_max_angular_plus_1 + n] =
    C3B[8] * s[8] * s[8] +
    2.0 * (C3B[9] * s[9] * s[9] + C3B[10] * s[10] * s[10] + C3B[11] * s[11] * s[11] +
           C3B[12] * s[12] * s[12] + C3B[13] * s[13] * s[13] + C3B[14] * s[14] * s[14]);
  q[3 * n_max_angular_plus_1 + n] =
    C3B[15] * s[15] * s[15] +
    2.0 * (C3B[16] * s[16] * s[16] + C3B[17] * s[17] * s[17] + C3B[18] * s[18] * s[18] +
           C3B[19] * s[19] * s[19] + C3B[20] * s[20] * s[20] + C3B[21] * s[21] * s[21] +
           C3B[22] * s[22] * s[22] + C3B[23] * s[23] * s[23]);
}

void find_q_with_4body(const int n_max_angular_plus_1, const int n, const double* s, double* q)
{
  find_q(n_max_angular_plus_1, n, s, q);
  q[4 * n_max_angular_plus_1 + n] =
    C4B[0] * s[3] * s[3] * s[3] + C4B[1] * s[3] * (s[4] * s[4] + s[5] * s[5]) +
    C4B[2] * s[3] * (s[6] * s[6] + s[7] * s[7]) + C4B[3] * s[6] * (s[5] * s[5] - s[4] * s[4]) +
    C4B[4] * s[4] * s[5] * s[7];
}

void find_q_with_5body(const int n_max_angular_plus_1, const int n, const double* s, double* q)
{
  find_q_with_4body(n_max_angular_plus_1, n, s, q);
  double s0_sq = s[0] * s[0];
  double s1_sq_plus_s2_sq = s[1] * s[1] + s[2] * s[2];
  q[5 * n_max_angular_plus_1 + n] = C5B[0] * s0_sq * s0_sq + C5B[1] * s0_sq * s1_sq_plus_s2_sq +
                                    C5B[2] * s1_sq_plus_s2_sq * s1_sq_plus_s2_sq;
}

void find_descriptor_for_lammps(
  NEP3_CPU::ParaMB& paramb,
  NEP3_CPU::ANN& annmb,
  int nlocal,
  int N,
  int* g_ilist,
  int* g_NN,
  int** g_NL,
  int* g_type,
  std::vector<int> map_atom_type_idx,
  double** g_pos,
  double* g_Fp,
  double* g_sum_fxyz,
  double& g_total_potential,
  double* g_potential,
  double* g_charge,
  double* g_charge_derivative)
{
  for (int ii = 0; ii < N; ++ii) {
    int n1 = g_ilist[ii];
    int t1 = map_atom_type_idx[g_type[n1] - 1];
    double q[MAX_DIM] = {0.0};

    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int n2 = g_NL[n1][i1];
      double r12[3] = {
        g_pos[n2][0] - g_pos[n1][0], g_pos[n2][1] - g_pos[n1][1], g_pos[n2][2] - g_pos[n1][2]};

      double d12sq = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
      if (d12sq >= paramb.rc_radial * paramb.rc_radial) {
        continue;
      }
      double d12 = sqrt(d12sq);
      int t2 = map_atom_type_idx[g_type[n2] - 1]; // from LAMMPS to NEP convention

      double fc12;
      find_fc(paramb.rc_radial, paramb.rcinv_radial, d12, fc12);
      double fn12[MAX_NUM_N];
      if (paramb.version == 2) {
        find_fn(paramb.n_max_radial, paramb.rcinv_radial, d12, fc12, fn12);
        for (int n = 0; n <= paramb.n_max_radial; ++n) {
          double c = (paramb.num_types == 1)
                       ? 1.0
                       : annmb.c[(n * paramb.num_types + t1) * paramb.num_types + t2];
          q[n] += fn12[n] * c;
        }
      } else {
        find_fn(paramb.basis_size_radial, paramb.rcinv_radial, d12, fc12, fn12);
        for (int n = 0; n <= paramb.n_max_radial; ++n) {
          double gn12 = 0.0;
          for (int k = 0; k <= paramb.basis_size_radial; ++k) {
            int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
            c_index += t1 * paramb.num_types + t2;
            gn12 += fn12[k] * annmb.c[c_index];
          }
          q[n] += gn12;
        }
      }
    }

    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      double s[NUM_OF_ABC] = {0.0};
      for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
        int n2 = g_NL[n1][i1];
        double r12[3] = {
          g_pos[n2][0] - g_pos[n1][0], g_pos[n2][1] - g_pos[n1][1], g_pos[n2][2] - g_pos[n1][2]};

        double d12sq = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
        if (d12sq >= paramb.rc_angular * paramb.rc_angular) {
          continue;
        }
        double d12 = sqrt(d12sq);
        int t2 = map_atom_type_idx[g_type[n2] - 1]; // from LAMMPS to NEP convention

        double fc12;
        find_fc(paramb.rc_angular, paramb.rcinv_angular, d12, fc12);

        if (paramb.version == 2) {
          double fn;
          find_fn(n, paramb.rcinv_angular, d12, fc12, fn);
          fn *=
            (paramb.num_types == 1)
              ? 1.0
              : annmb.c
                  [((paramb.n_max_radial + 1 + n) * paramb.num_types + t1) * paramb.num_types + t2];
          accumulate_s(d12, r12[0], r12[1], r12[2], fn, s);
        } else {
          double fn12[MAX_NUM_N];
          find_fn(paramb.basis_size_angular, paramb.rcinv_angular, d12, fc12, fn12);
          double gn12 = 0.0;
          for (int k = 0; k <= paramb.basis_size_angular; ++k) {
            int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
            c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
            gn12 += fn12[k] * annmb.c[c_index];
          }
          accumulate_s(d12, r12[0], r12[1], r12[2], gn12, s);
        }
      }
      if (paramb.num_L == paramb.L_max) {
        find_q(paramb.n_max_angular + 1, n, s, q + (paramb.n_max_radial + 1));
      } else if (paramb.num_L == paramb.L_max + 1) {
        find_q_with_4body(paramb.n_max_angular + 1, n, s, q + (paramb.n_max_radial + 1));
      } else {
        find_q_with_5body(paramb.n_max_angular + 1, n, s, q + (paramb.n_max_radial + 1));
      }
      for (int abc = 0; abc < NUM_OF_ABC; ++abc) {
        g_sum_fxyz[(n * NUM_OF_ABC + abc) * nlocal + n1] = s[abc];
      }
    }

    for (int d = 0; d < annmb.dim; ++d) {
      q[d] = q[d] * paramb.q_scaler[d];
      // std::cout << "scaler " << paramb.q_scaler[d] << " realq " <<  q[d] << std::endl;
    }

    double F = 0.0, Fp[MAX_DIM] = {0.0}, latent_space[MAX_NEURON] = {0.0};
    double charge = 0.0, charge_derivative[MAX_DIM] = {0.0};

    if (paramb.charge_mode == 2) {
        apply_ann_one_layer_charge(
          annmb.dim, annmb.num_neurons1, annmb.w0[t1], annmb.b0[t1], annmb.w1[t1], annmb.b1, q, F, Fp,
          charge, charge_derivative);
      } else if (paramb.version == 4) {
        apply_ann_one_layer(
          annmb.dim, annmb.num_neurons1, annmb.w0[t1], annmb.b0[t1], annmb.w1[t1], annmb.b1, q, F, Fp,
          latent_space, t1, paramb.version);
      } else if (paramb.version == 5) {
        apply_ann_one_layer_nep5(
          annmb.dim, annmb.num_neurons1, annmb.w0[t1], annmb.b0[t1], annmb.w1[t1], annmb.b1, q, F, Fp,
          latent_space, t1, paramb.version);        
      }

    g_total_potential += F; // always calculate this
    if (g_potential) {      // only calculate when required
      g_potential[n1] = F;
    }
    if (paramb.charge_mode == 2 && g_charge) {
      g_charge[n1] = charge;
    }
    for (int d = 0; d < annmb.dim; ++d) {
      g_Fp[d * nlocal + n1] = Fp[d] * paramb.q_scaler[d];
      if (paramb.charge_mode == 2 && g_charge_derivative) {
        g_charge_derivative[d * nlocal + n1] = charge_derivative[d] * paramb.q_scaler[d];
      }
    }
  }
}

void find_force_radial_for_lammps(
  NEP3_CPU::ParaMB& paramb,
  NEP3_CPU::ANN& annmb,
  int nlocal,
  int N,
  int* g_ilist,
  int* g_NN,
  int** g_NL,
  int* g_type,
  std::vector<int> map_atom_type_idx,
  double** g_pos,
  double* g_Fp,
  double** g_force,
  double g_total_virial[6],
  double** g_virial,
  int virial_components,
  int model_index)
{
  for (int ii = 0; ii < N; ++ii) {
    int n1 = g_ilist[ii];
    int t1 = map_atom_type_idx[g_type[n1] - 1]; // from LAMMPS to NEP convention
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int n2 = g_NL[n1][i1];
      int t2 = map_atom_type_idx[g_type[n2] - 1]; // from LAMMPS to NEP convention
      double r12[3] = {
        g_pos[n2][0] - g_pos[n1][0], g_pos[n2][1] - g_pos[n1][1], g_pos[n2][2] - g_pos[n1][2]};

      double d12sq = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
      if (d12sq >= paramb.rc_radial * paramb.rc_radial) {
        continue;
      }
      double d12 = sqrt(d12sq);
      double d12inv = 1.0 / d12;
      double f12[3] = {0.0};

      double fc12, fcp12;
      find_fc_and_fcp(paramb.rc_radial, paramb.rcinv_radial, d12, fc12, fcp12);
      double fn12[MAX_NUM_N];
      double fnp12[MAX_NUM_N];
      if (paramb.version == 2) {
        find_fn_and_fnp(paramb.n_max_radial, paramb.rcinv_radial, d12, fc12, fcp12, fn12, fnp12);
        for (int n = 0; n <= paramb.n_max_radial; ++n) {
          double tmp12 = g_Fp[n1 + n * nlocal] * fnp12[n] * d12inv;
          tmp12 *= (paramb.num_types == 1)
                     ? 1.0
                     : annmb.c[(n * paramb.num_types + t1) * paramb.num_types + t2];
          for (int d = 0; d < 3; ++d) {
            f12[d] += tmp12 * r12[d];
          }
        }
      } else {
        find_fn_and_fnp(
          paramb.basis_size_radial, paramb.rcinv_radial, d12, fc12, fcp12, fn12, fnp12);
        for (int n = 0; n <= paramb.n_max_radial; ++n) {
          double gnp12 = 0.0;
          for (int k = 0; k <= paramb.basis_size_radial; ++k) {
            int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
            c_index += t1 * paramb.num_types + t2;
            gnp12 += fnp12[k] * annmb.c[c_index];
          }
          double tmp12 = g_Fp[n1 + n * nlocal] * gnp12 * d12inv;
          for (int d = 0; d < 3; ++d) {
            f12[d] += tmp12 * r12[d];
          }
        }
      }

      g_force[n1][0] += f12[0];
      g_force[n1][1] += f12[1];
      g_force[n1][2] += f12[2];
      g_force[n2][0] -= f12[0];
      g_force[n2][1] -= f12[1];
      g_force[n2][2] -= f12[2];

      // always calculate the total virial:
      g_total_virial[0] -= r12[0] * f12[0]; // xx
      g_total_virial[1] -= r12[1] * f12[1]; // yy
      g_total_virial[2] -= r12[2] * f12[2]; // zz
      g_total_virial[3] -= r12[0] * f12[1]; // xy
      g_total_virial[4] -= r12[0] * f12[2]; // xz
      g_total_virial[5] -= r12[1] * f12[2]; // yz
      if (g_virial and virial_components >= 6 and model_index==0) { // only calculate the per-atom virial when required, for multi models, only calculate the first model
        g_virial[n2][0] -= r12[0] * f12[0]; // xx
        g_virial[n2][1] -= r12[1] * f12[1]; // yy
        g_virial[n2][2] -= r12[2] * f12[2]; // zz
        g_virial[n2][3] -= r12[0] * f12[1]; // xy
        g_virial[n2][4] -= r12[0] * f12[2]; // xz
        g_virial[n2][5] -= r12[1] * f12[2]; // yz
        if (virial_components >= 9) {
          g_virial[n2][6] -= r12[1] * f12[0]; // yx
          g_virial[n2][7] -= r12[2] * f12[0]; // zx
          g_virial[n2][8] -= r12[2] * f12[1]; // zy
        }
      }
    }
  }
}

void find_force_angular_for_lammps(
  NEP3_CPU::ParaMB& paramb,
  NEP3_CPU::ANN& annmb,
  int nlocal,
  int N,
  int* g_ilist,
  int* g_NN,
  int** g_NL,
  int* g_type,
  std::vector<int> map_atom_type_idx,
  double** g_pos,
  double* g_Fp,
  double* g_sum_fxyz,
  double** g_force,
  double g_total_virial[6],
  double** g_virial,
  int virial_components,
  int model_index)
{
  for (int ii = 0; ii < N; ++ii) {
    int n1 = g_ilist[ii];
    double Fp[MAX_DIM_ANGULAR] = {0.0};
    double sum_fxyz[NUM_OF_ABC * MAX_NUM_N];
    for (int d = 0; d < paramb.dim_angular; ++d) {
      Fp[d] = g_Fp[(paramb.n_max_radial + 1 + d) * nlocal + n1];
    }
    for (int d = 0; d < (paramb.n_max_angular + 1) * NUM_OF_ABC; ++d) {
      sum_fxyz[d] = g_sum_fxyz[d * nlocal + n1];
    }

    int t1 = map_atom_type_idx[g_type[n1] - 1]; // from LAMMPS to NEP convention

    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int n2 = g_NL[n1][i1];
      double r12[3] = {
        g_pos[n2][0] - g_pos[n1][0], g_pos[n2][1] - g_pos[n1][1], g_pos[n2][2] - g_pos[n1][2]};

      double d12sq = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
      if (d12sq >= paramb.rc_angular * paramb.rc_angular) {
        continue;
      }
      double d12 = sqrt(d12sq);
      int t2 = map_atom_type_idx[g_type[n2] - 1]; // from LAMMPS to NEP convention
      double f12[3] = {0.0};
      double fc12, fcp12;
      find_fc_and_fcp(paramb.rc_angular, paramb.rcinv_angular, d12, fc12, fcp12);
      if (paramb.version == 2) {
        for (int n = 0; n <= paramb.n_max_angular; ++n) {
          double fn;
          double fnp;
          find_fn_and_fnp(n, paramb.rcinv_angular, d12, fc12, fcp12, fn, fnp);
          const double c =
            (paramb.num_types == 1)
              ? 1.0
              : annmb.c
                  [((paramb.n_max_radial + 1 + n) * paramb.num_types + t1) * paramb.num_types + t2];
          fn *= c;
          fnp *= c;
          accumulate_f12(n, paramb.n_max_angular + 1, d12, r12, fn, fnp, Fp, sum_fxyz, f12);
        }
      } else {
        double fn12[MAX_NUM_N];
        double fnp12[MAX_NUM_N];
        find_fn_and_fnp(
          paramb.basis_size_angular, paramb.rcinv_angular, d12, fc12, fcp12, fn12, fnp12);
        for (int n = 0; n <= paramb.n_max_angular; ++n) {
          double gn12 = 0.0;
          double gnp12 = 0.0;
          for (int k = 0; k <= paramb.basis_size_angular; ++k) {
            int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
            c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
            gn12 += fn12[k] * annmb.c[c_index];
            gnp12 += fnp12[k] * annmb.c[c_index];
          }
          if (paramb.num_L == paramb.L_max) {
            accumulate_f12(n, paramb.n_max_angular + 1, d12, r12, gn12, gnp12, Fp, sum_fxyz, f12);
          } else if (paramb.num_L == paramb.L_max + 1) {
            accumulate_f12_with_4body(
              n, paramb.n_max_angular + 1, d12, r12, gn12, gnp12, Fp, sum_fxyz, f12);
          } else {
            accumulate_f12_with_5body(
              n, paramb.n_max_angular + 1, d12, r12, gn12, gnp12, Fp, sum_fxyz, f12);
          }
        }
      }

      g_force[n1][0] += f12[0];
      g_force[n1][1] += f12[1];
      g_force[n1][2] += f12[2];
      g_force[n2][0] -= f12[0];
      g_force[n2][1] -= f12[1];
      g_force[n2][2] -= f12[2];
      // always calculate the total virial:
      g_total_virial[0] -= r12[0] * f12[0]; // xx
      g_total_virial[1] -= r12[1] * f12[1]; // yy
      g_total_virial[2] -= r12[2] * f12[2]; // zz
      g_total_virial[3] -= r12[0] * f12[1]; // xy
      g_total_virial[4] -= r12[0] * f12[2]; // xz
      g_total_virial[5] -= r12[1] * f12[2]; // yz
      if (g_virial and virial_components >= 6 and model_index==0) { // only calculate the per-atom virial when required
        g_virial[n2][0] -= r12[0] * f12[0]; // xx
        g_virial[n2][1] -= r12[1] * f12[1]; // yy
        g_virial[n2][2] -= r12[2] * f12[2]; // zz
        g_virial[n2][3] -= r12[0] * f12[1]; // xy
        g_virial[n2][4] -= r12[0] * f12[2]; // xz
        g_virial[n2][5] -= r12[1] * f12[2]; // yz
        if (virial_components >= 9) {
          g_virial[n2][6] -= r12[1] * f12[0]; // yx
          g_virial[n2][7] -= r12[2] * f12[0]; // zx
          g_virial[n2][8] -= r12[2] * f12[1]; // zy
        }
      }
    }
  }
}

void find_force_ZBL_for_lammps(
  NEP3_CPU::ParaMB& paramb,
  const NEP3_CPU::ZBL& zbl,
  int N,
  int* g_ilist,
  int* g_NN,
  int** g_NL,
  int* g_type,
  std::vector<int> map_atom_type_idx,
  double** g_pos,
  double** g_force,
  double g_total_virial[6],
  double** g_virial,
  int virial_components,
  double& g_total_potential,
  double* g_potential,
  int model_index)
{
  for (int ii = 0; ii < N; ++ii) {
    int n1 = g_ilist[ii];
    int type1 = map_atom_type_idx[g_type[n1] - 1];
    int zi = zbl.atomic_numbers[type1]; // from LAMMPS to NEP convention
    double pow_zi = pow(zi, 0.23);
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int n2 = g_NL[n1][i1];
      double r12[3] = {
        g_pos[n2][0] - g_pos[n1][0], g_pos[n2][1] - g_pos[n1][1], g_pos[n2][2] - g_pos[n1][2]};

      double d12sq = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
      // float max_rc_outer = 2.5;  This restriction has been removed in the latest version of GPUMD.
      // if (d12sq >= max_rc_outer * max_rc_outer) {
      //   continue;
      // }
      double d12 = sqrt(d12sq);
      double d12inv = 1.0 / d12;
      double f, fp;
      int type2 = map_atom_type_idx[g_type[n2] - 1];
      int zj = zbl.atomic_numbers[type2]; // from LAMMPS to NEP convention
      double a_inv = (pow_zi + pow(zj, 0.23)) * 2.134563;
      double zizj = K_C_SP * zi * zj;
      if (zbl.flexibled) {
        int t1, t2;
        if (type1 < type2) {
          t1 = type1;
          t2 = type2;
        } else {
          t1 = type2;
          t2 = type1;
        }
        int zbl_index = t1 * zbl.num_types - (t1 * (t1 - 1)) / 2 + (t2 - t1);
        double ZBL_para[10];
        for (int i = 0; i < 10; ++i) {
          ZBL_para[i] = zbl.para[10 * zbl_index + i];
        }
        find_f_and_fp_zbl(ZBL_para, zizj, a_inv, d12, d12inv, f, fp);
      } else {
        double rc_inner = zbl.rc_inner;
        double rc_outer = zbl.rc_outer;
        if (paramb.use_typewise_cutoff_zbl) {
          // zi and zj start from 1, so need to minus 1 here
          rc_outer = std::min(
            (COVALENT_RADIUS[zi - 1] + COVALENT_RADIUS[zj - 1]) * paramb.typewise_cutoff_zbl_factor,
            rc_outer);
          rc_inner = 0.0;
        }
        find_f_and_fp_zbl(zizj, a_inv, rc_inner, rc_outer, d12, d12inv, f, fp);
      }
      double f2 = fp * d12inv * 0.5;
      double f12[3] = {r12[0] * f2, r12[1] * f2, r12[2] * f2};
      g_force[n1][0] += f12[0]; // accumulation here
      g_force[n1][1] += f12[1];
      g_force[n1][2] += f12[2];
      g_force[n2][0] -= f12[0];
      g_force[n2][1] -= f12[1];
      g_force[n2][2] -= f12[2];
      // always calculate the total virial:
      g_total_virial[0] -= r12[0] * f12[0]; // xx
      g_total_virial[1] -= r12[1] * f12[1]; // yy
      g_total_virial[2] -= r12[2] * f12[2]; // zz
      g_total_virial[3] -= r12[0] * f12[1]; // xy
      g_total_virial[4] -= r12[0] * f12[2]; // xz
      g_total_virial[5] -= r12[1] * f12[2]; // yz
      if (g_virial and virial_components >= 6 and model_index==0) { // only calculate the per-atom virial when required
        g_virial[n2][0] -= r12[0] * f12[0]; // xx
        g_virial[n2][1] -= r12[1] * f12[1]; // yy
        g_virial[n2][2] -= r12[2] * f12[2]; // zz
        g_virial[n2][3] -= r12[0] * f12[1]; // xy
        g_virial[n2][4] -= r12[0] * f12[2]; // xz
        g_virial[n2][5] -= r12[1] * f12[2]; // yz
        if (virial_components >= 9) {
          g_virial[n2][6] -= r12[1] * f12[0]; // yx
          g_virial[n2][7] -= r12[2] * f12[0]; // zx
          g_virial[n2][8] -= r12[2] * f12[1]; // zy
        }
      }
      g_total_potential += f * 0.5; // always calculate this
      if (g_potential) {            // only calculate when required
        g_potential[n1] += f * 0.5;
      }
    }
  }
}

struct NEP_CPU_Box {
  double h[9];
  double hi[9];
};

struct NEP_CPU_PPPM_Para {
  int K0K1K2 = 0;
  int K0K1 = 0;
  int K[3] = {0, 0, 0};
  int K_half[3] = {0, 0, 0};
  double alpha = 0.0;
  double alpha_factor = 0.0;
  double two_pi_over_V = 0.0;
  double b[3][3];
  double two_pi_over_K[3];
};

void cross_product(const double a[3], const double b[3], double c[3])
{
  c[0] = a[1] * b[2] - a[2] * b[1];
  c[1] = a[2] * b[0] - a[0] * b[2];
  c[2] = a[0] * b[1] - a[1] * b[0];
}

double vector_area(const double* a, const double* b)
{
  const double s1 = a[1] * b[2] - a[2] * b[1];
  const double s2 = a[2] * b[0] - a[0] * b[2];
  const double s3 = a[0] * b[1] - a[1] * b[0];
  return sqrt(s1 * s1 + s2 * s2 + s3 * s3);
}

void find_cpu_ewald_k_and_G(
  const int num_kpoints_max,
  const double alpha,
  const double alpha_factor,
  const NEP_CPU_Box& box,
  std::vector<double>& kx,
  std::vector<double>& ky,
  std::vector<double>& kz,
  std::vector<double>& G)
{
  const double det = box.h[0] * (box.h[4] * box.h[8] - box.h[5] * box.h[7]) +
                     box.h[1] * (box.h[5] * box.h[6] - box.h[3] * box.h[8]) +
                     box.h[2] * (box.h[3] * box.h[7] - box.h[4] * box.h[6]);
  const double a1[3] = {box.h[0], box.h[3], box.h[6]};
  const double a2[3] = {box.h[1], box.h[4], box.h[7]};
  const double a3[3] = {box.h[2], box.h[5], box.h[8]};
  double b1[3] = {0.0, 0.0, 0.0};
  double b2[3] = {0.0, 0.0, 0.0};
  double b3[3] = {0.0, 0.0, 0.0};
  cross_product(a2, a3, b1);
  cross_product(a3, a1, b2);
  cross_product(a1, a2, b3);

  const double two_pi = 2.0 * PI;
  const double two_pi_over_det = two_pi / det;
  for (int d = 0; d < 3; ++d) {
    b1[d] *= two_pi_over_det;
    b2[d] *= two_pi_over_det;
    b3[d] *= two_pi_over_det;
  }

  const double volume_k = two_pi * two_pi * two_pi / std::abs(det);
  const int n1_max = int(alpha * two_pi * vector_area(b2, b3) / volume_k);
  const int n2_max = int(alpha * two_pi * vector_area(b3, b1) / volume_k);
  const int n3_max = int(alpha * two_pi * vector_area(b1, b2) / volume_k);
  const double ksq_max = two_pi * two_pi * alpha * alpha;

  kx.clear();
  ky.clear();
  kz.clear();
  G.clear();
  for (int n1 = 0; n1 <= n1_max; ++n1) {
    for (int n2 = -n2_max; n2 <= n2_max; ++n2) {
      for (int n3 = -n3_max; n3 <= n3_max; ++n3) {
        const int nsq = n1 * n1 + n2 * n2 + n3 * n3;
        if (nsq == 0 || (n1 == 0 && n2 < 0) || (n1 == 0 && n2 == 0 && n3 < 0)) continue;
        const double this_kx = n1 * b1[0] + n2 * b2[0] + n3 * b3[0];
        const double this_ky = n1 * b1[1] + n2 * b2[1] + n3 * b3[1];
        const double this_kz = n1 * b1[2] + n2 * b2[2] + n3 * b3[2];
        const double ksq = this_kx * this_kx + this_ky * this_ky + this_kz * this_kz;
        if (ksq < ksq_max) {
          if (int(kx.size()) < num_kpoints_max) {
            kx.push_back(this_kx);
            ky.push_back(this_ky);
            kz.push_back(this_kz);
            G.push_back(2.0 * std::abs(two_pi_over_det) / ksq * exp(-ksq * alpha_factor));
          }
        }
      }
    }
  }
}

void zero_global_mean(
  const int nlocal,
  const long long natoms_global,
  MPI_Comm mpi_comm,
  std::vector<double>& values)
{
  double local_sum = 0.0;
  for (int n = 0; n < nlocal; ++n) {
    local_sum += values[n];
  }
  double global_sum = 0.0;
  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);
  const double mean = global_sum / double(natoms_global);
  for (int n = 0; n < nlocal; ++n) {
    values[n] -= mean;
  }
}

void find_force_charge_ewald_cpu(
  const NEP3_CPU::ParaMB& paramb,
  const int nlocal,
  const long long natoms_global,
  double** pos,
  const NEP_CPU_Box& box,
  MPI_Comm mpi_comm,
  std::vector<double>& charge,
  std::vector<double>& D_real,
  double** force,
  double total_virial[6],
  double** virial,
  int virial_components,
  const int model_index)
{
  zero_global_mean(nlocal, natoms_global, mpi_comm, charge);
  std::vector<double> kx, ky, kz, G;
  find_cpu_ewald_k_and_G(
    paramb.num_kpoints_max, paramb.charge_alpha, paramb.charge_alpha_factor, box, kx, ky, kz, G);
  const int num_kpoints = int(kx.size());
  std::vector<double> local(2 * num_kpoints, 0.0);
  std::vector<double> global(2 * num_kpoints, 0.0);
  for (int nk = 0; nk < num_kpoints; ++nk) {
    for (int n = 0; n < nlocal; ++n) {
      const double kr = kx[nk] * pos[n][0] + ky[nk] * pos[n][1] + kz[nk] * pos[n][2];
      local[nk] += charge[n] * cos(kr);
      local[num_kpoints + nk] -= charge[n] * sin(kr);
    }
  }
  if (num_kpoints > 0) {
    MPI_Allreduce(local.data(), global.data(), int(global.size()), MPI_DOUBLE, MPI_SUM, mpi_comm);
  }

  for (int n = 0; n < nlocal; ++n) {
    const double q = charge[n];
    double temp_force_sum[3] = {0.0, 0.0, 0.0};
    double temp_virial_sum[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double temp_D_real_sum = 0.0;
    for (int nk = 0; nk < num_kpoints; ++nk) {
      const double kr = kx[nk] * pos[n][0] + ky[nk] * pos[n][1] + kz[nk] * pos[n][2];
      const double sin_kr = sin(kr);
      const double cos_kr = cos(kr);
      const double S_real = global[nk];
      const double S_imag = global[num_kpoints + nk];
      const double imag_term = G[nk] * (S_real * sin_kr + S_imag * cos_kr);
      const double GSE = G[nk] * (S_real * cos_kr - S_imag * sin_kr);
      const double qGSE = q * GSE;
      const double ksq = kx[nk] * kx[nk] + ky[nk] * ky[nk] + kz[nk] * kz[nk];
      const double alpha_k_factor = 2.0 * paramb.charge_alpha_factor + 2.0 / ksq;
      temp_virial_sum[0] += qGSE * (1.0 - alpha_k_factor * kx[nk] * kx[nk]);
      temp_virial_sum[1] += qGSE * (1.0 - alpha_k_factor * ky[nk] * ky[nk]);
      temp_virial_sum[2] += qGSE * (1.0 - alpha_k_factor * kz[nk] * kz[nk]);
      temp_virial_sum[3] -= qGSE * (alpha_k_factor * kx[nk] * ky[nk]);
      temp_virial_sum[4] -= qGSE * (alpha_k_factor * ky[nk] * kz[nk]);
      temp_virial_sum[5] -= qGSE * (alpha_k_factor * kz[nk] * kx[nk]);
      temp_D_real_sum += GSE;
      temp_force_sum[0] += kx[nk] * imag_term;
      temp_force_sum[1] += ky[nk] * imag_term;
      temp_force_sum[2] += kz[nk] * imag_term;
    }
    D_real[n] = 2.0 * K_C_SP * temp_D_real_sum;
    const double charge_factor = 2.0 * K_C_SP * q;
    force[n][0] += charge_factor * temp_force_sum[0];
    force[n][1] += charge_factor * temp_force_sum[1];
    force[n][2] += charge_factor * temp_force_sum[2];

    const double virial_xx = K_C_SP * temp_virial_sum[0];
    const double virial_yy = K_C_SP * temp_virial_sum[1];
    const double virial_zz = K_C_SP * temp_virial_sum[2];
    const double virial_xy = K_C_SP * temp_virial_sum[3];
    const double virial_yz = K_C_SP * temp_virial_sum[4];
    const double virial_zx = K_C_SP * temp_virial_sum[5];
    total_virial[0] += virial_xx;
    total_virial[1] += virial_yy;
    total_virial[2] += virial_zz;
    total_virial[3] += virial_xy;
    total_virial[4] += virial_zx;
    total_virial[5] += virial_yz;
    if (virial && virial_components >= 6 && model_index == 0) {
      virial[n][0] += virial_xx;
      virial[n][1] += virial_yy;
      virial[n][2] += virial_zz;
      virial[n][3] += virial_xy;
      virial[n][4] += virial_zx;
      virial[n][5] += virial_yz;
      if (virial_components >= 9) {
        virial[n][6] += virial_xy;
        virial[n][7] += virial_zx;
        virial[n][8] += virial_yz;
      }
    }
  }
  zero_global_mean(nlocal, natoms_global, mpi_comm, D_real);
}

int get_best_pppm_K(const int m)
{
  int n = 16;
  while (n < m) {
    n *= 2;
  }
  return n;
}

int pppm_mesh_index(const int K, const int n)
{
  if (n >= K) return n - K;
  if (n < 0) return n + K;
  return n;
}

double pppm_sinc(const double x)
{
  static const double coeff[6] = {
    1.0, -1.6666667e-1, 8.3333333e-3, -1.9841270e-4, 2.7557319e-6, -2.5052108e-8};
  if (x * x > 1.0) return sin(x) / x;
  double y = 0.0;
  double term = 1.0;
  for (int i = 0; i < 6; ++i) {
    y += coeff[i] * term;
    term *= x * x;
  }
  return y;
}

void pppm_weights(const double d, double W[5])
{
  static const double coeff[5][5] = {
    {2.6041667e-03, -2.0833333e-02, 6.2500000e-02, -8.3333333e-02, 4.1666667e-02},
    {1.9791667e-01, -4.5833333e-01, 2.5000000e-01, 1.6666667e-01, -1.6666667e-01},
    {5.9895833e-01, 0.0000000e+00, -6.2500000e-01, 0.0000000e+00, 2.5000000e-01},
    {1.9791667e-01, 4.5833333e-01, 2.5000000e-01, -1.6666667e-01, -1.6666667e-01},
    {2.6041667e-03, 2.0833333e-02, 6.2500000e-02, 8.3333333e-02, 4.1666667e-02}};
  for (int i = 0; i < 5; ++i) {
    W[i] = (((coeff[i][4] * d + coeff[i][3]) * d + coeff[i][2]) * d + coeff[i][1]) * d + coeff[i][0];
  }
}

double box_volume(const NEP_CPU_Box& box)
{
  return box.h[0] * (box.h[4] * box.h[8] - box.h[5] * box.h[7]) -
         box.h[1] * (box.h[3] * box.h[8] - box.h[5] * box.h[6]) +
         box.h[2] * (box.h[3] * box.h[7] - box.h[4] * box.h[6]);
}

double box_area(const NEP_CPU_Box& box, const int d)
{
  const double a[3][3] = {
    {box.h[0], box.h[3], box.h[6]},
    {box.h[1], box.h[4], box.h[7]},
    {box.h[2], box.h[5], box.h[8]}};
  const int d1 = (d + 1) % 3;
  const int d2 = (d + 2) % 3;
  double cross[3];
  cross_product(a[d1], a[d2], cross);
  return sqrt(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]);
}

void find_pppm_para(
  const double alpha,
  const double alpha_factor,
  const NEP_CPU_Box& box,
  NEP_CPU_PPPM_Para& para)
{
  const double two_pi = 2.0 * PI;
  const double volume = box_volume(box);
  para.alpha = alpha;
  para.alpha_factor = alpha_factor;
  para.two_pi_over_V = two_pi / volume;
  for (int d = 0; d < 3; ++d) {
    const double box_thickness = volume / box_area(box, d);
    para.K[d] = get_best_pppm_K(int(box_thickness));
    para.K_half[d] = para.K[d] / 2;
    para.two_pi_over_K[d] = two_pi / para.K[d];
  }
  para.K0K1 = para.K[0] * para.K[1];
  para.K0K1K2 = para.K0K1 * para.K[2];
  for (int d = 0; d < 3; ++d) {
    para.b[0][d] = two_pi * box.hi[d];
    para.b[1][d] = two_pi * box.hi[3 + d];
    para.b[2][d] = two_pi * box.hi[6 + d];
  }
}

void pppm_direct_dft(
  const NEP_CPU_PPPM_Para& para,
  const std::vector<std::complex<double>>& in,
  std::vector<std::complex<double>>& out,
  const int sign)
{
  std::vector<std::complex<double>> tmp_x(para.K0K1K2, std::complex<double>(0.0, 0.0));
  std::vector<std::complex<double>> tmp_y(para.K0K1K2, std::complex<double>(0.0, 0.0));
  out.assign(para.K0K1K2, std::complex<double>(0.0, 0.0));

  for (int z = 0; z < para.K[2]; ++z) {
    for (int y = 0; y < para.K[1]; ++y) {
      for (int kx_i = 0; kx_i < para.K[0]; ++kx_i) {
        const int nk0 = kx_i >= para.K_half[0] ? kx_i - para.K[0] : kx_i;
        std::complex<double> sum(0.0, 0.0);
        for (int x = 0; x < para.K[0]; ++x) {
          const int ridx = x + para.K[0] * (y + para.K[1] * z);
          const double phase = sign * para.two_pi_over_K[0] * nk0 * x;
          sum += in[ridx] * std::complex<double>(cos(phase), sin(phase));
        }
        tmp_x[kx_i + para.K[0] * (y + para.K[1] * z)] = sum;
      }
    }
  }

  for (int z = 0; z < para.K[2]; ++z) {
    for (int ky_i = 0; ky_i < para.K[1]; ++ky_i) {
      const int nk1 = ky_i >= para.K_half[1] ? ky_i - para.K[1] : ky_i;
      for (int kx_i = 0; kx_i < para.K[0]; ++kx_i) {
        std::complex<double> sum(0.0, 0.0);
        for (int y = 0; y < para.K[1]; ++y) {
          const int ridx = kx_i + para.K[0] * (y + para.K[1] * z);
          const double phase = sign * para.two_pi_over_K[1] * nk1 * y;
          sum += tmp_x[ridx] * std::complex<double>(cos(phase), sin(phase));
        }
        tmp_y[kx_i + para.K[0] * (ky_i + para.K[1] * z)] = sum;
      }
    }
  }

  for (int kz_i = 0; kz_i < para.K[2]; ++kz_i) {
    const int nk2 = kz_i >= para.K_half[2] ? kz_i - para.K[2] : kz_i;
    for (int ky_i = 0; ky_i < para.K[1]; ++ky_i) {
      for (int kx_i = 0; kx_i < para.K[0]; ++kx_i) {
        std::complex<double> sum(0.0, 0.0);
        for (int z = 0; z < para.K[2]; ++z) {
          const int ridx = kx_i + para.K[0] * (ky_i + para.K[1] * z);
          const double phase = sign * para.two_pi_over_K[2] * nk2 * z;
          sum += tmp_y[ridx] * std::complex<double>(cos(phase), sin(phase));
        }
        out[kx_i + para.K[0] * (ky_i + para.K[1] * kz_i)] = sum;
      }
    }
  }
}

void find_force_charge_pppm_cpu(
  const NEP3_CPU::ParaMB& paramb,
  const int nlocal,
  const long long natoms_global,
  double** pos,
  const NEP_CPU_Box& box,
  MPI_Comm mpi_comm,
  std::vector<double>& charge,
  std::vector<double>& D_real,
  double** force,
  double total_virial[6],
  double** virial,
  int virial_components,
  const int model_index)
{
  zero_global_mean(nlocal, natoms_global, mpi_comm, charge);

  NEP_CPU_PPPM_Para para;
  find_pppm_para(paramb.charge_alpha, paramb.charge_alpha_factor, box, para);
  const int mesh_size = para.K0K1K2;
  std::vector<std::complex<double>> mesh(mesh_size, std::complex<double>(0.0, 0.0));

  for (int n = 0; n < nlocal; ++n) {
    const double sx = (box.hi[0] * pos[n][0] + box.hi[1] * pos[n][1] + box.hi[2] * pos[n][2]) * para.K[0];
    const double sy = (box.hi[3] * pos[n][0] + box.hi[4] * pos[n][1] + box.hi[5] * pos[n][2]) * para.K[1];
    const double sz = (box.hi[6] * pos[n][0] + box.hi[7] * pos[n][1] + box.hi[8] * pos[n][2]) * para.K[2];
    const int ix = int(sx + 0.5);
    const int iy = int(sy + 0.5);
    const int iz = int(sz + 0.5);
    double Wx[5], Wy[5], Wz[5];
    pppm_weights(sx - ix, Wx);
    pppm_weights(sy - iy, Wy);
    pppm_weights(sz - iz, Wz);
    for (int n0 = -2; n0 <= 2; ++n0) {
      const int neighbor0 = pppm_mesh_index(para.K[0], ix + n0);
      for (int n1 = -2; n1 <= 2; ++n1) {
        const int neighbor1 = pppm_mesh_index(para.K[1], iy + n1);
        for (int n2 = -2; n2 <= 2; ++n2) {
          const int neighbor2 = pppm_mesh_index(para.K[2], iz + n2);
          const int idx = neighbor0 + para.K[0] * (neighbor1 + para.K[1] * neighbor2);
          mesh[idx] += charge[n] * Wx[n0 + 2] * Wy[n1 + 2] * Wz[n2 + 2];
        }
      }
    }
  }

  std::vector<double> mesh_local(mesh_size * 2, 0.0), mesh_global(mesh_size * 2, 0.0);
  for (int i = 0; i < mesh_size; ++i) {
    mesh_local[i] = mesh[i].real();
    mesh_local[mesh_size + i] = mesh[i].imag();
  }
  MPI_Allreduce(mesh_local.data(), mesh_global.data(), int(mesh_global.size()), MPI_DOUBLE, MPI_SUM, mpi_comm);
  for (int i = 0; i < mesh_size; ++i) {
    mesh[i] = std::complex<double>(mesh_global[i], mesh_global[mesh_size + i]);
  }

  std::vector<std::complex<double>> S;
  pppm_direct_dft(para, mesh, S, -1);

  static const double G_coeff[5] = {
    1.0000000e+00, -1.6666667e+00, 7.7777778e-01, -8.9947090e-02, 7.0546737e-04};
  std::vector<double> kx(mesh_size), ky(mesh_size), kz(mesh_size), G(mesh_size);
  for (int n = 0; n < mesh_size; ++n) {
    int nk[3];
    nk[2] = n / para.K0K1;
    nk[1] = (n - nk[2] * para.K0K1) / para.K[0];
    nk[0] = n % para.K[0];
    double denominator[3] = {0.0, 0.0, 0.0};
    for (int d = 0; d < 3; ++d) {
      if (nk[d] >= para.K_half[d]) nk[d] -= para.K[d];
      double t = sin(0.5 * para.two_pi_over_K[d] * nk[d]);
      t *= t;
      t = (((G_coeff[4] * t + G_coeff[3]) * t + G_coeff[2]) * t + G_coeff[1]) * t + G_coeff[0];
      denominator[d] = t * t;
    }
    kx[n] = nk[0] * para.b[0][0] + nk[1] * para.b[1][0] + nk[2] * para.b[2][0];
    ky[n] = nk[0] * para.b[0][1] + nk[1] * para.b[1][1] + nk[2] * para.b[2][1];
    kz[n] = nk[0] * para.b[0][2] + nk[1] * para.b[1][2] + nk[2] * para.b[2][2];
    const double ksq = kx[n] * kx[n] + ky[n] * ky[n] + kz[n] * kz[n];
    double numerator = pppm_sinc(0.5 * para.two_pi_over_K[0] * nk[0]);
    numerator *= pppm_sinc(0.5 * para.two_pi_over_K[1] * nk[1]);
    numerator *= pppm_sinc(0.5 * para.two_pi_over_K[2] * nk[2]);
    numerator = numerator * numerator * numerator * numerator * numerator;
    numerator *= numerator;
    G[n] = ksq == 0.0 ? 0.0 :
      numerator * para.two_pi_over_V / ksq * exp(-ksq * para.alpha_factor) /
      (denominator[0] * denominator[1] * denominator[2]);
  }

  std::vector<std::complex<double>> mesh_G(mesh_size), mesh_x(mesh_size), mesh_y(mesh_size), mesh_z(mesh_size);
  std::vector<std::complex<double>> mesh_v[6];
  for (int d = 0; d < 6; ++d) mesh_v[d].resize(mesh_size);
  for (int n = 0; n < mesh_size; ++n) {
    mesh_x[n] = std::complex<double>(S[n].imag() * kx[n] * G[n], -S[n].real() * kx[n] * G[n]);
    mesh_y[n] = std::complex<double>(S[n].imag() * ky[n] * G[n], -S[n].real() * ky[n] * G[n]);
    mesh_z[n] = std::complex<double>(S[n].imag() * kz[n] * G[n], -S[n].real() * kz[n] * G[n]);
    mesh_G[n] = S[n] * G[n];
    const double ksq = kx[n] * kx[n] + ky[n] * ky[n] + kz[n] * kz[n];
    if (ksq != 0.0) {
      const double alpha_k_factor = 2.0 * para.alpha_factor + 2.0 / ksq;
      const double B[6] = {
        1.0 - alpha_k_factor * kx[n] * kx[n],
        1.0 - alpha_k_factor * ky[n] * ky[n],
        1.0 - alpha_k_factor * kz[n] * kz[n],
        -alpha_k_factor * kx[n] * ky[n],
        -alpha_k_factor * ky[n] * kz[n],
        -alpha_k_factor * kz[n] * kx[n]};
      for (int d = 0; d < 6; ++d) mesh_v[d][n] = mesh_G[n] * B[d];
    }
  }

  std::vector<std::complex<double>> field_G, field_x, field_y, field_z, field_v[6];
  pppm_direct_dft(para, mesh_G, field_G, +1);
  pppm_direct_dft(para, mesh_x, field_x, +1);
  pppm_direct_dft(para, mesh_y, field_y, +1);
  pppm_direct_dft(para, mesh_z, field_z, +1);
  for (int d = 0; d < 6; ++d) pppm_direct_dft(para, mesh_v[d], field_v[d], +1);

  for (int n = 0; n < nlocal; ++n) {
    const double q = K_C_SP * charge[n];
    const double sx = (box.hi[0] * pos[n][0] + box.hi[1] * pos[n][1] + box.hi[2] * pos[n][2]) * para.K[0];
    const double sy = (box.hi[3] * pos[n][0] + box.hi[4] * pos[n][1] + box.hi[5] * pos[n][2]) * para.K[1];
    const double sz = (box.hi[6] * pos[n][0] + box.hi[7] * pos[n][1] + box.hi[8] * pos[n][2]) * para.K[2];
    const int ix = int(sx + 0.5);
    const int iy = int(sy + 0.5);
    const int iz = int(sz + 0.5);
    double Wx[5], Wy[5], Wz[5];
    pppm_weights(sx - ix, Wx);
    pppm_weights(sy - iy, Wy);
    pppm_weights(sz - iz, Wz);
    double D = 0.0, E[3] = {0.0, 0.0, 0.0}, V[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (int n0 = -2; n0 <= 2; ++n0) {
      const int neighbor0 = pppm_mesh_index(para.K[0], ix + n0);
      for (int n1 = -2; n1 <= 2; ++n1) {
        const int neighbor1 = pppm_mesh_index(para.K[1], iy + n1);
        for (int n2 = -2; n2 <= 2; ++n2) {
          const int neighbor2 = pppm_mesh_index(para.K[2], iz + n2);
          const int idx = neighbor0 + para.K[0] * (neighbor1 + para.K[1] * neighbor2);
          const double W = Wx[n0 + 2] * Wy[n1 + 2] * Wz[n2 + 2];
          D += W * field_G[idx].real();
          E[0] += W * field_x[idx].real();
          E[1] += W * field_y[idx].real();
          E[2] += W * field_z[idx].real();
          for (int d = 0; d < 6; ++d) V[d] += W * field_v[d][idx].real();
        }
      }
    }
    D_real[n] = 2.0 * K_C_SP * D;
    force[n][0] += 2.0 * q * E[0];
    force[n][1] += 2.0 * q * E[1];
    force[n][2] += 2.0 * q * E[2];
    const double v[6] = {q * V[0], q * V[1], q * V[2], q * V[3], q * V[5], q * V[4]};
    for (int d = 0; d < 6; ++d) total_virial[d] += v[d];
    if (virial && virial_components >= 6 && model_index == 0) {
      for (int d = 0; d < 6; ++d) virial[n][d] += v[d];
      if (virial_components >= 9) {
        virial[n][6] += v[3];
        virial[n][7] += v[4];
        virial[n][8] += v[5];
      }
    }
  }
  zero_global_mean(nlocal, natoms_global, mpi_comm, D_real);
}


std::vector<std::string> get_tokens(std::ifstream& input)
{
  std::string line;
  std::getline(input, line);
  std::istringstream iss(line);
  std::vector<std::string> tokens{
    std::istream_iterator<std::string>{iss}, std::istream_iterator<std::string>{}};
  return tokens;
}

void print_tokens(const std::vector<std::string>& tokens)
{
  std::cout << "Line:";
  for (const auto& token : tokens) {
    std::cout << " " << token;
  }
  std::cout << std::endl;
}

int get_int_from_token(const std::string& token, const char* filename, const int line)
{
  int value = 0;
  try {
    value = std::stoi(token);
  } catch (const std::exception& e) {
    std::cout << "Standard exception:\n";
    std::cout << "    File:          " << filename << std::endl;
    std::cout << "    Line:          " << line << std::endl;
    std::cout << "    Error message: " << e.what() << std::endl;
    exit(1);
  }
  return value;
}

double get_double_from_token(const std::string& token, const char* filename, const int line)
{
  double value = 0;
  try {
    value = std::stod(token);
  } catch (const std::exception& e) {
    std::cout << "Standard exception:\n";
    std::cout << "    File:          " << filename << std::endl;
    std::cout << "    Line:          " << line << std::endl;
    std::cout << "    Error message: " << e.what() << std::endl;
    exit(1);
  }
  return value;
}

} // namespace

NEP3_CPU::NEP3_CPU() {}

NEP3_CPU::NEP3_CPU(const std::string& potential_filename) { read_neptxt(potential_filename, true); }

void NEP3_CPU::read_neptxt(const std::string& potential_filename, const bool is_rank_0)
{
  int neplinenums = countNonEmptyLines(potential_filename);

  std::ifstream input(potential_filename);
  if (!input.is_open()) {
    std::cout << "Failed to open " << potential_filename << std::endl;
    exit(1);
  }

  // nep3 1 C
  std::vector<std::string> tokens = get_tokens(input);
  if (tokens.size() < 3) {
    print_tokens(tokens);
    std::cout << "The first line of nep.txt should have at least 3 items." << std::endl;
    exit(1);
  }
  if (tokens[0] == "nep") {
    paramb.model_type = 0;
    paramb.version = 2;
    zbl.enabled = false;
  } else if (tokens[0] == "nep_zbl") {
    paramb.model_type = 0;
    paramb.version = 2;
    zbl.enabled = true;
  } else if (tokens[0] == "nep_dipole") {
    paramb.model_type = 1;
    paramb.version = 2;
    zbl.enabled = false;
  } else if (tokens[0] == "nep_polarizability") {
    paramb.model_type = 2;
    paramb.version = 2;
    zbl.enabled = false;
  } else if (tokens[0] == "nep3") {
    paramb.model_type = 0;
    paramb.version = 3;
    zbl.enabled = false;
  } else if (tokens[0] == "nep3_zbl") {
    paramb.model_type = 0;
    paramb.version = 3;
    zbl.enabled = true;
  } else if (tokens[0] == "nep3_dipole") {
    paramb.model_type = 1;
    paramb.version = 3;
    zbl.enabled = false;
  } else if (tokens[0] == "nep3_polarizability") {
    paramb.model_type = 2;
    paramb.version = 3;
    zbl.enabled = false;
  } else if (tokens[0] == "nep4") {
    paramb.model_type = 0;
    paramb.version = 4;
    zbl.enabled = false;
  } else if (tokens[0] == "nep4_zbl") {
    paramb.model_type = 0;
    paramb.version = 4;
    zbl.enabled = true;
  } else if (tokens[0] == "nep4_charge2") {
    paramb.model_type = 0;
    paramb.version = 4;
    paramb.charge_mode = 2;
    zbl.enabled = false;
  } else if (tokens[0] == "nep4_zbl_charge2") {
    paramb.model_type = 0;
    paramb.version = 4;
    paramb.charge_mode = 2;
    zbl.enabled = true;
  } else if (tokens[0] == "nep4_dipole") {
    paramb.model_type = 1;
    paramb.version = 4;
    zbl.enabled = false;
  } else if (tokens[0] == "nep4_polarizability") {
    paramb.model_type = 2;
    paramb.version = 4;
    zbl.enabled = false;
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
    print_tokens(tokens);
    std::cout << "The first line of nep.txt should have " << paramb.num_types << " atom symbols."
              << std::endl;
    exit(1);
  }

  element_list.resize(paramb.num_types);
  element_atomic_number_list.resize(paramb.num_types);
  for (int n = 0; n < paramb.num_types; ++n) {
    int atomic_number = 0;
    element_list[n] = tokens[2 + n];
    for (int m = 0; m < NUM_ELEMENTS; ++m) {
      if (tokens[2 + n] == ELEMENTS[m]) {
        atomic_number = m + 1;
        break;
      }
    }
    element_atomic_number_list[n] = atomic_number;
    zbl.atomic_numbers[n] = atomic_number;
  }

  // zbl
  if (zbl.enabled) {
    tokens = get_tokens(input);
    if (tokens.size() != 3 && tokens.size() != 4) {
      print_tokens(tokens);
      std::cout << "This line should be zbl rc_inner rc_outer [zbl_factor]." << std::endl;
      exit(1);
    }
    zbl.rc_inner = get_double_from_token(tokens[1], __FILE__, __LINE__);
    zbl.rc_outer = get_double_from_token(tokens[2], __FILE__, __LINE__);
    if (zbl.rc_inner == 0 && zbl.rc_outer == 0) {
      zbl.flexibled = true;
    } else {
      if (tokens.size() == 4) {
        paramb.typewise_cutoff_zbl_factor = get_double_from_token(tokens[3], __FILE__, __LINE__);
        paramb.use_typewise_cutoff_zbl = true;
      }
    }
  }

  // cutoff 4.2 3.7 80 47
  tokens = get_tokens(input);
  if (tokens.size() != 3 && tokens.size() != 5) {
    print_tokens(tokens);
    std::cout << "This line should be cutoff rc_radial rc_angular [MN_radial] [MN_angular].\n";
    exit(1);
  }
  paramb.rc_radial = get_double_from_token(tokens[1], __FILE__, __LINE__);
  paramb.rc_angular = get_double_from_token(tokens[2], __FILE__, __LINE__);
  if (paramb.charge_mode == 2) {
    paramb.charge_alpha = PI / paramb.rc_radial;
    paramb.charge_alpha_factor = 0.25 / (paramb.charge_alpha * paramb.charge_alpha);
  }

  // n_max 10 8
  tokens = get_tokens(input);
  if (tokens.size() != 3) {
    print_tokens(tokens);
    std::cout << "This line should be n_max n_max_radial n_max_angular." << std::endl;
    exit(1);
  }
  paramb.n_max_radial = get_int_from_token(tokens[1], __FILE__, __LINE__);
  paramb.n_max_angular = get_int_from_token(tokens[2], __FILE__, __LINE__);

  // basis_size 10 8
  if (paramb.version >= 3) {
    tokens = get_tokens(input);
    if (tokens.size() != 3) {
      print_tokens(tokens);
      std::cout << "This line should be basis_size basis_size_radial basis_size_angular."
                << std::endl;
      exit(1);
    }
    paramb.basis_size_radial = get_int_from_token(tokens[1], __FILE__, __LINE__);
    paramb.basis_size_angular = get_int_from_token(tokens[2], __FILE__, __LINE__);
  }

  // l_max
  tokens = get_tokens(input);
  if (paramb.version == 2) {
    if (tokens.size() != 2) {
      print_tokens(tokens);
      std::cout << "This line should be l_max l_max_3body." << std::endl;
      exit(1);
    }
  } else {
    if (tokens.size() != 4) {
      print_tokens(tokens);
      std::cout << "This line should be l_max l_max_3body l_max_4body l_max_5body." << std::endl;
      exit(1);
    }
  }

  paramb.L_max = get_int_from_token(tokens[1], __FILE__, __LINE__);
  paramb.num_L = paramb.L_max;

  if (paramb.version >= 3) {
    int L_max_4body = get_int_from_token(tokens[2], __FILE__, __LINE__);
    int L_max_5body = get_int_from_token(tokens[3], __FILE__, __LINE__);
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
    print_tokens(tokens);
    std::cout << "This line should be ANN num_neurons 0." << std::endl;
    exit(1);
  }
  annmb.num_neurons1 = get_int_from_token(tokens[1], __FILE__, __LINE__);
  annmb.dim = (paramb.n_max_radial + 1) + paramb.dim_angular;

  // calculated parameters:
  paramb.rcinv_radial = 1.0f / paramb.rc_radial;
  paramb.rcinv_angular = 1.0f / paramb.rc_angular;
  paramb.num_types_sq = paramb.num_types * paramb.num_types;

  annmb.num_c2   = paramb.num_types_sq * (paramb.n_max_radial + 1) * (paramb.basis_size_radial + 1);
  annmb.num_c3   = paramb.num_types_sq * (paramb.n_max_angular + 1) * (paramb.basis_size_angular + 1);
  
  int tmp = 0;
  if (paramb.charge_mode == 2) {
    annmb.num_para_ann = (annmb.dim + 3) * annmb.num_neurons1 * paramb.num_types + 2;
  } else if (paramb.version == 3) {
    annmb.num_para_ann = (annmb.dim + 2) * annmb.num_neurons1 + 1;
  } else if (paramb.version == 4) {
    annmb.num_para_ann = (annmb.dim + 2) * annmb.num_neurons1 * paramb.num_types;
  } else{
    annmb.num_para_ann = ((annmb.dim + 2) * annmb.num_neurons1 + 1) * paramb.num_types + 1;
  }

  tmp = annmb.num_para_ann + annmb.num_c2 + annmb.num_c3 + 6 + annmb.dim;

  int num_type_zbl = 0;
  if (zbl.enabled && zbl.flexibled) {
    num_type_zbl = (paramb.num_types * (paramb.num_types + 1)) / 2;
    neplinenums -= (1 + 10*num_type_zbl);// zbl 0 0; fixed zbl
  } else if (zbl.enabled) {
    neplinenums  -= 1; // zbl a b
  }

  bool is_gpumd_nep = false;
  if (paramb.charge_mode == 2) {
    is_gpumd_nep = true;
  } else if (paramb.num_types == 1) {
    is_gpumd_nep = false;
  } else if (paramb.version == 4) {
    if (neplinenums  == (tmp + 1)) {
      is_gpumd_nep = true;
      if (is_rank_0) {
        printf("    the input nep4 potential file is from GPUMD.\n");
      }
    } else if (neplinenums  == (tmp + paramb.num_types)) {
      if(is_rank_0) {
        printf("    the input nep4 potential file is from MatPL.\n");
      }
    } else {
    printf("    parameter parsing error, the number of nep parameters [PWMLFF %d, GPUMD %d] does not match the text lines %d.\n", tmp, (tmp-paramb.num_types+1), neplinenums);
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
  
  // annmb.num_para = (annmb.dim + 2) * annmb.num_neurons1 * (paramb.version == 4 ? paramb.num_types : 1) + (paramb.version == 4 ? paramb.num_types : 1);
  
  if (paramb.model_type == 2) {
    annmb.num_para *= 2;
  }
  int num_para_descriptor =annmb.num_c2 + annmb.num_c3;
  annmb.num_para += num_para_descriptor;
  paramb.num_c_radial =
    paramb.num_types_sq * (paramb.n_max_radial + 1) * (paramb.basis_size_radial + 1);

  if (paramb.version == 2) {
    paramb.num_c_radial =
      (paramb.num_types == 1) ? 0 : paramb.num_types_sq * (paramb.n_max_radial + 1);
  }

  // NN and descriptor parameters
  parameters.resize(annmb.num_para);
  for (int n = 0; n < annmb.num_para; ++n) {
    if (paramb.charge_mode == 0 && is_gpumd_nep == true && (n >= annmb.num_para_ann + 1) && (n < annmb.num_para_ann + paramb.num_types)) {
      parameters[n] = parameters[annmb.num_para_ann];
      if (is_rank_0) {
        printf("copy the last bias parameters[%d]=%f to parameters[%d]=%f \n", annmb.num_para_ann, parameters[annmb.num_para_ann], n, parameters[n]);
      }    
    } else {
      tokens = get_tokens(input);
      parameters[n] = get_double_from_token(tokens[0], __FILE__, __LINE__);
    }
  }
  update_potential(parameters.data(), annmb);
  for (int d = 0; d < annmb.dim; ++d) {
    tokens = get_tokens(input);
    paramb.q_scaler[d] = get_double_from_token(tokens[0], __FILE__, __LINE__);
  }

  // flexible zbl potential parameters if (zbl.flexibled)
  if (zbl.flexibled) {
    // int num_type_zbl = (paramb.num_types * (paramb.num_types + 1)) / 2;
    for (int d = 0; d < 10 * num_type_zbl; ++d) {
      tokens = get_tokens(input);
      zbl.para[d] = get_double_from_token(tokens[0], __FILE__, __LINE__);
    }
    zbl.num_types = paramb.num_types;
  }
  input.close();

  // only report for rank_0
  if (is_rank_0) {

    if (paramb.num_types == 1) {
      std::cout << "    Use the NEP" << paramb.version << " potential with " << paramb.num_types
                << " atom type.\n";
    } else {
      std::cout << "    Use the NEP" << paramb.version << " potential with " << paramb.num_types
                << " atom types.\n";
    }

    for (int n = 0; n < paramb.num_types; ++n) {
      std::cout << "    type " << n << "( " << element_list[n]
                << " with Z = " << zbl.atomic_numbers[n] << ").\n";
    }

    if (zbl.enabled) {
      if (zbl.flexibled) {
        std::cout << "    has flexible ZBL.\n";
      } else {
        std::cout << "    has universal ZBL with inner cutoff " << zbl.rc_inner
                  << " A and outer cutoff " << zbl.rc_outer << " A.\n";
        if (paramb.use_typewise_cutoff_zbl) {
          std::cout << "    ZBL typewise cutoff is enabled with factor "
                    << paramb.typewise_cutoff_zbl_factor << ".\n";
        }
      }
    }

    std::cout << "    radial cutoff = " << paramb.rc_radial << " A.\n";
    std::cout << "    angular cutoff = " << paramb.rc_angular << " A.\n";
    std::cout << "    n_max_radial = " << paramb.n_max_radial << ".\n";
    std::cout << "    n_max_angular = " << paramb.n_max_angular << ".\n";
    if (paramb.version >= 3) {
      std::cout << "    basis_size_radial = " << paramb.basis_size_radial << ".\n";
      std::cout << "    basis_size_angular = " << paramb.basis_size_angular << ".\n";
    }
    std::cout << "    l_max_3body = " << paramb.L_max << ".\n";
    std::cout << "    l_max_4body = " << (paramb.num_L >= 5 ? 2 : 0) << ".\n";
    std::cout << "    l_max_5body = " << (paramb.num_L >= 6 ? 1 : 0) << ".\n";
    std::cout << "    ANN = " << annmb.dim << "-" << annmb.num_neurons1 << "-1.\n";
    if (is_gpumd_nep) {
      std::cout << "    the input nep potential file is from GPUMD.\n";
      std::cout << "    number of neural network parameters = "
                << annmb.num_para - num_para_descriptor - paramb.num_types + 1 << ".\n";
      std::cout << "    number of descriptor parameters = " << num_para_descriptor << ".\n";
      std::cout << "    total number of parameters = " << annmb.num_para - paramb.num_types + 1 << ".\n";

    } else {
      std::cout << "    the input nep potential file is from MATPL.\n";
    
      std::cout << "    number of neural network parameters = "
                << annmb.num_para - num_para_descriptor << ".\n";
      std::cout << "    number of descriptor parameters = " << num_para_descriptor << ".\n";
      std::cout << "    total number of parameters = " << annmb.num_para << ".\n";
    }
  }
}

void NEP3_CPU::update_potential(double* parameters, ANN& ann)
{
  double* pointer = parameters;
  if (paramb.charge_mode == 2) {
    const int num_outputs = 2;
    for (int t = 0; t < paramb.num_types; ++t) {
      ann.w0[t] = pointer;
      pointer += ann.num_neurons1 * ann.dim;
      ann.b0[t] = pointer;
      pointer += ann.num_neurons1;
      ann.w1[t] = pointer;
      pointer += ann.num_neurons1 * num_outputs;
    }
    ann.sqrt_epsilon_inf = pointer;
    pointer += 1;
    ann.b1 = pointer;
    pointer += 1;
    ann.c = pointer;
    return;
  }
  for (int t = 0; t < paramb.num_types; ++t) {
    if (t > 0 && paramb.version == 3) { // Use the same set of NN parameters for NEP2 and NEP3_CPU
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
  pointer += (paramb.version == 4 ? paramb.num_types : 1); // the nep4 bias toghers
  // for (int ii = 0; ii < paramb.num_types; ++ii){
  //   std::cout << "last bias " << ann.b1[ii]<< " type " << ii << std::endl;
  // }
  if (paramb.model_type == 2) {
    for (int t = 0; t < paramb.num_types; ++t) {
      if (t > 0 && paramb.version != 4) { // Use the same set of NN parameters for NEP2 and NEP3_CPU
        pointer -= (ann.dim + 2) * ann.num_neurons1;
      }
      ann.w0_pol[t] = pointer;
      pointer += ann.num_neurons1 * ann.dim;
      ann.b0_pol[t] = pointer;
      pointer += ann.num_neurons1;
      ann.w1_pol[t] = pointer;
      pointer += ann.num_neurons1;
    }
    ann.b1_pol = pointer;
    pointer += 1;
  }

  ann.c = pointer;
}

void NEP3_CPU::compute_for_lammps(
  int nlocal,
  int N,
  int* ilist,
  int* NN,
  int** NL,
  int* type,
  double** pos,
  double& total_potential,
  double total_virial[6],
  double* potential,
  double** force,
  double** virial,
  int virial_components,
  int model_index,
  double xprd,
  double yprd,
  double zprd,
  double xy,
  double xz,
  double yz,
  const std::string& kspace_method,
  long long natoms_global,
  MPI_Comm mpi_comm)
{
  if (num_atoms < nlocal) {//num_atoms is 0,so the num_atoms = nlocal
    Fp.resize(nlocal * annmb.dim);
    Fp_charge.resize(nlocal * annmb.dim);
    charge.resize(nlocal);
    charge_derivative.resize(nlocal * annmb.dim);
    D_real.resize(nlocal);
    sum_fxyz.resize(nlocal * (paramb.n_max_angular + 1) * NUM_OF_ABC);
    num_atoms = nlocal;
  }
  find_descriptor_for_lammps(
    paramb, annmb, nlocal, N, ilist, NN, NL, type, map_atom_type_idx, pos,
    Fp.data(), sum_fxyz.data(), total_potential, potential,
    paramb.charge_mode == 2 ? charge.data() : nullptr,
    paramb.charge_mode == 2 ? charge_derivative.data() : nullptr);
  if (paramb.charge_mode == 2) {
    NEP_CPU_Box box;
    box.h[0] = xprd;
    box.h[1] = xy;
    box.h[2] = xz;
    box.h[3] = 0.0;
    box.h[4] = yprd;
    box.h[5] = yz;
    box.h[6] = 0.0;
    box.h[7] = 0.0;
    box.h[8] = zprd;
    const double det = box.h[0] * (box.h[4] * box.h[8] - box.h[5] * box.h[7]) -
                       box.h[1] * (box.h[3] * box.h[8] - box.h[5] * box.h[6]) +
                       box.h[2] * (box.h[3] * box.h[7] - box.h[4] * box.h[6]);
    box.hi[0] = (box.h[4] * box.h[8] - box.h[5] * box.h[7]) / det;
    box.hi[1] = (box.h[2] * box.h[7] - box.h[1] * box.h[8]) / det;
    box.hi[2] = (box.h[1] * box.h[5] - box.h[2] * box.h[4]) / det;
    box.hi[3] = (box.h[5] * box.h[6] - box.h[3] * box.h[8]) / det;
    box.hi[4] = (box.h[0] * box.h[8] - box.h[2] * box.h[6]) / det;
    box.hi[5] = (box.h[2] * box.h[3] - box.h[0] * box.h[5]) / det;
    box.hi[6] = (box.h[3] * box.h[7] - box.h[4] * box.h[6]) / det;
    box.hi[7] = (box.h[1] * box.h[6] - box.h[0] * box.h[7]) / det;
    box.hi[8] = (box.h[0] * box.h[4] - box.h[1] * box.h[3]) / det;
    if (kspace_method != "ewald" && kspace_method != "pppm") {
      std::cout << "NEP CPU charge_mode=2 kspace_method must be ewald or pppm, got "
                << kspace_method << std::endl;
      exit(1);
    }
    if (kspace_method == "pppm") {
      find_force_charge_pppm_cpu(
        paramb, nlocal, natoms_global, pos, box, mpi_comm, charge, D_real, force, total_virial, virial,
        virial_components, model_index);
    } else {
      find_force_charge_ewald_cpu(
        paramb, nlocal, natoms_global, pos, box, mpi_comm, charge, D_real, force, total_virial, virial,
        virial_components, model_index);
    }
    for (int d = 0; d < annmb.dim; ++d) {
      for (int n = 0; n < nlocal; ++n) {
        const int idx = d * nlocal + n;
        Fp[idx] += charge_derivative[idx] * D_real[n];
      }
    }
  }
  find_force_radial_for_lammps( 
    paramb, annmb, nlocal, N, ilist, NN, NL, type, map_atom_type_idx, pos, Fp.data(),
    force, total_virial, virial, virial_components, model_index);
  find_force_angular_for_lammps(
    paramb, annmb, nlocal, N, ilist, NN, NL, type, map_atom_type_idx, pos, Fp.data(), sum_fxyz.data(),
    force, total_virial, virial, virial_components, model_index);
  if (zbl.enabled) {
    find_force_ZBL_for_lammps(
      paramb, zbl, N, ilist, NN, NL, type, map_atom_type_idx,  pos, force, total_virial, virial,
      virial_components, total_potential, potential, model_index);
  }
}
