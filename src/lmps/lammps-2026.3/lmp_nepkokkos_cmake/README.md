# nep with KOKKOS

In this LAMMPS Kokkos version of NEP, neighbor list computations are handled by Kokkos on the GPU, while energy and force calculations are performed through custom C++/CUDA operators. Compared to the earlier MatPL-2025.3 version, memory consumption has been reduced by nearly one-third, and inference speed has been significantly accelerated. Furthermore, it demonstrates excellent parallel efficiency across nodes, enabling scalability to multi-node clusters.

* This Kokkos interface for NEP supports the nep.txt forcefield files trained with both `MatPL-2025.3(and later version)` and `GPUMD`.

* The code implementation is based on the NEP file parsing, descriptor calculation, and force evaluation routines from GPUMD 4.6 (https://github.com/brucefan1983/GPUMD). Building on this foundation, we have adapted it to LAMMPS's KOKKOS neighbor list conventions, meticulously optimized the kernels for descriptor and force calculations, and incorporated a multi-model deviation calculation feature for active learning.

# how to compile

## 1. copy files to lammpsfir

We recommend using the `kknep-pach.sh` script to automatically copy the nep code to the LAMMPS directory and modify the CMakeLists.txt file. The command is as follows:

```bash
bash kknep-patch.sh the/path/of/lammps/rootdir
```

* Currently, versions lammps-stable-22Jul2025updata4, lammps-stable_29Aug2024_update4 and lammps-stable_2Aug2023_update4 are supported. 

## 2. load the compilation environment

Compilation requires OpenMPI, CUDA, CMake, and GCC. We recommend using CUDA/11.6, OpenMPI/4.1.6, CMake/3.21.6, and GCC 8.5. Note that CUDA 11.8 and later versions require a more recent driver.

## 3. doing compilation

```bash
mkdir build & cd build

cmake -C ../cmake/presets/basic.cmake \
    -DPKG_MESONT=no \
    -DPKG_JPEG=no \
    -DPKG_KOKKOS=yes \
    -DPKG_NEP_KK=yes \
    -DKokkos_ENABLE_CUDA=yes \
    -DKokkos_ENABLE_OPENMP=yes \
    -DKokkos_ENABLE_CUDA_LAMBDA=yes \
    -DFFT_KOKKOS=CUFFT \
    -DKokkos_ARCH_AMPERE86=ON \
    -DTEST_TIME=ON \
    ../cmake

cmake --build . -j N #Number of cores for parallel compilation
```

```bash
# If you also need to compile the DP interface, please import the PyTorch path, import the MKL library, and enable the C++ STD17 standard for compilation.

export Torch_DIR=\$(python -c \import torch; print(torch.utils.cmake_prefix_path)\)/Torch

# Then, add the following option in cmake:
-DTorch_DIR=\${Torch_DIR}
-DCMAKE_CXX_STANDARD=17
-DPKG_MATPLDP=yes

# For the D3 interface, please add the following option in cmake. Note that D3 requires CUDA support and cannot be used in combination with matpl/nep/kk.
-DPKG_MATPLD3=yes

# NEP adopts single-precision inference by default. For double-precision inference, please add the following option in cmake.
-DPREC_NEPINFER=ON
```

# how to use
Reference case ./examples

```txt
pair_style   matpl/nep/kk  nep.txt
pair_coeff   * * Hf O
```

```txt
# for multi-model deviation
pair_style   matpl/nep/kk  nep0.txt nep1.txt nep2.txt nep3.txt out_freq ${DUMP_FREQ} out_file model_devi.out
pair_coeff       * * Hf O

```

## NEP-charge / qNEP

For `nep4_charge2` QNEP potentials, specify the reciprocal-space charge solver in the `pair_style` command with `kspace ewald` or `kspace pppm`:

```txt
# Ewald reciprocal-space solver
pair_style   matpl/nep/kk  nep.txt kspace ewald
pair_coeff   * * Hf O
```

```txt
# PPPM reciprocal-space solver
pair_style   matpl/nep/kk  nep.txt kspace pppm
pair_coeff   * * Hf O
```

By default, PPPM uses a power-of-two mesh selected from the simulation box size. For large systems, the
power-of-two mesh can consume substantial GPU memory; in that case, explicitly
request an FFT-friendly mesh whose prime factors are limited to `2`, `3`, `5`, and `7`.

```txt
# PPPM with memory-friendly FFT mesh
pair_style   matpl/nep/kk  nep.txt kspace pppm pppm_mesh friendly
pair_coeff   * * Hf O
```

The mesh size is calculated as follows:

$$
K=min\left \{ n \ge m | n = 2^a * 3^b * 5^c * 7 ^d \right \} 
$$


For example, with a 70-cell width of HfO2, the cell length is approximately 371.4 Å. Using a power of 2, the mesh size is 512 > 371.4Å. The entire mesh size is 512 * 512 * 512, which will consume a large amount of GPU memory. In this case, using the frinedly method, the mesh size is 375.

You can also set the mesh explicitly. This has the highest priority:

```txt
# PPPM with explicit mesh dimensions
pair_style   matpl/nep/kk  nep.txt kspace pppm pppm_mesh 384 384 384
pair_coeff   * * Hf O
```

The optional `pppm_spacing` keyword changes the target grid spacing used by
the automatic `power2` and `friendly` modes. Its default value is `1.0`.

The same option can also be written as `kspace_method ewald` or `kspace_method pppm`.

### Dump Born effective charge tensors

For QNEP potentials with BEC support, per-atom Born effective charge tensors can be exposed through the LAMMPS compute style `qnep/bec/atom` and written by a standard `dump custom` command.

```txt
compute      cbec all qnep/bec/atom

dump         mydump all custom 100 dump.peratom id type x y z fx fy fz &
             c_cbec[1] c_cbec[2] c_cbec[3] &
             c_cbec[4] c_cbec[5] c_cbec[6] &
             c_cbec[7] c_cbec[8] c_cbec[9]
```

The nine components are written in row-major tensor order:

```txt
c_cbec[1] = Zxx   c_cbec[2] = Zxy   c_cbec[3] = Zxz
c_cbec[4] = Zyx   c_cbec[5] = Zyy   c_cbec[6] = Zyz
c_cbec[7] = Zzx   c_cbec[8] = Zzy   c_cbec[9] = Zzz
```

The compute is intended for `pair_style matpl/nep/kk` with a `nep4_charge2` QNEP potential. It can be used with both `kspace ewald` and `kspace pppm`. The implementation follows normal LAMMPS per-atom output behavior, including reverse communication of ghost-atom BEC contributions before dumping owned atoms.

Example LAMMPS inputs are provided in:

```txt
examples/nep-charge/lmps-ewald
examples/nep-charge/lmps-pppm
```

# Citation
* If you use the LAMMPS interface of this nep-kokkos here, you are suggested to cite the following paper:

  * Pengfei Suo, Xingxing Wu, Hongzhen Tian, et al. Towards Scalable and Efficient Machine-Learning Force Fields: The MatPL package and Its Advancements on Neuroevolution Potentials. ChemRxiv. 06 May 2026.
DOI: https://doi.org/10.26434/chemrxiv.15001665/v3

* If you directly or indirectly use the `NEP` class here, you are suggested to cite the following paper:

  * Ke Xu, Hekai Bu, Shuning Pan, Eric Lindgren, Yongchao Wu, Yong Wang, Jiahui Liu, Keke Song, Bin Xu, Yifan Li, Tobias Hainer, Lucas Svensson, Julia Wiktor, Rui Zhao, Hongfu Huang, Cheng Qian, Shuo Zhang, Zezhu Zeng, Bohan Zhang, Benrui Tang, Yang Xiao, Zihan Yan, Jiuyang Shi, Zhixin Liang, Junjie Wang, Ting Liang, Shuo Cao, Yanzhou Wang, Penghua Ying, Nan Xu, Chengbing Chen, Yuwen Zhang, Zherui Chen, Xin Wu, Wenwu Jiang, Esme Berger, Yanlong Li, Shunda Chen, Alexander J. Gabourie, Haikuan Dong, Shiyun Xiong, Ning Wei, Yue Chen, Jianbin Xu, Feng Ding, Zhimei Sun, Tapio Ala-Nissila, Ari Harju, Jincheng Zheng, Pengfei Guan, Paul Erhart, Jian Sun, Wengen Ouyang, Yanjing Su, Zheyong Fan, [GPUMD 4.0: A high-performance molecular dynamics package for versatile materials simulations with machine-learned potentials]( https://doi.org/10.1002/mgea.70028), MGE Advances **3**, e70028 (2025).

* If you use QNEP/NEP-charge potentials with dynamic charges, you are suggested to cite the following paper:

  * Zheyong Fan*, Benrui Tang, Esmée Berger, Ethan Berger, Erik Fransson, Ke Xu, Zihan Yan, Zhoulin Liu, Zichen Song, Haikuan Dong, Shunda Chen, Lei Li, Ziliang Wang, Yizhou Zhu, Julia Wiktor, Paul Erhart*, [qNEP: A Highly Efficient Neuroevolution Potential with Dynamic Charges for Large-Scale Atomistic Simulations](https://doi.org/10.1021/acs.jctc.6c00146), J. Chem. Theory Comput. 2026.

* If you use the LAMMPS interface of the `NEP` class, a proper citation for LAMMPS is also suggested. 
