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

* Currently, only versions stable_29Aug2024_update4 and lammps-stable_2Aug2023_update4 are supported (igher versions are not supported ) 
* The lammps-stable_2Aug2023_update4 is more faster than stable_29Aug2024_update4 in our test.

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

# Citation
* If you use the LAMMPS interface of this nep-kokkos here, you are suggested to cite the following (`The article will be released soon`):

  * https://github.com/LonxunQuantum/MatPL

* If you directly or indirectly use the `NEP` class here, you are suggested to cite the following paper:

  * Ke Xu, Hekai Bu, Shuning Pan, Eric Lindgren, Yongchao Wu, Yong Wang, Jiahui Liu, Keke Song, Bin Xu, Yifan Li, Tobias Hainer, Lucas Svensson, Julia Wiktor, Rui Zhao, Hongfu Huang, Cheng Qian, Shuo Zhang, Zezhu Zeng, Bohan Zhang, Benrui Tang, Yang Xiao, Zihan Yan, Jiuyang Shi, Zhixin Liang, Junjie Wang, Ting Liang, Shuo Cao, Yanzhou Wang, Penghua Ying, Nan Xu, Chengbing Chen, Yuwen Zhang, Zherui Chen, Xin Wu, Wenwu Jiang, Esme Berger, Yanlong Li, Shunda Chen, Alexander J. Gabourie, Haikuan Dong, Shiyun Xiong, Ning Wei, Yue Chen, Jianbin Xu, Feng Ding, Zhimei Sun, Tapio Ala-Nissila, Ari Harju, Jincheng Zheng, Pengfei Guan, Paul Erhart, Jian Sun, Wengen Ouyang, Yanjing Su, Zheyong Fan, [GPUMD 4.0: A high-performance molecular dynamics package for versatile materials simulations with machine-learned potentials]( https://doi.org/10.1002/mgea.70028), MGE Advances **3**, e70028 (2025).

* If you use the LAMMPS interface of the `NEP` class, a proper citation for LAMMPS is also suggested. 
