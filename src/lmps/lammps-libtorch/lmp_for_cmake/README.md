# how to install

# load env
1. activate python env（pytorch）
2. load cuda/11.8-share intel/2020
3. load gcc11
4. load cmake/3. # if the cmake in python-env, skip 

# compile

This directory provides a compilation script named `install.sh`. After setting up the environment, execute:

```bash
    sh install.sh lammps_dir [install_dir]
```

- The `lammps_dir` is **mandatory**, specifying the root directory path of the LAMMPS source code.  
- The `install_dir` is **optional**. If not specified, the LAMMPS executable will be installed to `lammps_dir/build` by default. Otherwise, it will be installed to the `install_dir` directory.  

After installation, the target directory will contain:  
1. The compiled executable file
2. An environment configuration script `env.sh`
To use it, simply run `source /the/path/env.sh` 

# to use lmps

```bash
#1. load env
module load intel/2020 cuda/11.8-share
source /the/install/path/env.sh
#2. run lmp
mpirun -np N lmp -in in.lmp
```
