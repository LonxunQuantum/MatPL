# Heat Flux Example

This example calculates the thermal conductivity of graphene using the MatPL-NEP potential in LAMMPS, based on the non-equilibrium molecular dynamics (NEMD) method.

## Files

| File | Description |
|------|-------------|
| `graphene.in` | LAMMPS input script for heat flux simulation |
| `graphene.data` | Graphene structure data file |
| `nep.txt` | NEP potential file |
| `job.sh` | SLURM job submission script |
| `compute_Energy_Temp.out` | Reference output: energy and temperature from thermostats |
| `compute_HeatFlux.out` | Reference output: heat flux from atomic computations |
| `calc_heatflux.py` | Visualization script, generates `HeatFlux.png` |
| `check_test.py` | Automated test validation script |

## Method

The simulation applies Langevin thermostats at two ends of the graphene sheet (350 K and 250 K) to establish a temperature gradient. The accumulated heat is computed in two independent ways:

1. **From atoms**: integrating the heat flux computed from atomic stress tensors
2. **From thermostats**: tracking energy added/removed by the hot/cold thermostats

If the two curves agree, the heat flux calculation is validated.

## Running the Simulation

```bash
# Submit via SLURM
sbatch job.sh

# Or run directly (adjust GPU/MPI settings as needed)
mpirun -np 4 lmp -k on g 4 -sf kk -pk kokkos -in graphene.in
```

## Validating Results

After the simulation finishes, run the test validation:

```bash
python check_test.py
```

The script compares the two heat accumulation curves and reports PASS/FAIL based on:
- Relative max error < 5%
- R-squared > 0.95

To generate the comparison plot:

```bash
python calc_heatflux.py
```

This produces `HeatFlux.png` showing both curves overlaid.

## Expected Output

A successful test produces two nearly overlapping curves of accumulated heat vs. time, confirming that the MatPL-NEP potential correctly computes heat flux in LAMMPS.

![HeatFlux](HeatFlux.png)
