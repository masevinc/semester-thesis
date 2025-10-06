#!/bin/bash
#SBATCH --job-name=double_ramp_0p0125_0p0085_ma_2p735_pres_155157_interpolated_arrays_temperature_M2p735_T300p0_P155157p0
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --time=01:00:00
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=4

module load su2/8.0.0
srun SU2_CFD case.cfg
