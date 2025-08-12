#!/bin/bash
#SBATCH --job-name=double_ramp_0p046_0p0112_ma_2p477_pres_96187_interpolated_arrays_density_M2p477_T275p0_P96187p0
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --time=01:00:00
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=4

module load su2/4.1.0
srun SU2_CFD case.cfg
