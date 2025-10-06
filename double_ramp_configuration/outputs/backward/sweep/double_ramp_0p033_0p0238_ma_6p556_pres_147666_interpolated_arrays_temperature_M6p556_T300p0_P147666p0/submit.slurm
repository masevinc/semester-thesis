#!/bin/bash
#SBATCH --job-name=double_ramp_0p033_0p0238_ma_6p556_pres_147666_interpolated_arrays_temperature_M6p556_T300p0_P147666p0
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --time=01:00:00
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=4

module load su2/8.0.0
srun SU2_CFD case.cfg
