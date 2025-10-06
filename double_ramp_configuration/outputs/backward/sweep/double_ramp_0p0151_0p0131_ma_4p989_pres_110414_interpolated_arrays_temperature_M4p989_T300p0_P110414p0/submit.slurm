#!/bin/bash
#SBATCH --job-name=double_ramp_0p0151_0p0131_ma_4p989_pres_110414_interpolated_arrays_temperature_M4p989_T300p0_P110414p0
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --time=01:00:00
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=4

module load su2/8.0.0
srun SU2_CFD case.cfg
