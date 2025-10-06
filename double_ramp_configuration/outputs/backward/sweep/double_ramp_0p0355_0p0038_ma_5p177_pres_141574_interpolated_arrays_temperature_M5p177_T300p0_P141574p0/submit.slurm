#!/bin/bash
#SBATCH --job-name=double_ramp_0p0355_0p0038_ma_5p177_pres_141574_interpolated_arrays_temperature_M5p177_T300p0_P141574p0
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --time=01:00:00
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=4

module load su2/8.0.0
srun SU2_CFD case.cfg
