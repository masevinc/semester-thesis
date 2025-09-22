#!/bin/bash
#SBATCH --job-name=double_ramp_0p0491_0p0254_ma_3p506_pres_113826_interpolated_arrays_temperature_M3p506_T300p0_P113826p0
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --time=01:00:00
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=4

module load su2/8.0.0
srun SU2_CFD case.cfg
