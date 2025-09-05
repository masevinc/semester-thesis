#!/bin/bash
#SBATCH --job-name=double_ramp_0p011_0p0488_ma_6p859_pres_194500_interpolated_arrays_density_M6p859_T250p0_P194500p0
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --time=01:00:00
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=4

module load su2/8.1.0
srun SU2_CFD case.cfg
