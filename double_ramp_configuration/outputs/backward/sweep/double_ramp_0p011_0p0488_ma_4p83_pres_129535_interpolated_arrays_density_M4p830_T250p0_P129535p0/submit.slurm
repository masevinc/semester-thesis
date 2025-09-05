#!/bin/bash
#SBATCH --job-name=double_ramp_0p011_0p0488_ma_4p83_pres_129535_interpolated_arrays_density_M4p830_T250p0_P129535p0
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --time=01:00:00
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=4

module load su2/8.1.0
srun SU2_CFD case.cfg
