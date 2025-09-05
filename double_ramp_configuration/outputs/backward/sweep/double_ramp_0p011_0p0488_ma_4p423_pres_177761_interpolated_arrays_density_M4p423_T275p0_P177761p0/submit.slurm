#!/bin/bash
#SBATCH --job-name=double_ramp_0p011_0p0488_ma_4p423_pres_177761_interpolated_arrays_density_M4p423_T275p0_P177761p0
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --time=01:00:00
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=4

module load su2/8.1.0
srun SU2_CFD case.cfg
