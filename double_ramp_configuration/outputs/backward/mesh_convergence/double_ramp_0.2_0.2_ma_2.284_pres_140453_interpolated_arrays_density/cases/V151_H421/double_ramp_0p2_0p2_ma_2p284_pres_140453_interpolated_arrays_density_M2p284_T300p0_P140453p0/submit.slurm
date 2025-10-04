#!/bin/bash
#SBATCH --job-name=double_ramp_0p2_0p2_ma_2p284_pres_140453_interpolated_arrays_density_M2p284_T300p0_P140453p0
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --time=01:00:00
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=4

module load su2/8.0.0
srun SU2_CFD case.cfg
