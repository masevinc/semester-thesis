#!/bin/bash
#SBATCH --job-name=mesh_angle_r1_20_r2_2p5_M2p5_T300p0_P101325
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --time=01:00:00
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=4

module load su2/4.1.0
srun SU2_CFD case.cfg
