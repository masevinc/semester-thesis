#!/bin/bash
#SBATCH --job-name=mesh_angle_r1_10_r2_17p5_M2p5_T250p0_P110000
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --time=01:00:00
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=4

module load su2/4.1.0
srun SU2_CFD case.cfg
