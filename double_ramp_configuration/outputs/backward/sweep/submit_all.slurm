#!/bin/bash

cd ./double_ramp_configuration/outputs/backward/sweep/double_ramp_0p2_0p2_ma_2p284_pres_140453_interpolated_arrays_density_M2p284_T300p0_P140453p0
sbatch submit.slurm
cd - > /dev/null
