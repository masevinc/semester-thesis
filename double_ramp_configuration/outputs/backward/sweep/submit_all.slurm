#!/bin/bash

cd ./double_ramp_configuration/outputs/backward/sweep/double_ramp_0p0125_0p0085_ma_2p735_pres_155157_interpolated_arrays_temperature_M2p735_T300p0_P155157p0
sbatch submit.slurm
cd - > /dev/null
cd ./double_ramp_configuration/outputs/backward/sweep/double_ramp_0p0151_0p0131_ma_4p989_pres_110414_interpolated_arrays_temperature_M4p989_T300p0_P110414p0
sbatch submit.slurm
cd - > /dev/null
cd ./double_ramp_configuration/outputs/backward/sweep/double_ramp_0p033_0p0238_ma_6p556_pres_147666_interpolated_arrays_temperature_M6p556_T300p0_P147666p0
sbatch submit.slurm
cd - > /dev/null
cd ./double_ramp_configuration/outputs/backward/sweep/double_ramp_0p0355_0p0038_ma_5p177_pres_141574_interpolated_arrays_temperature_M5p177_T300p0_P141574p0
sbatch submit.slurm
cd - > /dev/null
