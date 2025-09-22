#!/bin/bash

cd ./double_ramp_configuration/outputs/backward/sweep/double_ramp_0p0199_0p0465_ma_6p671_pres_141047_interpolated_arrays_temperature_M6p671_T300p0_P141047p0
sbatch submit.slurm
cd - > /dev/null
cd ./double_ramp_configuration/outputs/backward/sweep/double_ramp_0p0443_0p0677_ma_5p163_pres_108345_interpolated_arrays_temperature_M5p163_T300p0_P108345p0
sbatch submit.slurm
cd - > /dev/null
cd ./double_ramp_configuration/outputs/backward/sweep/double_ramp_0p0491_0p0254_ma_3p506_pres_113826_interpolated_arrays_temperature_M3p506_T300p0_P113826p0
sbatch submit.slurm
cd - > /dev/null
