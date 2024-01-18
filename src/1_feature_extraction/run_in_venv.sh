#!/bin/bash

eval "$(conda shell.bash hook)"

conda activate /home/vkomeyer/miniconda3/envs/confound_continuum
if [ $? -ne 0 ]; then
    echo "Error activating the environment"
    exit -1
fi

conda info
echo $PATH

export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

python $@

