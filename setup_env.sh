#!/bin/bash
eval "$(conda shell.bash hook)"
mamba env create -f environment.yml && conda activate objaxdp && echo $(which pip) && pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_releases.html && pip install --upgrade objax