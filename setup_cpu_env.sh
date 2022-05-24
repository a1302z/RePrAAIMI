#!/bin/bash
eval "$(conda shell.bash hook)"
conda env create -f environment_cpu.yml && conda activate objaxdp_cpu && echo $(which pip) && pip install --upgrade jax[cpu]==0.3.7 jaxlib==0.3.7 -f https://storage.googleapis.com/jax-releases/jax_releases.html && pip install objax
# --upgrade jax==0.3 jaxlib==0.3+cuda11.cudnn805 -f https://storage.googleapis.com/jax-releases/jax_releases.html
# && pip install --upgrade "jax[cuda11_cudnn81]"