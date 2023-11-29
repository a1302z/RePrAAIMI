#!/bin/bash
eval "$(conda shell.bash hook)"
conda env create -f environment.yml && conda activate reconciling && CONDA_OVERRIDE_CUDA="11.4" conda install jaxlib=*=*cuda* jax cuda-nvcc -c conda-forge -c nvidia -y && pip install objax==1.6.0 && python -c "import torch, jax; print(f'Cuda available: {torch.cuda.is_available()}'); print(f'jax devices: {jax.devices()}');" 
