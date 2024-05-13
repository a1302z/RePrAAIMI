# Reconciling Privacy And Accuracy in Medical Imaging
What you can use this repository for: 
- Large-scale AI trainings under DP conditions on 2D and 3D imaging data
- Calculate theoretical risk profiles for settings under various privacy budgets
- Evaluate empirical reconstruction success for trainings with DP




## Getting Started
Prerequisites: Installation of [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).
### System requirements
Tested on following systems:
| Operating System | CPU | GPU | CUDA |
| --- | --- | --- | --- |
| Ubuntu 20.04 | Intel(R) Xeon(R) W-2245 CPU @ 3.90GHz | NVIDIA Quadro RTX 8000 | 11.2 |
| Ubuntu 20.04 | AMD Ryzen 9 5950X 16-Core Processor | NVIDIA GeForce RTX 3090 | 12.2 |
| Ubuntu 20.04 | AMD Ryzen Threadripper PRO 3995WX 64-Cores | NVIDIA Quadro RTX 8000 | 12.0 |

We recommend a similar or superior setup. 
Dependency versions are specified in [environment.yml](environment.yml) and the [environment setup file](setup_env.sh).
### Installation guide
Install environment via `bash setup_env.sh`. We note that installing jax based libraries has its intricacies and GPU support may require some finetuning to the specific system setup. One alternative is listed in the [Troubleshooting](#troubleshooting) section. Installation typically takes about 5-15 minutes. For a speedup of this process we recommend the use of the [libmamba solver](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community).
### Demo

#### Trainings
We provide many configs and examples in the config directory. Training can be started via:
```python dptraining/train.py -cn <name_of_config>.yaml```

Exemplarily a privacy-preserving training on CIFAR10 can be started via
```python dptraining/train.py -cn cifar10_dp.yaml``` This will automatically download the CIFAR10 dataset and train for 10 epochs taking about 5-10 minutes leading to an accuracy score of 68% on the test set. 

To reproduce our main results in the paper please first download the datasets from the respective sources and put them into the data directory. Afterwards, use the following configs: 
| Dataset | Non-private config | DP config |
| --- | --- | --- |
| RadImageNet | [radimagenet.yaml](configs/radimagenet.yaml) | [radimagenet_dp.yaml](configs/radimagenet_dp.yaml) |
| HAM10000 | [ham10000.yaml](configs/ham10000.yaml) | [ham10000_dp.yaml](configs/ham10000_dp.yaml) |
| MSD Liver | [msd_liver.yaml](configs/msd_liver.yaml) | [msd_liver_dp.yaml](configs/msd_liver_dp.yaml) |

All necessary hyperparameters are furthermore given in the methods section. For our experiments we used 10, 11, 100, 101, 110 as random seeds. 

#### Realistic Risk Assessment
Prerequisite: Clone [our modified version of breaching](https://github.com/a1302z/objaxbreaching) into this repository. 
1. Create reconstructions of a specific setup via ```python dptraining/vulnerability/create_reconstructions.py -cn configs/<name_of_config>.yaml -eb <minimal exponent of 10 for epsilon e.g. when input is 6 epsilon = 10^6> -ee <maximal exponent of 10 for epsilon e.g. 10^18> -es <step size between exponents>```
2. Match the reconstructions to the original data `python dptraining/vulnerability/match_reconstructions.py --recon_folder <path_to_reconstructions> --use_<error_metric>`
3. Visualize results `python dptraining/vulnerability/visualize_reconstructions.py --recon_csv <path_to_csv_file_created_by_previous_step> --metric_to_use <metric_from_previous_step>`

#### Theoretical Worst Case Risk Assessment
Run ```python dptraining/vulnerability/rero_bounds.py -cn <name_of_config>.yaml +eps_min=<minimal epsilon value> +eps_max=<maximal epsilon value> +N_eps=<number of samples>```


### Troubleshooting
```jaxlib.xla_extension.XlaRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 47579372416 bytes``` Please try again with a smaller batch size or a GPU with more VRAM.

```jaxlib.xla_extension.XlaRuntimeError: UNKNOWN: no kernel image is available for execution on the device in external/org_tensorflow/tensorflow/stream_executor/cuda/cuda_asm_compiler.cc(60): 'status'``` is [a commonly known bug of jax](https://github.com/google/jax/issues/5723) related to the cuda version of your device. A workaround is to set the environment flag ```XLA_FLAGS=--xla_gpu_force_compilation_parallelism=1```, which unfortunately slows down the compilation but often prevents this error. 

For debugging or other non-time critical workflows jax errors can typically be circumvented by deactivating GPU support via ```CUDA_VISIBLE_DEVICES=""```

Alternative environment installation: Some setups (for example [our second setup](#system-requirements)) require a different installation process. If installing via the bash script as is does not provide a GPU compatible installation you may try to exchange line 3 in [the bash file](setup_env.sh) with the following:
```
conda env create -f environment.yml && conda activate reconciling && conda install -c "nvidia/label/cuda-11.4.0" cuda -y && conda install -c "nvidia/label/cuda-11.4.0" cuda-nvcc -c conda-forge -c nvidia -y && nvcc --version && pip install --upgrade jax[cuda11_pip]==0.3.15 jaxlib==0.3.15+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && pip install objax==1.6.0
```

<!-- ## Contribute
Feel free to open Pull Requests or Issues. Please try to write code as configurable as possible and formatted by the black formatter. 


## Pretrained models
We provide several pretrained models, which can be downloaded via this [link](https://syncandshare.lrz.de/getlink/fiTqfRPfJK9iTbHDWLyny3/). -->

