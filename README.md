# Reconciling Privacy And Accuracy in Healthcare

Code Base for Training of AI models Under DP Guarantees + Evaluating Risk Profiles.

## Getting Started
Prerequisites: Installation of [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).
### System requirements
Tested on a System running under Ubuntu 20.04 using a Intel(R) Xeon(R) W-2245 CPU @ 3.90GHz CPU and an NVIDIA Quadro RTX 8000 GPU with driver version 460.27.04 and CUDA version 11.2. We recommend a similar or superior setup. \
Dependency versions are specified in [environment.yml](environment.yml) and the [environment setup file](setup_env.sh).
### Installation guide
Install environment via `bash setup_env.sh`. We note that installing jax based libraries has its intricacies and GPU support may require some finetuning to the specific system setup. Installation typically takes about 5-15 minutes. For a speedup of this process we recommend the use of the [libmamba solver](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community).
### Demo

We provide many configs and examples in the config directory. Training can be started via:
```python dptraining/train.py -cn <name_of_config.yaml```

Exemplarily a privacy-preserving training on CIFAR10 can be started via
```python dptraining/train.py -cn cifar10_dp.yaml``` This will automatically download the CIFAR10 dataset and train for 10 epochs. 

### Risk Assessment
1. Clone [this repository](https://github.com/a1302z/objaxbreaching) into the base level of this repository
2. Create reconstructions of a specific setup via ```python dptraining/vulnerability/create_reconstructions.py -cn <name_of_config.yaml>```
3. Match the reconstructions to the original data `python dptraining/vulnerability/match_reconstructions.py --recon_folder <path_to_reconstructions> --use_<error_metric>`
4. Visualize results `python dptraining/vulnerability/visualize_reconstructions.py --recon_csv <path_to_csv_file_created_by_previous_step>`

<!-- ## Contribute
Feel free to open Pull Requests or Issues. Please try to write code as configurable as possible and formatted by the black formatter. 


## Pretrained models
We provide several pretrained models, which can be downloaded via this [link](https://syncandshare.lrz.de/getlink/fiTqfRPfJK9iTbHDWLyny3/). -->

