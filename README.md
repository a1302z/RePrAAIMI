# ObjaxDPTraining

Code Base for Training of Neural Networks with Objax and DP-SGD.

## Setup
Prerequisites: 
 - installation of conda
 - cuda support

We provide an installation helper file which can be used as `bash setup_env.sh`. However, we note that installing jax based libraries has its intricacies an may require some finetuning to the specific system. 

## Use
### Training of Models
We provide many configs and examples in the config directory. Training can be started via:
```python dptraining/train.py -cn <name_of_config.yaml```

### Risk Assessment
1. Clone [this repository](https://github.com/a1302z/objaxbreaching) into the base level of this repository
2. Create reconstructions of a specific setup via ```python dptraining/vulnerability/create_reconstructions.py -cn <name_of_config.yaml>```
3. Match the reconstructions to the original data `python dptraining/vulnerability/match_reconstructions.py --recon_folder <path_to_reconstructions> --use_<error_metric>`
4. Visualize results `python dptraining/vulnerability/visualize_reconstructions.py --recon_csv <path_to_csv_file_created_by_previous_step>`

## Contribute
Feel free to open Pull Requests or Issues. Please try to write code as configurable as possible and formatted by the black formatter. 


## Pretrained models
We provide several pretrained models, which can be downloaded via this [link](https://syncandshare.lrz.de/getlink/fiTqfRPfJK9iTbHDWLyny3/).
