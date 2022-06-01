# ObjaxDPTraining

Code Base for Training of Neural Networks with Objax and DP-SGD.

## Setup
Prerequisites: 
 - installation of conda
 - cuda support

Environment can then be installed via `bash setup_env.sh` and activated with `conda activate objaxdp`

## Contribute
Please execute ```cp check_before_commit.sh .git/hooks/pre-commit && chmod u+x .git/hooks/pre-commit``` to ensure all commits are tested and adhere to pylint guidelines. To manually check use `bash check_before_commit.sh`.