GPU=0
NAME="train"
TMUX_NAME=$NAME$(date +'%M-%S')
FILE_NAME=$NAME$(date +'%Y-%m-%d_%H-%M-%S')
ENV_NAME="objaxdp"
COMMAND="python dptraining/train.py -cn radimagenet_dp.yaml general.parallel=False general.log_wandb=True wandb.notes=''"
tmux new -d -s $TMUX_NAME
tmux send-keys -t $TMUX_NAME "conda activate $ENV_NAME" ENTER "CUDA_VISIBLE_DEVICES=$GPU $COMMAND > out/$FILE_NAME.out 2>&1" ENTER "exit" ENTER