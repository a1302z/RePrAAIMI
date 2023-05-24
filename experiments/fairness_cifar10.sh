!/bin/bash

seeds="42 23 32 12 99"
for i in $seeds; do
    echo "NO BAM seed: $i"
    python dptraining/train.py -cn cifar10_dp.yaml project='cifar10-DP-fair' DP.bam=False general.seed=$i
done

for i in $seeds; do
    echo "BAM seed: $i"
    python dptraining/train.py -cn cifar10_dp.yaml project='cifar10-DP-fair' DP.bam=True DP.r=0.05 DP.alpha=0.1 general.seed=$i
done