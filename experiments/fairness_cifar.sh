!/bin/bash

seeds="42 23 32 12 99"
for i in $seeds; do
    echo "NO BAM seed: $i"
    python dptraining/train.py -cn cifar100_dp.yaml project='cifar100-DP-fair' DP.bam=False general.seed=$i dataset.undersample_class=True dataset.undersample_factor=0.2
done

for i in $seeds; do
    echo "BAM seed: $i"
    python dptraining/train.py -cn cifar100_dp.yaml project='cifar100-DP-fair' DP.bam=True DP.r=0.05 DP.alpha=0.1 general.seed=$i dataset.undersample_class=True dataset.undersample_factor=0.2
done