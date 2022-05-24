from argparse import Namespace
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from dptraining.train import main


def test_train_cifar_one_epoch():
    args = Namespace(
        epochs=1,
        batch_size=128,
        batch_size_test=1,
        disable_dp=False,
        lr=0.1,
        lr_schedule="cos",
        momentum=0.9,
        sigma=1.5,
        max_per_sample_grad_norm=10.0,
        delta=1e-5,
        norm_acc=False,
    )
    main(args)
