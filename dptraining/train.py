import argparse
import time

import jax
import jax.numpy as jn

import numpy as np

import objax

from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd()))

from dptraining.datasets import CIFAR10Creator
from dptraining.utils.loss import CSELogitsSparse
from dptraining.models.cifar10models import Cifar10ConvNet


def train(train_loader, train_op, lr):
    start_time = time.time()
    for x, y in train_loader:
        train_op(x, y, lr)
    return time.time() - start_time


def test(test_loader, predict_op):
    num_correct = 0
    for x, y in test_loader:
        y_pred = predict_op(x)
        num_correct += np.count_nonzero(np.argmax(y_pred, axis=1) == y)
    print(f"\tTest set:\tAccuracy: {num_correct/len(test_loader.dataset)}")


def main(args):
    train_ds, test_ds = CIFAR10Creator.make_datasets(
        (),
        {"root": "./data", "download": True},
        (),
        {"root": "./data", "download": True},
    )
    train_loader, test_loader = CIFAR10Creator.make_dataloader(
        train_ds,
        test_ds,
        (),
        {"batch_size": args.batch_size, "shuffle": True},
        (),
        {"batch_size": args.batch_size_test, "shuffle": False},
    )

    # Model
    model = Cifar10ConvNet()
    model_vars = model.vars()

    # Optimizer
    opt = objax.optimizer.Momentum(model_vars, momentum=args.momentum, nesterov=False)

    # Prediction operation
    predict_op = objax.Jit(lambda x: objax.functional.softmax(model(x)), model_vars)

    # Loss and training op
    loss_fn = CSELogitsSparse.create_loss_fn(model_vars, model)

    if args.disable_dp:
        loss_gv = objax.GradValues(loss_fn, model.vars())
    else:
        loss_gv = objax.privacy.dpsgd.PrivateGradValues(
            loss_fn,
            model_vars,
            args.sigma,
            args.max_per_sample_grad_norm,
            microbatch=1,
            batch_axis=(0, 0),
            use_norm_accumulation=args.norm_acc,
        )

    @objax.Function.with_vars(objax.random.DEFAULT_GENERATOR.vars())
    def augment_op(x):
        # random flip
        x = jax.lax.cond(
            objax.random.uniform(()) < 0.5,
            lambda t: t,
            lambda t: t[:, :, ::-1],
            operand=x,
        )
        # random crop
        x_pad = jn.pad(x, [[0, 0], [4, 4], [4, 4]], "reflect")
        offset = objax.random.randint((2,), 0, 4)
        return jax.lax.dynamic_slice(x_pad, (0, offset[0], offset[1]), (3, 32, 32))

    augment_op = objax.Vectorize(augment_op)

    @objax.Function.with_vars(model_vars + loss_gv.vars() + opt.vars())
    def train_op(x, y, learning_rate):
        if args.disable_dp:
            x = augment_op(x)
        grads, loss = loss_gv(x, y)
        opt(learning_rate, grads)
        return loss

    train_op = objax.Jit(train_op)

    epoch_time = []
    for epoch in range(args.epochs):
        if args.lr_schedule == "cos":
            lr = args.lr * 0.5 * (1 + np.cos(np.pi * (epoch + 1) / (args.epochs + 1)))
        else:
            lr = args.lr
        cur_epoch_time = train(train_loader, train_op, lr)
        print(f"Train Epoch: {epoch+1} \t took {cur_epoch_time} seconds")
        epoch_time.append(cur_epoch_time)
        test(test_loader, predict_op)
        if not args.disable_dp:
            epsilon = objax.privacy.dpsgd.analyze_dp(
                q=args.batch_size / len(train_ds),
                noise_multiplier=args.sigma,
                steps=len(train_loader) * (epoch + 1),
                delta=args.delta,
            )
            print(f"\tPrivacy: (ε = {epsilon:.2f}, δ = {args.delta})")

    print("Average epoch time (all epochs): ", np.average(epoch_time))
    print("Median epoch time (all epochs): ", np.median(epoch_time))
    if len(epoch_time) > 1:
        print("Average epoch time (except first): ", np.average(epoch_time[1:]))
        print("Median epoch time (except first): ", np.median(epoch_time[1:]))
    print("Total training time (excluding evaluation): ", np.sum(epoch_time))


def parse_args():
    parser = argparse.ArgumentParser(description="Objax CIFAR10 DP Training")
    parser.add_argument(
        "--epochs",
        default=90,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--batch-size",
        default=2000,
        type=int,
        metavar="N",
        help="Batch size for training",
    )
    parser.add_argument(
        "--batch-size-test",
        default=200,
        type=int,
        metavar="N",
        help="Batch size for test",
    )
    parser.add_argument(
        "--disable-dp",
        action="store_true",
        default=False,
        help="Disable privacy training and just train with vanilla SGD",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.1,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--lr-schedule", type=str, choices=["constant", "cos"], default="cos"
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="SGD momentum"
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.5,
        metavar="S",
        help="Noise multiplier (default 1.5)",
    )
    parser.add_argument(
        "-c",
        "--max-per-sample-grad_norm",
        type=float,
        default=10.0,
        metavar="C",
        help="Clip per-sample gradients to this norm (default 10.0)",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        metavar="D",
        help="Target delta (default: 1e-5)",
    )
    parser.add_argument(
        "--norm-acc",
        action="store_true",
        default=False,
        help="Enables norm accumulation in Objax DP-SGD code.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
