from typing import Iterable
import hydra
import wandb
import time

import numpy as np

import objax
import sys

from pathlib import Path
from sklearn import metrics
from tqdm import tqdm

sys.path.insert(0, str(Path.cwd()))

from dptraining.datasets import make_loader_from_config
from dptraining.models import make_model_from_config
from dptraining.utils import make_loss_from_config, make_scheduler_from_config
from dptraining.utils.augment import Transformation
from dptraining.privacy import EpsCalculator


def create_train_op(model_vars, loss_gv, opt, augment_op):
    @objax.Function.with_vars(model_vars + loss_gv.vars() + opt.vars())
    def train_op(x, y, learning_rate):  # pylint:disable=invalid-name
        x = augment_op(x)
        grads, loss = loss_gv(x, y)
        opt(learning_rate, grads)
        return loss

    train_op = objax.Jit(train_op)
    return train_op


def create_loss_gradient(config, model, model_vars, loss_fn, sigma):
    if config["DP"]["disable_dp"]:
        loss_gv = objax.GradValues(loss_fn, model.vars())
    else:
        loss_gv = objax.privacy.dpsgd.PrivateGradValues(
            loss_fn,
            model_vars,
            sigma,
            config["DP"]["max_per_sample_grad_norm"],
            microbatch=1,
            batch_axis=(0, 0),
            use_norm_accumulation=config["DP"]["norm_acc"],
        )

    return loss_gv


def train(config, train_loader, train_op, learning_rate):
    start_time = time.time()
    max_batches = (
        config["hyperparams"]["overfit"]
        if "overfit" in config["hyperparams"]
        else len(train_loader) + 1
    )
    for i, (x, y) in tqdm(  # pylint:disable=invalid-name
        enumerate(train_loader),  # pylint:disable=loop-invariant-statement
        total=len(train_loader),
        desc="Training",
        leave=False,
    ):
        train_loss = train_op(x, y, learning_rate)
        if config["log_wandb"]:  # pylint:disable=loop-invariant-statement
            wandb.log({"train_loss": train_loss[0].item()}, commit=False)
        if i > max_batches:
            break
    return time.time() - start_time


def test(config, test_loader, predict_op):
    correct, predicted = [], []
    max_batches = (
        config["hyperparams"]["overfit"]
        if "overfit" in config["hyperparams"]
        else len(test_loader) + 1
    )
    for i, (x, y) in tqdm(  # pylint:disable=invalid-name
        enumerate(test_loader),  # pylint:disable=loop-invariant-statement
        total=len(test_loader),
        desc="Testing",
        leave=False,
    ):
        y_pred = predict_op(x)
        # num_correct += np.count_nonzero(np.argmax(y_pred, axis=1) == y)
        correct.append(y)
        predicted.append(y_pred)
        if i > max_batches:
            break
    correct = np.concatenate(correct)
    predicted = np.concatenate(predicted)

    if config["log_wandb"]:
        wandb.log(
            {
                "val": metrics.classification_report(
                    correct, predicted.argmax(axis=1), output_dict=True
                )
            }
        )
    else:
        print(
            metrics.classification_report(
                correct, predicted.argmax(axis=1), output_dict=False, zero_division=0
            )
        )


@hydra.main(
    version_base=None,
    config_path=Path.cwd() / "configs",
)
def main(config):  # pylint:disable=too-many-locals
    if config["log_wandb"]:
        wandb.init(
            project=config["project"],
            config=config,
        )
    train_loader, test_loader = make_loader_from_config(config)
    model = make_model_from_config(config)
    model_vars = model.vars()

    opt = objax.optimizer.Momentum(
        model_vars, momentum=config["hyperparams"]["momentum"], nesterov=False
    )

    predict_op = objax.Jit(
        lambda x: objax.functional.softmax(model(x, training=False)), model_vars
    )

    sampling_rate: float = config["hyperparams"]["batch_size"] / len(
        train_loader.dataset
    )
    delta = config["DP"]["delta"]
    eps_calc = EpsCalculator(config, train_loader)
    sigma = eps_calc.calc_noise_for_eps()
    final_epsilon = objax.privacy.dpsgd.analyze_dp(
        q=sampling_rate,
        noise_multiplier=sigma,
        steps=len(train_loader) * config["hyperparams"]["epochs"],
        delta=delta,
    )

    print(
        f"This training will lead to a final epsilon of {final_epsilon:.2f}"
        f" at a noise multiplier of {sigma:.2f} and a delta of {delta:2f}"
    )

    loss_class = make_loss_from_config(config)
    loss_fn = loss_class.create_loss_fn(model_vars, model)
    loss_gv = create_loss_gradient(config, model, model_vars, loss_fn, sigma)

    augmenter = Transformation.from_dict_list(config["augmentations"])
    augment_op = augmenter.create_vectorized_transform()
    scheduler = make_scheduler_from_config(config)

    train_op = create_train_op(model_vars, loss_gv, opt, augment_op)

    epoch_time = []
    epoch_iter: Iterable
    if config["log_wandb"]:
        epoch_iter = tqdm(
            range(config["hyperparams"]["epochs"]),
            total=config["hyperparams"]["epochs"],
            desc="Epoch",
            leave=True,
        )
    else:
        epoch_iter = range(config["hyperparams"]["epochs"])
    for epoch, learning_rate in zip(epoch_iter, scheduler):
        cur_epoch_time = train(config, train_loader, train_op, learning_rate)
        if config["log_wandb"]:  # pylint:disable=loop-invariant-statement
            wandb.log({"epoch": epoch, "lr": learning_rate})
        else:
            print(f"Train Epoch: {epoch+1} \t took {cur_epoch_time} seconds")
        epoch_time.append(cur_epoch_time)
        test(config, test_loader, predict_op)
        if not config["DP"]["disable_dp"]:
            epsilon = objax.privacy.dpsgd.analyze_dp(
                q=sampling_rate,
                noise_multiplier=sigma,
                steps=len(train_loader)  # pylint:disable=loop-invariant-statement
                * (epoch + 1),  # pylint:disable=loop-invariant-statement
                delta=delta,
            )  # pylint:disable=loop-invariant-statement
            if config["log_wandb"]:  # pylint:disable=loop-invariant-statement
                wandb.log(
                    {"epsilon": epsilon}
                )  # pylint:disable=loop-invariant-statement
            else:
                print(
                    f"\tPrivacy: (ε = {epsilon:.2f}, δ = {delta})"  # pylint:disable=loop-invariant-statement
                )

    if not config["log_wandb"]:
        print("Average epoch time (all epochs): ", np.average(epoch_time))
        print("Median epoch time (all epochs): ", np.median(epoch_time))
        if len(epoch_time) > 1:
            print("Average epoch time (except first): ", np.average(epoch_time[1:]))
            print("Median epoch time (except first): ", np.median(epoch_time[1:]))
        print("Total training time (excluding evaluation): ", np.sum(epoch_time))


if __name__ == "__main__":
    main()  # pylint:disable=no-value-for-parameter
