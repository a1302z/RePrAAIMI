import hydra
from omegaconf import DictConfig
import time

import numpy as np

import objax

from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd()))

from dptraining.datasets import make_loader_from_config
from dptraining.models import make_model_from_config
from dptraining.utils import make_loss_from_config, make_scheduler_from_config
from dptraining.utils.augment import Augmentation


def create_train_op(model_vars, loss_gv, opt, augment_op):
    @objax.Function.with_vars(model_vars + loss_gv.vars() + opt.vars())
    def train_op(x, y, learning_rate):
        x = augment_op(x)
        grads, loss = loss_gv(x, y)
        opt(learning_rate, grads)
        return loss

    train_op = objax.Jit(train_op)
    return train_op


def create_loss_gradient(config, model, model_vars, loss_fn):
    if config["DP"]["disable_dp"]:
        loss_gv = objax.GradValues(loss_fn, model.vars())
    else:
        loss_gv = objax.privacy.dpsgd.PrivateGradValues(
            loss_fn,
            model_vars,
            config["DP"]["sigma"],
            config["DP"]["max_per_sample_grad_norm"],
            microbatch=1,
            batch_axis=(0, 0),
            use_norm_accumulation=config["DP"]["norm_acc"],
        )

    return loss_gv


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


@hydra.main(
    version_base=None,
    config_path=Path.cwd() / "configs",
    config_name="base_config.yaml",
)
def main(config: DictConfig):
    print(config)
    train_loader, test_loader = make_loader_from_config(config)
    model = make_model_from_config(config)
    model_vars = model.vars()

    opt = objax.optimizer.Momentum(
        model_vars, momentum=config["hyperparams"]["momentum"], nesterov=False
    )

    predict_op = objax.Jit(lambda x: objax.functional.softmax(model(x)), model_vars)

    loss_class = make_loss_from_config(config)
    loss_fn = loss_class.create_loss_fn(model_vars, model)
    loss_gv = create_loss_gradient(config, model, model_vars, loss_fn)

    augmenter = Augmentation.from_string_list(
        ["random_horizontal_flips", "random_vertical_flips", "random_img_shift"]
    )
    augment_op = augmenter.create_augmentation_op()
    scheduler = make_scheduler_from_config(config)

    train_op = create_train_op(model_vars, loss_gv, opt, augment_op)

    epoch_time = []
    for epoch in range(config["hyperparams"]["epochs"]):
        lr = next(scheduler)
        cur_epoch_time = train(train_loader, train_op, lr)
        print(f"Train Epoch: {epoch+1} \t took {cur_epoch_time} seconds")
        epoch_time.append(cur_epoch_time)
        test(test_loader, predict_op)
        if not config["DP"]["disable_dp"]:
            epsilon = objax.privacy.dpsgd.analyze_dp(
                q=config["hyperparams"]["batch_size"] / len(train_loader.dataset),
                noise_multiplier=config["DP"]["sigma"],
                steps=len(train_loader) * (epoch + 1),
                delta=config["DP"]["delta"],
            )
            print(f"\tPrivacy: (ε = {epsilon:.2f}, δ = {config['DP']['delta']})")

    print("Average epoch time (all epochs): ", np.average(epoch_time))
    print("Median epoch time (all epochs): ", np.median(epoch_time))
    if len(epoch_time) > 1:
        print("Average epoch time (except first): ", np.average(epoch_time[1:]))
        print("Median epoch time (except first): ", np.median(epoch_time[1:]))
    print("Total training time (excluding evaluation): ", np.sum(epoch_time))


if __name__ == "__main__":
    main()
