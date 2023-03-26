# # A combination of Recent Optimization-based Attacks

# This set of hyperparameters is a variation of the original Inverting Gradients hyperparameter set, including a few changes such as a small deep inversion prior (as in "See Through Gradients"), cosine decay of the step sizes and warmup (also "See Through Gradients"), feature regularization for the last linear layer in the style of some analytic inversion papers, and the structured initialization scheme of "A Framework for Evaluating Gradient Leakage Attacks in Federated Learning".
#
# Minor changes also include a sane color total variation regularization (via double-oppponents TV) and a "soft" signed gradient descent schedule.
#
# Paper URLs:
# * https://proceedings.neurips.cc/paper/2020/hash/c4ede56bbd98819ae6112b20ac6bf145-Abstract.html
# * https://openaccess.thecvf.com/content/CVPR2021/html/Yin_See_Through_Gradients_Image_Batch_Recovery_via_GradInversion_CVPR_2021_paper.html
# * https://arxiv.org/abs/2004.10397
# * https://link.springer.com/chapter/10.1007/978-3-319-46475-6_40


import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

import logging, sys

logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
    format="%(message)s",
)
logger = logging.getLogger()

from breaching import breaching

from breachingobjaxutils.initialize import make_configs, init_wandb, make_make_fns


def main():
    cfg, train_config = make_configs(
        ["attack=modern_simplified"], "breaching/test_train_config.yaml"
    )

    cfg.case.user.user_idx = 24
    # cfg.case.model = "resnet18"
    cfg.case.data.path = "/media/alex/NVME/ILSVRC2012/"
    device = torch.device(f"cuda") if torch.cuda.is_available() else torch.device("cpu")
    # torch.backends.cudnn.benchmark = cfg.case.impl.benchmark
    setup = dict(device=device, dtype=getattr(torch, cfg.case.impl.dtype))
    cfg.case.data.partition = "unique-class"

    run = init_wandb(cfg, train_config)

    make_model, make_loss_gv_from_model = make_make_fns(train_config)

    # cfg.attack.regularization.deep_inversion.scale = 1e-4

    # train_data, _, _ = make_loader_from_config(train_config)

    if train_config.DP or train_config.hyperparams.grad_acc_steps > 1:
        raise ValueError("Not yet supported")

    user, server, model = breaching.cases.construct_case(
        train_config,
        cfg.case,
        make_model,
        # dataloader=train_data,
        make_loss_grad=make_loss_gv_from_model,
    )
    attacker = breaching.attacks.prepare_attack(
        make_model, make_loss_gv_from_model, cfg.attack
    )
    breaching.utils.overview(server, user, attacker)

    # %% [markdown]
    # ### Simulate an attacked FL protocol

    # %% [markdown]
    # This exchange is a simulation of a single query in a federated learning protocol. The server sends out a `server_payload` and the user computes an update based on their private local data. This user update is `shared_data` and contains, for example, the parameter gradient of the model in the simplest case. `true_user_data` is also returned by `.compute_local_updates`, but of course not forwarded to the server or attacker and only used for (our) analysis.

    # %%
    server_payload = server.distribute_payload()
    shared_data, true_user_data = user.compute_local_updates(server_payload)

    # %%
    user.plot(true_user_data, print_labels=True)

    # %% [markdown]
    # ### Reconstruct user data:

    # %% [markdown]
    # Now we launch the attack, reconstructing user data based on only the `server_payload` and the `shared_data`.
    #
    # You can interrupt the computation early to see a partial solution.

    # %%
    reconstructed_user_data, stats = attacker.reconstruct(
        [server_payload], [shared_data], {}, dryrun=cfg.dryrun
    )

    # %% [markdown]
    # Next we'll evaluate metrics, comparing the `reconstructed_user_data` to the `true_user_data`.

    # %%
    metrics = breaching.analysis.report(
        reconstructed_user_data,
        true_user_data,
        [server_payload],
        # server.model,
        model_template=None,
        order_batch=True,
        compute_full_iip=False,
        cfg_case=cfg.case,
        setup=setup,
    )

    # %% [markdown]
    # And finally, we also plot the reconstructed data:

    # %%
    user.plot(reconstructed_user_data)

    # %% [markdown]
    # Notes:
    #    * Arguably a very good looking owl, even if the metrics do not reflect this well.
    #    * The color pattern in the background arises from the deep inversion prior.

    if run:
        run.finish()


if __name__ == "__main__":
    main()
