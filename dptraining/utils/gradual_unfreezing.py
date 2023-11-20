from bisect import bisect_right


from objax import VarCollection, Module

from dptraining.config.model import ModelName

# RESNET9_EXAMPLE = [
#     "(ResNet9).conv1(Sequential)[0](Conv2D).b",
#     "(ResNet9).conv1(Sequential)[0](Conv2D).w",
#     "(ResNet9).conv1(Sequential)[1](Sequential)[0](Conv2D).w",
#     "(ResNet9).conv1(Sequential)[1](Sequential)[1](GroupNorm2D).gamma",
#     "(ResNet9).conv1(Sequential)[1](Sequential)[1](GroupNorm2D).beta",
#     "(ResNet9).conv2(Sequential)[0](Conv2D).w",
#     "(ResNet9).conv2(Sequential)[1](GroupNorm2D).gamma",
#     "(ResNet9).conv2(Sequential)[1](GroupNorm2D).beta",
#     "(ResNet9).res1(Sequential)[0](Sequential)[0](Conv2D).w",
#     "(ResNet9).res1(Sequential)[0](Sequential)[1](GroupNorm2D).gamma",
#     "(ResNet9).res1(Sequential)[0](Sequential)[1](GroupNorm2D).beta",
#     "(ResNet9).res1(Sequential)[1](Sequential)[0](Conv2D).w",
#     "(ResNet9).res1(Sequential)[1](Sequential)[1](GroupNorm2D).gamma",
#     "(ResNet9).res1(Sequential)[1](Sequential)[1](GroupNorm2D).beta",
#     "(ResNet9).conv3(Sequential)[0](Conv2D).w",
#     "(ResNet9).conv3(Sequential)[1](GroupNorm2D).gamma",
#     "(ResNet9).conv3(Sequential)[1](GroupNorm2D).beta",
#     "(ResNet9).conv4(Sequential)[0](Conv2D).w",
#     "(ResNet9).conv4(Sequential)[1](GroupNorm2D).gamma",
#     "(ResNet9).conv4(Sequential)[1](GroupNorm2D).beta",
#     "(ResNet9).res2(Sequential)[0](Sequential)[0](Conv2D).w",
#     "(ResNet9).res2(Sequential)[0](Sequential)[1](GroupNorm2D).gamma",
#     "(ResNet9).res2(Sequential)[0](Sequential)[1](GroupNorm2D).beta",
#     "(ResNet9).res2(Sequential)[1](Sequential)[0](Conv2D).w",
#     "(ResNet9).res2(Sequential)[1](Sequential)[1](GroupNorm2D).gamma",
#     "(ResNet9).res2(Sequential)[1](Sequential)[1](GroupNorm2D).beta",
#     "(ResNet9).classifier(Linear).b",
#     "(ResNet9).classifier(Linear).w",
# ]


class UnfreezingScheduler:
    def __init__(
        self,
        total_model_vars: VarCollection,
        must_train_vars: VarCollection,
        epoch_triggers: list[int],
        unfreeze_keys: list[list[str]],
    ) -> None:
        """
        Scheduler to gradually unfreeze model.

        Args:
            total_model_vars (VarCollection): All model vars that shall be considered for training
            epoch_triggers (list[int]): triggerpoints in which next unfreeze step
                                shall be executed.
            unfreeze_keys (list[list[str]]): Must contain for each triggerpoint which
                                             parts of the model shall be trained.
                                             One more element than epoch_triggers, as the
                                             first element is executed from the start.
        """
        assert len(epoch_triggers) + 1 == len(
            unfreeze_keys
        ), "Trigger points must be one less than unfreeze keys"
        assert all(
            epoch_triggers[i] <= epoch_triggers[i + 1]
            for i in range(len(epoch_triggers) - 1)
        ), "trigger points must be in chronological order"
        assert all(
            (
                all((k in total_model_vars for k in trigger_keys))
                for trigger_keys in unfreeze_keys
            )
        )
        self.total_model_vars: VarCollection = total_model_vars
        self.must_train_vars: VarCollection = must_train_vars
        self.epoch_triggers: list[int] = epoch_triggers
        self.unfreeze_keys: list[list[str]] = unfreeze_keys

    def __call__(self, epoch: int) -> VarCollection:
        i = bisect_right(self.epoch_triggers, epoch)
        keys = self.unfreeze_keys[i]
        model_vars = VarCollection(
            **{k: v for k, v in self.total_model_vars.items() if k in keys},
            **{k: v for k, v in self.must_train_vars.items() if k not in keys},
        )
        return model_vars, self.epoch_triggers[i - 1] == epoch


def make_unfreezing_schedule(
    model_name: ModelName,
    epoch_triggers: list[int],
    model_vars: VarCollection,
    must_train_vars: VarCollection,
    model: Module,
) -> UnfreezingScheduler:
    unfreeze_keys: list[list[str]] = []
    all_keys = list(model_vars.keys())
    match model_name:
        case ModelName.resnet9:
            match len(epoch_triggers):
                case 2:
                    search_terms = (
                        (".classifier(",),
                        (".classifier(", ".res2(", ".conv4(", ".res3(", ".conv3("),
                        all_keys,
                    )
                case 3:
                    search_terms = (
                        (".classifier(",),
                        (
                            ".classifier(",
                            ".res2(",
                            ".conv4(",
                        ),
                        (".classifier(", ".res2(", ".conv4(", ".res3(", ".conv3("),
                        all_keys,
                    )
                case other:
                    raise ValueError(
                        f"No Scheduler for {model_name} with {other} trigger points yet defined"
                    )
            unfreeze_keys = [
                [k for k in all_keys if any((term in k for term in st))]
                for st in search_terms
            ]
        case ModelName.wide_resnet:
            assert len(epoch_triggers) == 3
            N = len(model)
            blocks_per_group = (len(model) - 4) // 6
            unfreeze_keys = [
                [
                    key
                    for key in model_vars.keys()
                    for i in range(N - 4, N)
                    if f"(WideResNet)[{i}]" in key
                ],
                [
                    key
                    for key in model_vars.keys()
                    for i in [0, *range(N - (4 + blocks_per_group), N)]
                    if f"(WideResNet)[{i}]" in key
                ],
                [
                    key
                    for key in model_vars.keys()
                    for i in [0, *range(N - (4 + 2 * blocks_per_group), N)]
                    if f"(WideResNet)[{i}]" in key
                ],
                [key for key in model_vars.keys()],
            ]
        case other:
            raise ValueError(f"No Scheduler for {other} yet defined.")
    return UnfreezingScheduler(
        model_vars, must_train_vars, epoch_triggers, unfreeze_keys
    )
