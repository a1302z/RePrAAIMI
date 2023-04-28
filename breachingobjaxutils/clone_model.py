import objax

from copy import deepcopy
from inspect import getfullargspec


# This is an awkward attempt in faking deepcopy for objax modules
def copy_model(model):
    class_type = type(model)
    signature = getfullargspec(class_type)
    if len(signature.args) == len(signature.defaults) + 1:
        new_model = class_type(*signature.defaults)
    else:
        # now we have to make happy guesses
        args = {}
        N_args = len(signature.args)
        defaults = len(signature.defaults)
        for i in range(defaults):
            args[signature.args[-i - 1]] = signature.defaults[-i - 1]
        try:
            for i in range(1, N_args - defaults):
                arg_name = signature.args[i]
                if hasattr(model, arg_name):
                    args[arg_name] = deepcopy(getattr(model, arg_name))
                else:
                    args[arg_name] = signature.annotations[arg_name]()
            new_model = class_type(**args)
        except Exception as e:
            raise ValueError(f"We cannot clone this model unfortunately. \n{e}")

    for var_name in dir(model):
        if var_name[:2] == "__" and var_name[-2:] == "__":
            continue
        old_value = getattr(model, var_name)
        new_value = None
        new_value = create_copy(old_value)
        if new_value:
            setattr(new_model, var_name, new_value)

    return new_model


def create_copy(old_value):
    if callable(old_value):
        if isinstance(old_value, objax.Module):
            new_value = copy_model(old_value)
    elif isinstance(old_value, list):
        new_value = type(old_value)(*[create_copy(ov) for ov in old_value])
    elif isinstance(old_value, objax.typing.JaxArray):
        new_value = old_value.copy()
    else:
        new_value = deepcopy(old_value)
    return new_value


if __name__ == "__main__":
    from objax.zoo import resnet_v2, wide_resnet

    m = resnet_v2.ResNet18(3, 1000)

    m2 = copy_model(m)

    print(m)
    print(m2)

    print(m.vars())
    print(m2.vars())
