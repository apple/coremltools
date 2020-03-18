_TORCH_OPS_REGISTRY = {}


def register_torch_op(_func=None, torch_alias=None):
    def func_wrapper(func):
        f_name = func.__name__
        if f_name in _TORCH_OPS_REGISTRY:
            raise ValueError("Torch Op {} already registered.".format(f_name))
        _TORCH_OPS_REGISTRY[f_name] = func
        if torch_alias is not None:
            for name in torch_alias:
                if name in _TORCH_OPS_REGISTRY:
                    msg = "Torch Op alias {} already registered."
                    raise ValueError(msg.format(name))
                _TORCH_OPS_REGISTRY[name] = func
        return func

    if _func is None:
        # decorator called without argument
        return func_wrapper
    return func_wrapper(_func)
