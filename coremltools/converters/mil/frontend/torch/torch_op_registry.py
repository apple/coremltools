_TORCH_OPS_REGISTRY = {}


def register_torch_op(_func=None, torch_alias=None, override=False):
    """
    Registration routine for PyTorch operators
    _func: (PyTorch conversion function) [Default=None]
        PyTorch conversion function to register

    torch_alias: (List of string) [Default=None]
        All other PyTorch operators that should also be mapped to
        current conversion routine.
        e.g. Sort aliased with SortV1, SortV2
        All provided alias operators must not be registered previously.

    override: (Boolean) [Default=False]
        If True, overrides earlier registration i.e. specified
        operator and alias will start pointing to current conversion
        function.
        Otherwise, duplicate registration will error out.
    """
    def func_wrapper(func):
        f_name = func.__name__
        if not override and f_name in _TORCH_OPS_REGISTRY:
            raise ValueError("Torch Op {} already registered.".format(f_name))
        _TORCH_OPS_REGISTRY[f_name] = func
        if torch_alias is not None:
            for name in torch_alias:
                if not override and name in _TORCH_OPS_REGISTRY:
                    msg = "Torch Op alias {} already registered."
                    raise ValueError(msg.format(name))
                _TORCH_OPS_REGISTRY[name] = func
        return func

    if _func is None:
        # decorator called without argument
        return func_wrapper
    return func_wrapper(_func)
