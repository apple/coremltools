_OPS_REGISTRY = {}


def register_tf_op(_func=None, tf_alias=None):
    def func_wrapper(func):
        _OPS_REGISTRY[func.__name__] = func
        if tf_alias is not None:
            for name in tf_alias:
                _OPS_REGISTRY[name] = func
        return func

    if _func is None:
        # decorator called without argument
        return func_wrapper
    return func_wrapper(_func)



