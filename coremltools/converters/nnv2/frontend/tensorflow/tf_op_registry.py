_OPS_REGISTRY = {}

def register_tf_op(_func=None, tf_alias=None):
    def func_wrapper(func):
        f_name = func.__name__
        if f_name in _OPS_REGISTRY:
            raise ValueError('TF Op {} already registered.'.format(f_name))
        _OPS_REGISTRY[f_name] = func
        if tf_alias is not None:
            for name in tf_alias:
                if name in _OPS_REGISTRY:
                    msg = 'TF Op alias {} already registered.'
                    raise ValueError(msg.format(name))
                _OPS_REGISTRY[name] = func
        return func

    if _func is None:
        # decorator called without argument
        return func_wrapper
    return func_wrapper(_func)

