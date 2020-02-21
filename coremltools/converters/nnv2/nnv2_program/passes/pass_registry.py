import logging

# str -> func (func takes SsaProgram as input)
PASS_REGISTRY = {}


def register_pass(pass_func):
    func_name = pass_func.__name__
    logging.debug("Registering pass: {}".format(func_name))
    if func_name in PASS_REGISTRY:
        raise ValueError("Pass {} already registered.".format(func_name))

    PASS_REGISTRY[func_name] = pass_func
    return pass_func
