from .builder import CoremlBuilder
import logging

# op_type (str) --> (Python class, doc_str)
SSA_OPS_REGISTRY = {}


def register_op(doc_str):
    def class_wrapper(op_cls):
        op_type = op_cls.__name__
        # op_cls.__doc__ = doc_str  # TODO: rdar://58622145
        logging.debug("Registering op {}".format(op_type))
        SSA_OPS_REGISTRY[op_type] = (op_cls, doc_str)

        @classmethod
        def add_op(cls, **kwargs):
            return cls._add_op(op_cls, **kwargs)

        setattr(CoremlBuilder, op_type, add_op)
        return op_cls

    return class_wrapper
