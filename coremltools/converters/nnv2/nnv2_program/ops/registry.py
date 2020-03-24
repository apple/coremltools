import logging
from .builder import CoremlBuilder

class SSAOpRegistry:
    ops = {}

    @staticmethod
    def register_op(doc_str):
        def class_wrapper(op_cls):
            op_type = op_cls.__name__
            # op_cls.__doc__ = doc_str  # TODO: rdar://58622145
            logging.debug("Registering op {}".format(op_type))
            if op_type in SSAOpRegistry.ops:
                raise ValueError(
                        'SSA Op {} already registered.'.format(op_type))
            SSAOpRegistry.ops[op_type] = {
                    'class':    op_cls,
                    'doc_str':  doc_str
            }

            @classmethod
            def add_op(cls, **kwargs):
                return cls._add_op(op_cls, **kwargs)

            setattr(CoremlBuilder, op_type, add_op)
            return op_cls
        return class_wrapper
