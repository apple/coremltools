import logging
from .builder import CoremlBuilder

class SSAOpRegistry:
    ops = {}
    custom_ops = {}

    @staticmethod
    def register_op(doc_str, is_custom_op=False):
        """
        Registration routine for NNV2 Program operators
        is_custom_op: (Boolean) [Default=False]
            If True, maps current operator to `custom_op`
            `custom_op` requires additional `bindings` which should be
            specified in operator.
            Current operator is registered as `SSARegistry.custom_ops`
            Otherwise, current operator is registered as usual operator,
            i.e. registered in `SSARegistry.ops'.
        """
        def class_wrapper(op_cls):
            op_type = op_cls.__name__
            # op_cls.__doc__ = doc_str  # TODO: rdar://58622145

            # Operation specific to custom op
            op_msg = 'Custom op' if is_custom_op else 'op'
            op_reg = SSAOpRegistry.custom_ops if is_custom_op else SSAOpRegistry.ops

            logging.debug("Registering {} {}".format(op_msg, op_type))
            if op_type in op_reg:
                raise ValueError(
                        'SSA {} {} already registered.'.format(op_msg, op_type))
            op_reg[op_type] = {
                'class':    op_cls,
                'doc_str':  doc_str
            }

            @classmethod
            def add_op(cls, **kwargs):
                return cls._add_op(op_cls, **kwargs)

            setattr(CoremlBuilder, op_type, add_op)
            return op_cls
        return class_wrapper
