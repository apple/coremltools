from coremltools.converters.nnv2.builtin_types import builtins
from coremltools.converters.nnv2.nnv2_program.program.program import Operation
from coremltools.converters.nnv2.nnv2_program.program.input_type import *
from coremltools.converters.nnv2.nnv2_program.ops.registry import SSAOpRegistry

register_op = SSAOpRegistry.register_op

# This file contains the TF dialect of SSA. Briefly, these ops are only
# understandable in the TF frontend and not acceptable in the standard op set.
# No backend would support any of the op here. These ops exist to facilitate
# frontend SSA passes, but must be replaced with standard ops during SSA
# passes.

# All tf op must start with 'tf_' prefix.
#
# tf_make_list allows elem_shape to be unspecified. core op make_list does
# not allow that.
@register_op(doc_str='TODO', namespace='tf')
class tf_make_list(Operation):
    input_spec = InputSpec(
        init_length = IntInputType(optional=True, default=1),
        elem_shape = TensorInputType(const=True, optional=True),
        dtype = StringInputType(const=True, optional=True, default='fp32'),
    )

    def __init__(self, **kwargs):
        super(tf_make_list, self).__init__(**kwargs)

    def type_inference(self):
        init_length = self.init_length.val
        if self.elem_shape is None or self.elem_shape.sym_val is None:
            return builtins.list(builtins.unknown,
                    init_length=init_length)
        builtin_dtype = builtins.string_to_builtin(dtype.val)
        if builtin_dtype is None:
            raise ValueError('Unsupported dtype {}'.format(self.dtype.val))
        elem_type = builtins.tensor(builtin_dtype, self.elem_shape.sym_val)
        return builtins.list(elem_type, init_length=init_length)
