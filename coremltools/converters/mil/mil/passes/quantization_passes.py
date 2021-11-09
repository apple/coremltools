#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np

from coremltools.converters.mil.mil.program import Program
from coremltools.converters.mil.mil.operation import Operation
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.types import builtin_to_string, is_tensor
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass

from enum import Enum

class ComputePrecision(Enum):
    FLOAT16 = "float16"
    FLOAT32 = "float32"

type_eps = {}
type_min = {}
type_negmin = {}

def _close_to_zero(val, np_type):
    if np_type not in type_eps:
        type_eps[np_type] = np.finfo(np_type).eps
        type_min[np_type] = np.nextafter(0., 1., dtype=np_type)
        type_negmin[np_type] = np.nextafter(0., -1., dtype=np_type)

    return np.isclose(val, 0, atol=type_min[np_type], rtol=type_eps[np_type])

class AbstractQuantizationPass(AbstractGraphPass):
    """
    Base class for Post-Training Quantization transforms.

    Derived class needs to implement following two methods:
        - is_valid_op(op)
        - transform_op(op)
    """

    def __init__(self, op_selector=None):
        if not callable(op_selector):
            msg = (
                'Argument "selector" needs to be a callable function which '
                "accepts a MIL operation object and returns a boolean value."
            )
            raise TypeError(msg)

        self.op_selector = op_selector

    def apply(self, prog):
        """
        Walks over each operation in the graph and performs following two steps,
        1. Checks whether an operation is valid for that quantized transform using `is_valid_op` method.
        2. If yes, calls `transform_op` method of the derived quantized transform class.

        :param prog: MIL program
        :return: Transformed MIL program
        """
        if not isinstance(prog, Program):
            raise TypeError(
                'Transform "{}" can only be applied on PyMIL programs.'.format(self)
            )

        def apply_block(block):
            for op in list(block.operations):
                for b in op.blocks:
                    apply_block(b)

                if self.is_valid_op(op) and self.op_selector(op):
                    self.transform_op(op)

        for f in prog.functions.values():
            apply_block(f)

    def transform_op(self, op):
        """
        Replaces an op with a transformed op.

        :param op: MIL operation
        :return: None
        """
        raise NotImplementedError(
            'Op transformation for quantization mode "{}" not implemented.'.format(self)
        )

    def is_valid_op(self, op):
        """
        Checks whether an operation is valid for given quantized transform.

        :param op: MIL operation
        :return: true | false
        """
        raise NotImplementedError(
            'Operation Preconditions for quantization mode "{}" not implemented.'.format(
                self
            )
        )

    def __repr__(self):
        return str(self)

    def __str__(self):
        return type(self).__name__


class FP16ComputePrecision(AbstractQuantizationPass):
    """
    This transform does the following, for each valid op and if the "op_selector" return True:
    - For each input of dtype float32, inject a "cast" op to change it to float16 dtype
    - For each output of dtype float16, inject a "cast" op to change it back to float32
    """

    def __init__(self, op_selector=None):
        super(FP16ComputePrecision, self).__init__(op_selector=op_selector)
        self.target_dtype = "fp16"

        # Var that feeds into multiple ops will be casted once and cached into this dict
        # For reference: Checkout test_single_input_to_multiple_operations in test_fp16_compute_precision.py
        self.cache_vars = {}

    def fp16_overflow(self, op):
        # Constants with values more than 65504 or less than -65504 overflows in FP16
        for _, inputs in op.inputs.items():
            is_list_input = isinstance(inputs, (list, tuple))
            if not is_list_input:
                inputs = [inputs]
            for var in inputs:
                if var.op is not None and var.op.op_type == "const" and var.is_tensor_or_scalar_of(dtype="fp32"):
                    if np.max(np.abs(var.op.val.val), initial=0.0) > 65504:
                        return True
        return False


    def is_valid_op(self, op):

        if op.op_type in ["cast", "while_loop", "cond"]:
            return False

        if op.op_type in ["make_list", "list_gather", "list_scatter", "list_read", "list_write", "list_length"]:
            return False  #  rdar://74458192

        if op.op_type in ["gru", "rnn", "lstm"]:
            return False

        if self.fp16_overflow(op):
            return False

        return True

    def is_valid_parameter(self, op, param_name):

        if op.op_type in ["resample"] and param_name == "coordinates":
            return False

        if op.op_type in ["crop_resize"] and param_name == "spatial_scale":
            return False

        if op.op_type in ["upsample_nearest_neighbor"] and param_name in ["scale_factor_height", "scale_factor_width"]:
            return False

        if op.op_type in ["upsample_bilinear"] and param_name in ["scale_factor_height", "scale_factor_width"]:
            return False

        return True

    def _check_underflow_to_zero(self, new_var, var):
        # We check whether there are casted values that "becomes" 0 which is not ideal for eps purposes.
        # However we skip arrays with more than 400 in case we compare through a large sparse matrix.
        if new_var.val is not None and \
             len(var.val.flatten()) < 400 and \
             _close_to_zero(new_var.val, np.float16).any():
            value_modified = False
            original_val = var.val.flatten()
            new_val = new_var.val.flatten()

            for idx in range(len(original_val)):
                if not _close_to_zero(original_val[idx], np.float32) and \
                     _close_to_zero(new_val[idx], np.float16):
                    new_val[idx] = type_min[np.float16] if np.sign(original_val[idx]) > 0 else type_negmin[np.float16]
                    value_modified = True

            if value_modified:
                if np.isscalar(new_var.val):
                    new_var._sym_val.val = new_val[0]
                else:
                    new_var._sym_val.val = new_val.reshape(new_var.val.shape)

    def transform_op(self, op):
        block = op.enclosing_block
        casted_inputs = {}
        inputs_modified = False

        for param, inputs in op.inputs.items():
            # First loop, iterates over all the input parameters of an operation.
            if not self.is_valid_parameter(op, param):
                continue

            is_list_input = isinstance(inputs, (list, tuple))
            if not is_list_input:
                inputs = [inputs]

            casted_inputs[param] = list(inputs[:])
            for i, var in enumerate(inputs):
                # Second loop, iterates over all the vars of a python list corresponding to an input parameter.
                if not var.is_tensor_or_scalar_of(dtype="fp32"):
                    continue

                inputs_modified = True
                with block:
                    casted_var_name = var.name + "_to_fp16"
                    if len(var._child_ops) > 1 and casted_var_name in self.cache_vars and (self.cache_vars[casted_var_name] in block._visible_vars_in_block()[1]):
                        casted_inputs[param][i] = self.cache_vars[casted_var_name]
                    else:
                        x = mb.cast(
                            x=var, dtype="fp16", name=casted_var_name, before_op= op
                        )
                        self._check_underflow_to_zero(x, var)

                        casted_inputs[param][i] = x
                        if len(var._child_ops) > 1:
                            self.cache_vars[casted_var_name] = casted_inputs[param][i]

            if not is_list_input:
                casted_inputs[param] = casted_inputs[param][0]

        if inputs_modified:
            casted_inputs.update(
                {k: v for k, v in op.inputs.items() if k not in casted_inputs}
            )
            casted_inputs["name"] = op.name + "_cast"
            casted_inputs["before_op"] = op
            with block:
                quant_output = getattr(mb, op.op_type)(**casted_inputs)

            if not isinstance(quant_output, (list, tuple)):
                quant_output = [quant_output]

            for old_output_var, new_output_var in zip(op.outputs, quant_output):
                if old_output_var.is_tensor_or_scalar_of(dtype="fp32") and (
                    not new_output_var.is_tensor_or_scalar_of(dtype="fp32")
                ):
                    with block:
                        x = mb.cast(
                            x=new_output_var,
                            dtype="fp32",
                            name=new_output_var.name + "_to_fp32",
                            before_op=op,
                        )
                        op.enclosing_block.replace_uses_of_var_after_op(
                            anchor_op=op, old_var=old_output_var, new_var=x
                        )
                else:
                    op.enclosing_block.replace_uses_of_var_after_op(
                        anchor_op=op, old_var=old_output_var, new_var=new_output_var
                    )

            block.remove_ops([op])
