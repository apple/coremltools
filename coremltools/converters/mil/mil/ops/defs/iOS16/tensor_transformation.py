#  Copyright (c) 2022, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clausefrom coremltools.converters.mil.mil import types

import numpy as np

from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.input_type import (InputSpec,
                                                       TensorInputType,
                                                       TupleInputType)
from coremltools.converters.mil.mil.operation import Operation
from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op
from coremltools.converters.mil.mil.ops.defs.iOS16 import _IOS16_TARGET
from coremltools.converters.mil.mil.types.symbolic import any_symbolic


@register_op(opset_version=_IOS16_TARGET)
class reshape_like(Operation):
    """
    Reshape a tensor to an output shape specified by some or all dimensions of a tuple of reference tensors ``ref_tensors``.

    Parameters
    ----------
    x: tensor<\\*?, T> (Required)
        * The input tensor to be reshaped.

    ref_tensors: Tuple[tensor<\\*?, R>] (Required)
        * A tuple of tensors that define the output shape.

    begins: Tuple[const<int32>] (Required)
        * A tuple of integers specifying the begin index into the shape vector of the corresponding ``ref_tensor``.

    ends: Tuple[const<int32>] (Required)
        * A tuple of integers specifying the end index into the shape vector of the corresponding ``ref_tensor``.

    end_masks: Tuple[const<bool>] (Required)
        * If ``True``, select all axes from the begin index until the end of the corresponding ``ref_tensor``, as in
          ``ref_tensors[i].shape[begins[i]:]``.

    Notes
    -----
    The output shape is computed as follows:

    .. sourcecode:: python

        output_shape = []
        num_of_refs = len(begins)
        for i in range(num_of_refs):
            if end_masks[i]:
                output_shape.append(ref_tensor_i.shape[begins[i]:])
            else:
                output_shape.append(ref_tensor_i.shape[begins[i]:ends[i]])
        output_shape = np.concat(output_shape, axis=0)

    The following is an example:

    .. sourcecode:: python

        ref_tensors=[tensor[2, 3, 4], tensor[1, 5, 6]]
        begins=[0, 1]
        ends=[2, 0]
        end_masks=[False, True]

    The output shape would be ``(2, 3, 5, 6)``.

    Returns
    -------
    tensor<\\*?, T>
        * Same type as input tensor ``x``.
        * Output shape is computed by ``ref_tensors``, ``begins``, ``ends``, and ``end_masks``.

    Attributes
    ----------
    T: fp16, fp32, i32, bool
    R: fp16, fp32, i32, bool
    """
    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"), 
        ref_tensors=TupleInputType(),
        begins=TupleInputType(),
        ends=TupleInputType(),
        end_masks=TupleInputType(),
    )
    
    type_domains = {
        "T": (types.fp16, types.fp32, types.int32, types.bool),
    }

    def _check_is_const_tuple_with_scalar(self, param, expected_type, param_name):
        """
        This utility function checks the param is a Tuple of scalar with expected data type.
        """
        for x in param:
            if x.dtype != expected_type or x.shape != ():
                msg = "In op reshape_like {}, {} must be a Tuple of scalar {}. Got a {} tensor with shape {}.".format(
                    self.name, 
                    param_name,
                    expected_type.__type_info__(),
                    x.dtype.__type_info__(),
                    x.shape,
                )
                raise ValueError(msg)          

    def type_inference(self):
        # Validation the inputs
        ref_number = len(self.ref_tensors)
        if len(self.begins) != ref_number or len(self.ends) != ref_number or len(self.end_masks) != ref_number:
            msg = (
                    "Op reshape_like {}'s ref_tensors, begins, ends and end_masks must have exactly the same length. "
                    "Got {}, {}, {} and {}."
            ).format(self.name, ref_number, len(self.begins), len(self.ends), len(self.end_masks))

        self._check_is_const_tuple_with_scalar(self.begins, types.int32, "begins")
        self._check_is_const_tuple_with_scalar(self.ends, types.int32, "ends")
        self._check_is_const_tuple_with_scalar(self.end_masks, types.bool, "end_masks")

        # Compute the output shape
        out_shape = ()
        for ref_tensor, begin, end, end_mask in zip(self.ref_tensors, self.begins, self.ends, self.end_masks):
            shape = ref_tensor.shape
            begin, end, end_mask = begin.val, end.val, end_mask.val
            ref_shape = shape[begin:end] if not end_mask else shape[begin:]
            out_shape += tuple(ref_shape)
            
        # Output shape must be known at compile time
        if any_symbolic(out_shape):
            msg = "Output shape of a reshape_like op {} must not be symbolic. Got {}".format(self.name, out_shape)
            raise ValueError(msg)
            
        # Output shape must be consistent with the input shape
        if not any_symbolic(self.x.shape):
            if np.prod(self.x.shape) != np.prod(out_shape):
                msg = "At reshape_like op {}, input shape {} not consistent with the output shape {}.".format(
                    self.name,
                    self.x.shape,
                    out_shape
                )
                raise ValueError(msg)

        return types.tensor(self.x.dtype, out_shape)

@register_op(opset_version=_IOS16_TARGET)
class pixel_unshuffle(Operation):
    """
    Rearrange elements in a tensor from spatial dimensions into depth (channel).
    It is basically the inverse operation of :py:class:`~.iOS15.tensor_transformation.pixel_shuffle`.
    Equivalent to `PyTorch PixelUnshuffle <https://pytorch.org/docs/stable/generated/torch.nn.PixelUnshuffle.html#pixelunshuffle>`_.

    Parameters
    ----------
    x: tensor<[n, C, H / f , W / f], T> (Required)
        * Input tensor of rank ``4``.

    downscale_factor: const<i32>
        * Factor to decrease spatial resolution by.

    Returns
    -------
    tensor<[n, C * f^2, H, W], T>
        * In which ``f`` is the downscale factor.

    Attributes
    ----------
    T: fp16, fp32

    References
    ----------
    `torch.nn.PixelUnshuffle <https://pytorch.org/docs/stable/generated/torch.nn.PixelUnshuffle.html>`_
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        downscale_factor=TensorInputType(const=True, type_domain=types.uint32),
    )
    
    type_domains = {
        "T": (types.fp16, types.fp32),
    }

    def type_inference(self):
        x_type = self.x.dtype
        n, c, h, w = self.x.shape
        f = self.downscale_factor.val
        ret_shape = (n, c * f * f, h / f, w / f)
        return types.tensor(x_type, ret_shape)
