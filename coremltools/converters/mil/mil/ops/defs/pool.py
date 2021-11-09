#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil import Operation, types
from coremltools.converters.mil.mil.input_type import (
    BoolInputType,
    DefaultInputs,
    InputSpec,
    IntTensorInputType,
    TensorInputType,
    StringInputType
)
from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op
from coremltools.converters.mil.mil.ops.defs._utils import spatial_dimensions_out_shape


class Pooling(Operation):
    """
    Pooling Op Superclass
    """
    input_spec = InputSpec(
        x=TensorInputType(),
        kernel_sizes=IntTensorInputType(const=True),
        strides=IntTensorInputType(const=True, optional=True),
        pad_type=StringInputType(const=True),
        pad=IntTensorInputType(const=True, optional=True),
        ceil_mode=BoolInputType(const=True, optional=True),
    )

    def default_inputs(self):
        num_spatial_dims = self.x.rank - 2
        return DefaultInputs(
            strides=[1]*num_spatial_dims,
            pad=[0]*2*num_spatial_dims,
            ceil_mode=False,
            )

    def __init__(self, **kwargs):
        super(Pooling, self).__init__(**kwargs)

    def type_inference(self):
        ksize = self.kernel_sizes.val
        x_shape = self.x.shape
        D_in_rank = len(x_shape) - 2

        strides = [1] * D_in_rank if self.strides is None else self.strides.val
        pad_type = "valid" if self.pad_type is None else self.pad_type.val.lower()
        if pad_type not in ["valid", "same", "custom"]:
            raise ValueError("Unrecognized value of pad_type : {}".format(pad_type))
        pad = None if self.pad is None else self.pad.val
        D_in = x_shape[2:]  # spatial dimensions

        if self.ceil_mode.val:
            if D_in_rank > 2:
                raise ValueError('pool: ceil_mode only supported for 1D or 2D pool')
            if pad_type == "same" and self.ceil_mode.val:
                raise ValueError("ceil_mode must be False when pad_type==same")
            if pad is not None:
                for i in range(D_in_rank):
                    if pad[2*i] != pad[2*i+1]:
                        raise ValueError("Padding must be symmetric if ceil_mode is True")

        D_out_shape = spatial_dimensions_out_shape(
            pad_type=pad_type,
            input_shape=D_in,
            kernel_shape=ksize,
            strides=strides,
            custom_pad=pad,
            ceil_mode=self.ceil_mode.val,
        )
        ret_shape = list(x_shape[:2]) + D_out_shape
        return types.tensor(self.x.dtype, tuple(ret_shape))


@register_op(doc_str="")
class avg_pool(Pooling):
    """
    Perform average pooling. Currently defined only for spatial 1-D and
    2-D cases. (Can be extended to the 3-D case easily.)
    
    Parameters
    ----------
    x: tensor<[n,C_in,\*D_in],T> (Required)
        *  ``3 <= rank <= 5``.
        *  ``D_in`` are spatial dimensions, ``1 <= len(D_in) <= 2``.
        *  ``C_in`` is the number of input channels or depth dimensions.
        *  ``n`` is the batch dimension.
    
    kernel_sizes: const tensor<[K],T> (Required)
        * The size of the window for each spatial dimension ``D_in`` of the
          input tensor.
        * ``K == len(D_in)``
    
    strides: const tensor<[S],i32> (Optional, default to all 1s)
        * Stride along each of the spatial dimensions.
        * ``S == len(D_in)``.
    
    pad_type: const str (Required)
        Must be one of ``valid``, ``same`` or ``custom``.
        
        * ``valid``: No padding. This is equivalent to custom pad with ``pad[i] = 0, for
          all i``.
        * ``same`` : This is equivalent to custom pad with ``pad[2*i] + pad[2*i+1] = kernel_size[i]``.
        * ``custom``: Specify custom padding in the parameter pad. note that "same"
          padding is equivalent to custom padding with
          ``pad[2*i] + pad[2*i+1] = kernel_size[i]``.
    
    pad: const<[P],i32> (Optional. Default to all 0s)
        * ``pad`` represents the number of elements to pad before and after each 
          dimension: `pad[2*i], pad[2*i+1]` are the pad size before and after spatial
          dimension ``i``.
        * ``P = 2 * len(D_in)``.
        * ``pad`` should be specified if and only if ``pad_type == custom``
    
    exclude_padding_from_average: const tensor<[], bool> (Optional, default to False)
        * If ''True'', padded values (0s) are excluded from the denominator count
          when computing the average over the kernel window.

    ceil_mode: const<bool>
        * same as PyTorch's ceil mode
        * ceil is used instead of floor in calculating the output size
        * only supported when its 1D or 2D pool
        * optional, defaults to False
        * only applicable when pad_type is ``valid`` or ``custom``
        * when ceil_mode is True, padding must be symmetric, that is, if specified, ``pad[2*i] == pad[2*i+1]`` must hold
    
    Returns
    -------
    tensor<[n, C_out,\*D_out],T>
        * Same rank as ``x``.
        * when ceil_mode= False:
            * ``D_out[i] = floor[(D_in[i] + pad[2*i] + pad[2*i+1] - kernel_sizes[i]) /
              strides[i]] +1, for i = 0, .., len(D_in) - 1`` is mathematically the same
              as (when all parameters involved are integers):

                  * ``D_out[i] = ceil [(D_in[i] + pad[2*i] + pad[2*i+1] - kernel_size[i] - 1) / stride[i]], for i = 0, .., len(D_in) - 1``.
                  * ``*D_out`` is all 1s if ``global_pooling`` is ``true``.

        * when ceil_mode= True
            * `D_out[i] = ceil[(D_in[i] + pad[2*i] + pad[2*i+1] - kernel_sizes[i]) / strides[i]] +1, for i = 0, .., len(D_in) - 1``
                * if  `(D_out[i] - 1) * strides[i] >= D_in[i] + pad[2*i] and (pad[2*i] + pad[2*i+1] > 0)`:
                        then `D_out[i] = D_out[i] - 1`

            * the first equation is same as :
                * `D_out[i] = floor[(D_in[i] + pad[2*i] + pad[2*i+1] - kernel_sizes[i] + strides[i] - 1) / strides[i]] +1, for i = 0, .., len(D_in) - 1``
    
    Attributes
    ----------
    T: fp32
    
    See Also
    --------
    l2_pool, max_pool
    """
    
    input_spec = (
        InputSpec(
          exclude_padding_from_average=BoolInputType(const=True,
                                                     optional=True))
        + Pooling.input_spec
    )

    def default_inputs(self):
        return super().default_inputs() + \
            DefaultInputs(
                exclude_padding_from_average=False,
                )

    def __init__(self, **kwargs):
        super(avg_pool, self).__init__(**kwargs)


@register_op(doc_str="")
class l2_pool(Pooling):
    """
    Perform L2 pooling. Currently supports only 1-D and 2-D.
    
    Parameters
    ----------
    x: tensor<[n,C_in,*D_in],T> (Required)
        * See ``avg_pool``.
    
    kernel_sizes: const tensor<[K],T> (Required)
        * See ``avg_pool``.
    
    strides: const tensor<[S],i32> (Optional, default to all 1s)
        * See ``avg_pool``.
    
    pad_type: const str (Required)
        * See ``avg_pool``.
    
    pad: const<[P],i32> (Optional, default to all 0s)
        * See ``avg_pool``.
    
    Returns
    -------
    tensor<[n, C_out,*D_out],T>
        * See ``avg_pool``.
    
    Attributes
    ----------
    T: fp32
    
    See Also
    --------
    avg_pool, max_pool
    """
    
    def __init__(self, **kwargs):
        super(l2_pool, self).__init__(**kwargs)


@register_op(doc_str="")
class max_pool(Pooling):
    """
    Perform max pooling. Currently supports only 1-D and 2-D.
    
    Parameters
    ----------
    x: tensor<[n,C_in,*D_in],T> (Required)
        * See ``avg_pool``.
    
    kernel_sizes: const tensor<[K],T> (Required)
        * See ``avg_pool``.
    
    strides: const tensor<[S],i32> (Optional, default to all 1s)
        * See ``avg_pool``.
    
    pad_type: const str (Required)
        * See ``avg_pool``.
    
    pad: const<[P],i32> (Optional, default to all 0s)
        * See ``avg_pool``.

    ceil_mode: const<bool>
        * see ``avg_pool``.
    
    Returns
    -------
    tensor<[n, C_out,*D_out],T>
        * See ``avg_pool``.
    
    Attributes
    ----------
    T: fp32

    See Also
    --------
    avg_pool, l2_pool
    """
    
    def __init__(self, **kwargs):
        super(max_pool, self).__init__(**kwargs)
