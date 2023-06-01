# Copyright (c) 2023, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools import _SPECIFICATION_VERSION_IOS_16
from coremltools.converters.mil import Operation as _Operation
from coremltools.converters.mil.converter import mil_convert as _mil_convert
from coremltools.converters.mil.frontend.milproto.load import load as _milproto_to_pymil
from coremltools.converters.mil.mil.passes.defs.quantization import (
    AbstractQuantizationPass as _AbstractQuantizationPass,
)
from ._quantization_passes import (
    linear_quantize_weights as _linear_quantize_weights,
    palettize_weights as _palettize_weights,
    prune_weights as _prune_weights,
    WeightDecompressor as _WeightDecompressor,
)
from coremltools.models import MLModel as _MLModel
from coremltools.optimize.coreml import OptimizationConfig as _OptimizationConfig

_DEFAULT_SPECIFICATION_VERSION_FOR_COMPRESSION = _SPECIFICATION_VERSION_IOS_16

def _apply_graph_pass(mlmodel, graph_pass):
    # Utility function which compresses a coreml model
    # convert the fully precision mlmodel into pymil program
    model_spec = mlmodel.get_spec()
    model_type = model_spec.WhichOneof("Type")
    if model_type in ("neuralNetwork", "neuralNetworkClassifier", "neuralNetworkRegressor", "pipeline", "PipelineClassifier", "PipelineRegressor"):
        msg = ("coremltools.optimize.coreml are meant to be used only with mlprogram typed coreml models. "
              "This model has type {}. Please use coremltools.models.neural_network.quantization_utils.quantize_weights"
              "instead to compress the weights of the model.")
        raise TypeError(msg.format(model_type))
    elif model_type == "mlProgram":
        pass
    else:
       raise TypeError("weight compression not applicable for model type {}".format(model_type))

    assert isinstance(graph_pass, _AbstractQuantizationPass), "compression pass must be an AbstractQuantizationPass instance"
    specification_version = max(model_spec.specificationVersion, _DEFAULT_SPECIFICATION_VERSION_FOR_COMPRESSION)
    prog = _milproto_to_pymil(
        model_spec=model_spec,
        specification_version=specification_version,
        file_weights_dir=mlmodel.weights_dir,
    )

    # apply compression graph pass
    graph_pass.apply(prog)

    # convert the pymil program back to mlmodel
    compressed_mlmodel = _mil_convert(
        prog,
        convert_to="mlprogram",
        convert_from="milinternal",
        specification_version=specification_version,
        compute_units=mlmodel.compute_unit,
        model_description=model_spec.description,
    )
    return compressed_mlmodel

def linear_quantize_weights(mlmodel: _MLModel, config: _OptimizationConfig):
    """
    Utility function to convert a float precision MLModel of type ``mlprogram``, which uses
    float-precision weights, into a compressed MLModel that uses 8-bit weights. This is
    achieved by converting the float weight values that are stored in the ``const`` op
    into the ``constexpr_affine_dequantize`` op.

    This function uses linear quantization on the float weights, providing up to 2x
    savings in storage compared to float 16, or up to 4x savings compared to float 32.
    All computation at runtime uses float precision; the precision of the intermediate
    tensors and the compute precision of the ops are not altered.

    For each weight, this utility function converts the weight into the int8 or uint8 type using
    either `linear interpolation` (``"linear"`` mode) or `linear symmetric
    interpolation` (``"linear_symmetric"`` mode, the default).

    **Linear interpolation**

    Linear interpolation (``"linear"`` mode) maps the min/max of the float
    range to the 8-bit integer range ``[low, high]`` using a zero point (also called quantization bias, or
    offset) and a scale factor. For the int8 quantization, ``[low, high] = [-128, 127]``, while uint8
    quantization uses range ``[0, 255]``.

    ``"linear"`` mode uses the quantization formula:

    .. math::
       w_r = s * (w_q - z)

    Where:

        * :math:`w_r` and  :math:`s` are of type float.
        * :math:`w_r`` represents the float precision weight.
        * :math:`s` represents the scale.
        * :math:`w_q` and :math:`z` are of type 8-bit integer.
        * :math:`w_q` represents quantized weight.
        * :math:`z` represents the zero point.

    Quantized weights are computed as follows:

    .. math::
       w_q = cast\_to\_8\_bit\_integer(w_r / s + cast\_to\_float(z))

    Note: :math:`cast\_to\_8\_bit\_integer` is the process of clipping the input to range ``[low, high]`` followed by rounding and casting to 8-bit integer.
    
    In ``"linear"`` mode, ``s, z`` are computed by mapping the original float range
    ``[A, B]`` into the 8-bit integer range ``[-128, 127]`` or ``[0, 255]``. That is, you are solving the
    following linear equations:

        * ``B = s * (high - z)``
        * ``A = s * (low - z)``

    The equations result in the following:

        * ``s = (B - A) / (high - low)``
        * ``z = cast_to_8_bit_integer((low * B - high * A) / (B - A))``

    When the rank of weight ``w`` is 1, then ``s`` and ``z`` are both scalars. When the
    rank of the weight is greater than 1, then ``s`` and ``z`` are both vectors. In that
    case, scales are computed per `channel`, in which `channel` is the output dimension,
    which corresponds to the first dimension for ops such as ``conv`` and ``linear``, and
    the second dimension for the ``conv_transpose`` op.
    
    For ``"linear"`` mode, :math:`A = min(w_r)`, :math:`B = max(w_r)`.
    
    **Linear symmetric interpolation**

    With linear symmetric interpolation (``"linear_symmetric"`` mode, the default), rather than
    mapping the exact min/max of the float range to the quantized range, the function
    chooses the maximum absolute value between the min/max, which results in a
    floating-point range that is symmetric with respect to zero. This also makes the resulting zero
    point ``0`` for int8 weight and ``127`` for uint8 weight.
    
    For ``"linear_symmetric"`` mode:
    
       * :math:`A = -R` and :math:`B = R`, where :math:`R = max(abs(w_r))`.
       * This function maps to the range of ``[-127, 127]`` for int8 weight and ``[0, 254]`` for uint8 weight.
       * The result is ``s=(B-A)/254`` -> ``s=2R/254`` -> ``s=R/127``.
       * Solving for ``z``:
            * int8:  ``z = (-127 * R + 127 * R)/2R`` -> ``z=0``.
            * uint8: ``z = (0 * R + 254 * R)/2R`` -> ``z=127``.

    Parameters
    ----------
    mlmodel: MLModel
        Model to be quantized. This MLModel should be of type ``mlprogram``.
        
    config: OptimizationConfig
        An :py:class:`OptimizationConfig` object that specifies the parameters for weight quantization.

    Returns
    -------

    model: MLModel
        The quantized MLModel instance.

    Examples
    --------
    .. sourcecode:: python
 
        import coremltools as ct
        import coremltools.optimize as cto
        
        model = ct.coreml.models.MLModel('my_model.mlpackage')
        config = cto.coreml.OptimizationConfig(
            global_config=cto.coreml.OpLinearQuantizerConfig(mode="linear_symmetric")
        )
        compressed_model = cto.coreml.linear_quantize_weights(model, config)

    """

    linear_weight_quantizer = _linear_quantize_weights(config, fake_compression=False)
    return _apply_graph_pass(mlmodel, linear_weight_quantizer)

def palettize_weights(mlmodel: _MLModel, config: _OptimizationConfig):
    """
    Utility function to convert a float precision MLModel of type ``mlprogram`` to a
    compressed MLModel by reducing the overall number of weights using a lookup table
    (LUT). A LUT contains a list of float values. An `nbit` LUT has 2\ :sup:`nbits` entries.

    For example, a float weight vector such as ``{0.3, 0.3, 0.5, 0.5}`` can be compressed
    using a 1-bit LUT: ``{0.3, 0.5}``. In this case the float vector can be replaced
    with a 1-bit vector ``{0, 0, 1, 1}``.

    This function iterates over all the weights in the ``mlprogram``, discretizes its values,
    and constructs the LUT according to the algorithm specified in ``mode``. The float
    values are then converted to the `nbit` values, and the LUT is saved alongside each
    weight. The ``const`` ops storing weight values are replaced by
    ``constexpr_lut_to_dense`` ops.

    At runtime, the LUT and the `nbit` values are used to reconstruct the float weight
    values, which are then used to perform the float operaton the weight is feeding into.

    Consider the following example of ``"uniform"`` mode (a linear histogram):

        * ``nbits = 4``
        * ``mode = "uniform"``
        * ``weight = [0.11, 0.19, 0.3, 0.08, 0.0, 0.02]``

    The weight can be converted to a palette with indices ``[0, 1, 2, 3]`` (2 bits). The
    indices are a byte array.

    The data range ``[0.0, 0.3]`` is divided into 4 partitions linearly, which is
    ``[0.0, 0.1, 0.2, 0.3]``.

        * The LUT would be ``[0.0, 0.1, 0.2, 0.3]``.

        * The weight is rounded to ``[0.1, 0.2, 0.3, 0.1, 0.0, 0.0]``, and represented in
          the palette as indices ``[01b, 10b, 11b, 01b, 00b, 00b]``.

    Parameters
    ----------
    mlmodel: MLModel
        Model to be converted by a LUT. This MLModel should be of type ``mlprogram``.
        
    config: OptimizationConfig
        An :py:class:`OptimizationConfig` object that specifies the parameters for weight palettization.

    Returns
    -------
    model: MLModel
        The palettized MLModel instance.

    Example
    -------

    .. sourcecode:: python

        import coremltools as ct
        import coremltools.optimize as cto
        
        model = ct.models.MLModel('my_model.mlpackage')
        config = cto.coreml.OptimizationConfig(
            global_config=cto.coreml.OpPalettizerConfig(mode="kmeans", nbits=4)
        )
        compressed_model = cto.coreml.palettize_weights(model, config)

    """

    weight_palettizer = _palettize_weights(config, fake_compression=False)
    return _apply_graph_pass(mlmodel, weight_palettizer)

def prune_weights(mlmodel: _MLModel, config: _OptimizationConfig):
    """
    Utility function to convert a float precision MLModel of type ``mlprogram`` to a
    compressed MLModel using sparse representation. The ``const`` ops storing weight
    values are replaced by ``constexpr_sparse_to_dense`` ops.

    This function is useful if the model is trained with pruning techniques so that
    a lot of weights have zero values. If a large percentage of weight values are zero,
    a sparse representation is more efficient than a dense one (the default).

    The sparsified weights are stored in a bit mask. If the weight values are
    ``{0, 0, 0, 0, 0, 0, 0, 56.3}``, its sparse representation contains a bit mask with
    ones on locations where the value is non-zero: ``00000001b``. This is accompanied by
    non-zero data, which is a size-1 vector of value ``{56.3}``.

    For example, given the following:

        * ``weight = [0.3, 0, 0, 0.5, 0, 0]``
        * ``non_zero_data, bit_mask = sparsify(weight)``

    The indices of the non-zero elements are:

        * ``non_zero_data = [0.3, 0.5]``
        * ``bit_mask = "100100"``

    Parameters
    ----------
    mlmodel: MLModel
        Model to be sparsified. This MLModel should be of type ``mlprogram``.

    config: OptimizationConfig
        An :py:class:`OptimizationConfig` object that specifies the parameters for weight pruning.

    Returns
    -------
    model: MLModel
        The sparse MLModel instance.

    Example
    -------
    .. sourcecode:: python

        import coremltools as ct
        import coremltools.optimize as cto
        
        model = ct.models.MLModel('my_model.mlpackage')
        config = cto.coreml.OptimizationConfig(
            global_config=cto.coreml.OpThresholdPrunerConfig(threshold=1e-3)
        )
        compressed_model = cto.coreml.prune_weights(model, config)

    """

    weight_pruner = _prune_weights(config, fake_compression=False)
    return _apply_graph_pass(mlmodel, weight_pruner)

def decompress_weights(mlmodel: _MLModel):
    """
    Utility function to convert weights that are sparse or palettized or affine quantized, back to the float format.
    That is, convert any of the following three ops to ``mb.const``:

    (1) ``constexpr_affine_dequantize``
    (2) ``constexpr_lut_to_dense``
    (3) ``constexpr_sparse_to_dense``

    Parameters
    ----------
    mlmodel: MLModel
        Model which will be decompressed.

    Returns
    -------
    model: MLModel
        The MLModel with no ``constexpr`` ops included.

    Example
    -------
    .. sourcecode:: python

        import coremltools as ct

        model = ct.models.MLModel("my_compressed_model.mlpackage")
        decompressed_model = ct.optimize.coreml.decompress_weights(model)

    """

    weight_decompressor = _WeightDecompressor(op_selector=lambda op: True)
    return _apply_graph_pass(mlmodel, weight_decompressor)
