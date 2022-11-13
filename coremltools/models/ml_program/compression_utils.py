# Copyright (c) 2022, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools import _SPECIFICATION_VERSION_IOS_16
from coremltools.converters.mil import Operation as _Operation
from coremltools.converters.mil.converter import mil_convert as _mil_convert
from coremltools.converters.mil.frontend.milproto.load import \
    load as _milproto_to_pymil
from coremltools.converters.mil.mil.passes.compression_passes import \
    WeightAffineQuantizer as _WeightAffineQuantizer
from coremltools.converters.mil.mil.passes.compression_passes import \
    WeightDecompressor as _WeightDecompressor
from coremltools.converters.mil.mil.passes.compression_passes import \
    WeightPalettizer as _WeightPalettizer
from coremltools.converters.mil.mil.passes.compression_passes import \
    WeightSparsifier as _WeightSparsifier
from coremltools.converters.mil.mil.passes.quantization_passes import \
    AbstractQuantizationPass as _AbstractQuantizationPass

_DEFAULT_MIN_WEIGHT_SIZE_TO_COMPRESS = 2048
_DEFAULT_SPECIFICATION_VERSION_FOR_COMPRESSION = _SPECIFICATION_VERSION_IOS_16


def _default_op_selector(const_op):
    if not isinstance(const_op, _Operation) or const_op.op_type != "const":
        raise ValueError("Input of the op_selector must be type of const Operation, got {}.".format(type(const_op)))
    return const_op.val.val.size > _DEFAULT_MIN_WEIGHT_SIZE_TO_COMPRESS

def _apply_graph_pass(mlmodel, graph_pass):
    # Utility function which compresses a coreml model
    # convert the fully precision mlmodel into pymil program
    model_spec = mlmodel.get_spec()
    model_type = model_spec.WhichOneof("Type")
    if model_type in ("neuralNetwork", "neuralNetworkClassifier", "neuralNetworkRegressor", "pipeline", "PipelineClassifier", "PipelineRegressor"):
        msg = ("coremltools.compression_utils are meant to be used only with mlprogram typed coreml models. "
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

def affine_quantize_weights(mlmodel, mode="linear_symmetric", op_selector=None):
    """
    Utility function to convert a float precision MLModel of type ``mlprogram`` that uses
    float-precision weights into a compressed MLModel that uses 8-bit weights. This is
    achieved by converting the float weight values that are stored in the ``const`` op
    into the ``constexpr_affine_dequantize`` op.
    
    This function uses affine quantization on the float weights, providing up to 2x
    savings in storage compared to float 16, or up to 4x savings compared to float 32.
    All computation at runtime uses float precision; the precision of the intermediate
    tensors and the compute precision of the ops are not altered.
    
    For each weight, this utility function converts the weight into the uint8 type using
    either `Linear interpolation` (``"linear"`` mode) or `Linear symmetric
    interpolation` (``"linear_symmetric"`` mode, the default).
    
    **Linear interpolation**
    
    Linear interpolation (``"linear"`` mode) maps the min/max of the float
    range to the range [0, 255] using a zero point (also called quantization bias, or
    offset) and a scale factor.
    
    ``"linear"`` mode uses the quantization formula ``w_r = s * (w_q - z)``, where:
    
        * ``w_r`` and  ``s`` are of type float.
        * ``w_r`` represents the float precision weight.
        * ``s`` represents the scale.
        * ``w_q`` and ``z`` are of type uint8.
        * ``w_q`` represents quantized weight.
        * ``z`` represents the zero point.
    
    Quantized weights are computed as follows:
    
        * ``w_q = cast_to_uint8(w_r / s + cast_to_float(z))``
        * Note: ``cast_to_uint8`` is the process of clipping the input to range [0, 255]
          followed by rounding and casting to uint8.
    
    In ``"linear"`` mode, ``s, z`` are computed by mapping the original float range
    ``[A, B]`` into the uint8 range [0, 255]. That is, you are solving the following
    linear equations:
    
        * ``B = s * (255 - z)``
        * ``A = s * (0 - z)``
    
    The equations result in the following:
    
        * ``s = (B - A) / 255``
        * ``z = cast_to_uint8(-255 * A / (B - A))``
    
    When the rank of weight ``w`` is 1, then ``s`` and ``z`` are both scalars. When the
    rank of the weight is greater than 1, then ``s`` and ``z`` are both vectors. In that
    case, scales are computed "per channel", in which "channel" is the output dimension,
    which corresponds to the first dimension for ops such as ``conv`` and ``linear``, and
    the second dimension for the ``conv_transpose`` op.
    
    For ``"linear"`` mode, ``A = min(w_r), B = max(w_r)``.
    
    **Linear symmetric interpolation**
    
    With linear symmetric interpolation (``"linear_symmetric"`` mode, the default), rather than
    mapping the exact min/max of the float range to the quantized range,
    the function chooses the maximum absolute value between the min/max, which results in
    a zero point value of 127. The floating-point range is symmetric with respect to zero,
    and so is the quantized range.
    
    For ``"linear_symmetric"`` mode:
    
       * ``A = -R`` and ``B = R``, where ``R = max(abs(w_r))``.
       * This function maps to the range [0, 254].
       * The result is ``s=(B-A)/254`` --> ``s=2R/254`` --> ``s=R/127``.
       * Solving for ``z``: ``z = (R/2R) * 254`` --> ``z=127``.

    Parameters
    ----------
    mlmodel: MLModel
        Model to be quantized. This MLModel should be of type ``mlprogram``.

    mode: str
        Mode for linear quantization:
        
        * ``"linear_symmetric"`` (default): Input data are quantized in the range
          ``[-R, R]``, where ``R = max(abs(w_r))``.
        * ``"linear"``: Input data are quantized in the range
          ``[min(w_r), max(w_r)]``.

    op_selector: callable 
        This function takes a single parameter with type ``coremltools.converters.mil.Const``;
        that is, a ``const`` operation. It returns a ``bool``: ``True`` to compress ``const_op``,
        otherwise ``False``. See the following examples:
        
        * All constants in the network are compressed:
          
          .. sourcecode:: python

              def op_selector(const_op):
                    return True

        * Only the constant with ``tensor.size > 2048`` is compressed:
          
          .. sourcecode:: python

              def op_selector(const_op): 
                    return const_op.val.val.size > 2048

        * Compress the constant if it is the weight of a convolution layer
          and ``tensor.size > 2048``:
          
          .. sourcecode:: python

              def op_selector(const_op):
                    return (const_op.val.val.size > 2048 
                            and const_op.val.child_ops[0].op_type == "conv" 
                            and const_op.val == const_op.val.child_ops[0].weight
                            )

        * When creating a custom ``op_selector`` function, the following attributes are helpful:
        
             * ``const_op.val.val``: The numpy array holding the value of the const.
             * ``const_op.val.child_ops``: A list of ops into which this constant is feeding.
             * ``const_op.val.child_ops[i].op_type``: The string corresponding to the op type
               of the i-th child op.
             * ``const_op.val.child_ops[i].name``: The string corresponding to the name the
               i-th child op.

        * If ``op_selector`` is not provided, it will be set to the behavior in which
          weights bigger than 2048 elements are compressed:
          
          .. sourcecode:: python

              def op_selector(const_op):
                    returm const_op.val.val.size > 2048:

    Returns
    -------
    
    model: MLModel
        The quantized MLModel instance.

    Examples
    --------
    
        >>> import coremltools as ct
        >>> model = ct.models.MLModel('my_model.mlpackage')
        >>> compressed_model = ct.compression_utils.affine_quantize_weights(model, mode="linear_symmetric")

    """
    if op_selector is None:
        op_selector = _default_op_selector
    affine_weight_quantizer = _WeightAffineQuantizer(fake_compression=False, mode=mode, op_selector=op_selector)
    return _apply_graph_pass(mlmodel, affine_weight_quantizer)


def palettize_weights(mlmodel, nbits=None, mode="kmeans", op_selector=None, lut_function=None):
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

    nbits: int
        Number of bits per weight. Required for ``kmeans`` or ``uniform`` mode, but must
        not be set for ``unique`` or ``custom`` mode. A LUT would have
        2\ :sup:`nbits` entries, where `nbits` can be ``{1, 2, 4, 6, 8}``.

    mode: str
        Determine how the LUT is constructed by specifying one of the following:
        
        * ``"kmeans"`` (default): The LUT is generated by `k-means clustering`, a method of vector
          quantization that groups similar data points together to discover underlying
          patterns by using a fixed number (`k`) of clusters in a dataset. A cluster
          refers to a collection of data points aggregated together because of certain
          similarities. `nbits` is required.

        * ``"uniform"``: The LUT is generated by a linear histogram.
        
           - ``[v_min, v_min + scale, v_min + 2 * scale, ..., v_max]``
           - Where the weight is in the range ``[v_min, v_max]``, and
             ``scale = (v_max - v_min) / (1 << nbits - 1)``.
           - ``nbits`` is required.
           
           A `histogram` is a representation of the distribution of a continuous variable,
           in which the entire range of values is divided into a series of intervals (or
           "bins") and the representation displays how many values fall into each bin.
           Linear histograms have one bin at even intervals, such as one bin per integer.
          
        * ``"unique"``: The LUT is generated by unique values in the weights. The weights
          are assumed to be on a discrete lattice but stored in a float data type. This
          parameter identifies the weights and converts them into the palettized representation.
          
          Do not provide ``nbits`` for this mode. ``nbits`` is picked up automatically,
          with the smallest possible value in ``{1, 2, 4, 6, 8}`` such that the
          number of the unique values is ``<= (1 << nbits)``. If the weight has ``> 256``
          unique values, the compression is skipped.
          
          For example:
          
          * If the weights are ``{0.1, 0.2, 0.3, 0.4}`` and ``nbits=2``, the weights are
            converted to ``{00b, 01b, 10b, 11b}``, and the generated LUT is
            ``[0.1, 0.2, 0.3, 0.4]``.
          * If the weights are ``{0.1, 0.2, 0.3, 0.4}`` and ``nbits=1``, nothing happens
            because the weights are not a 1-bit lattice.
          * If the weights are ``{0.1, 0.2, 0.3, 0.4, 0.5}`` and ``nbits=2``, nothing
            happens because the weights are not a 2-bit lattice.
          
        * ``"custom"``: The LUT and palettization parameters are calculated using a custom
          function. If this mode is selected then ``lut_function`` must be provided.

          Do not provide ``nbits`` for this mode. The user should customize ``nbits`` in the 
          ``lut_function`` implementation.

    op_selector: callable 
        This function takes a single parameter with type ``coremltools.converters.mil.Operation``.
        It returns a ``bool``: ``True`` to compress ``const_op``, otherwise ``False``.
        See the following examples:
        
        * All constants in the network are compressed:
          
          .. sourcecode:: python

              def op_selector(const_op):
                    return True

        * Only the constant with ``tensor.size > 2048`` is compressed:
          
          .. sourcecode:: python

              def op_selector(const_op): 
                    return const_op.val.val.size > 2048

        * Compress the constant if it is the weight of a convolution layer
          and ``tensor.size > 2048``:
          
          .. sourcecode:: python

              def op_selector(const_op):
                    return (const_op.val.val.size > 2048 
                            and const_op.val.child_ops[0].op_type == "conv" 
                            and const_op.val == const_op.val.child_ops[0].weight
                            )

        * When creating a custom ``op_selector`` function, the following attributes are helpful:

             * ``const_op.val.val``: The numpy array holding the value of the const.
             * ``const_op.val.child_ops``: A list of ops into which this constant is feeding.
             * ``const_op.val.child_ops[i].op_type``: The string corresponding to the op type
               of the i-th child op.
             * ``const_op.val.child_ops[i].name``: The string corresponding to the name the
               i-th child op.

        * If ``op_selector`` is not provided, it will be set to the behavior in which
          weights bigger than 2048 elements are compressed:
          
          .. sourcecode:: python

              def op_selector(const_op):
                    returm const_op.val.val.size > 2048:
  
    lut_function: callable
        A callable function which computes the weight palettization parameters. This must
        be provided if the mode is set to ``"custom"``.

        weight: np.ndarray
            A float precision numpy array.

        Returns: lut: list[float]
            The lookup table.

        indices: list[int]
            A list of indices for each element.

        The following is an example that extract the ``top_k`` elements as the LUT. Given
        that ``weight = [0.1, 0.5, 0.3, 0.3, 0.5, 0.6, 0.7]``, the ``lut_function``
        produces ``lut = [0, 0.5, 0.6, 0.7], indices = [0, 1, 0, 0, 2, 3]``.
          
        .. sourcecode:: python

           def lut_function(weight):
                # In this example, we assume elements in the weights >= 0
                weight = weight.flatten()
                nbits = 4

                # Get the LUT, from extracting top k maximum unique elements in the weight to be the LUT
                # Note that k = 1 << nbits - 1, so we have the first element be 0
                unique_elements = np.unique(weight)
                k = (1 << nbits) - 1
                top_k = np.partition(weight, -k)[-k:]
                np.sort(top_k)
                lut = [0.] + top_k.tolist()

                # Compute the indices
                mapping = {v: idx for idx, v in enumerate(lut)}
                indices = [mapping[v] if v in mapping else 0 for v in weight]

                return lut, indices

    Returns
    -------
    model: MLModel
        The palettized MLModel instance.

    Examples
    --------
    
    .. sourcecode:: python

        >>> import coremltools as ct
        >>> model = ct.models.MLModel('my_model.mlpackage')
        >>> compressed_model = ct.compression_utils.palettize_weights(model, mode="kmeans", nbits=4)
    
    
    """
    if op_selector is None:
        op_selector = _default_op_selector        
    weight_palettizer = _WeightPalettizer(nbits=nbits, fake_compression=False, op_selector=op_selector, mode=mode, lut_function=lut_function)
    return _apply_graph_pass(mlmodel, weight_palettizer)
    

def sparsify_weights(mlmodel, mode="threshold_based", threshold=1e-3, target_percentile=1.0, op_selector=None):
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

    mode: str
        Determine the scheme to sparsify the model by specifying one of the following:
        
        * ``"threshold_based"`` (default): All the absolute weight values that are smaller
          than ``threshold`` are changed to 0, and the tensor is stored in a sparse format.
          For example, given the following:

               * ``weight = [0.3, -0.2, -0.01, 0.05]``
               * ``threshold = 0.03``

          The sparsified weight would be ``[0.3, -0.2, 0, 0.05]``.

        * ``"percentile_based"``: Sparsify the weight with a constant sparsity percentile,
          which is ``target_percentile``. Where 
          ``n = floor(size_of_weight_tensor * target_percentile)``, the ``n`` lowest
          absolute weight values are changed to 0. For example, given the following:

               * ``weight = [0.3, -0.2, -0.01, 0.05]``
               * ``target_percentile = 0.75``

          The sparsified weight would be ``[0.3, 0, 0, 0]``.

    threshold: float
        Required when ``mode = "prune_threshold"``. The absolute threshold to sparsify the weight.

    target_percentile: float
        Required when ``mode = "percentile_based"``. The percentage of sparsity for
        compression, which needs to be in the range [0, 1]. When 0, no sparsification
        occurs. For 1, all weights become 0.

    op_selector: callable 
        This function takes a single parameter with type ``coremltools.converters.mil.Operation``.
        It returns a ``bool``: ``True`` to compress ``const_op``, otherwise ``False``.
        See the following examples:
        
        * All constants in the network are compressed:
          
          .. sourcecode:: python

              def op_selector(const_op):
                    return True

        * Only the constant with ``tensor.size > 2048`` is compressed:
          
          .. sourcecode:: python

              def op_selector(const_op): 
                    return const_op.val.val.size > 2048

        * Compress the constant if it is the weight of a convolution layer
          and ``tensor.size > 2048``:
          
          .. sourcecode:: python

              def op_selector(const_op):
                    return (const_op.val.val.size > 2048 
                            and const_op.val.child_ops[0].op_type == "conv" 
                            and const_op.val == const_op.val.child_ops[0].weight
                            )

        * When creating a custom ``op_selector`` function, the following attributes are helpful:
        
             * ``const_op.val.val``: The numpy array holding the value of the const.
             * ``const_op.val.child_ops``: A list of ops into which this constant is feeding.
             * ``const_op.val.child_ops[i].op_type``: The string corresponding to the op type
               of the i-th child op.
             * ``const_op.val.child_ops[i].name``: The string corresponding to the name the
               i-th child op.

        * If ``op_selector`` is not provided, it will be set to the behavior in which
          weights bigger than 2048 elements are compressed:
          
          .. sourcecode:: python

              def op_selector(const_op):
                    returm const_op.val.val.size > 2048:
  
    Returns
    -------
    model: MLModel
        The sparse MLModel instance.

    Examples
    --------
    .. sourcecode:: python

        >>> import coremltools as ct
        >>> model = ct.models.MLModel('my_model.mlpackage')
        >>> compressed_model = ct.compression_utils.sparsify_weights(model, mode="threshold_based", threshold=0.01)

    """
    if op_selector is None:
        op_selector = _default_op_selector
    weight_sparsifier = _WeightSparsifier(mode=mode, threshold=threshold, target_percentile=target_percentile, op_selector=op_selector)
    return _apply_graph_pass(mlmodel, weight_sparsifier)

def decompress_weights(mlmodel):
    """
    Utility function to convert weights that are sparse or palettized or affine quantized, back to the float format.
    That is, convert any of the follwing three ops:
    
    (1) constexpr_affine_dequantize
    (2) constexpr_lut_to_dense
    (3) constexpr_sparse_to_dense
    
    to mb.const

    Parameters
    ----------
    mlmodel: MLModel
        Model which will be decompressed.

    Returns
    -------
    model: MLModel
        The MLModel with no constexpr ops included.

    Examples
    --------
    .. sourcecode:: python

        >>> import coremltools as ct
        >>> model = ct.models.MLModel('my_compressed_model.mlpackage')
        >>> decompressed_model = ct.compression_utils.decompress_weights(model)

    """
    weight_decompressor = _WeightDecompressor(op_selector=lambda op: True)
    return _apply_graph_pass(mlmodel, weight_decompressor)
