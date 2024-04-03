# Copyright (c) 2023, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from collections import OrderedDict
from typing import Callable, Dict, List, Optional

import numpy as np
from attrs import define, field, validators
from tqdm import tqdm

from coremltools import _SPECIFICATION_VERSION_IOS_16
from coremltools.converters.mil.converter import mil_convert as _mil_convert
from coremltools.converters.mil.frontend.milproto import load as _milproto_to_pymil
from coremltools.converters.mil.mil.passes.defs.quantization import (
    AbstractQuantizationPass as _AbstractQuantizationPass,
)
from coremltools.converters.mil.mil.passes.graph_pass import PassOption
from coremltools.converters.mil.mil.passes.pass_registry import PASS_REGISTRY
from coremltools.models import MLModel as _MLModel
from coremltools.optimize.coreml import OptimizationConfig as _OptimizationConfig
from coremltools.optimize.coreml._config import _MetaDataDict

from ._quantization_passes import WeightDecompressor as _WeightDecompressor
from ._quantization_passes import linear_quantize_weights as _linear_quantize_weights
from ._quantization_passes import palettize_weights as _palettize_weights


def _convert_model_spec_to_pymil_prog(
    mlmodel: _MLModel, specification_version: int, pymil_load_func: Callable
):
    """
    An utility that converts a ml program model into PyMIL program.
    """
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

    prog = pymil_load_func(
        model_spec=model_spec,
        specification_version=specification_version,
        file_weights_dir=mlmodel.weights_dir,
    )
    return prog


def _apply_graph_pass(
    mlmodel: _MLModel,
    graph_pass: _AbstractQuantizationPass,
    spec_version: int = _SPECIFICATION_VERSION_IOS_16,
    skip_model_load: bool = False,
    pymil_load_func: Callable = _milproto_to_pymil.load,
):
    # Utility function which compresses a Core ML model
    # converts the full precision mlmodel into a pymil program
    model_spec = mlmodel.get_spec()
    specification_version = max(model_spec.specificationVersion, spec_version)
    prog = _convert_model_spec_to_pymil_prog(mlmodel, specification_version, pymil_load_func)

    # apply compression graph pass
    assert isinstance(
        graph_pass, _AbstractQuantizationPass
    ), "compression pass must be an AbstractQuantizationPass instance"
    graph_pass.apply(prog)

    # convert the pymil program back to mlmodel
    compressed_mlmodel = _mil_convert(
        prog,
        convert_to="mlprogram",
        convert_from="milinternal",
        specification_version=specification_version,
        compute_units=mlmodel.compute_unit,
        model_description=model_spec.description,
        skip_model_load=skip_model_load,
    )
    return compressed_mlmodel


def _is_valid_const(val, weight_threshold):
    return isinstance(val, np.ndarray) and val.size >= weight_threshold

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

        model = ct.coreml.models.MLModel("my_model.mlpackage")
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
    values, which are then used to perform the float operation the weight is feeding into.

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

    Examples
    --------
    .. sourcecode:: python

        import coremltools as ct
        import coremltools.optimize as cto

        model = ct.models.MLModel("my_model.mlpackage")
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

    Examples
    --------
    .. sourcecode:: python

        import coremltools as ct
        import coremltools.optimize as cto

        model = ct.models.MLModel("my_model.mlpackage")
        config = cto.coreml.OptimizationConfig(
            global_config=cto.coreml.OpThresholdPrunerConfig(threshold=1e-12)
        )
        compressed_model = cto.coreml.prune_weights(model, config)

    """
    weight_pruner = PASS_REGISTRY["compression::prune_weights"]
    weight_pruner.set_options([PassOption("config", config)])
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

    Examples
    --------
    .. sourcecode:: python

        import coremltools as ct

        model = ct.models.MLModel("my_compressed_model.mlpackage")
        decompressed_model = ct.optimize.coreml.decompress_weights(model)

    """

    weight_decompressor = _WeightDecompressor(op_selector=lambda op: True)
    return _apply_graph_pass(mlmodel, weight_decompressor)



def get_weights_metadata(mlmodel: _MLModel, weight_threshold: int = 2048):
    """
    Utility function to get the weights metadata as a dictionary, which maps the weight's name to its corresponding CoreMLWeightMetaData.

    CoreMLWeightMetaData contains the following attributes:

    1. ``val``: The weight data.
    2. ``sparsity``: the percentile of the element whose absolute value ``<= 1e-12``.
    3. ``unique_values``: number of unique values in the weight.
    4. ``child_ops``: meta information of the child ops in which the weight is feeding into.

    Parameters
    ----------
    mlmodel: MLModel
        Model in which the weight metadata is retrieved from.

    weight_threshold: int
        * The size threshold, above which weights are returned. That is, a weight tensor is included in the resulting dictionary only if its total number of elements are greater than ``weight_threshold``.
          For example, if ``weight_threshold = 1024`` and a weight tensor is of shape ``[10, 20, 1, 1]``, hence ``200``
          elements, it will not be returned by the ``get_weights_metadata`` API.

        * If not provided, it will be set to ``2048``, in which weights bigger than ``2048`` elements are returned.

    Returns
    -------
    dict[str, CoreMLWeightMetaData]
        A dict that maps weight's name to its metadata.

    Examples
    --------
    In this example, there are two weights whose sizes are greater than ``2048``.
    A weight named ``conv_1_weight`` is feeding into a ``conv`` op named ``conv_1``,
    while another weight named ``linear_1_weight`` is feeding into a ``linear`` op named ``linear_1``.
    You can access the metadata by ``weight_metadata_dict["conv_1_weight"]``, and so on.

    .. sourcecode:: python

        import coremltools as ct

        mlmodel = ct.models.MLModel("my_model.mlpackage")
        weight_metadata_dict = ct.optimize.coreml.get_weights_metadata(
            mlmodel, weight_threshold=2048
        )

        # get the weight names with size > 25600
        large_weights = []
        for k, v in weight_metadata_dict.items():
            if v.val.size >= 25600:
                large_weights.append(k)

        # get the weight names with sparsity >= 50%
        sparse_weights = []
        for k, v in weight_metadata_dict.items():
            if v.sparsity >= 0.5:
                sparse_weights.append(k)

        # get the weight names with unique elements <= 16
        palettized_weights = []
        for k, v in weight_metadata_dict.items():
            if v.unique_values <= 16:
                palettized_weights.append(k)

        # print out the dictionary
        print(weight_metadata_dict)

    The output from the above example would be:

    ::

        conv_1_weight
        [
            val: np.ndarray(shape=(32, 64, 2, 2), dtype=float32)
            sparsity: 0.5
            unique_values: 4097
            child_ops: [
                conv(name=conv_1, weight=conv_1_weight, ...)
            ]
        ]
        linear_1_weight
        [
            val: np.ndarray(shape=(128, 64), dtype=float32)
            sparsity: 0.2501220703125
            unique_values: 4
            child_ops: [
                linear(name=linear_1, weight=linear_1_weight, ...)
            ]
        ]
    """
    def _get_weight_metadata(op):
        """
        Returns a CoreMLWeightMetaData object given a const operation.
        """
        assert op.op_type == "const", f"Expect op be type of 'const', got '{op.op_type}'"
        child_ops = []
        visited = set()
        for child_op in op.outputs[0].child_ops:
            if child_op in visited:
                continue
            visited.add(child_op)
            params_name_mapping = OrderedDict()
            for k, v in child_op.inputs.items():
                if _is_valid_const(v.val, weight_threshold):
                    params_name_mapping[k] = v.op.name
            child_ops.append(
                CoreMLOpMetaData(
                    op_type=child_op.op_type,
                    name=child_op.name,
                    params_name_mapping=params_name_mapping,
                )
            )
        return CoreMLWeightMetaData(op.val.val, child_ops=child_ops)

    prog = _convert_model_spec_to_pymil_prog(mlmodel, mlmodel.get_spec().specificationVersion,
                                             _milproto_to_pymil.load)
    res = _MetaDataDict({})

    def get_weights_meta_block(block):
        # get the candidates ops with the given op_type
        candidate_ops = []
        for op in block.operations:
            for b in op.blocks:
                get_weights_meta_block(b)

            if op.op_type == "const" and _is_valid_const(op.val.val, weight_threshold):
                candidate_ops.append(op)

        for op in tqdm(
            candidate_ops,
            desc="Getting Core ML weights meta data",
            unit=" ops",
        ):
            res[op.name] = _get_weight_metadata(op)

    for f in prog.functions.values():
        get_weights_meta_block(f)

    return res


@define(frozen=True)
class CoreMLOpMetaData:
    """
    A container class that stores op meta data.

    The class has the following attributes:

    Parameters
    ----------
    op_type: str
        The type of the op. For instance: ``conv``, ``linear``, and so on.

    name: str
        The name of the op.

    params_name_mapping: dict[str, str]
        A dict that maps the op's constant parameters to its corresponding weight name.
        For instance, given a ``conv`` op with ``params_name_mapping``,

        .. sourcecode:: python

            {
                "weight": "conv_1_weight",
                "bias": "conv_1_bias",
            }

        means that the weight and bias of this op are named ``conv_1_weight``,  ``conv_1_bias``, respectively.

    """

    op_type: str = field(validator=validators.instance_of(str))
    name: str = field(validator=validators.instance_of(str))
    params_name_mapping: Dict[str, str] = field(validator=validators.instance_of(dict))

    def __str__(self):
        res = f"{self.op_type}(name={self.name}"
        for k, v in self.params_name_mapping.items():
            res += f", {k}={v}"
        res += ", ...)"
        return res


@define(frozen=True)
class CoreMLWeightMetaData:
    """
    A container class that stores weight meta data.

    The class has the following attributes:

    Parameters
    ----------
    val: numpy.ndarray
        The weight data.

    sparsity: float
        The percentile of the element whose absolute value ``<= 1e-12``.

    unique_values: int
        Number of unique values in the weight.

    child_ops: list[CoreMLOpMetaData]
        A list of ``CoreMLOpMetaData`` which contains information of child ops in which the weight is feeding into.

        The attributes can be accessed by:
        ``child_ops[idx].op_type``: The operation type of the ``idx`` 'th child op.
        ``child_ops[idx].name``: The name of the ``idx`` 'th child op.

        Other op-dependant attributes also can be accessed. For instance, if ``idx`` 'th child op is a ``conv`` layer,
        ``child_ops[idx].weight`` will return its weight name.

        For more details, please refer to the ``CoreMLOpMetaData`` doc string.

    Examples
    --------
    .. sourcecode:: python

        import numpy as np
        from coremltools.optimize.coreml import CoreMLWeightMetaData

        data = np.array([[1.0, 0.0], [0.0, 6.0]], dtype=np.float32)
        meta_data = CoreMLWeightMetaData(data)
        print(meta_data)

    Outputs::
    
        [
            val: np.ndarray(shape=(2, 2), dtype=float32)
            sparsity: 0.5
            unique_values: 3
        ]

    """

    val: np.ndarray = field(validator=validators.instance_of(np.ndarray))
    sparsity: Optional[float] = field(validator=validators.instance_of(float))
    unique_values: Optional[int] = field(validator=validators.instance_of(int))
    child_ops: Optional[List[CoreMLOpMetaData]] = field(
        default=None, validator=validators.optional(validators.instance_of(list))
    )

    @sparsity.default
    def _get_sparsity(self):
        num_of_zeros = np.sum(np.abs(self.val) <= 1e-12)
        return num_of_zeros / self.val.size

    @unique_values.default
    def _get_unique_values(self):
        return len(np.unique(self.val))

    def __str__(self):
        res = "[  \n"
        res += f"  val: np.ndarray(shape={self.val.shape}, dtype={self.val.dtype})\n"
        res += f"  sparsity: {self.sparsity}\n"
        res += f"  unique_values: {self.unique_values}\n"
        if self.child_ops is not None:
            res += "  child_ops: [\n"
            for child_op in self.child_ops:
                res += f"    {child_op}\n"
            res += "  ]\n"
        res += "]"
        return res
