#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import sys
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import IO, Any, Callable, Dict, Optional, Tuple, Union

import cattrs
import numpy as np
import yaml
from attrs import define, field, validators

from coremltools.converters.mil.mil import Operation, types
from coremltools.converters.mil.mil.types.type_mapping import is_builtin, numpy_type_to_builtin_type


class OpCompressorConfig(ABC):
    """
    An abstract class for the compressor configuration
    """

    def _validate_op_type(self, op_type):
        """
        A utility function checking if an op type is valid for the configuration
        """
        pass

    @classmethod
    @abstractmethod
    def _from_dict(cls, config_dict: Dict[str, Any]) -> "OpCompressorConfig":
        """
        An abstract method that construct an OpCompressorConfig from a dictionary.
        It must be implemented in the child class.
        """
        raise ValueError("_from_dict must be implemented in the subclasses of OpCompressorConfig.")

def _check_weight_threshold(instance, attribute, value):
    if value is not None and value < 0:
        raise ValueError(f"\"weight_threshold\" must be a non-negative integer. Got {value}.")

"""
Linear Quantization configuration
"""

@define
class OpLinearQuantizerConfig(OpCompressorConfig):
    """
    Parameters
    ----------

    mode: str
        Mode for linear quantization:

        * ``"linear_symmetric"`` (default): Input data are quantized in the range
          ``[-R, R]``, where :math:`R = max(abs(w_r))`.
        * ``"linear"``: Input data are quantized in the range
          :math:`[min(w_r), max(w_r)]`.

    dtype: np.generic or mil.type type
        Determines the quantized data type (int8/uint8).

        * The allowed values are:
            * ``np.int8`` (the default)
            * ``np.uint8``
            * ``coremltools.converters.mil.mil.types.int8``
            * ``coremltools.converters.mil.mil.types.uint8``

    weight_threshold: int
        The size threshold, above which weights are pruned.
        That is, a weight tensor is pruned only if its total number of elements are greater than ``weight_threshold``.

        For example, if ``weight_threshold = 1024`` and a weight tensor is of shape ``[10, 20, 1, 1]``, hence ``200``
        elements, it will not be pruned.

        * If not provided, it will be set to ``2048``, in which weights bigger than ``2048`` elements are compressed.
    """
    mode: str = field(default="linear_symmetric", validator=validators.instance_of(str))
    dtype: type = field(default=np.int8, validator=validators.instance_of(type))
    weight_threshold: Optional[int] = field(default=2048, validator=validators.optional([validators.instance_of(int), _check_weight_threshold]))

    _WEIGHT_AFFINE_QUANTIZATION_MODES = ("LINEAR_SYMMETRIC", "LINEAR")
    _WEIGHT_AFFINE_DTYPES = (types.int8, types.uint8)

    @mode.validator
    def check_mode(self, attr, mode):
        if not mode.upper() in self._WEIGHT_AFFINE_QUANTIZATION_MODES:
            raise ValueError(f"Only mode {self._WEIGHT_AFFINE_QUANTIZATION_MODES} supported for weight affine quantization. Got mode: \"{mode}\".")

    @dtype.validator
    def check_dtype(self, attr, dtype):
        msg = f"dtype={dtype} is unsupported for affine_quantize_weights."
        if not is_builtin(dtype):
            try:
                dtype = numpy_type_to_builtin_type(dtype)
            except TypeError:
                raise ValueError(msg)

        if dtype not in self._WEIGHT_AFFINE_DTYPES:
            raise ValueError(msg)

    def __attrs_post_init__(self):
        self.mode = self.mode.upper()
        if not is_builtin(self.dtype):
            self.dtype = numpy_type_to_builtin_type(self.dtype)

    @classmethod
    def _from_dict(cls, config_dict: Dict[str, Any]) -> "OpLinearQuantizerConfig":
        def _structure_type(value, dtype):
            if isinstance(value, type):
                return value
            else:
                if not isinstance(value, str) or value not in ("int8", "uint8"):
                    raise ValueError(
                        f'"dtype" must be type of type or str ["int8", "uint8"]. Got {value}'
                    )
                return getattr(np, value)

        converter = cattrs.Converter(forbid_extra_keys=True)
        converter.register_structure_hook(type, _structure_type)
        return converter.structure(config_dict, cls)

"""
Pruner configurations
"""
@define
class OpThresholdPrunerConfig(OpCompressorConfig):
    """
    All weights with absolute value smaller than ``threshold`` are changed to ``0``,
    and the tensor is stored in a sparse format.

    For example, given the following:

       * ``weight = [0.3, -0.2, -0.01, 0.05]``
       * ``threshold = 0.03``

    The sparsified weight would be ``[0.3, -0.2, 0, 0.05]``.

    Parameters
    ----------

    threshold: float
        All weight values above this threshold are set to ``0``.

        * Default value is ``1e-12``.

    minimum_sparsity_percentile: float
        The sparsity level must be above this value for the weight representation to be stored in the sparse format rather than the dense format.

        For example, if ``minimum_sparsity_percentile = 0.6`` and the sparisty level is ``0.54``; that is, ``54%`` of the
        weight values are exactly ``0``, then the resulting weight tensor will be stored as a dense const op,
        and not converted to the ``constsexpr_sparse_to_dense`` op (which stores the weight values in a sparse format).

        * Must be a value between ``0`` and ``1``.
        * Default value is ``0.5``.

    weight_threshold: int
        The size threshold, above which weights are pruned.
        That is, a weight tensor is pruned only if its total number of elements are greater than ``weight_threshold``.

        For example, if ``weight_threshold = 1024`` and a weight tensor is of shape ``[10, 20, 1, 1]``, hence ``200``
        elements, it will not be pruned.

        * If not provided, it will be set to ``2048``, in which weights bigger than ``2048`` elements are compressed.
    """

    threshold: float = field(default=1e-12, validator=validators.instance_of(float))
    minimum_sparsity_percentile: float = field(default=0.5, validator=validators.instance_of(float))
    weight_threshold: Optional[int] = field(
                                        default=2048,
                                        validator=validators.optional([validators.instance_of(int), _check_weight_threshold])
                                      )

    @threshold.validator
    def check_threshold(self, attr, threshold):
        if threshold < 0:
            raise ValueError(
                f"Invalid value of \"threshold\": {threshold}. Needs to be in [0, inf)"
            )

    @minimum_sparsity_percentile.validator
    def check_minimum_sparsity_percentile(self, attr, minimum_sparsity_percentile):
        if minimum_sparsity_percentile < 0 or minimum_sparsity_percentile > 1:
            raise ValueError(
                f"Invalid value of \"minimum_sparsity_percentile\": {minimum_sparsity_percentile}. Needs to be in [0, 1]"
            )

    @classmethod
    def _from_dict(cls, config_dict: Dict[str, Any]) -> "OpThresholdPrunerConfig":
        converter = cattrs.Converter(forbid_extra_keys=True)
        return converter.structure(config_dict, cls)

@define
class OpMagnitudePrunerConfig(OpCompressorConfig):
    """
    Prune the weight with a constant sparsity percentile, which can be specified by either ``target_sparsity`` or ``n_m_ratio``.

    If ``target_sparsity`` is set, where ``n = floor(size_of_weight_tensor * target_sparsity)``, the ``n`` lowest
    absolute weight values are changed to ``0``. For example, given the following:

       * ``weight = [0.3, -0.2, -0.01, 0.05]``
       * ``target_sparsity = 0.75``

    The sparsified weight would be ``[0.3, 0, 0, 0]``.

    If ``block_size`` is set, then weights are pruned in a block structured manner; that is, chunks of weight values, as big as the ``block_size``, will be set to ``0``.
    Block sparsity can only be applied to ``linear`` and ``conv`` layers.
    For example:

        .. code-block:: python

            # Given a 4 x 2 weight with the following value, and block_size = 2, dim = 0.
            [
                [1, 3],
                [-6, -7],
                [0, 3],
                [-9, 2],
            ]

            # We first flatten the matrix along axis = 0.
            [1, -6, 0, -9, 3, -7, 3, 2]

            # For block size 2, the L2 norm will be compute of first 2 elements, then the second and 3rd element and so on.
            [6.08, 9.00, 7.62, 3.61]

            # Then the smallest values will be picked to prune. So if target_sparsity = 0.5, then the blocks that will be
            # pruned will be with ones with L2 norm value of 6.08 and 3.61. And hence, the elements in the first and third
            # block are pruned. Resulting in the following flatten pruned tensor:
            [0, 0, 0, -9, 3, -7, 0, 0]

            # The final pruned tensor is:
            [
                [0, 3],
                [0, -7],
                [0, 0],
                [-9, 0],
            ]

    The ``n_m_ratio`` triggers ``n:m`` pruning along the ``dim`` axis. In ``n:m`` pruning,
    out of every ``m`` elements, ``n`` with lowest magnitude are set to ``0``.
    For more information, see
    `Learning N:M Fine-Grained Structured Sparse Neural Networks From Scratch <https://arxiv.org/abs/2102.04010>`_.

    ``n:m`` pruning can be applied only to ``linear`` and ``conv`` layers.

    Example:

        .. code-block:: python

            # Given a 4 x 4 weight of
            [
                [3, 4, 7, 6],
                [1, 8, -3, -8],
                [-2, -3, -4, 0],
                [5, 4, -3, -2],
            ]

            # For n_m_ratio = (1, 2) with axis = 1 (default), the resulting pruned weight is
            [
                [0, 4, 7, 0],
                [0, 8, 0, -8],
                [0, -3, -4, 0],
                [5, 0, -3, 0],
            ]

            # For axis = 0, we get
            [
                [3, 0, 7, 0],
                [0, 8, 0, -8],
                [0, 0, -4, 0],
                [5, 4, 0, -2],
            ]

    Parameters
    ----------

    target_sparsity: float
        The percentage of sparsity for compression, which needs to be in the range ``[0, 1]``. When ``0``, no sparsification
        occurs. For ``1``, all weights become ``0``.

    block_size: int
        Block size for inducing block sparsity.
        This is applied on the ``dim`` dimension of the parameter.
        Having the zeros aligned in the parameter helps gain latency/memory performance on-device.

        * If set, must be greater than ``1`` to enable block sparsity.
        * Block sparsity can be applied only to ``linear`` and ``conv`` layers.
        * The channel will be padded with ``0`` if it is not divisble by ``block_size``.

    n_m_ratio: tuple[int]
        A tuple of two integers which specify the ratio for ``n:m`` pruning.

        * ``n`` must be smaller or equal to ``m``.
        * The channel would be padded with ``0`` if it is not divisble by ``m``.

    dim: int
        Dimension where the block sparsity or ``n:m`` sparsity is applied.

        * Must be either ``0`` or ``1``.
        * The default value for block sparsity is ``0`` (output channel).
        * The default value for ``n:m`` sparsity is ``1`` (input channel).

    weight_threshold: int
        The size threshold, above which weights are pruned.
        That is, a weight tensor is pruned only if its total number of elements is greater than ``weight_threshold``.

        For example, if ``weight_threshold = 1024`` and a weight tensor is of shape ``[10, 20, 1, 1]``, hence ``200``
        elements, it will not be pruned.

        * If not provided, it will be set to ``2048``, in which weights bigger than ``2048`` elements are compressed.
    """
    target_sparsity: Optional[float] = field(default=None, validator=validators.optional(validators.instance_of(float)))
    block_size: Optional[int] = field(default=None, validator=validators.optional(validators.instance_of(int)))
    n_m_ratio: Optional[Tuple[int, int]] = field(default=None, validator=validators.optional(validators.instance_of((list, tuple))))
    dim: Optional[int] = field(default=None, validator=validators.optional(validators.instance_of(int)))
    weight_threshold: Optional[int] = field(
                                        default=2048,
                                        validator=validators.optional([validators.instance_of(int), _check_weight_threshold])
                                      )

    _SUPPORTED_OPS_FOR_STRUCTURAL_PRUNING = {
        "conv": ["weight"],
        "linear": ["weight"],
    }

    def _is_structural_pruning(self):
        return self.n_m_ratio is not None or self.block_size is not None

    def _validate_op_type(self, op_type):
        """
        Structural sparsity can only be applied to conv / linear weight.
        """
        if self._is_structural_pruning() and op_type not in self._SUPPORTED_OPS_FOR_STRUCTURAL_PRUNING:
            raise ValueError(f"block sparsity or n:m pruning does not support op type {op_type}.")

    def _check_const_op_is_valid(self, op):
        def _get_child_op_and_input(op):
            assert op.op_type == "const"
            res = []
            for child in op.outputs[0].child_ops:
                child_op_type = child.op_type
                child_op_input = ""
                for k, v in child.inputs.items():
                    if v is op.outputs[0]:
                        child_op_input = k
                        break
                assert child_op_input != ""
                res.append((child_op_type, child_op_input))
            return res

        if not self._is_structural_pruning():
            return True

        child_op_type_and_input = _get_child_op_and_input(op)
        for op_type, input in child_op_type_and_input:
            if op_type not in self._SUPPORTED_OPS_FOR_STRUCTURAL_PRUNING:
                return False
            if input not in self._SUPPORTED_OPS_FOR_STRUCTURAL_PRUNING[op_type]:
                return False

        return True

    @target_sparsity.validator
    def check_target_sparsity(self, attr, target_sparsity):
        msg = "Either \"target_sparsity\" or \"n_m_ratio\" need to be set. They cannot be set at the same time."
        if target_sparsity is not None and self.n_m_ratio is not None:
            raise ValueError(msg)
        if target_sparsity is None and self.n_m_ratio is None:
            raise ValueError(msg)

        if target_sparsity is None:
            return
        if target_sparsity < 0 or target_sparsity > 1:
            raise ValueError(
                f"Invalid value of \"target_sparsity\": {target_sparsity}. Needs to be in [0, 1]."
            )

    @block_size.validator
    def check_block_size(self, attr, block_size):
        if block_size is not None and self.n_m_ratio is not None:
            raise ValueError(
                "\"block_size\" and \"n_m_ratio\" cannot be set at the same time."
            )
        if block_size is None:
            return
        if  block_size is not None and block_size <= 1:
            raise ValueError(f"\"block_size\" must be an integer > 1. Got {block_size}.")

    @n_m_ratio.validator
    def check_n_m_ratio(self, attr, n_m_ratio):
        if n_m_ratio is None:
            return
        if len(n_m_ratio) != 2 or n_m_ratio[0] > n_m_ratio[1]:
            raise ValueError(f"\"n_m_ratio\" must be a tuple of two integers (n, m). n <= m. Got {n_m_ratio}")

    @dim.validator
    def check_dim(self, attr, dim):
        if dim is None:
            return
        if self.block_size is None and self.n_m_ratio is None:
            raise ValueError("\"dim\" can only be set along with \"block_size\" or \"n_m_ratio\".")
        if dim not in [0, 1]:
            raise ValueError(f"\"dim\" must be 1 or 0. Got {dim}.")

    def __attrs_post_init__(self):
        if self.block_size is not None and self.dim is None:
            self.dim = 0
        if self.n_m_ratio is not None and self.dim is None:
            self.dim = 1

    @classmethod
    def _from_dict(cls, config_dict: Dict[str, Any]) -> "OpMagnitudePrunerConfig":
        converter = cattrs.Converter(forbid_extra_keys=True)
        return converter.structure(config_dict, cls)

"""
Palettizer configuration
"""

@define
class OpPalettizerConfig(OpCompressorConfig):
    """
    Parameters
    ----------

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
           `bins`) and the representation displays how many values fall into each bin.
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
               lut = [0.0] + top_k.tolist()

               # Compute the indices
               mapping = {v: idx for idx, v in enumerate(lut)}
               indices = [mapping[v] if v in mapping else 0 for v in weight]

               return lut, indices

    weight_threshold: int
        The size threshold, above which weights are pruned.
        That is, a weight tensor is pruned only if its total number of elements are greater than ``weight_threshold``.

        For example, if ``weight_threshold = 1024`` and a weight tensor is of shape ``[10, 20, 1, 1]``, hence ``200``
        elements, it will not be pruned.

        * If not provided, it will be set to ``2048``, in which weights bigger than ``2048`` elements are compressed.
    """
    mode: str = field(default="kmeans", validator=validators.instance_of(str))
    nbits: Optional[int] = field(default=None)
    lut_function: Optional[Callable] = field(default=None)
    weight_threshold: Optional[int] = field(default=2048, validator=validators.optional([validators.instance_of(int), _check_weight_threshold]))

    _WEIGHT_PALETTIZATION_MODES = ("KMEANS", "UNIFORM", "UNIQUE", "CUSTOM")

    @nbits.validator
    def check_nbits(self, attr, nbits):
        mode = self.mode.upper()

        if nbits is None and mode in ("KMEANS", "UNIFORM"):
            raise ValueError(f"\"nbits\" must be provided for {self.mode} mode")

        if nbits is not None and mode in ("UNIQUE", "CUSTOM"):
            raise ValueError(f"\"nbits\" must NOT be provided for {self.mode} mode")

        if nbits is not None and nbits not in [1, 2, 4, 6, 8]:
            raise ValueError(
                f"Invalid value of \"nbits\" ({nbits}) for palettization. Supported \"nbits\" are {{1, 2, 4, 6, 8}}"
            )

    @mode.validator
    def check_mode(self, attr, mode):
        if not mode.upper() in self._WEIGHT_PALETTIZATION_MODES:
            raise ValueError(f"Only modes {self._WEIGHT_PALETTIZATION_MODES} are supported for weight palettization. Got \"mode\": \"{mode}\".")


    @lut_function.validator
    def check_lut_function(self, attr, lut_function):
        mode = self.mode.upper()

        if lut_function is None and mode == "CUSTOM":
            raise ValueError("\"lut_function\" can not be None, if \"mode\" is \"custom\".")

        if lut_function is not None and mode != "CUSTOM":
            raise ValueError("\"lut_function\" must be None, if \"mode\" is not \"custom\".")

        if lut_function is not None and not callable(lut_function):
            raise ValueError(f"A function object must be provided as \"lut_function\". Got a \"lut_function\" as type {type(self.lut_function)}")

    def __attrs_post_init__(self):
        self.mode = self.mode.upper()

    @classmethod
    def _from_dict(cls, config_dict: Dict[str, Any]) -> "OpPalettizerConfig":
        if "lut_function" in config_dict:
            raise ValueError(
                "_from_dict method does not support lut_function. Please create the OpPalettizerConfig from scratch."
            )
        converter = cattrs.Converter(forbid_extra_keys=True)
        return converter.structure(config_dict, cls)

@define
class OptimizationConfig:
    """
    A configuration wrapper that enables fine-grained control when compressing a model,
    Providing the following levels: `global`, `op type`, and `op name`.

    1. ``global_config``: The default configuration applied to all ops / consts.
    2. ``op_type_configs``: Configurations applied to specific op type. It overrides ``global_config``.
    3. ``op_name_configs``: Confgurations applied to specific constant or op instance. It overrides ``global_config`` and ``op_type_configs``.

    The following is an example that constructs an optimization config for weight palettization.

        .. code-block:: python

            from coremltools.optimize.coreml import OpPalettizerConfig, OptimizationConfig

            # The default global configuration is 8 bits palettization with kmeans
            global_config = OpPalettizerConfig(mode="kmeans", nbits=8)

            # We use 2 bits palettization for convolution layers, and skip the compression for linear layers
            op_type_configs = {
                "conv": OpPalettizerConfig(mode="kmeans", nbits=2),
                "linear": None,
            }

            # We want a convolution layer named "conv_1" to have a 4 bits palettization with a different mode
            op_name_configs = {
                "conv_1": OpPalettizerConfig(mode="uniform", nbits=4),
            }

            # Now we can put all configuration across three levels to construct an OptimizationConfig object
            config = OptimizationConfig(
                global_config=global_config,
                op_type_configs=op_type_configs,
                op_name_configs=op_name_configs,
            )


    Parameters
    ----------

    global_config: OpCompressorConfig
        Config to be applied globally to all supported ops.

    op_type_configs: dict[str, OpCompressorConfig]
        Op type level configs applied to a specific op class.

        * The keys of the dictionary are the string of the op type, and the values are the corresponding :py:class:`OpCompressorConfig`.
        * An op type will not be compressed if the value is set to ``None``.

    op_name_configs: dict[str, OpCompressorConfig]
        Op instance level configs applied to a specific constant or op.

        * The keys of the dictionary are the name of a constant or an op instance, and the values are the corresponding :py:class:`OpCompressorConfig`.
        * An op instance will not be compressed if the value is set to ``None``.
        * You can use ``coremltools.optimize.coreml.get_weights_metadata`` to get the name of the constants / op instances in the model.
    """
    global_config: Optional[OpCompressorConfig] = field(default=None)
    op_type_configs: Optional[OpCompressorConfig] = field(default=None)
    op_name_configs: Optional[OpCompressorConfig] = field(default=None)

    # The following two private attributes is aim for backward compatibility for ct.compression_utils implementation
    # They need to be removed in the future once we deprecate ct.compression_utils
    _is_deprecated: bool = field(default=False, validator=validators.instance_of(bool))
    _op_selector: Optional[Callable] = field(default=None)

    @staticmethod
    def _check_op_config_type(config):
        if config is None:
            return
        if not isinstance(config, OpCompressorConfig):
            raise ValueError(f"config must be type of OpCompressorConfig. Got {type(config)}.")

    def set_global(self, op_config: OpCompressorConfig):
        """
        Sets the global config that would be applied to all constant ops.

        .. code-block:: python

            from coremltools.optimize.coreml import OpPalettizerConfig, OptimizationConfig

            config = OptimizationConfig()
            global_config = OpPalettizerConfig(mode="kmeans", nbits=8)
            config.set_global(global_config)

        Parameters
        ----------

        op_config: OpCompressorConfig
            Config to be applied globally to all supported ops.
        """
        self._check_op_config_type(op_config)
        self.global_config = op_config

    def set_op_type(
        self,
        op_type: str,
        op_config: OpCompressorConfig,
    ):
        """
        Sets the compression config at the level of op type.

        .. code-block:: python

            from coremltools.optimize.coreml import OpPalettizerConfig, OptimizationConfig

            config = OptimizationConfig()
            conv_config = OpPalettizerConfig(mode="kmeans", nbits=2)
            config.set_op_type("conv", conv_config)

        Parameters
        ----------

        op_type: str
            The type of an op. For instance, ``"conv", "linear"``.

        op_config: OpCompressorConfig
            Op type level config applied to a specific op class ``op_type``.
        """
        if self._is_deprecated:
            raise ValueError("set_op_type is not exposed through the coremltools.compression_utils API.")
        self._check_op_config_type(op_config)
        if op_config is not None:
            op_config._validate_op_type(op_type)
        self.op_type_configs[op_type] = op_config

    def set_op_name(
        self,
        op_name: str,
        op_config: OpCompressorConfig,
    ):
        """
        Sets the compression config at the level of constant / op instance by name.

        .. code-block:: python

            from coremltools.optimize.coreml import OpPalettizerConfig, OptimizationConfig

            config = OptimizationConfig()
            op_config = OpPalettizerConfig(mode="kmeans", nbits=2)
            config.set_op_name("conv_1", op_config)

        Note that, in order to get the name of a constant or an op instance, please refer to the ``coremltools.optimize.coreml.get_weights_metadata`` API.

        Parameters
        ----------

        op_name: str
            The name of a constant or an op instance.

        op_config: OpCompressorConfig
            Op instance level config applied to a specific constant or op with name ``op_name``.
        """
        if self._is_deprecated:
            raise ValueError("set_op_name is not exposed through the coremltools.compression_utils API.")
        self._check_op_config_type(op_config)
        self.op_name_configs[op_name] = op_config

    @_is_deprecated.validator
    def check_is_deprecated(self, attr, _is_deprecated):
        if not _is_deprecated and self._op_selector is not None:
            raise ValueError("op_selector is supported only through the coremltools.compression_utils API.")

    @op_type_configs.validator
    def check_op_type_configs(self, attr, op_type_configs):
        if op_type_configs is None:
            return
        for v in op_type_configs.values():
            self._check_op_config_type(v)
        for k, v in op_type_configs.items():
            if v is not None:
                v._validate_op_type(k)

    @op_name_configs.validator
    def check_op_name_configs(self, attr, op_name_configs):
        if op_name_configs is None:
            return
        for v in op_name_configs.values():
            self._check_op_config_type(v)

    @global_config.validator
    def check_global_configs(self, attr, global_config):
        if global_config is None:
            return
        self._check_op_config_type(global_config)


    def _get_op_config(self, op: Operation):
        """
        This utility function retrieve the compression config for an non-const Operation instance.
        The priority is by: op name -> op type -> global
        """
        if not isinstance(op, Operation):
           raise TypeError(f"op must be type of Operation. Got {type(op)}")

        if op.op_type == "const":
            raise TypeError("op must not be of type const")

        if op.name in self.op_name_configs:
            return self.op_name_configs[op.name]
        elif op.op_type in self.op_type_configs:
            return self.op_type_configs[op.op_type]

        return self.global_config

    def _get_const_op_config(self, op: Operation):
        """
        This utility function retrieves the compression config by an const Operation instance.
        If the const is fed into multiple operations, an error would be thrown if a conflict is detected.
        """
        if not isinstance(op, Operation):
            raise TypeError(f"op must be type of Operation. Got {type(op)}")

        if op.op_type != "const":
            raise TypeError(f"op must be of type const. Got {op.op_type}")

        if op.name in self.op_name_configs:
            return self.op_name_configs[op.name]

        if op.op_type in self.op_type_configs:
            # We don't allow users to call set_op_type for "const" ops.
            # The users are supposed to use set_global instead
            raise ValueError("const ops cannot be set by the `set_op_type` function. Please use `set_global`")

        # If the constant's output is only connected to the block output, we don't do compression
        # Due to this bug: rdar://108274019 ([Bug] constexpr ops cannot be directly fed to block output)
        child_ops = op.outputs[0].child_ops
        if len(child_ops) == 0:
            return None

        op_configs = [self._get_op_config(op) for op in child_ops]

        for i, config in enumerate(op_configs):
            if config != op_configs[0]:
                raise ValueError(
                    f"compression config conflict detected between ops {child_ops[0]} and {child_ops[i]}. "
                    f"{child_ops[0]} has config {op_configs[0]} while {child_ops[i]} has {config}."
                )
        return op_configs[0]

    def __attrs_post_init__(self):
        if self.op_type_configs is None:
            self.op_type_configs = {}
        if self.op_name_configs is None:
            self.op_name_configs = {}

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "OptimizationConfig":
        """
        Construct an ``OptimizationConfig`` instance from a nested dictionary.
        The dictionary should have the structure that only contains (if any) the following four ``str`` keys:

        * ``"config_type"``: Specify the configuration class type.
        * ``"global_config"``: Parameters for ``global_config``.
        * ``"op_type_configs"``: A nested dictionary for ``op_type_configs``.
        * ``"op_name_config"``: A nested dictionary for ``op_name_configs``.

        The following is a nested dictionary that creates an optimization config for weight palettization:

        .. code-block:: python

            config_dict = {
                "config_type": "OpPalettizerConfig",
                "global_config": {
                    "mode": "kmeans",
                    "nbits": 4,
                },
                "op_type_configs": {
                    "conv": {
                        "mode": "uniform",
                        "nbits": 1,
                    }
                },
                "op_name_configs": {
                    "conv_1": {
                        "mode": "unique",
                    }
                },
            }

        Note that you can override the ``config_type``. For instance, if you want to do threshold-based
        pruning to the model in addition to the convolution layers in which magnitude pruning is applied, the following is an
        example of the nested dictionary:

        .. code-block:: python

            config_dict = {
                "config_type": "OpThresholdPrunerConfig",
                "global_config": {
                    "threshold": 0.01,
                },
                "op_type_configs": {
                    "conv": {
                        "config_type": "OpMagnitudePrunerConfig",
                        "n_m_ratio": [3, 4],
                    }
                },
            }

        Parameters
        ----------

        config_dict: dict[str, Any]
            A dictionary that represents the configuration structure.
        """
        def _get_cls_instance(cls_type, cls_attrs):
            if cls_attrs is None:
                return None
            converter = cattrs.Converter(forbid_extra_keys=True)
            if "config_type" in cls_attrs:
                cls_type = cls_attrs["config_type"]
                del cls_attrs["config_type"]
            class_type = getattr(sys.modules[__name__], cls_type)
            return class_type._from_dict(cls_attrs)

        def _check_config_dict(config_dict):
            valid_keys = ("config_type", "global_config", "op_name_configs", "op_type_configs")
            for k in config_dict:
                if k not in valid_keys:
                    raise ValueError(
                        f"Invalid key {k} to construct an OptimizationConfig object. Supported keys are {valid_keys}."
                    )

        _check_config_dict(config_dict)

        config_type = config_dict.get("config_type", None)
        if config_type is None or not isinstance(config_type, str):
            raise ValueError("config_type must be provided with type of string.")

        cls_attrs = {}
        if config_dict.get("global_config", None) is not None:
            cls_attrs["global_config"] = _get_cls_instance(
                config_type, config_dict["global_config"]
            )
        for key in ["op_type_configs", "op_name_configs"]:
            if config_dict.get(key, None) is None:
                continue
            if not isinstance(config_dict[key], dict):
                raise ValueError(f"{key} must be type of dict. Got {type(config_dict[key])}")
            cls_attrs[key] = {
                k: _get_cls_instance(config_type, v) for k, v in config_dict[key].items()
            }

        return cls(**cls_attrs)

    @classmethod
    def from_yaml(cls, yml: Union[IO, str]) -> "OptimizationConfig":
        """
        Construct an ``OptimizationConfig`` instance from a YAML file.
        The YAML file should have the structure that only contains (if any) the following four ``str`` keys:

        * ``"config_type"``: Specify the configuration class type.
        * ``"global_config"``: Parameters for ``global_config``.
        * ``"op_type_configs"``: A nested dictionary for ``op_type_configs``.
        * ``"op_name_config"``: A nested dictionary for ``op_name_configs``.

        The following is a YAML file that creates an optimization config for weight palettization:

        ::

            config_type: OpPalettizerConfig
            global_config:
                mode: kmeans
                nbits: 4
            op_type_configs:
                conv:
                    mode: uniform
                    nbits: 1
            op_name_configs:
                conv_1:
                    mode: unique

        Note that you can override the ``config_type``. For instance, if you want to do threshold-based
        pruning to the model in addition to the convolution layers in which magnitude pruning is applied, the following is an
        example of the YAML file:

        ::

            config_type: OpThresholdPrunerConfig
            global_config:
                threshold: 0.01
            op_type_configs:
                conv:
                    config_type: OpMagnitudePrunerConfig
                    n_m_ratio: [3, 4]

        Parameters
        ----------

        yml: str, IO
            A YAML file or the path to the file.
        """
        if isinstance(yml, str):
            with open(yml, "r") as file:
                config_dict = yaml.safe_load(file)
        else:
            config_dict = yaml.safe_load(yml)
        return cls.from_dict(config_dict)

class _MetaDataDict(OrderedDict):
    """
    A dictionary class with nice print out str
    """

    def __init__(self, mapping=None, str_prefix=""):
        super().__init__(mapping)
        self._str_prefix = str_prefix

    def __str__(self):
        res = ""
        for k, v in self.items():
            res += f"{self._str_prefix}{k}\n"
            res += f"{v}\n"
        return res
