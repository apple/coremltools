#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import math
from collections import OrderedDict
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import attrs
import numpy as np
import torch as torch
from torch.jit._script import RecursiveScriptModule

from coremltools import _logger as logger
from coremltools._deps import _HAS_TORCH_EXPORT_API
from coremltools.converters.mil import mil
from coremltools.converters.mil._deployment_compatibility import AvailableTarget as _target
from coremltools.converters.mil.frontend import _utils as frontend_utils
from coremltools.converters.mil.input_types import (
    EnumeratedShapes,
    ImageType,
    InputType,
    StateType,
    TensorType,
)
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Function, Placeholder, Program, types
from coremltools.converters.mil.mil.block import is_current_opset_version_compatible_with
from coremltools.converters.mil.mil.scope import ScopeInfo, ScopeSource
from coremltools.converters.mil.mil.types import builtin_to_string, is_float
from coremltools.converters.mil.mil.types.symbolic import any_symbolic, is_symbolic
from coremltools.converters.mil.mil.var import Var
from coremltools.optimize.coreml import _utils as optimize_utils
from coremltools.optimize.coreml._quantization_passes import prune_weights

from .exir_utils import WRAPPED_SCALAR_INPUT_SUFFIX
from .internal_graph import InternalTorchIRGraph, InternalTorchIRNode
from .ops import convert_nodes
from .quantization_ops import _dequantized_weight
from .torch_op_registry import _TORCH_OPS_REGISTRY
from .torchir_passes import (
    flatten_graph_input_values,
    flatten_graph_output_values,
    generate_tensor_assignment_ops,
    populate_native_const_model_hierarchy,
    remove_getattr_nodes,
    transform_inplace_ops,
)
from .utils import (
    NUM_TO_NUMPY_DTYPE,
    TORCH_DTYPE_TO_MIL_DTYPE,
    TORCH_DTYPE_TO_NUM,
    TORCH_EXPORT_BASED_FRONTENDS,
    TorchFrontend,
)

if _HAS_TORCH_EXPORT_API:
    from torch.export import ExportedProgram


# The compression info is stored in state_dict with the prefix, e.g. "dense2._COREML_n_bits".
_COMPRESSION_INFO_PREFIX = "_COREML_"


# TODO: Share the enum between cto.coreml and cto.torch (rdar://124409664).
class CompressionType(Enum):
    PRUNING = 1
    PALETTIZATION = 2
    QUANTIZATION = 3


@attrs.define(kw_only=True)
class CompressionInfo:
    """
    This class stores the compression info carried by the traced torch model.
    """

    # Quantization related fields.
    quantization_n_bits: Optional[int] = attrs.field(
        default=None,
        validator=attrs.validators.optional([attrs.validators.instance_of(int)]),
        converter=attrs.converters.optional(int),
    )
    quantization_scale: Optional[torch.Tensor] = attrs.field(
        default=None,
        validator=attrs.validators.optional([attrs.validators.instance_of(torch.Tensor)]),
    )
    zero_point: Optional[torch.Tensor] = attrs.field(
        default=None,
        validator=attrs.validators.optional([attrs.validators.instance_of(torch.Tensor)]),
    )

    # Palettization related fields.
    lut: Optional[torch.Tensor] = attrs.field(
        default=None,
        validator=attrs.validators.optional([attrs.validators.instance_of(torch.Tensor)]),
    )
    palettization_scale: Optional[torch.Tensor] = attrs.field(
        default=None,
        validator=attrs.validators.optional([attrs.validators.instance_of(torch.Tensor)]),
    )
    vector_axis: Optional[int] = attrs.field(
        default=None,
        validator=attrs.validators.optional([attrs.validators.instance_of(int)]),
        converter=attrs.converters.optional(int),
    )

    # Compression type indication fields.
    compression_type: Optional[List[int]] = attrs.field(
        default=None,
        converter=attrs.converters.optional(lambda tensor: tensor.tolist()),
    )

    @quantization_n_bits.validator
    def check_n_bits(self, attribute, n_bits):
        if n_bits is not None and not 1 <= n_bits <= 8:
            raise ValueError(f"Only support quantization_n_bits between 1 and 8, but got {n_bits}")

    @compression_type.validator
    def check_compression_type(self, attribute, compression_type):
        if compression_type is not None:
            if not all(isinstance(type_val, int) for type_val in compression_type):
                raise ValueError(
                    f"Only support int compression_type, but got {type(compression_type)}"
                )


def _convert_to_torch_inputtype(
    inputs: List[TensorType], allow_default_shape: bool = True
) -> List[TensorType]:
    input_type = []
    for _input in inputs:
        if isinstance(_input, (list, tuple)):
            input_type.append(_convert_to_torch_inputtype(_input))
        elif isinstance(_input, InputType):
            if _input.shape is None and not allow_default_shape:
                raise ValueError(
                    "'shape' must be provided in the 'inputs' argument for pytorch conversion"
                )
            input_type.append(_input)
        elif isinstance(_input, torch.Tensor):
            input_type.append(
                TensorType(shape=_input.shape, dtype=TORCH_DTYPE_TO_MIL_DTYPE[_input.dtype])
            )
        else:
            raise ValueError("Unknown type {} for conversion to InputType.".format(type(_input)))
    return input_type


class QuantizationContext:
    """
    Utilities to manage information pertaining to quantization of tensors in a PyTorch graph.

    This is necessary only for TorchScript (not ExecuTorch)
    """

    def __init__(self, context: "TranscriptionContext") -> None:
        if context.frontend != TorchFrontend.TORCHSCRIPT:
            raise ValueError("QuantizationContext is necessary only for TorchScript")
        self._context = context

        # Maps var name to tuple of (torch dtype, scale, zero_point)
        # zero_point is in a NumPy dtype corresponding to torch one (for e.g. np.uint8 for torch.quint8).
        self._quant_param_map = {}
        # In MIL Programs, if a MIL op doesn't support quantized I/O but the PyTorch ops do,
        # we just use floating-point tensors after dequantization. This means that information about
        # what dtype (int8/uint8) quantized tensors had in the PyTorch graph is not carried into
        # in the MIL graph.
        # To simplify, we only support a single dtype for activation quantizations throughout the
        # incoming graph.
        # The other option is to remember dtypes across ops, including MIL ones that don't support
        # quantized I/O. We will need to be careful about edge cases like conflicting dtypes, etc.
        self._quant_dtype = None

    def add_quantization_info(self, name, torch_dtype, scale, zero_point, axis=None):
        """
        Stores the quantization parameters (torch dtype, scale, zero_point) corresponding to a named
        var in the graph.
        zero_point should be in a NumPy dtype corresponding to torch one (for e.g. np.uint8 for torch.quint8).
        """
        self._quant_param_map[name] = (torch_dtype, scale, zero_point, axis)

    def get_quantization_info(self, name: str) -> None:
        """
        Retrieves the information added via add_quantization_info, if applicable.
        Returns None if quantization parameters could not be found.
        """
        if name not in self._quant_param_map:
            return None
        return self._quant_param_map[name]

    def maybe_handle_quantized_inputs(self, node: InternalTorchIRNode) -> None:
        """
        If a node's op doesn't support quantized inputs but gets one, this will wire it to
        receive a dequantized version of it.
        """

        op_type = node.kind
        if op_type in {"quantize_per_tensor", "dequantize"} or "quantized::" in op_type:
            # Op can handle quantized inputs. Nothing to do here.
            return

        for input in node.inputs:
            # In EXIR, input can be a literal and thus have no name
            if not isinstance(input, str) or self.get_quantization_info(input) is None:
                # Not a quantized tensor
                continue

            # We need a dequantized version of the input to feed to the op.
            dequantized_var, _ = self.get_dequantized_var(input)
            node.replace_name(input, dequantized_var.name)

    def get_quantized_per_tensor(self, name, torch_dtype, scale, zero_point, quantized_name):
        """
        Quantizes the provided named var as per quantization params.
        zero_point will be cast to the appropriate dtype based on torch_dtype.
        """
        if self._quant_dtype is None:
            self._quant_dtype = torch_dtype
        elif self._quant_dtype != torch_dtype:
            raise NotImplementedError(
                "Currently we only support a single activation dtype throughout the model"
            )

        if torch_dtype == torch.quint8:
            zero_point = np.uint8(zero_point)
            output_dtype = "uint8"
        elif torch_dtype == torch.qint8:
            zero_point = np.int8(zero_point)
            output_dtype = "int8"
        else:
            raise ValueError(f"Invalid torch dtype for quantization: {torch_dtype}")
        if np.isscalar(zero_point):
            # MIL allows skipping zero_point if its zero.
            if zero_point == 0:
                zero_point = None
            # TODO (rdar://107718371): skip 128 for uint8 by switching to int8

        result = mb.quantize(
            input=self._context[name], zero_point=zero_point, scale=scale, output_dtype=output_dtype
        )
        self._context.add(result, quantized_name)
        self._context.quant_context.add_quantization_info(
            quantized_name, torch_dtype, scale, zero_point
        )
        return result

    def get_dequantized_var(self, name: str, dequantized_name: str = None):
        """
        Returns dequantized var & torch dtype corresponding to the named var.
        """

        original_var = self._context[name]
        if is_float(original_var.dtype):
            # Input doesn't need dequantization.
            # This might happen if in the PyTorch graph the upstream nodes supported quantized inputs,
            # but MIL does not. In that case, we already dequantized the vars before feeding them to
            # the MIL op.
            if dequantized_name is not None:
                self._context.add(original_var, dequantized_name)
            return original_var, self._quant_dtype

        quant_params = self.get_quantization_info(name)
        if quant_params is None:
            raise ValueError(
                f"Could not find quantization parameters for quantized var {original_var.name}"
            )
        torch_dtype, scale, zero_point, axis = quant_params

        # We add a new var corresponding to each dequantized value.
        # This ensures the atomicity of quantized op patterns in MIL.
        dequantized_var = mb.dequantize(
            input=original_var, scale=scale, zero_point=zero_point, axis=axis
        )
        if dequantized_name is not None:
            dequantized_var_name = dequantized_name
        else:
            dequantized_var_name = dequantized_var.name
        self._context.add(dequantized_var, dequantized_var_name)

        return dequantized_var, torch_dtype


class TranscriptionContext:
    """
    Maintains a map from torch operations to their MIL values
    while building the graph. Can be used to process subgraphs recursively
    by pushing new context when stepping into a subgraph and popping that
    context when stepping out.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        frontend: TorchFrontend = TorchFrontend.TORCHSCRIPT,
    ) -> None:
        self.name = name if name else ""
        self.frontend = frontend
        self._current_graph = [{}]
        self._torch_graph = None
        if frontend == TorchFrontend.TORCHSCRIPT:
            self._quant_context = QuantizationContext(self)
        # Dict to map a var's name into its corresponding source state var.
        self.name_to_source_state = dict()

    @property
    def torch_graph(self):
        if self._torch_graph is None:
            raise ValueError("InternalTorchIRGraph not set yet on context")
        return self._torch_graph

    @property
    def quant_context(self) -> QuantizationContext:
        return self._quant_context

    @torch_graph.setter
    def torch_graph(self, graph: InternalTorchIRGraph):
        self._torch_graph = graph

    def convert_input_to_tensor_type(self, node: InternalTorchIRNode) -> None:
        """
        Convert non-tensor type input of a node into tensor type.

        This utility check if the input is a function state input, and
        convert it into a tensor type.

        For instance, given the following torchscript graph:

            %x(state, fp16), %y(tensor, fp32) -> {
                %1 = add(%x, %y)
            }

        The graph is translated into:

            %x(state, fp16), %y(tensor, fp32) -> {
                %read_x = read_state(%x)
                %read_x_cast = cast(%read_x, "fp32")
                %1 = add(%read_x_cast, %y)
            }

        ``%read_x_cast`` is cached in ``name_to_source_state``, to make sure one
        state feeds into only one ``read_state`` op.
        """

        for val in node.inputs:
            if val is None:
                continue
            if val not in self:
                continue
            in_node = self[val]
            if in_node is None or not isinstance(in_node, Var):
                continue
            if types.is_state(in_node.sym_type):
                self.name_to_source_state[val] = self[val]
                assert (
                    in_node.op is None
                ), f"A state type var must come from a placeholder. Got parent op {in_node.op.op_type} instead."
                read_state = mb.read_state(input=in_node)
                read_state_fp32 = mb.cast(x=read_state, dtype="fp32")
                self.add(read_state_fp32, torch_name=val, override=True)
        return

    def process_inplace_op(self, node: InternalTorchIRNode) -> None:
        """
        This utility:

        1. adds ``mb.coreml_update_state`` after each torch inplace ops.
        2. adjusts the dtype across state / tensor.

        In torch, inplaces ops have the following properties:

        1. op type has the suffix of ``_``. For instance, ``add_``, ``mul_``, etc.
        2. The op does an inplace update for the first input tensor.

        For instance, the following syntax of a TorchScript:

            %3 = add_(%1, %2)

        denotes an inplace ``add`` operation on the ``%1`` tensor. The memory buffer
        of ``%1`` is updated and returned as a reference ``%3``.

        Here are the steps what this utility does, lets use the above
        simple torch script as an example, after adding the ``add_`` in the context,
        we currently have a MIL graph as ``%3 = add(x=%1, y=%2)``:

        1. Validate the first input (``%1``) comes from a state source by checking if the tensor's name ``1`` is in ``name_to_source_state``. If not, this utility does nothing.
        2. Say ``name_to_source_state["1"] = %state``. ``%state, %3`` can potentially has different dtype.
           For instance, the user could specify ``%state`` in fp16, while
           the MIL program in the front end conversion stage is
           still in fp32. Hence we cast ``%3`` into ``%state``'s dtype:

           (%state: fp16) -> {
                ...
                %3_<fp32> = add(x=%1, y=%2)
                %3_cast<fp16> = cast(x=%3_, dtype="fp16")
            }
        3. Insert a ``coreml_update_state`` and cast the output back to ``%3``'s original dtype:

           (%state: fp16) -> {
                ...
                %3_<fp32> = add(x=%1, y=%2)
                %3_cast<fp16> = cast(x=%3_, dtype="fp16")
                %3_update<fp16> = coreml_update_state(state=%state, value=%3_cast)
                %3<fp32> = cast(x=%3_update, dtype="fp32")
            }
        4. Set ``name_to_source_state["3"] = %state``, so the state chain can be used in the downstream.

        The below Torch Script model,

            (%state: fp16) -> {
                ...
                %3 = add_(%1, %2)
                %out = sub_(%3, %4)
            }

        will result in:

           (%state: fp16) -> {
                %1_<fp16> = read_state(%state)
                %1<fp32> = cast(x=%1_, dtype="fp32")
                %3_<fp32> = add(x=%1, y=%2)
                %3_cast<fp16> = cast(x=%3_, dtype="fp16")
                %3_update<fp16> = coreml_update_state(state=%state, value=%3_cast)
                %3<fp32> = cast(x=%3_update, dtype="fp32")
                %out_<fp32> = sub(x=%3, y=%4)
                %out_cast<fp16> = cast(x=%out_, dtype="fp16")
                %out_update<fp16> = coreml_update_state(state=%state, value=%out_cast)
                %out<fp32> = cast(x=%out_update, dtype="fp32")
            }

        Please note that, the intermediate ``cast`` ops would be removed
        by the ``add_fp16_cast`` + ``cast_optimization`` graph passes:

           (%state: fp16) -> {
                %1<fp16> = read_state(%state)
                %3_<fp16> = add(x=%1, y=%2)
                %3<fp16> = coreml_update_state(state=%state, value=%3_)
                %out_<fp16> = sub(x=%3, y=%4)
                %out<fp16> = coreml_update_state(state=%state, value=%out_)
            }

        """
        assert self.frontend == TorchFrontend.TORCHSCRIPT, "Only torch script has no in-place op"

        if len(node.inputs) == 0:
            return

        if node.inputs[0] not in self.name_to_source_state:
            return

        source_state = self.name_to_source_state[node.inputs[0]]
        self.name_to_source_state[node.name] = source_state
        value_node = self[node.name]
        cast_value = mb.cast(x=value_node, dtype=builtin_to_string(source_state.dtype))
        update = mb.coreml_update_state(state=source_state, value=cast_value)
        cast_update = mb.cast(x=update, dtype=builtin_to_string(value_node.dtype), name=node.name)
        self.add(cast_update, torch_name=node.name, override=True)

    def add(self, ssa_var: Var, torch_name: Optional[str] = None, override=False) -> None:
        """
        Arguments:
            ssa_var: Variable to add to the graph being constructed.
            torch_name: Optional unique string identifier of the operation. If
                omitted, it will use @ssa_var.name.
        """
        if torch_name is None:
            torch_name = ssa_var.name
        if torch_name in self._current_graph[-1] and not override:
            logger.warning(f"Torch var {torch_name} is added again.")
            return
        self._current_graph[-1][torch_name] = ssa_var

    def __getitem__(self, torch_name: str) -> Var:
        """
        Lookup a name in the context. Note that since nested blocks must be
        able to access anything that was defined before them, we have to
        search all contexts for a name, starting with the most local scope.
        """
        for idx in reversed(range(len(self._current_graph))):
            current_graph = self._current_graph[idx]
            if torch_name in current_graph:
                return self._current_graph[idx][torch_name]
        raise ValueError(f"Torch var {torch_name} not found in context {self.name}")

    def __contains__(self, torch_name):
        """Returns whether or not the torch var exist in context."""
        for idx in reversed(range(len(self._current_graph))):
            current_graph = self._current_graph[idx]
            if torch_name in current_graph:
                return True
        return False

    def push(self, inputs=None):
        """
        Add another frame to the context. Optionally provide a tuple of
        (name list, Var list) to populate the new context frame.
        """
        self._current_graph.append({})

        if inputs is not None:
            if len(inputs[0]) != len(inputs[1]):
                raise ValueError("name list and Var list must be the same length")
            for name, var in zip(inputs[0], inputs[1]):
                self.add(var, torch_name=name)

    def pop(self):
        """
        Remove and discard the top context frame.
        """
        self._current_graph = self._current_graph[:-1]

    def __str__(self):
        _str = ""
        for current_graph in reversed(self._current_graph):
            __str = ""
            for k, v in current_graph.items():
                if hasattr(v, "shape_str"):
                    shape_str = v.shape_str()
                elif hasattr(v, "sym_shape"):
                    shape_str = v.sym_shape()
                else:
                    shape_str = "None"
                __str += f"%{k} : {shape_str}\n"
            _str += __str + "\n"
        return _str

    def __repr__(self):
        return str(self)


class TorchConverter:
    """
    Class that handles conversion of pytorch models to the MIL format.

    Models passed to the @TorchConverter go from:
    Loaded-Torch Model -> Internal Graph -> PyMIL
    """

    def __init__(
        self,
        loaded_model: Union[RecursiveScriptModule, "ExportedProgram"],
        inputs: Optional[List[TensorType]] = None,
        outputs: Optional[List[TensorType]] = None,
        cut_at_symbols: Optional[List[str]] = None,
        opset_version: Optional[int] = None,
        use_default_fp16_io: bool = False,
        states: Optional[List[StateType]] = None,
    ) -> None:
        """
        Arguments:
            loaded_model: It could be one of the following:
                    - In-memory TorchScript model of type torch.jit.ScriptModule
                    - In-memory EXIR program of type ExportedProgram
            inputs: Input values and optional names. See kwarg in load.py for full description.
            outputs: List of outputs as ct.InputType. See kwarg in load.py for full description.
            cut_at_symbols: A list of internal symbol name strings. Graph conversion will
                terminate once these symbols have been generated. For debugging use
                only. See kwarg in load.py.
            opset_version: An int represents the Core ML opset version.
            use_default_fp16_io (optional): bool. Defaults to False.
                When minimum_deployment_target set >= ct.target.iOS16 (the same as ct.target.macOS13),
                and the compute precision set to fp16, this flag is True.
                When True, fp32 i/o defaults to fp16.
        """
        self.use_default_fp16_io = use_default_fp16_io

        # process inputs
        if inputs is None:
            inputs = []
        allow_default_shape = _HAS_TORCH_EXPORT_API and isinstance(loaded_model, ExportedProgram)
        self.inputs = _convert_to_torch_inputtype(inputs, allow_default_shape=allow_default_shape)
        for idx, inp in enumerate(self.inputs):
            if isinstance(inp, ImageType) and self.inputs[idx].channel_first is None:
                self.inputs[idx].channel_first = True

        # process states
        if states is None:
            states = []
        self.states = states

        if self.use_default_fp16_io:
            # If the input type is not specified by the user and use_default_fp16_io
            # is True. Make the default input type to fp16
            self._adjust_default_input_to_fp16()

        self.outputs = outputs
        self.output_names = frontend_utils.get_output_names(self.outputs)
        self.opset_version = _target(opset_version) if opset_version is not None else None
        self._prog = mil.Program()

        if isinstance(loaded_model, torch.jit.ScriptModule):
            self.context = TranscriptionContext(frontend=TorchFrontend.TORCHSCRIPT)
            self.graph = InternalTorchIRGraph.from_torchscript(
                torchscript=loaded_model, inputs=self.inputs, cut_at_symbols=cut_at_symbols
            )

            # TODO (rdar://106161395): Register Torch IR passes and unify them into the pass pipeline.
            # Apply Torch IR passes
            passes = [
                transform_inplace_ops,
                flatten_graph_input_values,
                flatten_graph_output_values,
                remove_getattr_nodes,
                generate_tensor_assignment_ops,
                populate_native_const_model_hierarchy,
            ]
            for p in passes:
                p(self.graph)

            # finalize inputs after internal graph gets settled
            self.inputs = list(self.graph.inputs.values())

        elif _HAS_TORCH_EXPORT_API and isinstance(loaded_model, ExportedProgram):
            if loaded_model.dialect == "ATEN":
                frontend = TorchFrontend.TORCHEXPORT
            elif loaded_model.dialect == "EDGE":
                frontend = TorchFrontend.EXECUTORCH
            else:
                raise NotImplementedError(
                    "Conversion for models with only ATEN or EDGE dialect is supported/tested. "
                    f"Provided Dialect: {loaded_model.dialect}"
                )
            self.context = TranscriptionContext(frontend=frontend)
            self.graph = InternalTorchIRGraph.from_exir(
                exir=loaded_model, cut_at_symbols=cut_at_symbols
            )

            # finalize inputs after internal graph gets settled
            self.inputs = self._match_user_exir_inputs(inputs)

            if states is None or len(states) == 0:
                # For torch.export, we default to create states from torch mutable buffers
                self.states = []
                for name, tensor in self.graph.buffers.items():
                    dtype = NUM_TO_NUMPY_DTYPE[TORCH_DTYPE_TO_NUM[tensor.dtype]]
                    if dtype != np.float16:
                        logger.warning(
                            "Core ML only supports fp16 states, "
                            f"so buffer {name} has been cast to fp16"
                        )
                        dtype = np.float16
                    state = StateType(
                        wrapped_type=TensorType(shape=tensor.shape, dtype=dtype), name=name
                    )
                    self.states.append(state)

        else:
            raise ValueError(
                "Model should be an instance of either torch.jit.ScriptModule or ExportedProgram"
            )

        self.context.torch_graph = self.graph
        self._validate_states()

        # Store the mapping from parameter name (such as "dense1.weight") to the compression info.
        self.param_to_compression_info: Dict[str, CompressionInfo] = dict()
        if self.opset_version is not None and self.opset_version >= _target.iOS16:
            # Notice that even the compression info in registered buffer is kept in self.graph,
            # we still want to explicitly construct it here, to make it useful for both TorchScript
            # and ExportedProgram.
            state_dict = loaded_model.state_dict
            self.param_to_compression_info = self._construct_compression_info(
                state_dict() if callable(state_dict) else state_dict
            )
            if self.context.frontend in TORCH_EXPORT_BASED_FRONTENDS:
                # For EXIR, all param names are lifted as input names (in the format of `argx_x`), so we need to
                # change names accordingly to make sure the compression info could be found later.
                for (
                    arg_name,
                    param_name,
                ) in loaded_model.graph_signature.inputs_to_parameters.items():
                    if param_name in self.param_to_compression_info:
                        self.param_to_compression_info[arg_name] = self.param_to_compression_info[
                            param_name
                        ]
                        del self.param_to_compression_info[param_name]

    def _match_user_exir_inputs(self, user_inputs: List[TensorType]) -> List[TensorType]:
        """
        check consistency between user-specified `InputType`s and EXIR inputs
        inherit missing user specifications from EXIR
        """
        if user_inputs is None or len(user_inputs) == 0:
            # user did not specify inputs, default to use EXIR inputs
            return list(self.graph.inputs.values())
        if len(user_inputs) != len(self.graph.inputs):
            raise ValueError("Inconsistent number of inputs between user and EXIR specifications")

        for user_input, (_, exir_input) in zip(user_inputs, self.graph.inputs.items()):
            # user specified shape, then check consistency with EXIR
            if user_input.shape is not None:
                if isinstance(user_input.shape, EnumeratedShapes):
                    user_shapes = [enum_shape.shape for enum_shape in user_input.shape.shapes]
                else:
                    user_shapes = [tuple(user_input.shape.to_list(allow_symbolic=True))]
                exir_shape = tuple(exir_input.shape.to_list(allow_symbolic=True))
                for user_shape in user_shapes:
                    for user_size, exir_size in zip(user_shape, exir_shape):
                        # Dynamic size can be changed (almost) arbitrarily
                        # Static size, however, is dangerous to change, since the EXIR graph
                        # is very likely to have been specialized using the static size value
                        if not is_symbolic(exir_size) and user_size != exir_size:
                            raise ValueError(
                                f"inconsistent shape between "
                                f"EXIR input {exir_input.name} shape {exir_shape} and "
                                f"user specified input {user_input.name} shape {user_shape}"
                            )
            # shape not specified, inherit from EXIR
            else:
                user_input.shape = exir_input.shape

            # inherit dtype from EXIR if not specified
            if user_input.dtype is None:
                user_input.dtype = exir_input.dtype

            # inherit name from EXIR if not specified
            if user_input.name is None:
                user_input.name = exir_input.name

        return user_inputs

    def _validate_states(self) -> None:
        """
        Validate that the user provided states is consistent with the
        registered buffer in the torch model, and add states to inputs
        """
        if len(self.states) > 0:
            for state in self.states:
                if state.name is None or state.name not in self.graph.buffers:
                    raise ValueError(
                        f"StateType named {state.name} not provided or "
                        "not found in the source torch model. "
                        "Please make sure the name in "
                        "'ct.StateType(name=..., wrapped_type=ct.TensorType(...))' "
                        f"match the 'named_buffers()' in the source torch model: {list(self.graph.buffers.keys())}"
                    )

                state_shape = tuple(state.shape.symbolic_shape)
                buffer_shape = tuple(self.graph.buffers[state.name].size())
                # If Core ML state has fixed shape, then we make sure it matches torch buffer shape
                # Note: Although dynamic-shape state does not make sense at runtime,
                #       for flexibility in graph manipulation, pymil allows symbolic-shape state
                if not any_symbolic(state_shape):
                    if state_shape != buffer_shape:
                        raise ValueError(
                            f"StateType shape {state_shape} must match the torch buffer shape {buffer_shape}."
                        )

            if self.opset_version is None or self.opset_version < _target.iOS18:
                raise ValueError(
                    "State model is supported only >= iOS18. "
                    "Please update the minimum_deployment_target to at least coremltools.target.iOS18"
                )
            self.inputs.extend(self.states)

    def _adjust_default_input_to_fp16(self) -> None:
        """
        An utility function that sets the default input dtype to fp16
        """

        def _adjust_default_input_to_fp16_helper(inputs: InputType):
            assert isinstance(inputs, list), "inputs must be type of list"
            # Adjust inputs dtype to fp16
            for val in inputs:
                if isinstance(val, (StateType, TensorType)) and val.dtype is None:
                    val.dtype = types.fp16

        _adjust_default_input_to_fp16_helper(self.inputs)
        _adjust_default_input_to_fp16_helper(self.states)

    def _adjust_default_output_to_fp16(self, graph_outputs):
        """
        An utility function that sets the default outputs with inferred type fp32 to fp16.

        - If the inferred output dtype is fp32, and the user doesn't provide dtype, it defaults to fp16.
        - If the inferred output dtype is not fp32, nothing would change.
        """
        if self.outputs is None:
            self.outputs = []
            for val in graph_outputs:
                dtype = types.fp16 if val.dtype == types.fp32 else val.dtype
                self.outputs.append(TensorType(dtype=dtype))
        else:
            for i, val in enumerate(self.outputs):
                if (
                    isinstance(val, TensorType)
                    and val.dtype is None
                    and graph_outputs[i].dtype == types.fp32
                ):
                    val.dtype = types.fp16

    @staticmethod
    def _check_ops(graph):
        """
        Returns the set of ops in @graph that are implemented, and the set
        for which no conversion function is registered. @graph can be
        either InternalTorchIRGraph or InternalTorchIRBlock.
        """
        implemented_ops = set()
        missing_ops = set()
        for node in graph.nodes:
            _add_op = _TORCH_OPS_REGISTRY.get_func(node.kind)
            if _add_op is None:
                missing_ops.add(node.kind)
            else:
                implemented_ops.add(node.kind)
            for block in node.blocks:
                _impl, _miss = TorchConverter._check_ops(block)
                implemented_ops.update(_impl)
                missing_ops.update(_miss)
        return implemented_ops, missing_ops

    @staticmethod
    def _create_placeholder(_input: InputType) -> Placeholder:
        """
        Converts an InputType into a Placeholder.

        1. ``StateType`` into ``mb.state_tensor_placeholder``.
        2. ``TensorType`` and ``ImageType`` into ``mb.placeholder``.
        """
        shape = _input.shape.symbolic_shape
        dtype = _input.dtype
        # int64 and fp64 are not supported, so they are mapped to int32 / fp32 accordingly
        if dtype == types.int64:
            dtype = types.int32
            logger.warning(f"int64 dtype input {_input.name} down casted to int32.")
        elif dtype == types.fp64:
            dtype = types.fp32
            logger.warning(f"fp64 dtype input {_input.name} down casted to fp32.")

        if isinstance(_input, StateType):
            return mb.state_tensor_placeholder(shape, dtype=dtype)

        return mb.placeholder(shape, dtype=dtype)

    @staticmethod
    def _construct_compression_info(
        state_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, CompressionInfo]:
        """
        Construct compression info from the traced model's state_dict.

        The state_dict of the traced model is something like
        {
            'dense1.weight': xxx, 'dense1.bias': xxx,
            'dense1._COREML_/weight/quantization_n_bits': tensor(4),
            'dense1._COREML_/weight/quantization_scale': xxx,
            'dense1._COREML_/weight/zero_point': xxx,
            'dense1._COREML_/weight/compression_type': tensor([3]),
            'dense2.weight': xxx,
            ...
        }

        We extract the compression info and store it as a dict
        {
            'dense1.weight': CompressionInfo(quantization_n_bits=4, quantization_scale=xxx,
                                            zero_point=xxx, compression_type=[QUANTIZATION]),
            'dense2.weight': ...
        }
        """
        compression_info = dict()
        for torch_key_name in state_dict.keys():
            if f"{_COMPRESSION_INFO_PREFIX}/metadata_version" in torch_key_name:
                # TODO: rdar://124707382 ([Compression] Support versioning in CompressionInfo)
                continue

            if _COMPRESSION_INFO_PREFIX in torch_key_name:
                module_name = None
                buffer_name = torch_key_name
                if not torch_key_name.startswith(_COMPRESSION_INFO_PREFIX):
                    module_name, buffer_name = torch_key_name.rsplit(".", 1)
                _, param_name, compression_key = buffer_name.rsplit("/", 2)
                if module_name:
                    param_name = f"{module_name}.{param_name}"

                if param_name not in compression_info:
                    compression_info[param_name] = CompressionInfo()
                setattr(
                    compression_info[param_name],
                    compression_key,
                    state_dict[torch_key_name],
                )

        return compression_info

    def _has_compression_info(self, param_name: str) -> bool:
        """Check if the parameter carries compression info."""
        return param_name in self.param_to_compression_info

    @staticmethod
    def _interleave_repeat_scale_zp(
        weight: np.ndarray, scale: np.ndarray, zero_point: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        The scale and zero-point both have shape [.., block_num, ..], which means each scale is
        for one block. As weight has shape [.., block_num*block_size, ..], we need to interleave
        repeat them, so they can be applied to all blocks at once.
        """
        scale_repeated = scale
        zero_point_repeated = zero_point
        for axis, weight_dim_size in enumerate(weight.shape):
            scale_dim_size = scale.shape[axis]
            if weight_dim_size != scale_dim_size and scale_dim_size != 1:
                # Only repeat axis where dim size is not 1, because 1 will be auto-broadcast by np.
                block_size = weight_dim_size // scale.shape[axis]
                scale_repeated = np.repeat(scale_repeated, block_size, axis=axis)
                if zero_point_repeated is not None:
                    zero_point_repeated = np.repeat(zero_point_repeated, block_size, axis=axis)
        return scale_repeated, zero_point_repeated

    def _construct_quantization_op(
        self,
        weight: np.ndarray,
        compression_info: CompressionInfo,
        name: str,
        compressed_var: Optional[Var] = None,
    ) -> Var:
        """
        The weight is constructed by `weight = scale * (quantized_data - zero_point)`.
        We need to restore the quantized_data to construct the quantization op.

        If compressed_var is not None, it's the var constructed by a previous compression function,
        which means this is a joint compression. For example, if the compression_info.compression_type
        is [CompressionType.PRUNING, CompressionType.QUANTIZATION], the compressed_var is the var
        produced by the pruning.
        """
        if compression_info.quantization_n_bits is None:
            raise ValueError("quantization_n_bits must be specified in quantization.")
        if compression_info.quantization_scale is None:
            raise ValueError("quantization_scale must be specified in quantization.")

        scale = compression_info.quantization_scale.detach().numpy()
        zero_point: Optional[np.ndarray] = None
        if compression_info.zero_point is not None:
            zero_point = compression_info.zero_point.detach().numpy()
        # For conv/conv_transpose, the weight has rank=4, so we auto-expand scale and zero-point if
        # it only has two elements.
        if len(weight.shape) == 4 and len(scale.shape) == 2:
            scale = np.expand_dims(np.expand_dims(scale, axis=-1), axis=-1)
            if zero_point is not None:
                zero_point = np.expand_dims(np.expand_dims(zero_point, axis=-1), axis=-1)

        if compressed_var is not None and compressed_var.op.op_type == "constexpr_lut_to_dense":
            # The quantization on lut could lead to extra two dims at the end.
            if len(scale.shape) == len(weight.shape) + 2 and scale.shape[-2:] == (1, 1):
                scale = np.squeeze(np.squeeze(scale, axis=-1), axis=-1)
                if zero_point is not None:
                    zero_point = np.squeeze(np.squeeze(zero_point, axis=-1), axis=-1)

        if len(weight.shape) != len(scale.shape):
            raise ValueError(
                f"In {name}, the `weight` should have same rank as `scale`, but got {weight.shape} vs {scale.shape}"
            )
        if zero_point is not None:
            if len(weight.shape) != len(zero_point.shape):
                raise ValueError(
                    f"In {name}, the `weight` should have same rank as `zero_point`, but got {weight.shape} vs {zero_point.shape}"
                )

        scale_repeated, zero_point_repeated = self._interleave_repeat_scale_zp(
            weight, scale, zero_point
        )
        quantized_data = np.round(weight / scale_repeated)
        if zero_point_repeated is not None:
            quantized_data += zero_point_repeated

        # Adjust dtype based on nbits.
        dtype_str_prefix = "int"
        if quantized_data.min() >= 0 and (zero_point is None or zero_point.min() >= 0):
            dtype_str_prefix = "uint"
        dtype_str = dtype_str_prefix + str(compression_info.quantization_n_bits)
        builtin_dtype = types.string_to_builtin(dtype_str)
        np_dtype = types.nptype_from_builtin(builtin_dtype)

        builtin_range = types.type_mapping.builtin_to_range(builtin_dtype)
        quantized_data = np.clip(quantized_data, builtin_range.low, builtin_range.high).astype(
            np_dtype
        )
        if zero_point is not None:
            zero_point = zero_point.astype(np_dtype)

        if compressed_var is None:
            return frontend_utils._construct_constexpr_dequant_op(
                quantized_data, zero_point, scale, name=name
            )
        else:
            # Specially handles joint compression, such as using sparse op if joint with pruning.
            if compressed_var.op.op_type == "constexpr_sparse_to_dense":
                mask, nonzero_data = mb.constexpr_sparse_blockwise_shift_scale(
                    data_mask=compressed_var.op.mask,
                    nonzero_data=quantized_data[compressed_var.op.mask.val != 0].flatten(),
                    scale=scale,
                    offset=zero_point,
                    before_op=compressed_var.op,
                    name=compressed_var.op.name + "_quantized",
                )
                return mb.constexpr_sparse_to_dense(nonzero_data=nonzero_data, mask=mask, name=name)
            elif compressed_var.op.op_type == "constexpr_lut_to_dense":
                if not types.is_int(compressed_var.dtype):
                    raise ValueError(
                        "The joint palettization+quantization only supports lut with "
                        f"int entries, but got {types.builtin_to_string(compressed_var.dtype)}"
                    )
                return mb.constexpr_blockwise_shift_scale(
                    data=compressed_var,
                    scale=scale,
                    offset=zero_point,
                    name=name,
                )
            else:
                raise ValueError(
                    "Unsupported joint compression combination. The quantization can only be joint "
                    f"with pruning or palettization, but got {compressed_var.op.op_type}. Please check the value of "
                    "'compression_type' in your registered buffers."
                )

    def _construct_palettization_op(
        self,
        weight: np.ndarray,
        compression_info: CompressionInfo,
        name: str,
        compressed_var: Optional[Var] = None,
    ) -> Var:
        """
        The weight is constructed by 2**nbits unique values in each group.

        When `palettization_scale` is provided, it means the weight has scales before got palettized.
        More specifically, the diagram is:

        lut(fp16) \
                    -> constexpr_lut_to_dense -> dense(fp16) -> constexpr_blockwise_shift_scale -> dense(fp16)
        indices  /

        If compressed_var is not None, it's the var constructed by a previous compression function,
        which means this is a joint compression. For example, if the compression_info.compression_type
        is [CompressionType.PRUNING, CompressionType.PALETTIZATION], the compressed_var is the var
        produced by the pruning.
        """
        if compression_info.lut is None:
            raise ValueError("Missing lut in compression info. Please register a buffer for lut.")

        lut = compression_info.lut.detach().numpy()
        if len(lut.shape) == len(weight.shape) + 1:
            # The last dim to indicate vector size is by default 1 for scalar palettization.
            lut = np.expand_dims(lut, axis=-1)
        if len(lut.shape) != len(weight.shape) + 2:
            raise ValueError(
                f"In {name}, The rank of lut is invalid. It should match weight's rank where"
                f"lut.rank == weight.rank + 2). Got lut.rank {len(lut.shape)} and weight.rank {len(weight.shape)}"
            )

        num_palettes = lut.shape[-2]
        nbits = int(math.ceil(math.log2(num_palettes)))
        if 2**nbits != num_palettes:
            # Padding lut to make it has 2**nbits dim size on -2 axis.
            padding_shape = list(lut.shape)
            padding_shape[-2] = 2**nbits - num_palettes
            lut = np.concatenate([lut, np.zeros(padding_shape, dtype=lut.dtype)], axis=-2)
            num_palettes = lut.shape[-2]

        if compression_info.palettization_scale is not None:
            # The weight has scales, which means the palettization is on the pre-scale data.
            scale = compression_info.palettization_scale.detach().numpy()
            # For conv/conv_transpose, the weight has rank=4, so we auto-expand scale and zero-point if
            # it only has two elements.
            if len(weight.shape) == 4 and len(scale.shape) == 2:
                scale = np.expand_dims(np.expand_dims(scale, axis=-1), axis=-1)
            if len(scale.shape) != len(weight.shape):
                raise ValueError(
                    f"In {name}, the scale should have the same rank as weight, but got "
                    f"{scale.shape} vs {weight.shape}."
                )
            weight = weight / scale

        vector_axis = compression_info.vector_axis
        if lut.shape[-1] > 1:
            if vector_axis is None:
                # The cto.torch uses 0 for vector axis.
                logger.warning(
                    "It's recommended to provide vector_axis for vector palettization. "
                    "Defaulting to axis zero."
                )
                vector_axis = 0
        indices = optimize_utils.find_indices_for_lut(weight, lut, vector_axis)

        if CompressionType.QUANTIZATION.value in compression_info.compression_type:
            # In joint palettization + quantization, the `lut` in the palettization op should be
            # quantized, so we calculate the quantized lut on-the-fly.
            tmp_quant_var = self._construct_quantization_op(
                lut, compression_info, name + "_tmp_quant"
            )
            lut = tmp_quant_var.op.data.val

        if compressed_var is None:
            result = frontend_utils._construct_constexpr_lut_op(indices, lut, vector_axis, name)
        else:
            # Specially handles joint compression, such as using sparse op if joint with pruning.
            if compressed_var.op.op_type == "constexpr_sparse_to_dense":
                mask, nonzero_data = mb.constexpr_lut_to_sparse(
                    indices_mask=compressed_var.op.mask,
                    indices_nonzero_data=indices[compressed_var.op.mask.val != 0].flatten(),
                    lut=lut,
                    vector_axis=vector_axis,
                    before_op=compressed_var.op,
                    name=compressed_var.op.name + "_palettized",
                )
                result = mb.constexpr_sparse_to_dense(
                    nonzero_data=nonzero_data, mask=mask, name=name
                )
            else:
                raise ValueError(
                    "Unsupported joint compression combination. The palettization can only be joint "
                    f"with pruning, but got {compressed_var.op.op_type}. Please check the value of "
                    "'compression_type' in your registered buffers."
                )

        if compression_info.palettization_scale is not None:
            if not is_current_opset_version_compatible_with(_target.iOS18):
                raise ValueError(
                    "The palettization with per-channel-scale is only supported in iOS18+. Please "
                    "set the minimum_deployment_target to iOS18 or later."
                )
            result = mb.constexpr_blockwise_shift_scale(
                data=result, scale=scale, offset=None, name=name
            )
        return result

    @staticmethod
    def _construct_sparsification_op(
        weight: np.ndarray,
        compression_info: CompressionInfo,
        name: str,
        compressed_var: Optional[Var] = None,
    ) -> Var:
        sparse_params = prune_weights.compress_by_threshold(
            weight, threshold=np.finfo(np.float16).eps, minimum_sparsity_percentile=0
        )
        if sparse_params is None:
            raise ValueError(
                f"Unable to construct sparsified op. Please check if the weight {name} "
                "is sparse."
            )
        if is_current_opset_version_compatible_with(_target.iOS18):
            sparse_params_ios18 = optimize_utils.ios16_sparse_params_to_ios18(sparse_params)
            return mb.constexpr_sparse_to_dense(
                nonzero_data=sparse_params_ios18.nonzero_data,
                mask=sparse_params_ios18.mask,
                name=name,
            )
        else:
            return mb.constexpr_sparse_to_dense(
                nonzero_data=sparse_params.nonzero_data,
                mask=sparse_params.mask,
                shape=np.uint32(sparse_params.shape),
                name=name,
            )

    def _construct_compression_op(self, val: np.ndarray, param_name: str) -> Var:
        """Construct the compression op based on the compression info."""
        compression_info: CompressionInfo = self.param_to_compression_info[param_name]

        shared_msg = (
            "There are coreml compression related buffers registered in the torch "
            f"model (with {_COMPRESSION_INFO_PREFIX} in the buffer's name) for {param_name}"
        )
        if not compression_info.compression_type:
            raise ValueError(
                shared_msg + ", but the 'compression_type' is not set. Please set it to indicate "
                "the type of compression used on the weight."
            )
        if len(compression_info.compression_type) > 3:
            raise ValueError(
                shared_msg + ", but the 'compression_type' has too many values. Support at most 3 "
                "values."
            )

        if len(compression_info.compression_type) > 1:
            if not is_current_opset_version_compatible_with(_target.iOS18):
                raise ValueError(
                    "The joint compression (more than one values in 'compression_type') is only "
                    "supported in iOS18+. Please set minimum_deployment_target to iOS18 or later."
                )

        result: Optional[Var] = None
        for type_val in compression_info.compression_type:
            if type_val == CompressionType.QUANTIZATION.value:
                result = self._construct_quantization_op(val, compression_info, param_name, result)
            elif type_val == CompressionType.PALETTIZATION.value:
                result = self._construct_palettization_op(val, compression_info, param_name, result)
            else:
                assert type_val == CompressionType.PRUNING.value
                result = self._construct_sparsification_op(
                    val, compression_info, param_name, result
                )

        if result is None:
            raise AssertionError(shared_msg + f", but unable to compress weight {param_name}")
        return result

    def _add_const(self, name: str, val: Union[torch.Tensor, torch._C.ScriptObject]) -> None:
        """Create a const op and add it to the graph."""
        if isinstance(val, torch.Tensor) and self._has_compression_info(name):
            try:
                compression_op = self._construct_compression_op(val.detach().numpy(), name)
                self.context.add(compression_op)
                return
            except NotImplementedError as e:
                logger.warning(
                    "Failed to create a compression op based on the compression info "
                    f"carried by {name} in the torch model. Ignored the compression info "
                    f"and constructed a normal const. Detailed error message:\n{e}"
                )

        if isinstance(val, torch._C.ScriptObject):
            logger.info(f"Encountered constant {name} of type _torch._C.ScriptObject")
            return
        elif isinstance(val, torch.Tensor) and val.is_quantized:
            const = _dequantized_weight(val.cpu(), name)
            self.context.add(const)
            return
        elif not isinstance(val, torch.Tensor):
            raise ValueError(f"unsupported class for {name} in PyTorch graph: {type(val)}")
        val = val.detach().cpu().numpy()
        # TODO (rdar://107718371): support uint8 activation quantization in torchscript
        # Some torchscript models store indices with uint8, which are unrelated to quantization and
        # need to be cast to int32 since many non-quantized Core ML ops do not support int8.
        # We need a way to distinguish whether an uint8 is quantization (so should be kept)
        # or not (so should be cast to int32).
        if self.context.frontend == TorchFrontend.TORCHSCRIPT and val.dtype == np.uint8:
            val = val.astype(np.int32)
        const = mb.const(val=val, name=name)
        self.context.add(const)

    def check_ops(self):
        """
        Returns the set of ops in @self.graph that are implemented, and
        the set for which no conversion function is registered.
        """
        return TorchConverter._check_ops(self.graph)

    def convert_const(self) -> None:
        for name, val in self.graph.params.items():
            if self.context.frontend == TorchFrontend.TORCHSCRIPT:
                scope_name, scope_type = self.graph.params_scope[name]
                with mb.scope(
                    ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_TYPE, data=scope_type),
                    ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_NAME, data=scope_name),
                ):
                    self._add_const(name, val)
            elif self.context.frontend in TORCH_EXPORT_BASED_FRONTENDS:
                # Torch.Export has constants lifted as inputs, yet we have not sorted out
                # how to support IO metadata, so for now just put a dummy metadata
                # since inputs/constants will not contribute to debugging/profiling
                # TODO (rdar://125572392): Support torch.export IO metadata
                scopes = [ScopeInfo(source=ScopeSource.EXIR_STACK_TRACE, data=[None])]
                if self.context.frontend == TorchFrontend.EXECUTORCH:
                    scopes.append(ScopeInfo(source=ScopeSource.EXIR_DEBUG_HANDLE, data=[None]))
                with mb.scope(*scopes):
                    self._add_const(name, val)
            else:
                raise ValueError(f"Invalid PyTorch frontend {self.context.frontend}")

    def convert(self) -> Program:
        logger.info("Converting graph.")

        # Set SSA function input name to user defined name if provided.
        for index, (name, spec) in enumerate(self.graph.inputs.items()):
            if spec.name is not None:
                name = spec.name
            self.inputs[index].name = name

        # This will hold the converted model.
        prog = self._prog

        # Construct placeholder for input to SSA function
        ssa_func_inputs = OrderedDict()
        for spec in self.inputs:
            ssa_func_inputs[spec.name] = self._create_placeholder(spec)

        # Initialize the SSA for conversion
        with Function(ssa_func_inputs, opset_version=self.opset_version) as ssa_func:

            # Map internal @self.graph.inputs to user specified @ssa_func_inputs
            # If @self.graph.inputs == @ssa_func_inputs this just adds the inputs
            # to the context.
            # Convert input placeholders
            user_names = list(ssa_func_inputs.keys())
            internal_names = list(self.graph.inputs.keys())
            internal_names.extend(user_names[len(internal_names) :])
            for torch_name, ssa_name in zip(internal_names, user_names):
                input_var = ssa_func.inputs[ssa_name]
                if self.context.frontend == TorchFrontend.TORCHSCRIPT:
                    # To create fp16 Core ML model from fp32 torch script model, we
                    # 1. Cast input to fp32 (if specified fp16 input)
                    # 2. Convert fp32 torch model to fp32 Core ML model
                    # 3. Graph passes `add_fp16_cast` and `cast_optimization`
                    #    then cast fp32 Core ML model to fp16
                    # So here we perform the "cast input to fp32" step
                    if (
                        types.is_tensor(input_var.sym_type) or types.is_scalar(input_var.sym_type)
                    ) and input_var.dtype == types.fp16:
                        # This cast should have placeholder scope
                        with mb.scope(
                            ScopeInfo(
                                source=ScopeSource.TORCHSCRIPT_MODULE_TYPE, data="placeholder"
                            ),
                            ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_NAME, data=torch_name),
                        ):
                            input_var = mb.cast(x=input_var, dtype="fp32")
                elif self.context.frontend in TORCH_EXPORT_BASED_FRONTENDS:
                    # When handle user provided IO dtypes, we will also need to handle IO metadata
                    # TODO (rdar://125572392): Support torch.export IO metadata
                    if types.is_tensor(input_var.sym_type) or types.is_scalar(input_var.sym_type):
                        # cast and minimum deployment target check may be needed
                        user_input_dtype = input_var.dtype
                        exir_input_dtype = self.graph.inputs[torch_name].dtype
                        if user_input_dtype != exir_input_dtype:
                            scopes = [
                                ScopeInfo(source=ScopeSource.EXIR_STACK_TRACE, data=torch_name)
                            ]
                            if self.context.frontend == TorchFrontend.EXECUTORCH:
                                scopes.append(
                                    ScopeInfo(source=ScopeSource.EXIR_DEBUG_HANDLE, data=[None])
                                )
                            with mb.scope(*scopes):
                                input_var = mb.cast(
                                    x=input_var, dtype=builtin_to_string(exir_input_dtype)
                                )
                        if (
                            user_input_dtype == types.fp16
                            and not is_current_opset_version_compatible_with(_target.iOS16)
                        ):
                            raise ValueError(
                                "To use fp16 input, please set minimum deployment target to iOS16+"
                            )
                    if torch_name.endswith(WRAPPED_SCALAR_INPUT_SUFFIX):
                        # Torch.export may produce scalar input,
                        # which then gets wrapped as rank-1 size-1 tensor for Core ML residency
                        # during our internal graph construction.
                        # Here we squeeze it back to scalar
                        torch_name = torch_name[: -len(WRAPPED_SCALAR_INPUT_SUFFIX)]
                        scopes = [
                            ScopeInfo(
                                source=ScopeSource.EXIR_STACK_TRACE,
                                data=f"unwrap_scalar_input_{torch_name}",
                            )
                        ]
                        if self.context.frontend == TorchFrontend.EXECUTORCH:
                            scopes.append(
                                ScopeInfo(source=ScopeSource.EXIR_DEBUG_HANDLE, data=[None])
                            )
                        with mb.scope(*scopes):
                            input_var = mb.squeeze(x=input_var, name=torch_name)
                else:
                    raise ValueError(f"Invalid PyTorch frontend {self.context.frontend}")
                self.context.add(input_var, torch_name=torch_name)

            # EXIR lifts buffer references as inputs, so we need to create them by reading states
            if self.context.frontend in TORCH_EXPORT_BASED_FRONTENDS:
                for (
                    input_name,
                    buffer_name,
                ) in self.context.torch_graph.input_name_to_source_buffer_name.items():
                    buffer_var = self.context[buffer_name]
                    scopes = [
                        ScopeInfo(source=ScopeSource.EXIR_STACK_TRACE, data=f"read_{buffer_name}")
                    ]
                    if self.context.frontend == TorchFrontend.EXECUTORCH:
                        scopes.append(ScopeInfo(source=ScopeSource.EXIR_DEBUG_HANDLE, data=[None]))
                    with mb.scope(*scopes):
                        input_var = mb.read_state(input=buffer_var)
                        # As of iOS 18, Core ML state can only be fp16
                        # In torch converter, we convert everything under fp32
                        # (then cast everything to fp16 if specified fp16 compute precision)
                        # so we need to (temporarily) cast read result to fp32
                        input_var_fp32 = mb.cast(x=input_var, dtype="fp32", name=input_name)
                    self.context.add(input_var_fp32)
                    self.context.name_to_source_state[input_name] = buffer_var

            # Convert constants
            self.convert_const()

            # Add the rest of the operations
            has_states = len(getattr(self, "states", [])) > 0
            convert_nodes(self.context, self.graph, early_exit=not has_states)

            # EXIR represents stateful execution as buffer mutation at output,
            # i.e. buffer.copy_(...) at the end of EXIR program,
            # so analogously we update state at the end of pymil function
            if self.context.frontend in TORCH_EXPORT_BASED_FRONTENDS:
                for (
                    output_name,
                    buffer_name,
                ) in self.context.torch_graph.output_name_to_target_buffer_name.items():
                    output_var = self.context[output_name]
                    buffer_var = self.context[buffer_name]
                    scopes = [
                        ScopeInfo(source=ScopeSource.EXIR_STACK_TRACE, data=f"write_{buffer_name}")
                    ]
                    if self.context.frontend == TorchFrontend.EXECUTORCH:
                        scopes.append(ScopeInfo(source=ScopeSource.EXIR_DEBUG_HANDLE, data=[None]))
                    with mb.scope(*scopes):
                        cast_value = mb.cast(
                            x=output_var, dtype=builtin_to_string(buffer_var.dtype)
                        )
                        mb.coreml_update_state(state=buffer_var, value=cast_value)

            graph_outputs = [self.context[name] for name in self.graph.outputs]

            # An output can be None when it's a None constant, which happens
            # in Fairseq MT.
            for g in graph_outputs:
                if g is None:
                    logger.warning(f"Dropping output {g} which is None")
            graph_outputs = [g for g in graph_outputs if g is not None]

            # Output renaming occurs
            if self.outputs is not None:
                if len(self.outputs) != len(graph_outputs):
                    raise ValueError(
                        f"Number of outputs provided, {len(self.outputs)}, do not match the number of outputs detected in the model, {len(graph_outputs)}."
                    )
            if self.output_names:
                for index, var in enumerate(graph_outputs):
                    if self.output_names[index] is not None:
                        output_rename = self.output_names[index]
                        var.name = output_rename

            ssa_func.set_outputs(graph_outputs)
            prog.add_function("main", ssa_func)
            if self.use_default_fp16_io:
                # If the output type is not specified by the user and use_default_fp16_io
                # is True. Make the default output type to fp16
                self._adjust_default_output_to_fp16(graph_outputs)
            if self.outputs is not None:
                prog.functions["main"].set_output_types(self.outputs)

            prog.functions["main"].set_input_types(tuple(self.inputs))

            # Make sure the prog is not missing any scope information
            essential_scope_sources = []
            if self.context.frontend == TorchFrontend.TORCHSCRIPT:
                essential_scope_sources = [
                    ScopeSource.TORCHSCRIPT_MODULE_NAME,
                    ScopeSource.TORCHSCRIPT_MODULE_TYPE,
                ]
            elif self.context.frontend in TORCH_EXPORT_BASED_FRONTENDS:
                essential_scope_sources = [ScopeSource.EXIR_STACK_TRACE]
                if self.context.frontend == TorchFrontend.EXECUTORCH:
                    essential_scope_sources.append(ScopeSource.EXIR_DEBUG_HANDLE)
            else:
                raise ValueError(f"Invalid PyTorch frontend {self.context.frontend}")
            prog._add_essential_scope_source(essential_scope_sources)
            prog.validate(check_essential_scope=True)
        return prog
