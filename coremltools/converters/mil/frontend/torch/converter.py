#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from collections import OrderedDict
from typing import List, Optional, Union

import numpy as np
import torch as torch
from torch.jit._script import RecursiveScriptModule

from coremltools import _logger as logger
from coremltools._deps import _HAS_TORCH_EXPORT_API
from coremltools.converters.mil import mil
from coremltools.converters.mil._deployment_compatibility import AvailableTarget as _target
from coremltools.converters.mil.input_types import ImageType, InputType, TensorType
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Function, Placeholder, Program, types
from coremltools.converters.mil.mil.block import is_current_opset_version_compatible_with
from coremltools.converters.mil.mil.scope import ScopeInfo, ScopeSource
from coremltools.converters.mil.mil.types import is_float
from coremltools.converters.mil.mil.var import Var

from .._utils import get_output_names
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
from .torchscript_utils import torch_to_mil_types
from .utils import TorchFrontend

if _HAS_TORCH_EXPORT_API:
    from torch.export import ExportedProgram


def _convert_to_torch_inputtype(inputs: List[TensorType]) -> List[TensorType]:
    input_type = []
    for _input in inputs:
        if isinstance(_input, (list, tuple)):
            input_type.append(_convert_to_torch_inputtype(_input))
        elif isinstance(_input, InputType):
            if _input.shape is None:
                raise ValueError(
                    "'shape' must be provided in the 'inputs' argument for pytorch conversion"
                )
            input_type.append(_input)
        elif isinstance(_input, torch.Tensor):
            input_type.append(
                TensorType(shape=_input.shape, dtype=torch_to_mil_types[_input.dtype])
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

    def prepare_for_conversion(self, node: InternalTorchIRNode) -> None:
        """
        Perform any preparation necessary before node-specific frontend conversion
        is invoked.
        """
        return

    def process_inplace_op(self, node: InternalTorchIRNode):
        return

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
        return torch_name in self._current_graph[-1]

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

        if inputs is not None:
            inputs = _convert_to_torch_inputtype(inputs)
            self.inputs = inputs
            for idx, inp in enumerate(self.inputs):
                if isinstance(inp, ImageType) and self.inputs[idx].channel_first is None:
                    self.inputs[idx].channel_first = True

            if self.use_default_fp16_io:
                # If the input type is not specified by the user and use_default_fp16_io
                # is True. Make the default input type to fp16
                self._adjust_default_input_to_fp16()

        self.outputs = outputs
        self.output_names = get_output_names(self.outputs)
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

        elif _HAS_TORCH_EXPORT_API and isinstance(loaded_model, ExportedProgram):
            self.context = TranscriptionContext(frontend=TorchFrontend.EXIR)
            self.graph = InternalTorchIRGraph.from_exir(exir=loaded_model)
        else:
            raise ValueError(
                "Model should be an instance of either torch.jit.ScriptModule or ExportedProgram"
            )

        self.context.torch_graph = self.graph

        self.inputs = list(self.graph.inputs.values())

    def _adjust_default_input_to_fp16(self):
        """
        An utility function that sets the default input dtype to fp16
        """
        assert isinstance(self.inputs, list), "inputs must be type of list"
        # Adjust inputs dtype to fp16
        for val in self.inputs:
            if isinstance(val, TensorType) and val.dtype is None:
                val.dtype = types.fp16

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
    def _create_placeholder(
        _input: TensorType,
    ) -> Placeholder:
        """
        Converts an InputType into a Placeholder.

        _input: TensorType
        """
        shape = _input.shape.symbolic_shape
        dtype = _input.dtype
        # int64 and fp64 are not supported, so they are mapped to int32 / fp32 accordingly
        if dtype == types.int64:
            dtype = types.int32
        elif dtype == types.fp64:
            dtype = types.fp32
        return mb.placeholder(shape, dtype=dtype)

    def _add_const(self, name: str, val: Union[torch.Tensor, torch._C.ScriptObject]) -> None:
        """Create a const op and add it to the graph."""
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
            elif self.context.frontend == TorchFrontend.EXIR:
                # ExecuTorch has constants lifted as inputs, yet we have not sorted out
                # how to support IO metadata, so for now just put a dummy metadata
                # since inputs/constants will not contribute to debugging/profiling
                # TODO (rdar://125572392): Support torch.export IO metadata
                with mb.scope(
                    ScopeInfo(source=ScopeSource.EXIR_DEBUG_HANDLE, data=[None]),
                ):
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
                    # To create fp16 Core ML model from fp32 torch model, we
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
                elif self.context.frontend == TorchFrontend.EXIR:
                    # EXIR has dtypes all determined, so for now we just stick to EXIR dtypes
                    # TODO (rdar://115845792): Handle fp16 IO dtypes
                    # When handle user provided IO dtypes, we will also need to handle IO metadata
                    # TODO (rdar://125572392): Support torch.export IO metadata
                    if (
                        input_var.dtype == types.fp16
                        and not is_current_opset_version_compatible_with(_target.iOS16)
                    ):
                        raise ValueError(
                            "To use fp16 input, please set minimum deployment target to iOS16+"
                        )
                else:
                    raise ValueError(f"Invalid PyTorch frontend {self.context.frontend}")
                self.context.add(input_var, torch_name=torch_name)

            # Convert constants
            self.convert_const()

            # Add the rest of the operations
            convert_nodes(self.context, self.graph)

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
            elif self.context.frontend == TorchFrontend.EXIR:
                essential_scope_sources = [ScopeSource.EXIR_DEBUG_HANDLE]
            else:
                raise ValueError(f"Invalid PyTorch frontend {self.context.frontend}")
            prog._add_essential_scope_source(essential_scope_sources)
            prog.validate(check_essential_scope=True)
        return prog
