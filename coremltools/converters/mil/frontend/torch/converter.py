#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from collections import OrderedDict

import numpy as np
import torch as torch

from coremltools import _logger as logger
from coremltools._deps import version_lt
from coremltools.converters.mil._deployment_compatibility import AvailableTarget as _target
from coremltools.converters.mil.input_types import ImageType, TensorType
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Function, Program, types
from coremltools.converters.mil.mil.types import is_float

from .._utils import get_output_names
from .internal_graph import InternalTorchIRGraph, InternalTorchIRNode
from .ops import convert_nodes
from .quantization_ops import _dequantized_weight
from .torch_op_registry import _TORCH_OPS_REGISTRY
from .torchir_passes import (
    flatten_graph_input_values,
    flatten_graph_output_values,
    generate_tensor_assignment_ops,
    remove_getattr_nodes,
    transform_inplace_ops,
)

torch_to_mil_types = {
    torch.bool: types.bool,
    torch.float16: types.fp16,
    torch.float32: types.fp32,
    torch.float64: types.fp32,
    torch.int32: types.int32,
    torch.int64: types.int32,
}


mil_to_torch_types = {v: k for k, v in torch_to_mil_types.items()}


class QuantizationContext:
    """
    Utilities to manage information pertaining to quantization of tensors in a PyTorch graph.
    """

    def __init__(self, context):
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

    def get_quantization_info(self, name):
        """
        Retrieves the information added via add_quantization_info, if applicable.
        Returns None if quantization parameters could not be found.
        """
        if name not in self._quant_param_map:
            return None
        return self._quant_param_map[name]

    def maybe_handle_quantized_inputs(self, node: InternalTorchIRNode):
        """
        If a node's op doesn't support quantized inputs but gets one, this will wire it to
        receive a dequantized version of it.
        """

        op_type = node.kind
        if op_type in {"quantize_per_tensor", "dequantize"} or "quantized::" in op_type:
            # Op can handle quantized inputs. Nothing to do here.
            return

        for input_name in node.inputs:
            if self.get_quantization_info(input_name) is None:
                # Not a quantized tensor
                continue

            # We need a dequantized version of the input to feed to the op.
            dequantized_var, _ = self.get_dequantized_var(input_name)
            node.replace_name(input_name, dequantized_var.name)

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

    def __init__(self, name=None):
        self.name = name if name else ""
        self._current_graph = [{}]
        self._torch_graph = None
        self._quant_context = QuantizationContext(self)

    @property
    def torch_graph(self):
        if self._torch_graph is None:
            raise ValueError("InternalTorchIRGraph not set yet on context")
        return self._torch_graph

    @property
    def quant_context(self):
        return self._quant_context

    @torch_graph.setter
    def torch_graph(self, graph: InternalTorchIRGraph):
        self._torch_graph = graph

    def prepare_for_conversion(self, node: InternalTorchIRNode):
        """
        Perform any preparation necessary before node-specific frontend conversion
        is invoked.
        """
        self.quant_context.maybe_handle_quantized_inputs(node)

    def add(self, ssa_var, torch_name=None):
        """
        Arguments:
            ssa_var: Variable to add to the graph being constructed.
            torch_name: Optional unique string identifier of the operation. If
                omitted, it will use @ssa_var.name.
        """
        if torch_name is None:
            torch_name = ssa_var.name
        if torch_name in self._current_graph[-1]:
            print(f"Torch var {torch_name} is added again.")
            return
        self._current_graph[-1][torch_name] = ssa_var

    def __getitem__(self, torch_name):
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
    Class that handles conversion of pytorch models represented in TorchScript
    format to the MIL format.

    Models passed to the @TorchConverter go from:
    TorchScript -> Expanded/Optimized Torch IR -> Internal Graph -> CoreML SSA
    The internal graph representation was added to make testing easier.
    """

    def __init__(
        self,
        torchscript,
        inputs,
        outputs=None,
        cut_at_symbols=None,
        opset_version=None,
        use_default_fp16_io=False,
    ):
        """
        Arguments:
            torchscript: torch.jit.ScriptModule object representing the model to convert.
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
        assert isinstance(torchscript, torch.jit.ScriptModule)

        self.inputs = inputs
        for idx, inp in enumerate(self.inputs):
            if isinstance(inp, ImageType) and self.inputs[idx].channel_first is None:
                self.inputs[idx].channel_first = True

        self.torchscript = torchscript
        self.outputs = outputs
        self.use_default_fp16_io = use_default_fp16_io

        if self.use_default_fp16_io:
            # If the input type is not specified by the user and use_default_fp16_io
            # is True. Make the default input type to fp16
            self._adjust_default_input_to_fp16()

        self.output_names = get_output_names(self.outputs)
        self.opset_version = _target(opset_version) if opset_version is not None else None
        self.context = TranscriptionContext()
        raw_graph, params_dict = self._expand_and_optimize_ir(self.torchscript)
        self.params_dict = params_dict
        self.graph = InternalTorchIRGraph(
            raw_graph, params_dict, self.inputs, cut_at_symbols
        )
        self.context.torch_graph = self.graph

        # TODO (rdar://106161395): Register Torch IR passes and unify them into the pass pipeline.
        # Apply Torch IR passes
        passes = [
            transform_inplace_ops,
            flatten_graph_input_values,
            flatten_graph_output_values,
            remove_getattr_nodes,
            generate_tensor_assignment_ops,
        ]
        for p in passes:
            p(self.graph)

        self.inputs = list(self.graph.inputs.values())
        self._prog = Program()

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
            _add_op = _TORCH_OPS_REGISTRY.get(node.kind, None)
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
    def _create_placeholder(_input):
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

    def check_ops(self):
        """
        Returns the set of ops in @self.graph that are implemented, and
        the set for which no conversion function is registered.
        """
        return TorchConverter._check_ops(self.graph)

    def convert_const(self):
        for name, val in self.graph.params.items():
            if isinstance(val, torch._C.ScriptObject):
                logger.info(f"Encountered constant {name} of type _torch._C.ScriptObject")
                continue
            elif isinstance(val, torch.Tensor) and val.is_quantized:
                const = _dequantized_weight(val.cpu(), name)
                self.context.add(const)
                continue
            elif not isinstance(val, np.ndarray):
                raise ValueError(f"unsupported class for {name} in PyTorch graph: {type(val)}")
            # TODO (rdar://107718371): support uint8 quantization
            # Some torch models store indices with uint8, which are unrelated to quantization and
            # need to be cast to int32 since Core ML does not support int8.
            # We need a way to distinguish whether an uint8 is quantization (so should be kept)
            # or not (so should be cast to int32).
            if val.dtype == np.uint8:
                val = val.astype(np.int32)
            const = mb.const(val=val, name=name)
            self.context.add(const)

    def convert(self):
        logger.info("Converting graph.")

        # This will hold the converted model.
        prog = self._prog

        # Construct placeholder for input to SSA function
        # This is where input renaming occurs
        ssa_func_inputs = OrderedDict()
        for index, (name, spec) in enumerate(self.graph.inputs.items()):
            placeholder = self._create_placeholder(spec)
            # Set SSA function input name to user defined name if provided.
            if spec.name is not None:
                name = spec.name
            self.inputs[index].name = name
            ssa_func_inputs[name] = placeholder
        prog.set_main_input_types(tuple(self.inputs))

        # Initialize the SSA for conversion
        with Function(ssa_func_inputs, opset_version=self.opset_version) as ssa_func:

            # Map internal @self.graph.inputs to user specified @ssa_func_inputs
            # If @self.graph.inputs == @ssa_func_inputs this just adds the inputs
            # to the context.
            for internal_name, users_name in zip(
                self.graph.inputs.keys(), ssa_func_inputs.keys()
            ):
                input_var = ssa_func.inputs[users_name]
                if (
                    types.is_tensor(input_var.sym_type) or types.is_scalar(input_var.sym_type)
                ) and input_var.dtype == types.fp16:
                    input_var = mb.cast(x=input_var, dtype="fp32")
                self.context.add(input_var, torch_name=internal_name)

            self.convert_const()

            # Add the rest of the operations
            convert_nodes(self.context, self.graph)

            graph_outputs = [self.context[name] for name in self.graph.outputs]

            # An output can be None when it's a None constant, which happens
            # in Fairseq MT.
            for g in graph_outputs:
                if g is None:
                    logger.warning(f"Droping output {g} which is None")
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
                prog.set_main_output_types(self.outputs)
        return prog

    def _jit_pass_lower_graph(graph, torchscript):
        """
        This graph pass does a similar thing as torch._C._jit_pass_lower_graph does.
        It does two things:
        1. Rename getattr nodes which produce a torch tensor to match the keys in torch model's state_dict
        2. Construct the params_dict, with the keys similar to state_dict

        To be more specific, this graph pass traces down series of GetAttr ops, and rename the final node to match the torch model state_dict.
        It also replaces the node inputs by the first created tensor node with the same name.

        Example:
        Input graph:
        graph(%self.1 : __torch__.torch.nn.modules.Sequential, %input.1 : Tensor):
        %2 : prim::GetAttr[name="linear"](%self.1)
        %3 : prim::GetAttr[name="weight"](%2)
        %4 : prim::GetAttr[name="bias"](%2)
        %5 : prim::GetAttr[name="bias"](%2) # duplicated node
        %6 : conv(%input.1, %3, %4)
        %7 : add(%input.1, %5)
        return (%6, %7)

        Output graph:
        graph(%self.1 : __torch__.torch.nn.modules.Sequential, %input.1 : Tensor):
        %2 : prim::GetAttr[name="linear"](%self.1)
        %linear.weight : prim::GetAttr[name="weight"](%2)
        %linear.bias : prim::GetAttr[name="bias"](%2)
        %5 : prim::GetAttr[name="bias"](%2) # duplicated node, it is not used now
        %6 : conv(%input.1, %linear.weight, %linear.bias)
        %7 : add(%input.1, %linear.bias) # the second input is replaced
        return (%6, %7)

        And a dictionary {"linear.weight": ..., "linear.bias": ...} is returned, to record the parameters values.
        Note that, those GetAttr nodes are still in the torch ir graph, but they would be removed in a latter
        graph pass in the coremltools torch internal graph

        """

        """
        Each getattr node corresponds to a torch object in the torch IR,
        it could be either:
        1. torch.nn.modules: submodule in a torch model. For instance, a linear layer in a MLP network.
        2. torch.Tensor: torch model parameters. For instance, weight for a conv layer.
        3. torch._C.ScriptObject: quantized torch model parameters.
        For example, in the graph above, %2 is pointing to the __torch__.torch.nn.modules.Sequential.linear torch submodule.
        node_to_module_map tracks these mapping.

        node_to_prefic_map track the name for each module,
        for example, %2 has the prefix name linear and %3 is linear.weight.
        These names are also keys in the state_dict
        """
        node_to_module_map = {}
        node_to_prefix_map = {}
        first_node_with_prefix = {}
        replace_input = {}

        base_module_node = list(graph.inputs())[0]
        node_to_module_map[base_module_node] = torchscript
        node_to_prefix_map[base_module_node] = ""

        """
        params_dict will be contructed in this graph pass. It contains all const tensors needed for the graph computation.
        And the value is validated against the state_dict if the key is presented in both dictionaries.
        In some rare cases, state_dict lacks parameters / buffers, so we still need to go through the while graph ourselves.
        """
        params_dict = {}
        state_dict = torchscript.state_dict(keep_vars=True)

        def _check_is_tensor(node, module):
            if not isinstance(module, torch.Tensor):
                return False
            if str(node.output().type()) not in ("Tensor", "Optional[Tensor]"):
                raise TypeError(f'Type "{node.output().type()}" not supported')
            return True

        def _check_is_quantized_tensor(node, module):
            if not isinstance(module, torch._C.ScriptObject):
                return False
            # We only support ScriptObjects that correspond to quantized packed params.
            assert "PackedParams" in node.output().type().name()
            return True

        def _lower_graph_block(graph):
            for node in list(graph.nodes()):

                for block in node.blocks():
                    _lower_graph_block(block)

                for idx, _input in enumerate(list(node.inputs())):
                    if _input in replace_input:
                        node.replaceInput(idx, replace_input[_input])

                kind = node.kind().split("::")[1].lower()
                if kind != "getattr":
                    continue

                _input = node.input()
                _output = node.output()
                attr_name = getattr(node, node.kindOf("name"))("name")

                module = getattr(node_to_module_map[_input], attr_name)
                node_to_module_map[_output] = module

                input_prefix = node_to_prefix_map[_input]
                prefix = input_prefix + '.' + attr_name if input_prefix != "" else attr_name
                node_to_prefix_map[_output] = prefix

                is_tensor = _check_is_tensor(node, module)
                is_quantized_tensor = _check_is_quantized_tensor(node, module)

                if is_tensor or is_quantized_tensor:
                    if is_tensor and prefix in state_dict:
                        assert torch.equal(
                            module.cpu(), state_dict[prefix].cpu()
                        ), "tensor value not consistent between torch ir and state_dict"
                    if prefix in params_dict:
                        assert torch.equal(module.cpu(), params_dict[prefix].cpu())
                        replace_input[_output] = first_node_with_prefix[prefix]
                    else:
                        params_dict[prefix] = module
                        first_node_with_prefix[prefix] = _output
                        _output.setDebugName(prefix)

        _lower_graph_block(graph)

        return graph, params_dict

    @staticmethod
    def _expand_and_optimize_ir(torchscript):
        """
        Given a torch.jit.ScriptModule, convert it to a optimized
        torch._C.Graph and dict of model parameter's names to tensors.
        """
        graph = torchscript.forward.graph

        # From PyTorch code: Inline function and method calls.
        torch._C._jit_pass_inline(graph)
        # From PyTorch code: This inlines the forked section in the fork()
        # callsite and replaces uses of the result of wait() calls with the
        # values produced from the (now-inlined) forked section.
        torch._C._jit_pass_inline_fork_wait(graph)
        # Starting from the return node, marks all nodes that feed into the
        # output, as well as nodes with side effects. Any nodes not marked are
        # eliminated.
        torch._C._jit_pass_dce(graph)
        # From PyTorch code: checks well-formedness and invariants of graph.
        torch._C._jit_pass_lint(graph)
        # Replaces a couple specific ops patterns (add, sub, mul, div, chunk).
        if version_lt(torch, "1.6.0"):
            torch._C._jit_pass_canonicalize_ops(graph)
            torch._C._jit_pass_lint(graph)

            # From PyTorch code: This pass catches all of the small, easy to catch
            # peephole optimizations you might be interested in doing.
            #     Eliminate no-op 'expand' nodes
            #     Simplify x.t().t() to x
            # pass disabled for v1.6.0 and onwards, wrongly captures the shape of dummy inputs during tracing.
            torch._C._jit_pass_peephole(graph, addmm_fusion_enabled=False)
        else:
            # v1.6.0 pass renamed
            torch._C._jit_pass_canonicalize_graph_fuser_ops(graph)
        torch._C._jit_pass_lint(graph)

        # From PyTorch docs: Renumber the graph so that all structurally
        # equivalent graphs have same numbers.
        graph = torch._C._jit_pass_canonicalize(graph)
        torch._C._jit_pass_lint(graph)
        if version_lt(torch, "1.6.0"):
            # v1.6.0 JIT changes disallows pulling list values out of
            # prim::Constant. We can only pull scalar values. constant
            # propagation removes `listConstruct` and results in list values.
            # We disallow constant prop pass to keep them as scalars, and rely
            # on our own constant prop to interpret `listConstruct`.
            torch._C._jit_pass_constant_propagation(graph)
        # NOTE: Don't need another DCE, it's included in constant propagation.
        torch._C._jit_pass_lint(graph)

        # Get the params_dict and rename the getattr nodes in the graph
        graph, params_dict = TorchConverter._jit_pass_lower_graph(graph, torchscript)

        return graph, params_dict
