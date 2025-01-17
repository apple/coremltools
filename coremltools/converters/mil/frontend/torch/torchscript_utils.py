#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import torch

from coremltools._deps import version_lt


def _jit_pass_lower_graph(graph, torchscript):
    """
    This graph pass does a similar thing as torch._C._jit_pass_lower_graph does.
    It does three things:
    1. Rename getattr nodes which produce a torch tensor to match the keys in torch model's state_dict
    2. Construct the params_dict, with the keys similar to state_dict
    3. Get the named_buffer dict in torch model

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
    buffer_dict = {k: v for k, v in torchscript.named_buffers()}

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
    return graph, params_dict, buffer_dict

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
    graph, params_dict, buffer_dict = _jit_pass_lower_graph(graph, torchscript)

    return graph, params_dict, buffer_dict
