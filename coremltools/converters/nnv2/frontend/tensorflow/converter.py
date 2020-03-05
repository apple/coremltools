import six
import logging
from coremltools.converters.nnv2.nnv2_program.program.var import Var
from coremltools.converters.nnv2.builtin_types.symbolic import any_symbolic

from ...builtin_types import builtins
from .basic_graph_ops import topsort, simple_topsort

from .tf_op_registry import _OPS_REGISTRY
from .ops import *

from coremltools.converters.nnv2.nnv2_program.ops import CoremlBuilder as cb
from coremltools.converters.nnv2.nnv2_program.program import SsaProgram, SsaFunction


def check_output_shapes(context, node):
    """
    x: Var (single output ops) or list[Var] (multi-output ops)
    node: ParsedTFNode
    """
    tf_shapes = node.attr.get('_output_shapes', None)
    if tf_shapes is None:
        return
    x = context[node.name]
    if not isinstance(x, (tuple, list)):
        x = [x]
    inf_shapes = [list(y.sym_type.get_shape()) \
            if y is not None and builtins.is_tensor(y.sym_type) else None for y in x]
    for t, s in zip(tf_shapes, inf_shapes):
        if t is None or s is None or any_symbolic(s):
            continue
        for dt, ds in zip(t, s):
            if dt is not None and dt != ds:
                msg = "Op {} ({}) type inference ({}) and TF output shape " + \
                        "({}) mismatch"
                raise ValueError(msg.format(node.name, node.op, s, t))
            
# TranscriptionContext maintains a map of tf_node.name --> ssa_var available
# to the current TF --> tfssa transcription.
class TranscriptionContext:
    def __init__(self, name=None):
        self.name = name if name is not None else ""
        self.context = {}

    def add(self, tf_name, ssa_vars):
        """
        ssa_vars: list[Var] (multiple outputs) or Var (single_output)
        """
        if tf_name in self.context:
            print("TF var %s is added again.", tf_name)
            return
        if isinstance(ssa_vars, Var) and tf_name != ssa_vars.name:
            msg = 'tf_name: != ssa_vars.name: {} != {}'
            raise ValueError(msg.format(tf_name, ssa_vars.name))
        self.context[tf_name] = ssa_vars

    def __getitem__(self, tf_name):
        if tf_name not in self.context:
            raise ValueError("TF var %s not found in context %s" \
                    % (tf_name, self.name))
        return self.context[tf_name]

class TFConverter:
    def __init__(self, tfssa, inputs=None, outputs=None, **kwargs):
        self.tfssa = tfssa
        self.global_type = {}

        if inputs is None:
            self.inputs = []
        else:
            if not isinstance(inputs, list):
                inputs = [inputs]
            self.inputs = [self._get_tensor_name(x) for x in inputs]
        if outputs is None:
            self.outputs = []
        else:
            if not isinstance(outputs, list):
                outputs = [outputs]
            self.outputs = [self._get_tensor_name(x) for x in outputs]

        # We would like a stack so that we run conversion sequentially.
        self.graph_stack = self._get_stack(tfssa, root="main")
        self.context = TranscriptionContext()

    def _get_stack(self, tfssa, root="main"):
        # We're trying to get a order of how to loop through the graphs.
        # This is NOT necessarily a DAG.
        dep = {x: [] for x in tfssa.functions}
        for fname in tfssa.functions:
            for node in tfssa.functions[fname].graph.values():
                if node.op == 'while':
                    bfunc = node.attr['body_function']
                    cfunc = node.attr['cond_function']
                    if fname not in dep[bfunc]:
                        dep[bfunc].append(fname)
                    if fname not in dep[cfunc]:
                        dep[cfunc].append(fname)

        assert (len(dep[root]) == 0)
        graph_stack = simple_topsort(dep)

        return graph_stack

    def _get_tensor_name(self, tensor):
        if isinstance(tensor, six.string_types):
            return tensor
        ret = tensor.name
        return ret.split(":")[0]

    @staticmethod
    def _create_placeholder(node):
        node.parse_from_attr()
        shape = []
        dtype = node.attr['dtype']
        if builtins.is_tensor(node.datatype):
            shape = node.datatype.get_shape()
        return cb.placeholder(shape, dtype=dtype)

    def convert_graph(self, prog, graph, gname):
        nodes = topsort(graph)
        graph_inputs = {}
        for node_name in nodes:
            node = graph[node_name]
            if node.op == 'Placeholder':
                graph_inputs[node.name] = \
                    TFConverter._create_placeholder(node)

        with SsaFunction(graph_inputs) as ssa_func:
            # Get the input Var
            for name in graph_inputs.keys():
                self.context.add(name, ssa_func.inputs[name])

            # Translate the non-placeholder ops.
            for node_name in nodes:
                node = graph[node_name]
                if node.op == 'Placeholder':
                    continue
                _add_op = _OPS_REGISTRY.get(node.op, None)
                logging.info("Converting op {} ({})".format(node.op, node_name))

                if _add_op is None:
                    raise RuntimeError(
                        "TensorFlow convert function for op '" + node.op + "' not implemented")
                _add_op(self.context, node)
                check_output_shapes(self.context, node)
            outputs = [self.context[o] for o in self.outputs]
            ssa_func.set_outputs(outputs)
            prog.add_function(gname, ssa_func)

    def convert(self):
        prog = SsaProgram()
        for gname in self.graph_stack:
            if gname != "main":
                raise NotImplementedError()
            graph = self.tfssa.functions[gname].graph
            self.convert_graph(prog, graph, gname)
        return prog

