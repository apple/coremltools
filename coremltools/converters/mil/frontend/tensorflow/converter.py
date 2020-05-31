import six
import logging
from coremltools.converters.mil.mil.var import Var
from coremltools.converters.mil.mil import get_new_symbol

from coremltools.converters.mil.mil import types
from .basic_graph_ops import topsort, simple_topsort

from .ops import *  # register the ops
from .convert_utils import convert_graph

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Program
from coremltools.converters.mil.mil import Function
from .ssa_passes.tf_passes import tensorflow_passes
from coremltools.converters._profile_utils import profile

# make tensorflow_passes a variable because it could be overwritten
# later on. E.g., TF2 uses different set of graph passes then TF1.
tensorflow_passes = tensorflow_passes


# TranscriptionContext maintains a map of tf_node.name --> ssa_var available
# to the current TF --> tfssa transcription.
class TranscriptionContext:
    def __init__(self, name=None):
        self.name = name if name is not None else ""
        self.context = {}
        self.graphs = {}

        # TF loops are represented as functions, so nested loops becomes
        # stacked functions. Stacked functions are translated to nested
        # blocks in Program, like
        #
        # while_loop(loop_vars=(%a, %b))
        #  cond_block1(%a.x, %b.x) {
        #    ...some ops
        #  } -> (%bool_var1)
        #  body_block1(%a.x, %b.x) {
        #    %ret_axx = while_loop(loop_vars=(%a.x,))
        #      cond_block2(%a.x.x) {
        #        ...some ops
        #      } -> (%bool_var2)
        #      body_block2(%a.x.x) {
        #       ...some ops
        #      } -> (%new_a.x.x)
        #    } -> (%ret_axx)
        #    ....some ops using %ret_a
        #  } -> (%ret_ax, %ret_bx)
        #
        # During the translation of cond_block2, we'd have func_input_stack
        #
        # (%a.x.x,)
        # (%a.x, %b.x)
        #
        # where [%a.x.x] would be unstacked once cond_block2 is done.
        self.func_input_stack = []  # list of tuple[Var]

    def add(self, tf_name, ssa_vars, is_new_var=True):
        """
        ssa_vars: list[Var] / tuple[Var] (multiple outputs) or
        Var (single_output)
        is_new_var: True if ssa_vars are newly created for tf_name.
        """
        if tf_name in self.context:
            logging.warning("TF var %s is added again.", tf_name)
            return
        if is_new_var and isinstance(ssa_vars, Var) and \
                tf_name != ssa_vars.name:
            msg = 'tf_name: != ssa_vars.name: {} != {}'
            raise ValueError(msg.format(tf_name, ssa_vars.name))
        self.context[tf_name] = ssa_vars

    def add_graph(self, graph_name, graph):
        self.graphs[graph_name] = graph

    def get_graph(self, graph_name):
        if graph_name not in self.graphs:
            msg = "Graph '{}' not found in: {}"
            raise KeyError(msg.format(graph_name, list(self.graphs.keys())))
        return self.graphs[graph_name]

    def stack_func_inputs(self, inputs):
        self.func_input_stack.append(inputs)

    def unstack_func_inputs(self):
        if len(self.func_input_stack) == 0:
            raise ValueError('No func input available')
        self.func_input_stack.pop()

    def get_func_inputs(self):
        if len(self.func_input_stack) == 0:
            raise ValueError('No func input available')
        return self.func_input_stack[-1]

    def __getitem__(self, tf_name):
        if tf_name not in self.context:
            msg = "TF var {} not found in context {}"
            raise KeyError(msg.format(tf_name, self.name))
        return self.context[tf_name]

    def __contains__(self, tf_name):
        return tf_name in self.context


class TFConverter:
    def __init__(self, tfssa, inputs=None, outputs=None, **kwargs):
        """
        tfssa: TensorFlow IR.
        inputs: dict[str -> list/tuple] or list[str], optional, defaults to None.
            Dictionary containing {name: shape} for each input or list of input
            names. It None, the converter will try to extract the input information
            by assuming all Placeholder or PlaceholderWithDefault as inputs.
        outputs: list of str or str, optional, defaults to None.
            A list of names of the output nodes or a str for single output name.
            If None, the converter will try to extract the output information from
            TensorFlow model.
        """
        self.tfssa = tfssa
        self.global_type = {}

        main_func = tfssa.functions['main']
        graph = main_func.graph

        # Filter the inputs to only Placeholder names
        placeholder_names = [n for n in graph if graph[n].op == 'Placeholder']
        if isinstance(inputs, dict):
            inputs = {name: shape for name, shape in inputs.items() \
                    if name in placeholder_names}
        elif isinstance(inputs, list):
            inputs = [name for name in inputs if name in placeholder_names]

        # Instantiate self.inputs as list[str] of node names
        if inputs is None:
            self.inputs = main_func.inputs
        elif isinstance(inputs, list):  # set given input shape
            self.inputs = inputs
        elif isinstance(inputs, dict):  # set given input shape
            self.inputs = list(inputs.keys())
            for name, shape in inputs.items():
                node = graph[name]
                node.attr['_output_shapes'] = [shape] # list of length 1

        # infer outputs if not provided
        self._validate_outputs(tfssa, outputs)
        outputs = main_func.outputs if outputs is None else outputs
        outputs = outputs if isinstance(outputs, (tuple, list)) else [outputs]
        outputs = [x if isinstance(x, six.string_types) else x.name for x in outputs]
        self.outputs = outputs

        # We would like a stack so that we run conversion sequentially.
        self.graph_stack = self._get_stack(tfssa, root="main")
        self.context = TranscriptionContext()

    def _get_stack(self, tfssa, root="main"):
        # We're trying to get a order of how to loop through the graphs.
        # This is NOT necessarily a DAG.
        dep = {x: [] for x in tfssa.functions}
        for fname in tfssa.functions:
            for node in tfssa.functions[fname].graph.values():
                func_x, func_y = None, None

                if node.op in {'StatelessIf', 'If'}:
                    func_x = node.attr.get('then_branch')
                    func_y = node.attr.get('else_branch')
                elif node.op in {'StatelessWhile', 'While'}:
                    func_x = node.attr.get('body')
                    func_y = node.attr.get('cond')
                elif node.op == 'while':
                    func_x = node.attr['body_function']
                    func_y = node.attr['cond_function']

                if func_x and fname not in dep[func_x]:
                    dep[func_x].append(fname)
                if func_y and fname not in dep[func_y]:
                    dep[func_y].append(fname)

        assert (len(dep[root]) == 0)
        graph_stack = simple_topsort(dep)

        return graph_stack

    @staticmethod
    def _get_tensor_name(tensor):
        ret = None
        if isinstance(tensor, six.string_types):
            ret = tensor
        else:
            ret = tensor.name
        return ret.split(":")[0]

    @staticmethod
    def _create_placeholder(node):
        node.parse_from_attr()
        shape = []
        dtype = node.attr['dtype']
        if types.is_tensor(node.datatype):
            shape = node.datatype.get_shape()
            shape = tuple(get_new_symbol() if s is None or s < 0 else s for s in shape)
        return mb.placeholder(shape, dtype=dtype)

    def _validate_outputs(self, tfssa, outputs):
        if outputs is None:
            return
        outputs = outputs if isinstance(outputs, (tuple, list)) else [outputs]
        output_nodes = []
        for f in tfssa.functions.values():
            output_nodes += list(f.outputs)
        all_nodes = []
        for f in tfssa.functions.values():
            all_nodes += list(f.graph.keys())
        for n in outputs:
            if self._get_tensor_name(n) not in output_nodes + all_nodes:
                raise KeyError('Output node name "{}" does exist.'.format(n))

    def convert_main_graph(self, prog, graph):
        func_inputs = {}
        for name in self.inputs:
            node = graph[name]
            func_inputs[name] = TFConverter._create_placeholder(node)

        with Function(func_inputs) as ssa_func:
            # Get the input Var
            for name in func_inputs.keys():
                self.context.add(name, ssa_func.inputs[name])
            outputs = convert_graph(self.context, graph, self.outputs)
            ssa_func.set_outputs(outputs)
            prog.add_function('main', ssa_func)

        # Rename outputs to TF's name. This is needed when the last op doesn't
        # generate a new Var (e.g., get_tuple, Identity etc.), and thus the
        # last Var would have a different name than the last TF op's name.
        #
        # Example:
        #
        # TF code:
        #    x = tf.placeholder(tf.float32, shape=(1,))
        #    y = tf.placeholder(tf.float32, shape=(1,))
        #    c = lambda i, j: \
        #            tf.less(tf.math.reduce_mean(i), tf.math.reduce_mean(j))
        #    b = lambda i, j: (tf.add(i, 1), j)
        #    res = tf.while_loop(c, b, [x, y])
        #
        # Resulting nodes (excluding the nodes in while loop cond & body):
        #
        # node name: Placeholder op type: Placeholder inputs: []
        # node name: Placeholder_1 op type: Placeholder inputs: []
        # node name: make_input_0 op type: make_tuple inputs: ['Placeholder',
        #         'Placeholder_1']
        # node name: while_0 op type: while inputs: ['make_input_0']
        # node name: while/Exit op type: get_tuple inputs: ['while_0']
        # node name: while/Exit_1 op type: get_tuple inputs: ['while_0']
        #
        # Observe that return node `while/Exit` is an output from get_tuple,
        # which in our translation simply unpack a python tuple of Vars
        # ('while_0:0', 'while_0:1') returned from while_0 SSA op. We need to
        # rename `while_0:0` to `while/Exit` in order for users to find the
        # output.

        for v_o, out_name in zip(prog['main'].outputs, self.outputs):
            if v_o.name != out_name:
                logging.info("Renaming output var: '{}' -> '{}'".format(
                    v_o.name, out_name))
                v_o.name = out_name

    @profile
    def convert(self):
        prog = Program()
        if len(self.graph_stack) == 0:
            raise ValueError('At least one TF function must be present')
        if self.graph_stack[0] != 'main':
            msg = 'TF root graph must be named \'main\'. Got {}'
            raise ValueError(msg.format(self.graph_stack[0]))
        graph = self.tfssa.functions['main'].graph
        for g_name in self.graph_stack[1:]:
            self.context.add_graph(g_name, self.tfssa.functions[g_name].graph)
        self.convert_main_graph(prog, graph)

        # Apply TF frontend passes on Program. These passes are different
        # from passes applied to tfssa.
        tensorflow_passes(prog)

        return prog
