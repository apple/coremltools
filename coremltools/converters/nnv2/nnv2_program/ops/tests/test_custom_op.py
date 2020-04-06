from coremltools.converters.nnv2 import testing_reqs
from coremltools.converters.nnv2.testing_reqs import *
backends = testing_reqs.backends
from coremltools.converters.nnv2.frontend.tensorflow.tf_op_registry import register_tf_op
from coremltools.converters.nnv2.nnv2_program.ops.defs._op_reqs import *
from coremltools.converters.nnv2.nnv2_program.ops import CoremlBuilder as cb
from coremltools.converters.nnv2.builtin_types.symbolic import (
                            any_symbolic, any_variadic, is_symbolic)
from coremltools.converters.nnv2.frontend.tensorflow.test.testing_utils import tf_graph_to_proto

class TestCustomMatMul:
    # Define SSA Custom Op for Sparse MatMul
    # This will map to `custom_op` in SSA with binding information
    # to bind input spec to the custom implementation
    @register_op(doc_str='Sparse MatMul Layer', is_custom_op=True)
    class sparse_matmul(Operation):
        # Defining input spec for current op
        input_spec = InputSpec(
                      x = TensorInputType(),
                      y = TensorInputType(),
            transpose_x = BoolInputType(const=True, default=False),
            transpose_y = BoolInputType(const=True, default=False),
            x_is_sparse = BoolInputType(const=True, default=False),
            y_is_sparse = BoolInputType(const=True, default=False),
        )

        # Specifying binding for custom op for specifying inputs,
        # parameters required for creating custom op to be synced with Swift API
        bindings = { 'class_name'  : 'SparseMatMul',
                     'input_order' : ['x', 'y'],
                     'parameters'  : ['transpose_x', 'transpose_y', 'x_is_sparse', 'y_is_sparse'],
                     'description' : "Custom Sparse MatMul Layer"
                    }

        def __init__(self, **kwargs):
            super(TestCustomMatMul.sparse_matmul, self).__init__(**kwargs)

        def type_inference(self):
            x_type = self.x.dtype
            x_shape = self.x.shape
            y_shape = self.y.shape
            # For illustration purpose, assumming getting valid shape
            # Ideally, should consider transpose_?, ?_is_sparse parameters into consideration
            # for computing output shape
            ret_shape = [x_shape[0], y_shape[1]]
            return builtins.tensor(x_type, [x_shape[0], y_shape[1]])

    # TensorFlow Sort Op
    @register_tf_op()
    def SparseMatMul(context, node):
        a = context[node.inputs[0]]
        b = context[node.inputs[1]]
        transpose_a = node.attr.get('transpose_a', False)
        transpose_b = node.attr.get('transpose_b', False)
        a_is_sparse = node.attr.get('a_is_sparse', False)
        b_is_sparse = node.attr.get('b_is_sparse', False)
        
        x = cb.sparse_matmul(x=a, y=b, transpose_x=transpose_a, transpose_y=transpose_b,
                             x_is_sparse=a_is_sparse, y_is_sparse=b_is_sparse, name=node.name)
        context.add(node.name, x)

    @pytest.mark.skipif(not HAS_TF1, reason=MSG_TF1_NOT_FOUND)
    @pytest.mark.parametrize('use_cpu_only, backend, transpose_a,transpose_b,'
                             'a_is_sparse, b_is_sparse',
                             itertools.product(
                                 [True],
                                 backends,
                                 [True, False],
                                 [True, False],
                                 [True, False],
                                 [True, False]
                             ))
    def test_tf1(self, use_cpu_only, backend, transpose_a, transpose_b,
                 a_is_sparse, b_is_sparse):
        rank = 2
        shape = list(np.random.randint(low=3, high=100, size=1)) * rank
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=shape)
            y = tf.placeholder(tf.float32, shape=shape)
            ref = tf.sparse_matmul(x, y, transpose_a=transpose_a, transpose_b=transpose_b, a_is_sparse=a_is_sparse, b_is_sparse=b_is_sparse)
            tf.io.write_graph(graph, '/tmp/', 'sort.pb')
            spec, _, _, _ = tf_graph_to_proto(graph, {x: random_gen(shape, rand_min=-100, rand_max=100),
                                                      y: random_gen(shape, rand_min=-100, rand_max=100)},
                                              ref, backend=backend, add_custom_layer=True)
            layers = spec.neuralNetwork.layers
            assert layers[-1].custom is not None, "Expecting a custom layer"
            assert 'SparseMatMul' == layers[-1].custom.className, "Custom Layer class name mis-match"
            assert transpose_a == layers[-1].custom.parameters['transpose_x'].boolValue, "Incorrect parameter value k"
            assert transpose_b == layers[-1].custom.parameters['transpose_y'].boolValue, "Incorrect parameter value k"
            assert a_is_sparse == layers[-1].custom.parameters['x_is_sparse'].boolValue, "Incorrect parameter value k"
            assert b_is_sparse == layers[-1].custom.parameters['y_is_sparse'].boolValue, "Incorrect parameter value k"


# Example and test for Overriding TF op
# Following example is disabled because
# 1. Registering tf op overrides existing behavior and leads to
#    failure of `top_k` op tests.
# 2. Commenting the example so that later can be moved to an example section
#    for developers
# TODO: rdar://61241807 ([NNv2] [Polish] Custom layer operator documentation)

# class TestCustomTopK:
#     # Defining SSA TopK Op
#     @register_op(doc_str='Custom TopK Layer', is_custom_op=True)
#     class custom_topk(Operation):
#         input_spec = InputSpec(
#                  x = TensorInputType(),
#                  k = IntInputType(const=True, default=1),
#               axis = IntInputType(const=True, default=-1),
#             sorted = BoolInputType(const=True, default=False),
#         )

#         bindings = { 'class_name'  : 'TopK',
#                      'input_order' : ['x'],
#                      'parameters'  : ['k', 'axis', 'sorted'],
#                      'description' : "Top K Custom layer"
#                     }

#         def __init__(self, **kwargs):
#             super(TestCustomTopK.custom_topk, self).__init__(**kwargs)

#         def type_inference(self):
#             x_type = self.x.dtype
#             x_shape = self.x.shape
#             k = self.k.val
#             axis = self.axis.val

#             if not is_symbolic(x_shape[axis]) and k > x_shape[axis]:
#                 msg = 'K={} is greater than size of the given axis={}'
#                 raise ValueError(msg.format(k, axis))

#             ret_shape = list(x_shape)
#             ret_shape[axis] = k
#             return builtins.tensor(x_type, ret_shape), builtins.tensor(builtins.int32, ret_shape)

#     # Override TopK op with override=True flag
#     @register_tf_op(tf_alias=['TopKV2'], override=True)
#     def CustomTopK(context, node):
#         x = context[node.inputs[0]]
#         k = context[node.inputs[1]]
#         sorted = node.attr.get('sorted', False)
#         x = cb.custom_topk(x=x, k=k.val, axis=-1, sorted=sorted, name=node.name)
#         context.add(node.name, x)

#     @pytest.mark.skipif(not HAS_TF1, reason=MSG_TF1_NOT_FOUND)
#     @pytest.mark.parametrize('use_cpu_only, backend, rank, k',
#                              itertools.product(
#                                  [True],
#                                  backends,
#                                  [rank for rank in range(1, 4)],
#                                  [1, 2],
#                              ))
#     def test_tf1(self, use_cpu_only, backend, rank, k):
#         shape = np.random.randint(low=3, high=6, size=rank)
#         with tf.Graph().as_default() as graph:
#             x = tf.placeholder(tf.float32, shape=shape)
#             ref = tf.math.top_k(x, k=k, sorted=True)
#             ref = (ref[1], ref[0])
#             spec, _, _, _ = tf_graph_to_proto(graph, {x: random_gen(shape, rand_min=-100, rand_max=100)},
#                                               ref, backend=backend, add_custom_layer=True)
#             layers = spec.neuralNetwork.layers
#             assert layers[-1].custom is not None, "Expecting a custom layer"
#             assert 'TopK' == layers[-1].custom.className, "Custom Layer class name mis-match"
#             assert k == layers[-1].custom.parameters['k'].intValue, "Incorrect parameter value k"
#             assert True == layers[-1].custom.parameters['sorted'].boolValue, "Incorrect parameter value for Sorted"