```{eval-rst}
.. index::
    single: custom operators
```

# Custom Operators

While converting a model to [Core ML](https://developer.apple.com/documentation/coreml), you may encounter an unsupported operation that can't be represented by a [composite operator](composite-operators).

In such cases you can create a _custom layer_ in your model for the custom operator, and implement the [Swift](https://developer.apple.com/swift/) classes that define the operator's computational behavior. For instructions, see [Creating and Integrating a Model with Custom Layers](https://developer.apple.com/documentation/coreml/core_ml_api/creating_and_integrating_a_model_with_custom_layers).

```{admonition} Use Custom Operators as a Last Resort

A custom operator is not easy to implement. Use a custom operator only if you can't get the performance you want, and only as a last resort if:

- The functionality you need is not supported in the [Core ML API](https://developer.apple.com/documentation/coreml/core_ml_api).
- You can't represent the functionality with a [composite operator](composite-operators). Whenever possible, use a composite operator, which is more efficient than a custom operation, and compiles down to various hardware backends available on your device.
```

## Developer Workflow

The following example uses a custom operator to define a TopK operator. The custom operator needs a custom optimized implementation to apply sorting in place within TopK, which avoids the need for an extra composite operator.

The workflow for creating a custom operator is as follows:

1. Register a [Model Intermediate Language (MIL)](model-intermediate-language) operator.
2. Define the operator to use the custom operator from step 1.
3. Convert the model.
4. Implement the custom operator in Swift, adhering to the binding information provided in step 1.

## Step 1: Register the MIL Operator

1. Define the [MIL](https://coremltools.readme.io/docs/model-intermediate-language) operator using the `register_op` decorator. To specify that the given operator is custom, set `is_custom_op` to `True`.

2. As part of the operator input specification, type inference, and (optionally) value inference, specify bindings as a member of a given operator.

The binding dictionary that communicates with the Swift API is specified as `binding` and has the following properties:

- `class_name`: The name of the class. This is the interface name of the custom layer implementation.
- `input_order`: The input order, from the above named input used in the custom implementation. Inputs will be packed as a `List` of `Multi-Array` and passed in this order to the [`evaluate(with:)`](https://developer.apple.com/documentation/foundation/nspredicate/1417924-evaluate) Swift API.
- `parameters`: The parameters that should be passed as operator attributes and known statically.
- `description`: A short description of the current operator.

The following code snippet shows how to define a custom operator. You may not need all of the imported input types:

```python custom_mil_ops.py
# Imports for custom ops (not all may be required)
from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op
from coremltools.converters.mil.mil.types.symbolic import is_symbolic
from coremltools.converters.mil.mil import (
    Builder as mb,
    Operation,
    types
)
from coremltools.converters.mil.mil.input_type import (
    BoolInputType,
    DefaultInputs,
    InputSpec,
    TensorInputType,
    IntInputType,
    FloatInputType,
    ListInputType,
    StringInputType,
)


@register_op(doc_str='Custom TopK Layer', is_custom_op=True)
class custom_topk(Operation):
    input_spec = InputSpec(
             x = TensorInputType(type_domain="T"),
             k = TensorInputType(const=True, optional=True, type_domain=types.int32),
          axis = TensorInputType(const=True, optional=True, type_domain=types.int32),
        sorted = TensorInputType(const=True, optional=True, type_domain=types.bool),
    )
    
    type_domains = {
        "T": (types.fp16, types.fp32),
    }

    bindings = { 'class_name'  : 'CustomTopK',
                 'input_order' : ['x'],
                 'parameters'  : ['k', 'axis', 'sorted'],
                 'description' : "Top K Custom layer"
                }

    def __init__(self, **kwargs):
        super(custom_topk, self).__init__(**kwargs)

    def type_inference(self):
        x_type = self.x.dtype
        x_shape = self.x.shape
        k = self.k.val
        axis = self.axis.val

        if not is_symbolic(x_shape[axis]) and k > x_shape[axis]:
            msg = 'K={} is greater than size of the given axis={}'
            raise ValueError(msg.format(k, axis))

        ret_shape = list(x_shape)
        ret_shape[axis] = k
        return types.tensor(x_type, ret_shape), types.tensor(types.int32, ret_shape)
```

## Step 2: Define a TensorFlow Composite Operator

[TensorFlow](https://www.tensorflow.org/) and [PyTorch](https://pytorch.org/) operators are used to define conversion to [MIL](model-intermediate-language) operators. It is therefore mandatory to define new TensorFlow or PyTorch operators to use the custom operator introduced in Step 1. After defining the custom MIL op, define a translation function for conversion (which is similar to defining a composite op). For this example use the `mb.custom_topk()` custom op defined in Step 1:

```python custom_tf_ops.py
# Import MIL builder
from coremltools.converters.mil.mil import Builder as mb
# Import TensorFlow registration utility
from coremltools.converters.mil.frontend.tensorflow.tf_op_registry import register_tf_op
# Import custom MIL op defined above
from custom_mil_ops import custom_topk

# Override TopK op with override=True flag
@register_tf_op(tf_alias=['TopKV2'], override=True)
def CustomTopK(context, node):
    x = context[node.inputs[0]]
    k = context[node.inputs[1]]
    sorted = node.attr.get('sorted', False)
    x = mb.custom_topk(x=x, k=k.val, axis=-1, sorted=sorted, name=node.name)
    context.add(x)
```

## Step 3: Convert the Model

Since the TopK MIL TensorFlow implementation is overridden, import it before the conversion to put it into use:

```python model_convert.py
import coremltools as ct
from custom_tf_ops import CustomTopK
// ..
// tf_model loaded here
// ..
model_from_tf = ct.convert(tf_model)
```

## Step 4: Implement Classes in Swift

The Python code defines only the custom op's name, properties, and so on. You need to code its actual implementation in Swift.

The Swift implementation must provide the API endpoints specified in [the MLCustomLayer interface](https://developer.apple.com/documentation/coreml/mlcustomlayer).

```{warning}

Binding information provided while creating the MIL operator in Step 1 must match the binding with the Swift API.
```

For a complete example of implementing a custom layer, see [this detailed example](https://machinethink.net/blog/coreml-custom-layers/).

```{admonition} Custom Layer Support

Custom layers are supported when the conversion is targeted at the "Neural Network" backend. They are not available when using the [ML Programs](convert-to-ml-program) backend.
```

