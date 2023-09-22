# MIL Graph Pass

## For Users

This guide describes the passes that optimize an MIL Program.

### Overview

In Core ML Tools, the conversion process, as described in [Model Intermediate Language](https://coremltools.readme.io/docs/model-intermediate-language#overview),
is roughly divided into the following stages based on the model representation:

1. Frontend (PyTorch/TensorFlow/etc --> Model Intermediate Language (MIL) Program)
2. MIL-based Graph Optimizations
3. Backend (MIL --> NeuralNetworks/MLProgram Proto)

The Program is a Python class for Core ML Tools's internal in-memory and Pythonic representation. It's the same class you would use when using the Core ML Tools [Python MIL Builder](https://coremltools.readme.io/docs/model-intermediate-language#create-a-mil-program) directly.

The Program consists of a `main` function implemented as a [`Block`](https://github.com/apple/coremltools/blob/main/coremltools/converters/mil/mil/block.py). Each `Block` contains a list of [`Operators`](https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html). Passes are applied to the Program representation to simplify and canonicalize it.

During each conversion, the graph passes are specified in a pass pipeline (the `pass_pipeline` parameter in `ct.convert`).
All available passes are recorded in `_PIPELINE_NAME_TO_PASSES` in `ct.PassPipeline`.
For a detailed description of each pass (including what it does and examples), see the
[coremltools API Reference](https://apple.github.io/coremltools/index.html).

You can find the code for all MIL passes at `coremltools/converters/mil/mil/passes/defs`.

In addition to the using default setting, you can:

* Use predefined builtin PassPipelines (see [Predefined Builtin PassPipeline Factory](#predefined-builtin-passpipeline)). We provide a set of predefined commonly used passpipelines that you can directly call.

* Decide which passes and the order of passes to run (see
[Specify Passes To Run](#specify-passes-to-run)). For example,
   - Switching off certain fusions to correctly export optimized models from coremltools.optimize.torch for palettization.
   - Skipping all passes to keep the MIL Program untouched.

* Set options for a specific pass to control the behaviour (see [Set Pass Option](#set-pass-option)). For example,
   - Setting a threshold in a constant elimination pass to trade off computation and model size.
   - Skipping ops in fp16 quantization casting.

* Define a custom graph pass to do fully customized optimization on the Program (see [Define Custom Graph Pass](#define-custom-graph-pass)).

### Predefined Builtin PassPipeline

We provide a set of predefined commonly used passpipeline to the users. Which includes:

* `coremltools.PassPipeline.EMPTY`: This skips all passes.

* `coremltools.PassPipeline.DEFAULT`: This is used by the converter by default.

* `coremltools.PassPipeline.CLEANUP`: This contains cleanup graph passes. For instance, `const_elimination`, `dead_code_elimination`, etc.

* `coremltools.PassPipeline.DEFAULT_PALETTIZATION`: This is used for the conversion of a palettized source model.

* `coremltools.PassPipeline.DEFAULT_PRUNING`: This is used for the conversion of a sparse source model.

### Specify Passes To Run

If no pass pipeline is specified, a default pipeline will be used:

```python
# The following two conversions are equivalent.
ct.convert(model, pass_pipeline=ct.PassPipeline.DEFAULT)
ct.convert(model)
```

To skip all passes, use an empty pipeline:

```python
pipeline = ct.PassPipeline.EMPTY
ct.convert(model, pass_pipeline=pipeline)
```

To run some specific passes in specific orders:

```python
pipeline = ct.PassPipeline(
    pass_names=["common::fuse_conv_batchnorm", "common::const_elimination"],
    pipeline_name="my_pipeline",
)
ct.convert(model, pass_pipeline=pipeline)
```

To inspect passes and their corresponding indexes in the pipeline:

```python
pipeline = ct.PassPipeline.DEFAULT
# Find indexes of a specific pass.
pass_indexes = [
    idx
    for idx, pass_name in enumerate(pipeline.passes)
    if pass_name == "common::reduce_transposes"
]
```

You can skip specific passes to avoid unwanted side effects. For example, to avoid fusing
the `conv` and `batchnorm`:

```python
pipeline = ct.PassPipeline.DEFAULT
pipeline.remove_passes({"common::fuse_conv_batchnorm"})
ct.convert(model, pass_pipeline=pipeline)
```

### Set Pass Option

You can set options specific to a certain pass.
Each pass option is an attribute of the corresponding pass class.
In the following example, you can see how `skip_const_by_size` is supported in `const_elimination`. You can also add options to existing passes or your custom passes.

The following example shows how to avoid folding too-large `const` ops that would lead to a large model:

```python
pipeline = ct.PassPipeline.DEFAULT
pipeline.set_options("common::const_elimination", {"skip_const_by_size": "1e6"})
ct.convert(model, pass_pipeline=pipeline)
```

You can also add options to existing passes or to your custom passes.

Another example is to skip ops during an fp16 quantization pass:

```python
pipeline = ct.PassPipeline.DEFAULT
pipeline.set_options("common::add_fp16_cast", {"skip_ops_by_type": "mul,const"})
ct.convert(model, pass_pipeline=pipeline)
```


## Define Custom Graph Pass

If the currently available
[MIL Graph Passes in the coremltools API Reference](https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.passes.defs.html#mil-graph-passes) do not meet your goal, you can  define custom graph passes.

To illustrate how to define a custom graph pass, the following example demonstrates merging consecutive `relu` ops using a PyTorch model with 2 `relu` layers. You can directly convert this model using the following script:

```python
import coremltools as ct
import torch


class TestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()

    def forward(self, x):
        return self.relu2(self.relu1(x))


model = TestModel()
model.eval()
x = torch.rand(1, 2, 3)
traced_model = torch.jit.trace(model, x)
converted_model = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=(1, 2, 3), name="input")],
    convert_to="mlprogram",
    compute_units=ct.ComputeUnit.CPU_ONLY,
    compute_precision=ct.precision.FLOAT32,
)

print(converted_model._get_mil_internal())
```

You should get the internal MIL graph as shown in the following output. There is only one `relu` op in the MIL graph because the pass `common::merge_consecutive_relus` in the default pipeline merges consecutive `relu` ops into a single `relu` layer.

```text
main[CoreML5](%input: (1, 2, 3, fp32)(Tensor)) {
  block0() {
    %var_3: (1, 2, 3, fp32)(Tensor) = relu(x=%input, name="input")
  } -> (%var_3)
}
```

You can then use the `pass_pipeline` API to remove that pass from the pipeline with the following code:

```python
pipeline = ct.PassPipeline.DEFAULT
pipeline.remove_passes(["common::merge_consecutive_relus"])
converted_model = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=(1, 2, 3), name="input")],
    convert_to="mlprogram",
    compute_units=ct.ComputeUnit.CPU_ONLY,
    compute_precision=ct.precision.FLOAT32,
    pass_pipeline=pipeline,
)

print(converted_model._get_mil_internal())
```

You will then get the following internal MIL graph in which there are two `relu` ops.

```text
main[CoreML5](%input: (1, 2, 3, fp32)(Tensor)) {
  block0() {
    %input_1: (1, 2, 3, fp32)(Tensor) = relu(x=%input, name="input")
    %var_3: (1, 2, 3, fp32)(Tensor) = relu(x=%input_1, name="op_3")
  } -> (%var_3)
}
```

Now, define your custom graph to do the same thing (merging consecutive relus):

```python
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import (
    _check_child_op_type,
    block_context_manager,
)
from coremltools.converters.mil.mil.passes.pass_registry import register_pass


@register_pass(namespace="mypass")
class my_merge_consecutive_relus(AbstractGraphPass):
    """
    Our custom graph pass, which merge consecutive relus.
    """

    def apply(self, prog):
        for f in prog.functions.values():
            self._merge_relus_in_block(f)

    def _match_and_replace_pattern(self, block, relu_op):
        if not (relu_op.op_type == "relu" and _check_child_op_type(relu_op, "relu")):
            return False

        child_relu_op = list(relu_op.outputs[0].child_ops)[0]
        return self._replace_ops(block, relu_op, child_relu_op)

    @staticmethod
    def _replace_ops(block, relu_op, child_relu_op):
        if relu_op.enclosing_block.try_replace_uses_of_var_after_op(
            anchor_op=relu_op,
            old_var=child_relu_op.outputs[0],
            new_var=relu_op.outputs[0],
        ):
            block.remove_ops([child_relu_op])
            return True
        return False

    @block_context_manager
    def _merge_relus_in_block(self, block):
        def help_merge_relu_ops(block):
            for op in list(block.operations):
                if self._match_and_replace_pattern(block, op):
                    return True
            return False

        block_changed = True
        while block_changed:
            block_changed = help_merge_relu_ops(block)
```

You can then remove the `common::merge_consecutive_relus` and insert your custom pass `mypass::my_merge_consecutive_relus`:

```python
pipeline = ct.PassPipeline.DEFAULT
# Find the index of the merge_consecutive_relus pass, where we will insert our custom pass.
pass_index = pipeline.passes.index("common::merge_consecutive_relus")
pipeline.remove_passes(["common::merge_consecutive_relus"])
pipeline.insert_pass(index=pass_index, pass_name="mypass::my_merge_consecutive_relus")

converted_model = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=(1, 2, 3), name="input")],
    convert_to="mlprogram",
    compute_units=ct.ComputeUnit.CPU_ONLY,
    compute_precision=ct.precision.FLOAT32,
    pass_pipeline=pipeline,
)

print(converted_model._get_mil_internal())
```

You find that the `relu` ops are successfully merged:

```text
main[CoreML5](%input: (1, 2, 3, fp32)(Tensor)) {
  block0() {
    %var_3: (1, 2, 3, fp32)(Tensor) = relu(x=%input, name="input")
  } -> (%var_3)
}
```

The custom graph pass is useful for local debugging. If you find that a custom graph pass may also benefit other use cases, please consider adding it as a new graph pass to the code base as described in the next section.


## Add a New Graph Pass

If you feel that a new graph pass is necessary, please follow these steps to contribute a new graph pass to the codebase.

1. Determine which category the graph pass would fall into, and then create a new graph pass class in the corresponding `defs/*.py`.

2. Once the new graph pass class is implemented, add a test for it in `tests/test_passes.py`.

3. Update `coremltools/docs/source/coremltools.converters.mil.mil.passes.defs.rst` to include the new pass in the documentation.

**Note:** The docstring for the pass class may need editing for correct Sphinx rendering. You can preview the documentation by first installing the packages in `reqs/docs.pip`, and then switching to the `coremltools/docs` directory and using the command `make html`. Open the generated `coremltools/docs/_build/html/index.html` to see if the docstring is formatted correctly.


## Code Style: Reduce Block Context Instantiation

Using the `block_context_manager` decorator is highly recommended, especially when the
original function involves calling `with block` multiple times. However, you may want to avoid recursively calling the function decorated with `block_context_manager`, since it involves expensive `_propagate_nonreplaceable_vars()`.

For details about how to use a `_noop_elimination_block_wrapper` to avoid that recursive calling, see  [noop_elimination](https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.passes.defs.html#coremltools.converters.mil.mil.passes.defs.cleanup.noop_elimination).
