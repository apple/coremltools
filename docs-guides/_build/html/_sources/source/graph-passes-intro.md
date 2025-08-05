```{eval-rst}
.. index::
    single: graph passes
    single: pass_pipeline
```

# Graph Passes

During conversion, Core ML Tools optimizes the model by applying graph transformations, called _graph passes_, which simplify and canonicalize the representation for a more efficient execution by the Core ML runtime. 

For example, [dead_code_elimination](https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.passes.defs.html#coremltools.converters.mil.mil.passes.defs.cleanup.dead_code_elimination) eliminates unused ops whose outputs do not contribute to final outputs. The [const_elimination](https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.passes.defs.html#coremltools.converters.mil.mil.passes.defs.cleanup.const_elimination) pass fuses ops with multiple constant inputs into a single constant, saving compute at runtime. The [fuse_elementwise_to_batchnorm](https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.passes.defs.html#coremltools.converters.mil.mil.passes.defs.optimize_elementwise_binary.fuse_elementwise_to_batchnorm) pass detects combinations of multiplication and add ops, and fuses them to a single `batchnorm` op.

Using the `pass_pipeline` parameter in [`ct.convert`](https://apple.github.io/coremltools/source/coremltools.converters.convert.html#coremltools.converters._converters_entry.convert), you can control which graph passes to run and the order of the graph passes. You can also specify options for each pass. For an overview, see the [MIL Graph Pass Guide](https://github.com/apple/coremltools/blob/main/coremltools/converters/mil/mil/passes/graph_pass.md) in the repo (`coremltools/coremltools/converters/mil/mil/passes/graph_pass.md`).

For a description of each graph pass, see [MIL Graph Passes](https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.passes.defs.html#mil-graph-passes) in the API Reference.
