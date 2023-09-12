```{eval-rst}
.. index::
    single: ML program; compared to neural network
    single: neural network; compared to ML program
```


# Comparing ML Programs and Neural Networks

As ML models evolve in sophistication and complexity, their [representations](#model-representations) are also evolving to describe how they work. _ML programs_ are models that are represented as operations in code. The ML program model type is the foundation for future Core ML improvements.

A few of the major the differences between a neural network and an ML program are as follows:

| Neural Network | ML Program |
| ----------- | ----------- |
| Layers with a computational graph representation | Operations with a programmatic representation |
| Weights embedded in layers | Weights decoupled and serialized |
| Intermediate tensor type implicit | Intermediate tensor type explicit |
| Limited control over precision | More granular control over precision |


```{eval-rst}
.. index:: 
	single: typed execution
	single: GPU runtime
	single: Metal Performance Shaders Graph framework
	single: compiling an ML program
	single: ML program; compiling
```

## ML Program Benefits

Converting to an ML program provides the following advantages over converting to a neural network:

- _Typed execution_ for control of numeric precision:
    
    An ML program defines types for the intermediate tensors that are produced by one segment of the model and consumed by another. The ML program runtime respects these specified types as the minimum compute precision. As a result, you can easily control the precision when you convert the model. By comparison, the neural network precision type is tied to the compute unit. For details, see [Typed Execution](typed-execution).

- _GPU runtime_ for float 32 precision:
    
    ML programs use a GPU runtime that is backed by the [Metal Performance Shaders Graph framework](https://developer.apple.com/documentation/metalperformanceshadersgraph). While the GPU with neural networks supports only float 16 precision at runtime, this execution path supports both float 16 and float 32 precision. This path also provides performance improvements, especially for newer devices.

- More efficient compiling:
    
    Since ML programs do not store weights in the protobuf file format, the models are more memory-efficient to compile. You can  significantly improve performance by using Core ML's on-device compilation API. For details, see [Downloading and Compiling a Model on the User’s Device](https://developer.apple.com/documentation/coreml/core_ml_api/downloading_and_compiling_a_model_on_the_user_s_device).

## Model Representations

An `mlprogram` model type represents a neural network in a different way than the original `neuralnetwork` model type, and the Core ML framework uses different runtimes when executing them.

There are several ways to represent a deep learning model. At a foundational level, models can be represented as a set of mathematical equations (as shown in the left side of the following figure), which is how they are often described in a machine learning course for beginners.


```{figure} images/model-representations-no-numbers.png
:alt: Ways to represent a deep learning model
:align: center
:class: imgnoborder

Three ways to represent a deep learning model.
```

```{eval-rst}
.. index:: Core ML NeuralNetwork
```

To express a neural network, the mathematical descriptions are often abstracted into a _computational graph_, which is a more concise and scalable representation (as shown in the center of the previous figure). A computational graph is a directed graph in which computational layers connect to each other — the input feeds into the _source_ layers and undergoes a series of mathematical transformations to generate the outputs through the _sink_ layers. 

At the center of the Core ML [NeuralNetwork](https://apple.github.io/coremltools/mlmodel/Format/NeuralNetwork.html) are the layers that form the nodes of the graph. There are many different types of layers, and each describes a particular mathematical operation. Each layer specifies its input and output names, which are the _edges_ of the graph. Connections are established by matching an output name of a layer to another layer’s input name. 

Although the graphical representation is useful for understanding and visualization, the graphs are actually special kinds of programs. A neural network is just like any other program and can be directly represented and stored in a code-like form (as shown in the right side of the previous figure). By representing the model as code, the ML program model provides more flexibility in defining neural networks by incorporating programmatic features such as functions, blocks, control flow, and strongly typed variables. 

## Differences in Detail

An ML program differs from a neural network in the following ways:

- _Ops vs. layers_:
    
    An ML Program consists of a _main_ function, which includes one or more blocks. Each block consists of a series of operations ("ops" for short), which are versioned as a set ("opset" for short).
    
    While the layers in a neural network are fully specified in the protobuf serialization file format itself ([`Neuralnetwork.proto`](https://github.com/apple/coremltools/blob/main/mlmodel/format/NeuralNetwork.proto)), the supported ops for an ML program are specified in [MIL ops/defs](https://github.com/apple/coremltools/tree/main/coremltools/converters/mil/mil/ops/defs) (see the [MIL Ops Reference](https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html#mil-ops)), not in [MIL proto](https://github.com/apple/coremltools/blob/main/mlmodel/format/MIL.proto) messages.

- _Inputs without parameters_:
    
    A neural network layer distinguishes between the inputs and the parameters comprised of trained weights and other constants. By comparison, an ML program op has only inputs. (An exception is the `const` op which has attributes.) All the inputs are named, so their position is irrelevant. Some of the inputs are constrained to be constants. 

- _Variables are typed_: 
    
     An ML program is typed (for details, see [Typed Execution](typed-execution)). The inputs and outputs of an op are called variables ("vars" for short). Each var has a type and shape associated with it.

- _Weights are serialized separately_:
    
    The weights for an ML program are serialized outside of the protobuf format. Since the architecture and weights are separated, an ML program can be saved only as a [Core ML model package](https://developer.apple.com/documentation/coreml/core_ml_api/updating_a_model_file_to_a_model_package). By comparison, neural networks save its entire representation in a single protobuf file and therefore can be saved as either an `.mlmodel` file or as an `.mlpackage` model package. 

To see the ML program representation that is executed, compile a model package of type `mlprogram` to the `mlmodelc` archive. You can then find the `model.mil` file inside the contents, which is the ML program expressed in a text format. To compile an `.mlpackage`, either compile your Xcode project with the `.mlpackage` added to it, or execute the following on the command line, which places the compiled archive in the current working directory:

```shell
xcrun coremlcompiler compile model.mlpackage .
```

While the new ML program model type supports most of the functionality supported by neural networks, the following components are not yet supported. If you are using one of these features, you may want to continue to use the neural network format:

- [On-device update](updatable-model-examples):
    
    You can still get the benefit of the ML program runtime by using a `pipeline` of the ML program, followed by a neural network consisting of just the updatable portion.

- [Custom operators](custom-operators) for neural networks, and your own Swift implementations:
    
    While custom layers may be useful for new layer types, in most cases [composite operators](composite-operators) built using the MIL builder library are sufficient for converter to handle unsupported layer types. Composite layers, which are built with existing layers, perform better than custom operators and work seamlessly with both neural network and ML program model types. 

- [Integer and LUT weight-only quantization](quantization-overview):
    
    ML programs currently support both float 16 and float 32 typed weights and activations.

```{eval-rst}
.. index:: MIL, Model Intermediate Language
```

## ML Programs and MIL

ML program refers to the model type and the language in which they are described is referred to as the [Model Intermediate Language](model-intermediate-language) (MIL). MIL was introduced with coremltools version 4 as an internal intermediate representation (IR) in the converter stack. 

```{admonition} Core ML model support

In addition to ML programs and neural networks, Core ML supports several other model types including trees, support vector machines (SVMs), and general linear models (GLMs).
```

MIL enables the unification of TensorFlow 1, TensorFlow 2, and PyTorch converters through the [Unified Conversion API](unified-conversion-api). The converter works as follows: 

1. The beginning representation (TensorFlow or PyTorch), referred to as the "frontend", is converted to a Python MIL object with TensorFlow/PyTorch centric MIL opsets. This is then simplified by frontend specific passes.

2. The result is translated into the common MIL intermediate representation (IR). 

3. MIL is simplified through a sequence of passes. 

4. Finally, MIL is translated into one of the backends for serialization on disk. In coremltools version 4, the backend was `neuralnetwork`. In subsequent versions `mlprogram` is available as another backend.

The MIL IR for the `neuralnetwork` model type must be further converted to the [`NeuralNetwork`](https://apple.github.io/coremltools/mlmodel/Format/NeuralNetwork.html) specification. By comparison, the translation to the ML program specification is straightforward because the same internal MIL Python object is converted to the protobuf format ([MIL.proto](https://github.com/apple/coremltools/blob/main/mlmodel/format/MIL.proto)). The opset of ML program is identical to the ops available in the MIL builder. For details, see the [MIL Ops Reference](https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html) and the [Python source code](https://github.com/apple/coremltools/tree/main/coremltools/converters/mil/mil/ops/defs).

