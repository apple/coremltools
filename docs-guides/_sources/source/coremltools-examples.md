# Examples

The following are code example snippets and full examples of using Core ML Tools to convert models.

## For a Quick Start

Full example:

- [Getting Started](introductory-quickstart): Demonstrates how to convert an image classifier model trained using the TensorFlow Keras API to the Core ML format.

## ML Program with Typed Execution

Full example:

- [Typed Execution Workflow Example](typed-execution-example): Demonstrates a workflow for checking accuracy using [ML Programs](convert-to-ml-program) with [Typed Execution](typed-execution).

## TensorFlow 2

- [Load and Convert Model Workflow](load-and-convert-model)
- [TensorFlow 2 Workflow](tensorflow-2)
- [Convert a Pre-trained Model](tensorflow-2.md#convert-a-pre-trained-model)
- [Convert a User-defined Model](tensorflow-2.md#convert-a-user-defined-model)

Full examples:

- [Getting Started](introductory-quickstart): Demonstrates how to convert an image classifier model trained using the TensorFlow Keras API to the Core ML format.
- [Converting TensorFlow 2 BERT Transformer Models](convert-tensorflow-2-bert-transformer-models): Converts an object of the tf.keras.Model class and a SavedModel in the TensorFlow 2 format.

## TensorFlow 1

- [Convert From TensorFlow 1](load-and-convert-model.md#convert-from-tensorflow-1).
- [Export as Frozen Graph and Convert](tensorflow-1-workflow.md#export-as-a-frozen-graph-and-convert).
- [Convert a Pre-trained Model](tensorflow-1-workflow.md#convert-a-pre-trained-model).

Full examples:

- [Converting a TensorFlow 1 Image Classifier](convert-a-tensorflow-1-image-classifier): Demonstrates the importance of setting the image preprocessing parameters correctly during conversion to get the right results.
- [Converting a TensorFlow 1 DeepSpeech Model](convert-a-tensorflow-1-deepspeech-model): Demonstrates automatic handling of flexible shapes using automatic speech recognition.

## PyTorch

- [Convert from PyTorch](load-and-convert-model.md#convert-from-pytorch).
- [PyTorch Conversion Workflow](convert-pytorch-workflow).
- [Model Tracing](model-tracing).
- [Model Scripting](model-scripting)

Full examples:
- [Converting a Natural Language Processing Model](convert-nlp-model): Combines tracing and scripting to convert a PyTorch natural language processing model.
- [Converting a torchvision Model from PyTorch](convert-a-torchvision-model-from-pytorch): Traces a torchvision MobileNetV2 model, adds preprocessing for image input, and then converts it to Core ML.
- [Converting a PyTorch Segmentation Model](pytorch-conversion-examples): Converts a PyTorch segmentation model that takes an image and outputs a class prediction for each pixel of the image.

## Model Intermediate Language (MIL)

Full example:

- [Model Intermediate Language](model-intermediate-language): Construct a MIL program using the Python builder."

## Conversion Options

### Image Inputs

- [Use an MLMultiArray](image-inputs.md#use-an-mlmultiarray).
- [Use an ImageType](image-inputs.md#use-an-imagetype).
- [Add Image Preprocessing Options](image-inputs.md#add-image-preprocessing-options).

### Classifiers

- [Produce a Classifier Model](classifiers.md#produce-a-classifier-model).

### Flexible Input Shapes
  
- [Select from Predetermined Shapes](flexible-inputs.md#select-from-predetermined-shapes).
- [Set the Range for Each Dimension](flexible-inputs.md#set-the-range-for-each-dimension).
- [Update a Core ML Model to Use Flexible Input shapes](flexible-inputs.md#update-a-core-ml-model-to-use-flexible-input-shapes)

### Composite and Custom Operators

- [Composite Operators](composite-operators): Defining a composite operation by decomposing it into MIL operations.  

Full example:

- [Custom Operators](custom-operators): Augment Core ML with your own operators and implement them in Swift.

## Optimization

Full examples:

- [Training-Time Compression Examples](https://apple.github.io/coremltools/source/coremltools.optimize.torch.examples.html): Use magnitude pruning, linear quantization, or palettization while training your model, or start from a pre-trained model and fine-tune it with training data.
- [Compressing Neural Network Weights](quantization-neural-network): Reduce the size of a neural network by reducing the number of bits that represent a number.

## Trees and Linear Models

- [LibSVM](libsvm-conversion)
- [Scikit-learn](sci-kit-learn-conversion)
- [XGBoost](xgboost-conversion)

## MLModel

### MLModel Overview

- [Load and save the MLModel](mlmodel.md#load-and-save-the-mlmodel).
- [Use the MLModel for Prediction](mlmodel.md#use-the-mlmodel-for-prediction).
- [Work with the Spec](mlmodel.md#work-with-the-spec).
- [Update the Metadata and Input/output Descriptions](mlmodel.md#update-the-metadata-and-inputoutput-descriptions).

### Model Prediction

- [Make Predictions](model-prediction)
- [Multi-array Prediction](model-prediction.md#multi-array-prediction)
- [Image Prediction](model-prediction.md#image-prediction)
- [Image Prediction for a Multi-array Model](model-prediction.md#image-prediction-for-a-multi-array-model)

### Xcode Model Preview Types

Full examples:

- [Segmentation Example](xcode-model-preview-types.md#segmentation-example)
- [BERT QA Example](xcode-model-preview-types.md#bert-qa-example)
- [Body Pose Example](xcode-model-preview-types.md#body-pose-example)

### MLModel Utilities

- [Rename a Feature](mlmodel-utilities.md#rename-a-feature).
- [Convert All Double Multi-array Feature Descriptions to Float](mlmodel-utilities.md#convert-all-double-multi-array-feature-descriptions-to-float).
- [Evaluate Classifier, Regressor, and Transformer models](mlmodel-utilities.md#evaluate-classifier-regressor-and-transformer-models).

## Updatable Models

Full examples:

- [Nearest Neighbor Classifier](updatable-nearest-neighbor-classifier): Create an updatable empty k-nearest neighbor. 
- [Neural Network Classifier](updatable-neural-network-classifier-on-mnist-dataset): Create a simple convolutional model with Keras, convert the model to Core ML, and make the model updatable.
- [Pipeline Classifier](updatable-tiny-drawing-classifier-pipeline-model): Use a pipeline composed of a drawing-embedding model and a nearest neighbor classifier to create a model for training a sketch classifier.
If you have a code example you'd like to submit, see [Contributing](how-to-contribute).

