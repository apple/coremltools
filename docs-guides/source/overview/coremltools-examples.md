# Examples

The following are code example snippets and full examples of using Core ML Tools to convert models.

## For a Quick Start

[Getting Started](../essentials/introductory-quickstart): Demonstrates how to convert an image classifier model trained using the TensorFlow Keras API to the Core ML format.

## ML Program with Typed Execution

[Typed Execution Workflow Example](../unified/ml-programs/typed-execution-example): Demonstrates a workflow for checking accuracy using [ML Programs](../unified/ml-programs/ml-programs) with [Typed Execution](../unified/typed-execution).


```
    "2-0": "TensorFlow 2",
    "2-1": "[Load and Convert a Model](doc:unified-conversion-api#load-and-convert-a-model) (in [Unified Conversion API](doc:unified-conversion-api))  \n[TensorFlow 2 Workflow](doc:tensorflow-2)  \n[Convert a Pre-trained Model](doc:tensorflow-2#convert-a-pre-trained-model)  \n[Convert a User-defined Model](doc:tensorflow-2#convert-a-user-defined-model)  \n  \nFull examples:  \n[Getting Started](doc:introductory-quickstart): Demonstrates how to convert an image classifier model trained using the TensorFlow Keras API to the Core ML format.  \n[Converting TensorFlow 2 BERT Transformer Models](doc:convert-tensorflow-2-bert-transformer-models): Converts an object of the tf.keras.Model class and a SavedModel in the TensorFlow 2 format.",
    "3-0": "TensorFlow 1",
    "3-1": "[Convert From TensorFlow 1](doc:unified-conversion-api#convert-from-tensorflow-1) (in [Unified Conversion API](doc:unified-conversion-api))  \n[Export as Frozen Graph and Convert](doc:tensorflow-1#export-as-a-frozen-graph-and-convert)  \n[Convert a Pre-trained Model](doc:tensorflow-1#convert-a-pre-trained-model)  \n  \nFull examples:  \n[Converting a TensorFlow 1 Image Classifier](doc:convert-a-tensorflow-1-image-classifier): Demonstrates the importance of setting the image preprocessing parameters correctly during conversion to get the right results.  \n[Converting a TensorFlow 1 DeepSpeech Model](doc:convert-a-tensorflow-1-deepspeech-model): Demonstrates automatic handling of flexible shapes using automatic speech recognition.",
    "4-0": "PyTorch",
    "4-1": "[Convert from PyTorch](doc:unified-conversion-api#convert-from-pytorch) (in [Unified Conversion API](doc:unified-conversion-api))  \n[Converting from PyTorch](doc:pytorch-conversion)  \n[Model Tracing](doc:model-tracing)  \n[Model Scripting](doc:model-scripting)  \n  \nFull examples:  \n[Converting a Natural Language Processing Model](doc:convert-nlp-model): Combines tracing and scripting to convert a PyTorch natural language processing model.  \n[Converting a torchvision Model from PyTorch](doc:convert-a-torchvision-model-from-pytorch): Traces a torchvision MobileNetV2 model, adds preprocessing for image input, and then converts it to Core ML.  \n[Converting a PyTorch Segmentation Model](doc:pytorch-conversion-examples): Converts a PyTorch segmentation model that takes an image and outputs a class prediction for each pixel of the image.",
    "5-0": "Model Intermediate Language (MIL)",
    "5-1": "[Model Intermediate Language](doc:model-intermediate-language): Construct a MIL program using the Python builder.",
    "6-0": "Conversion Options",
    "6-1": "Image Inputs:  \n[Convert a Model with a MultiArray](doc:image-inputs#convert-a-model-with-a-multiarray)  \n[Convert a Model with an ImageType](doc:image-inputs#convert-a-model-with-an-imagetype)  \n[Add Image Preprocessing Options](doc:image-inputs#add-image-preprocessing-options)  \n  \nClassifiers: [Produce a Classifier Model](doc:classifiers#produce-a-classifier-model)  \n  \nFlexible Input Shapes:  \n[Select from Predetermined Shapes](doc:flexible-inputs#select-from-predetermined-shapes)  \n[Set the Range for Each Dimension](doc:flexible-inputs#set-the-range-for-each-dimension)  \n[Update a Core ML Model to Use Flexible Input shapes](doc:flexible-inputs#update-a-core-ml-model-to-use-flexible-input-shapes)  \n  \n[Composite Operators](doc:composite-operators): Defining a composite operation by decomposing it into MIL operations.  \n  \nFull examples:  \n[Custom Operators](doc:custom-operators): Augment Core ML with your own operators and implement them in Swift.",
    "7-0": "Optimization",
    "7-1": "[Training-Time Compression Examples](https://apple.github.io/coremltools/source/coremltools.optimize.torch.examples.html): Use magnitude pruning, linear quantization, or palettization while training your model, or start from a pre-trained model and fine-tune it with training data.  \n[Compressing Neural Network Weights](doc:quantization): Reduce the size of a neural network by reducing the number of bits that represent a number.",
    "8-0": "Other Converters",
    "8-1": "[Multi-backend Keras](doc:kerasio-conversion)  \n[ONNX](doc:onnx-conversion)  \n[Caffe](doc:caffe-conversion)",
    "9-0": "Trees and Linear Models",
    "9-1": "[LibSVM](doc:libsvm-conversion)  \n[Scikit-learn](doc:sci-kit-learn-conversion)  \n[XGBoost](doc:xgboost-conversion)",
    "10-0": "MLModel",
    "10-1": "MLModel Overview:  \n[Load and save the MLModel](doc:mlmodel#load-and-save-the-mlmodel)  \n[Use the MLModel for Prediction](doc:mlmodel#use-the-mlmodel-for-prediction)  \n[Work with the spec Object](doc:mlmodel#work-with-the-spec-object)  \n[Update the Metadata and Input/output Descriptions](doc:mlmodel#update-the-metadata-and-inputoutput-descriptions)  \n  \nModel Prediction:  \n[Make Predictions](doc:model-prediction)  \n[Multi-array Prediction](doc:model-prediction#multi-array-prediction)  \n[Image Prediction](doc:model-prediction#image-prediction)  \n[Image Prediction for a Multi-array Model](doc:model-prediction#image-prediction-for-a-multi-array-model)  \n  \nXcode Model Preview Types:  \n[Segmentation Example](doc:xcode-model-preview-types#segmentation-example)  \n[BERT QA Example](doc:xcode-model-preview-types#bert-qa-example)  \n[Body Pose Example](doc:xcode-model-preview-types#body-pose-example)  \n  \nMLModel Utilities:  \n[Rename a Feature](doc:mlmodel-utilities#rename-a-feature)  \n[Convert All Double Multi-array Feature Descriptions to Float](doc:mlmodel-utilities#convert-all-double-multi-array-feature-descriptions-to-float)  \n[Evaluate Classifier, Regressor, and Transformer models](doc:mlmodel-utilities#evaluate-classifier-regressor-and-transformer-models)",
    "11-0": "Updatable Models",
    "11-1": "[Nearest Neighbor Classifier](doc:updatable-nearest-neighbor-classifier): Create an updatable empty k-nearest neighbor.  \n[Neural Network Classifier](doc:updatable-neural-network-classifier-on-mnist-dataset): Create a simple convolutional model with Keras, convert the model to Core ML, and make the model updatable.  \n[Pipeline Classifier](doc:updatable-tiny-drawing-classifier-pipeline-model): Use a pipeline composed of a drawing-embedding model and a nearest neighbor classifier to create a model for training a sketch classifier."
  },
  "cols": 2,
  "rows": 12,
  "align": [
    "left",
    "left"
  ]
}
[/block]
```

If you have a code example you'd like to submit, see [Contributing](doc:how-to-contribute).

