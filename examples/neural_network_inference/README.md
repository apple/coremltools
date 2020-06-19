Neural Network Inference Examples
=================================

In this set of notebook examples, we show examples for building and editing mlmodels via coremltools. 

- [Setting per channel scale image pre-processing](Image_preprocessing_per_channel_scale.ipynb)  
This notebook shows how an mlmodel can be edited after conversion, to add to it a per channel scale pre-processing layer.

- [Power iteration algorithm using a neural network Core ML model](Neural_network_control_flow_power_iteration.ipynb)  
This notebook shows how to build an mlmodel from scratch using the neural network `builder` API in coremltools.
In particular, this examples shows how a non-neural network algorithm involving control flow features can still be 
expressed as an mlmodel. The feature of using `flexible shaped` inputs is also touched upon at the end of the notebook.

- [Tensorflow 1 conversion examples](tensorflow_converter/Tensorflow_1)

- [Tensorflow 2 conversion examples](tensorflow_converter/Tensorflow_2)
  
- [ONNX conversion examples](onnx_converter)

