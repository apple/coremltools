**(This is the readme for Apple internal release of coremltools)**


External (Github) coremltools readme is [here](https://github.com/apple/coremltools)

To install the latest wheels:

```bash
pip install coremltools==4.0a5 -i https://pypi.apple.com/simple
```

coremltools package contains the following converters:

```python
import coremltools

# libsvm
coremltools.converters.libsvm.convert(...)

# sklearn
coremltools.converters.sklearn.convert(...)

# xgboost
coremltools.converters.xgboost.convert(...)

# Tensorflow and Pytorch [NEW API in coremltools 4.x, not present in coremltools 3.x]
coremltools.converters.convert(...)


# keras [for models defined directly using the keras.io api, NOT for tf.keras]
coremltools.converters.keras.convert(...)

# onnx
coremltools.converters.onnx.convert(...)
```


## Tensorlfow and Pytorch Converters


For coremltools 3.x converters see [here](https://github.com/apple/coremltools/blob/master/examples/NeuralNetworkGuide.md)

For coremltools 4.x new converters, use the following: 

```python

import coremltools

coremltools.converters.convert(model,
                               source="auto",
                               inputs=None,
                               outputs=None)
```

    
   

Method to convert neural networks represented in Tensorflow or Pytorch formats to the Core ML model format.

#### Parameters

- model:  
    an object representing a neural network model defined in one of Tensorflow 1, Tensorflow 2 or Pytorch formats

Depending on the source framework, type of model is one of the following:

For Tensorflow versions 1.x:  
    - frozen tf.Graph object  
    - path to a frozen .pb file  
    
For Tensorflow versions 2.x:  
    - tf.Graph object  
    - tf.keras model object    
    - path to a .h5 saved keras model file  
    - path to a saved model directory  
    - list of concrete functions  

For Pytorch:  
    - a TorchScript object  
    - path to a .pt file  

- source: str (optional)  
    one of "auto" (default), "tensorflow", "pytorch"  

- inputs: list (optional)  
    For Tensorflow:  
        list of tuples or list of strings  
        - If [tuple] : each tuple contains input tensor name and shape  
        - If [str]: each string is the name of the Placeholder input op in the TF graph
          
    For Pytorch
        a list of example inputs, which are any of:
        1. tensor
        2. tuple shape
        3. tuple of (string name, (1. or 2.))


- outputs: list[str] (optional)  
    For Tensorflow:  
        (required)  
        list of output op names  
        
    For Pytorch:  
        (not required)
        list of output op names    

#### Returns

model: MLModel
A Core ML MLModel object

### Examples

Tensorflow 1:

```python

mlmodel = coremltools.converters.convert(model='frozen_model_mobilenet.pb',
                                         inputs=[('input', (1, 224, 224, 3))],
                                         outputs=['Softmax'])

mlmodel.save('model_mobilenet.mlmodel')

```




Tensorflow 2:  

```python
mlmodel = coremltools.converters.convert(model='frozen_model_mobilenet.h5',
                                         outputs=['Softmax'])

mlmodel.save('model_mobilenet.mlmodel')

```



Pytorch :  

```python

model = torchvision.models.mobilenet_v2()
model.eval()
example_input = torch.rand(1, 3, 256, 256)
traced_model = torch.jit.trace(model, example_input)

mlmodel = coremltools.converters.convert(traced_model,
                                        inputs=[('input_name', example_input)]
                                        outputs=['output_name'])

mlmodel.save('mobilenetv2.mlmodel')

```


    
  

### Other utilities:

```python
import coremltools

mlmodel = coremltools.converters.convert(model='frozen_model_mobilenet.pb',
                                                 inputs=[('input', (1, 224, 224, 3))],
                                                 outputs=['Softmax'])
                                                 
# To convert input type from multi-array to image                                          
from coremltools.models.neural_network.utils import make_image_input

mlmodel = make_image_input(mlmodel, "input",
                           red_bias=-5, green_bias=-6, blue_bias=-2.5,
                           scale=10.0,
                           image_format='NCHW')
                           
mlmodel.save("/tmp/image_input_model.mlmodel")
                           

# To convert neural network model to a classifier model
from coremltools.models.neural_network.utils import make_nn_classifier

mlmodel = make_nn_classifier(mlmodel, class_labels=['a', 'b', 'c'],
                             predicted_feature_name='Softmax',
                             predicted_probabilities_output='output_class_prob')
                            
mlmodel.save("/tmp/classifier_model.mlmodel")               
```


   