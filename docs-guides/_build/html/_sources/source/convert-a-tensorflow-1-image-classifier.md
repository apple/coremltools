```{eval-rst}
.. index:: 
    single: TensorFlow 1; convert image classifier
    single: classifier; convert TensorFlow 1
    single: classifier; image
```


# Converting a TensorFlow 1 Image Classifier

The following example converts the TensorFlow Inception V1 image classifier to a Core ML classifier model that directly predicts the class label of the input image. It demonstrates the importance of setting the image preprocessing parameters correctly to get the right results.

## Requirements

This model requires TensorFlow 1, which is deprecated and difficult to install directly with pip. You can use the appropriate [Miniconda installer](https://docs.conda.io/en/latest/miniconda.html) for your operating system and create a Miniconda environment specifically for Python 3.7, and then use conda to install TensorFlow 1.15:

```shell
conda create -n tensorflow1-env python=3.7
conda activate tensorflow1-env
conda install tensorflow==1.15
```

```{note}
For alternatives, see [How to pip install old version of library(tensorflow)](https://stackoverflow.com/questions/41937915/how-to-pip-install-old-version-of-librarytensorflow) on StackOverflow.
```

In addition, you need to install the following for this environment:

```
pip install -U coremltools
pip install pillow
conda install requests
conda install matplotlib
```

## Download the Model

The following code downloads the Inception V1 frozen TF graph (the `.pb` file):

```python
# Download the model and class label package
from __future__ import print_function
import  os, sys
import tarfile

def download_file_and_unzip(url, dir_path='.'):
    """Download the frozen TensorFlow model and unzip it.
    url - The URL address of the frozen file
    dir_path - local directory
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    k = url.rfind('/')
    fname = url[k+1:]
    fpath = os.path.join(dir_path, fname)

    if not os.path.exists(fpath):
        if sys.version_info[0] < 3:
            import urllib
            urllib.urlretrieve(url, fpath)
        else:
            import urllib.request
            urllib.request.urlretrieve(url, fpath)

    tar = tarfile.open(fpath)
    tar.extractall(dir_path)
    tar.close()

inception_v1_url = 'https://storage.googleapis.com/download.tensorflow.org/models/inception_v1_2016_08_28_frozen.pb.tar.gz'
download_file_and_unzip(inception_v1_url)
```

## Load the Graph Definition

The following code loads the TensorFlow graph to find the input and output tensor names. You use them in the conversion process and for running the graph for a numerical accuracy check:

```python
# Load the TF graph definition
import tensorflow as tf # 1.x

tf_model_path = './inception_v1_2016_08_28_frozen.pb'
with open(tf_model_path, 'rb') as f:
    serialized = f.read()
tf.reset_default_graph()
original_gdef = tf.GraphDef()
original_gdef.ParseFromString(serialized)

# Lets get some details about a few ops in the beginning and the end of the graph
with tf.Graph().as_default() as g:
    tf.import_graph_def(original_gdef, name='')
    ops = g.get_operations()
    N = len(ops)
    for i in [0,1,2,N-3,N-2,N-1]:
        print('\n\nop id {} : op type: "{}"'.format(str(i), ops[i].type));
        print('input(s):'),
        for x in ops[i].inputs:
            print("name = {}, shape: {}, ".format(x.name, x.get_shape())),
        print('\noutput(s):'),
        for x in ops[i].outputs:
            print("name = {}, shape: {},".format(x.name, x.get_shape())),
```

If you run the code at this point, you can see that the output of the `Placeholder` op is the input (`input:0`), and the output of the `Softmax` op (near the end of the graph) is the output (`InceptionV1/Logits/Predictions/Softmax:0`). 

## Convert to Core ML

The following code sets the `image_inputs` for `inputs` and the output name (`'InceptionV1/Logits/Predictions/Softmax'`) for `outputs` in order to use them with the [`convert()`](https://apple.github.io/coremltools/source/coremltools.converters.convert.html#module-coremltools.converters._converters_entry) method. The `convert()` method produces an ML program by default:

```python
import coremltools as ct

image_inputs = ct.ImageType(shape=(1, 224, 224, 3))
classifier_config = ct.ClassifierConfig('imagenet_slim_labels.txt')
coreml_model_file = './inception_v1.mlpackage'
output = ['InceptionV1/Logits/Predictions/Softmax']

coreml_model = ct.convert(tf_model_path, 
                          inputs=[image_inputs], 
                          classifier_config=classifier_config,
                          outputs=output)

coreml_model.save(coreml_model_file)
```

The result shows the progress of the conversion, but also includes the following warning:

```
UserWarning: Output, 'InceptionV1/Logits/Predictions/Softmax', of the source model, has been renamed to 'InceptionV1_Logits_Predictions_Softmax' in the Core ML model.
```

You will use the new name when making a prediction.

## Load a Test Image

To make predictions on the same image using both the original model and the converted model, right-click the following image and save it as `Golden_Retriever_Carlos.jpg` in the same folder as your Python project:

```{figure} images/Golden_Retriever_Carlos_full.jpg
:alt: Golden_Retriever_Carlos.jpg
:align: center
:class: imgnoborder

This image of a golden retriever is from [Wikipedia](https://en.m.wikipedia.org/wiki/File:Golden_Retriever_Carlos_%2810581910556%29.jpg).
```

The following code loads the image:

```python
# Load an image
import numpy as np
from PIL import Image
img = Image.open("Golden_Retriever_Carlos.jpg")
```

## Input the Image and Make a Prediction

The following code passes the PIL image into the Core ML model after resizing it, and uses a NumPy array of the image to make a prediction. It also fixes the output name to use the renamed output (`'InceptionV1_Logits_Predictions_Softmax'`) in the Core ML model:

```python
img = img.resize([224,224], Image.LANCZOS)
coreml_inputs = {'input': img}

# Fix output name
output = ['InceptionV1_Logits_Predictions_Softmax']

coreml_output = coreml_model.predict(coreml_inputs)
coreml_pred_dict = coreml_output[output[0]]

coreml_predicted_class_label = coreml_output['classLabel']

#for getting TF prediction we get the numpy array of the image
img_np = np.array(img).astype(np.float32)
print( 'image shape:', img_np.shape)
print( 'first few values: ', img_np.flatten()[0:4], 'max value: ', np.amax(img_np))
img_tf = np.expand_dims(img_np, axis = 0) #now shape is [1,224,224,3] as required by TF

# Evaluate TF and get the highest label 
tf_input_name = 'input:0'
tf_output_name = 'InceptionV1/Logits/Predictions/Softmax:0'
# tf_output_name = 'InceptionV1_Logits_Predictions_Softmax:0'

img_tf = (2.0/255.0) * img_tf - 1
with tf.Session(graph = g) as sess:
    tf_out = sess.run(tf_output_name, 
                      feed_dict={tf_input_name: img_tf})

tf_out = tf_out.flatten()
idx = np.argmax(tf_out)
label_file = 'imagenet_slim_labels.txt' 
with open(label_file) as f:
     labels = f.readlines()

# print TF prediction
print("TF prediction class = {}, probability = {}".format(labels[idx],
                                            str(tf_out[idx])))

#print Core ML prediction
print('\n')

print("CoreML prediction class = {}, probability = {}".format(coreml_predicted_class_label,
                                str(coreml_pred_dict[0])))
```

The result shows that both predictions match, which ensures that the conversion is correct. However, for better results, ensure that the image is preprocessed correctly before passing it to the ML program. 

## Preprocess the Image Before Converting

Preprocessing is always a crucial step when using ML programs and neural networks on images. The best approach is to find the source of the pre-trained model and check for the preprocessing that the model's author used during training and evaluation.

In this case, the TensorFlow model comes from the
[SLIM library](https://github.com/tensorflow/models/tree/edb6ed22a801665946c63d650ab9a0b23d98e1b1/research/slim "tensorflow/models/slim/"),
and the preprocessing steps are defined in the `preprocess_for_eval` definition in [inception_preprocessing.py](https://github.com/tensorflow/models/blob/edb6ed22a801665946c63d650ab9a0b23d98e1b1/research/slim/preprocessing/inception_preprocessing.py#L243). The image pixels have to be scaled to lie within the interval `[-1,1]`.
("models/research/slim/preprocessing/inception_preprocessing.py").
The following code preprocesses the image and makes a new prediction:

```python
img_tf = (2.0/255.0) * img_tf - 1
with tf.Session(graph = g) as sess:
    tf_out = sess.run(tf_output_name, 
                      feed_dict={tf_input_name: img_tf})
tf_out = tf_out.flatten()    
idx = np.argmax(tf_out)
print("TF prediction class = {}, probability = {}".format(labels[idx],
                                            str(tf_out[idx])))
```

The TensorFlow model predicts an English Setter as the highest class, (with a probability of 0.301507):

```text Result
TF prediction class = English setter
, probability = 0.301507
```

Core ML automatically handles the image preprocessing when the input is of type image. However, the image biases and scale are not correct. The channel scale should be multiplied first before adding the bias. The following code converts the model again with this correction, and saves the newly converted model. It also makes the prediction again with the newly converted Core ML model:


```python
image_inputs = ct.ImageType(shape=(1, 224, 224, 3), bias=[-1,-1,-1], scale=2.0/255)
classifier_config = ct.ClassifierConfig('imagenet_slim_labels.txt')
coreml_model_file = './inception_v1.mlpackage'
output = ['InceptionV1/Logits/Predictions/Softmax']

coreml_model = ct.convert(tf_model_path, 
                          inputs=[image_inputs], 
                          classifier_config=classifier_config,
                          outputs=output)

coreml_model.save(coreml_model_file)

# Call CoreML predict again

# Fix output name
output = ['InceptionV1_Logits_Predictions_Softmax']

coreml_output = coreml_model.predict(coreml_inputs)
coreml_pred_dict = coreml_output[output[0]]
coreml_predicted_class_label = coreml_output['classLabel']
print("CoreML prediction class = {}, probability = {}".format(coreml_predicted_class_label,
                        str(coreml_pred_dict[0])))

```

The output predicts the English Setter with higher probability (1.68707207e-04):

```text Result
CoreML prediction class = English setter, probability = [1.68707207e-04 4.01963953e-05 2.33356332e-04 ... 1.15576135e-04
 3.79885838e-04 2.21910377e-04]
```

```{admonition} Predictions Can Vary Slightly
Predictions with the default Core ML `predict` call may vary slightly, since by default it uses a lower-precision optimized path for faster execution. In previous versions of Core ML Tools, you would restrict execution to the CPU by specifying the `useCPUOnly=True` flag. This flag is now deprecated. Instead, use the `compute_units` parameter at load time or conversion time (that is, in [`coremltools.models.MLModel`](https://apple.github.io/coremltools/source/coremltools.models.html#module-coremltools.models.model) or [`convert()`](https://apple.github.io/coremltools/source/coremltools.converters.convert.html#module-coremltools.converters._converters_entry). For more information, see [Set the compute units](load-and-convert-model.md#set-the-compute-units).

```

