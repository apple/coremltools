```{eval-rst}
.. index::
    single: classifier; neural network updatable
    single: updatable neural network classifier
    single: neural network; classifier
```

# Neural Network Classifier

A neural network is a collection of layers, each containing weights that get used alongside its other inputs to produce an output. To be able to update weights of a layer, we need to mark them as "updatable". 

Loss function tells us how much the output of the neural network was off for a given input. Output of loss is feed to an optimizer that uses that loss along with our output to adjust the weights in our layers so that our neural network can provide outputs more in-line with what we’d expect.

```{admonition} Support

We currently support updating convolutional and fully-connected layers. In terms of loss functions, mean squared error and categorical cross-entropy are supported. For optimizers we support both Stochastic Gradient Descent and Adam optimization strategies.
```

The following example demonstrates how to:

1. Create a simple convolutional model on the MNIST dataset with Keras.
2. Convert the model to a Core ML model.
3. Make the model updatable.

The updatable model has two updatable layers and uses categorical cross-entropy loss and Stochastic Gradient Descent (SGD) optimizer.

## Create the Base Model

```python
def create_keras_base_model(url):
    """This method creates a convolutional neural network model using Keras.
    url - The URL that the keras model will be saved as h5 file.
    """
    
    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
    
    keras.backend.clear_session()
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(lr=0.01),
                  metrics=['accuracy'])

    model.save(url)

keras_model_path = './KerasMnist.h5'
create_keras_base_model(keras_model_path)
```

## Convert the Model to the ML Model Format

```python
def convert_keras_to_mlmodel(keras_url, mlmodel_url):
    """This method simply converts the keras model to a mlmodel using coremltools.
    keras_url - The URL the keras model will be loaded.
    mlmodel_url - the URL the Core ML model will be saved.
    """
    from keras.models import load_model
    keras_model = load_model(keras_url)
    
    from coremltools.converters import keras as keras_converter
    class_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    mlmodel = keras_converter.convert(keras_model, input_names=['image'],
                                output_names=['digitProbabilities'],
                                class_labels=class_labels,
                                predicted_feature_name='digit')
    
    mlmodel.save(mlmodel_url)
     
coreml_model_path = './MNISTDigitClassifier.mlmodel'
convert_keras_to_mlmodel(keras_model_path , coreml_model_path)
```

## Load the Spec and Apply its Settings

1. Inspect the last 3 layers of the model:
    
	```python
	# let's inspect the last few layers of this model
	import coremltools
	spec = coremltools.utils.load_spec(coreml_model_path)
	builder = coremltools.models.neural_network.NeuralNetworkBuilder(spec=spec)
	builder.inspect_layers(last=3)
	```
	
	```text
	[Id: 9], Name: dense_2__activation__ (Type: softmax)
			  Updatable: False
			  Input blobs: [u'dense_2_output']
			  Output blobs: [u'digitProbabilities']
	[Id: 8], Name: dense_2 (Type: innerProduct)
			  Updatable: False
			  Input blobs: [u'dense_1__activation___output']
			  Output blobs: [u'dense_2_output']
	[Id: 7], Name: dense_1__activation__ (Type: activation)
			  Updatable: False
			  Input blobs: [u'dense_1_output']
			  Output blobs: [u'dense_1__activation___output']
	```

2. Inspect the input of the model. This information is needed later for the `make_updatable()` method:
    
	```python
	# let's inspect the input of the model as we need this information later on the make_updatable method
	builder.inspect_input_features()

	neuralnetwork_spec = builder.spec
	```
	
	```text
	[Id: 0] Name: image
			  Type: multiArrayType {
	  shape: 1
	  shape: 28
	  shape: 28
	  dataType: DOUBLE
	}
	```

3. Change the input so the model can accept 28 x 28 grayscale images and inspect the input to confirm the changes:
    
	```python
	# change the input so the model can accept 28x28 grayscale images
	neuralnetwork_spec.description.input[0].type.imageType.width = 28
	neuralnetwork_spec.description.input[0].type.imageType.height = 28

	from coremltools.proto import FeatureTypes_pb2 as _FeatureTypes_pb2
	grayscale = _FeatureTypes_pb2.ImageFeatureType.ColorSpace.Value('GRAYSCALE')
	neuralnetwork_spec.description.input[0].type.imageType.colorSpace = grayscale

	# let's inspect the input again to confirm the change in input type
	builder.inspect_input_features()
	```
	
	```text
	[Id: 0] Name: image
			  Type: imageType {
	  width: 28
	  height: 28
	  colorSpace: GRAYSCALE
	}
	```

4. Set the input and output descriptions:
    
	```python
	# Set input and output description
	neuralnetwork_spec.description.input[0].shortDescription = 'Input image of the handwriten digit to classify'
	neuralnetwork_spec.description.output[0].shortDescription = 'Probabilities / score for each possible digit'
	neuralnetwork_spec.description.output[1].shortDescription = 'Predicted digit'
	```

5. Provide the metadata:
    
	```python
	# Provide metadata
	neuralnetwork_spec.description.metadata.author = 'Core ML Tools'
	neuralnetwork_spec.description.metadata.license = 'MIT'
	neuralnetwork_spec.description.metadata.shortDescription = (
			'An updatable hand-written digit classifier setup to train or be fine-tuned on MNIST like data.')
	```

## Define `make_updatable`

```python
def make_updatable(builder, mlmodel_url, mlmodel_updatable_path):
    """This method makes an existing non-updatable mlmodel updatable.
    mlmodel_url - the path the Core ML model is stored.
    mlmodel_updatable_path - the path the updatable Core ML model will be saved.
    """
    import coremltools
    model_spec = builder.spec

    # make_updatable method is used to make a layer updatable. It requires a list of layer names.
    # dense_1 and dense_2 are two innerProduct layer in this example and we make them updatable.
    builder.make_updatable(['dense_1', 'dense_2'])

    # Categorical Cross Entropy or Mean Squared Error can be chosen for the loss layer.
    # Categorical Cross Entropy is used on this example. CCE requires two inputs: 'name' and 'input'.
    # name must be a string and will be the name associated with the loss layer
    # input must be the output of a softmax layer in the case of CCE. 
    # The loss's target will be provided automatically as a part of the model's training inputs.
    builder.set_categorical_cross_entropy_loss(name='lossLayer', input='digitProbabilities')

    # in addition of the loss layer, an optimizer must also be defined. SGD and Adam optimizers are supported.
    # SGD has been used for this example. To use SGD, one must set lr(learningRate) and batch(miniBatchSize) (momentum is an optional parameter).
    from coremltools.models.neural_network import SgdParams
    builder.set_sgd_optimizer(SgdParams(lr=0.01, batch=32))

    # Finally, the number of epochs must be set as follows.
    builder.set_epochs(10)
        
    # Set training inputs descriptions
    model_spec.description.trainingInput[0].shortDescription = 'Example image of handwritten digit'
    model_spec.description.trainingInput[1].shortDescription = 'Associated true label (digit) of example image'

    # save the updated spec
    from coremltools.models import MLModel
    mlmodel_updatable = MLModel(model_spec)
    mlmodel_updatable.save(mlmodel_updatable_path)

coreml_updatable_model_path = './UpdatableMNISTDigitClassifier.mlmodel'
make_updatable(builder, coreml_model_path, coreml_updatable_model_path)
```

## Make the Model Updatable

1. Add the input `digitProbabilities_true` as the target for the categorical cross-entropy loss layer:
    
	```python
	# let's inspect the loss layer of the Core ML model
	import coremltools
	spec = coremltools.utils.load_spec(coreml_updatable_model_path)
	builder = coremltools.models.neural_network.NeuralNetworkBuilder(spec=spec)

	builder.inspect_loss_layers()
	```
	
	```text
	[Id: 0], Name: lossLayer (Type: categoricalCrossEntropyLossLayer)
			  Loss Input: digitProbabilities
			  Loss Target: digitProbabilities_true
	```

2. Inspect the optimizer of the Core ML model:
    
	```python
	# let's inspect the optimizer of the Core ML model
	builder.inspect_optimizer()
	```
	
	```text
	Optimizer Type: sgdOptimizer
	lr: 0.01, min: 0.0, max: 1.0
	batch: 32, allowed_set: [32L]
	momentum: 0.0, min: 0.0, max: 1.0
	```

3. Inspect the layers to see which are updatable:
    
	```python
	# let's see which layers are updatable
	builder.inspect_updatable_layers()
	```
	
	```text
	Name: dense_2 (Type: innerProduct)
			  Input blobs: [u'dense_1__activation___output']
			  Output blobs: [u'dense_2_output']
	Name: dense_1 (Type: innerProduct)
			  Input blobs: [u'flatten_1_output']
			  Output blobs: [u'dense_1_output']
	```


