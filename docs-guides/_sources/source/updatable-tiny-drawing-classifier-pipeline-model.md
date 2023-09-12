```{eval-rst}
.. index::
    single: classifier; pipeline updatable
    single: updatable pipeline classifier
    single: pipeline classifier
```

# Pipeline Classifier

This example creates a model which can be used to train a simple drawing or sketch classifier based on user examples. The model is a pipeline composed of a drawing-embedding model and a nearest-neighbor classifier.

The model is updatable and starts off empty, meaning that the nearest-neighbor classifier has no examples or labels. Before updating with training examples, the model predicts "unknown" for all input.

The input to the model is a 28 x 28 grayscale drawing. The background is expected to be black (`0`), while the strokes of the drawing should be rendered as white (`255`). Right-click these 28 x 28 images for the following example:

* Drawing of a star:
    
    ![Drawing of a star](images/star28x28.png)

* Drawing of a heart:
    
    ![Drawing of a heart](images/heart28x28.png)

* Drawing of 5:

    ![Drawing of 5](images/five28x28.png)


## Get the Embedding Model

The drawing-embedding model is used as a feature extractor. Start by getting the first part of the model, the spec:

```python
import coremltools
from coremltools.models import MLModel

embedding_path = './models/TinyDrawingEmbedding.mlmodel'
embedding_model = MLModel(embedding_path)

embedding_spec = embedding_model.get_spec()
print embedding_spec.description
```

In the following output, the `shortDescription` indicates that the embedding model takes in a 28 x 28 grayscale image about outputs a 128 dimensional float vector:

```text
tf.estimator package not installed.
tf.estimator package not installed.
input {
  name: "drawing"
  shortDescription: "Input sketch image with black background and white strokes"
  type {
    imageType {
      width: 28
      height: 28
      colorSpace: GRAYSCALE
    }
  }
}
output {
  name: "embedding"
  shortDescription: "Vector embedding of sketch in 128 dimensional space"
  type {
    multiArrayType {
      shape: 128
      dataType: FLOAT32
    }
  }
}
metadata {
  shortDescription: "Embeds a 28 x 28 grayscale image of a sketch into 128 dimensional space. The model was created by removing the last layer of a simple convolution based neural network classifier trained on the Quick, Draw! dataset (https://github.com/googlecreativelab/quickdraw-dataset)."
  author: "Core ML Tools Example"
  license: "MIT"
}
```


## Create the Nearest Neighbor Classifier

Now that the feature extractor is in place, create the second model of your pipeline model. It is a nearest-neighbor classifier operating on the embedding:

```python
from coremltools.models.nearest_neighbors import KNearestNeighborsClassifierBuilder
import coremltools.models.datatypes as datatypes

knn_builder = KNearestNeighborsClassifierBuilder(input_name='embedding',
                                                 output_name='label',
                                                 number_of_dimensions=128,
                                                 default_class_label='unknown',
                                                 k=3,
                                                 weighting_scheme='inverse_distance',
                                                 index_type='linear')

knn_builder.author = 'Core ML Tools Example'
knn_builder.license = 'MIT'
knn_builder.description = 'Classifies 128 dimension vector based on 3 nearest neighbors'

knn_spec = knn_builder.spec
knn_spec.description.input[0].shortDescription = 'Input vector to classify'
knn_spec.description.output[0].shortDescription = 'Predicted label. Defaults to \'unknown\''
knn_spec.description.output[1].shortDescription = 'Probabilities / score for each possible label.'

# print knn_spec.description
```

## Create an Updatable Pipeline Model

The last step is to create the pipeline model and insert the feature extractor and the nearest-neighbor classifier. The model will be set to be updatable. Follow these steps:

1. Create the spec, set it to be updatable, and set the specification version:
    
	```python
	pipeline_spec = coremltools.proto.Model_pb2.Model()
	pipeline_spec.specificationVersion = coremltools._MINIMUM_UPDATABLE_SPEC_VERSION
	pipeline_spec.isUpdatable = True
	```

2. Set the inputs to the inputs from the embedding model:
    
	```python
	# Inputs are the inputs from the embedding model
	pipeline_spec.description.input.extend(embedding_spec.description.input[:])
	```

3. Set the outputs to the outputs from the classification model:
    
	```python
	# Outputs are the outputs from the classification model
	pipeline_spec.description.output.extend(knn_spec.description.output[:])
	pipeline_spec.description.predictedFeatureName = knn_spec.description.predictedFeatureName
	pipeline_spec.description.predictedProbabilitiesName = knn_spec.description.predictedProbabilitiesName
	```

4. Set the training inputs:
    
	```python
	# Training inputs
	pipeline_spec.description.trainingInput.extend([embedding_spec.description.input[0]])
	pipeline_spec.description.trainingInput[0].shortDescription = 'Example sketch'
	pipeline_spec.description.trainingInput.extend([knn_spec.description.output[0]])
	pipeline_spec.description.trainingInput[1].shortDescription = 'Associated true label of example sketch'
    ```

5. Provide the metadata:
    
	```python
	# Provide metadata
	pipeline_spec.description.metadata.author = 'Core ML Tools'
	pipeline_spec.description.metadata.license = 'MIT'
	pipeline_spec.description.metadata.shortDescription = ('An updatable model which can be used to train a tiny 28 x 28 drawing classifier based on user examples.'
														   ' It uses a drawing embedding trained on the Quick, Draw! dataset (https://github.com/googlecreativelab/quickdraw-dataset)')
	```

6. Construct the pipeline by adding the embedding and the nearest-neighbor classifier:
    
	```python
	# Construct pipeline by adding the embedding and then the nearest neighbor classifier
	pipeline_spec.pipelineClassifier.pipeline.models.add().CopyFrom(embedding_spec)
	pipeline_spec.pipelineClassifier.pipeline.models.add().CopyFrom(knn_spec)
	```

7. Save the updated spec:
    
	```python
	# Save the updated spec.
	from coremltools.models import MLModel
	mlmodel = MLModel(pipeline_spec)

	output_path = './TinyDrawingClassifier.mlmodel'
	from coremltools.models.utils import save_spec
	mlmodel.save(output_path)
	```


