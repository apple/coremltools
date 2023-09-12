```{eval-rst}
.. index::
    single: classifier; nearest neighbor updatable
    single: updatable nearest neighbor classifier
    single: nearest neighbor classifier
```

# Nearest Neighbor Classifier

This topic demonstrates the process of creating an updatable empty k-nearest neighbor model using Core ML Tools.

## Create the Classifier

1. Create the classifier and apply its properties:
    
	```python
	number_of_dimensions = 128

	from coremltools.models.nearest_neighbors import KNearestNeighborsClassifierBuilder
	builder = KNearestNeighborsClassifierBuilder(input_name='input',
												 output_name='output',
												 number_of_dimensions=number_of_dimensions,
												 default_class_label='defaultLabel',
												 number_of_neighbors=3,
												 weighting_scheme='inverse_distance',
												 index_type='linear')

	builder.author = 'Core ML Tools Example'
	builder.license = 'MIT'
	builder.description = 'Classifies {} dimension vector based on 3 nearest neighbors'.format(number_of_dimensions)

	builder.spec.description.input[0].shortDescription = 'Input vector to classify'
	builder.spec.description.output[0].shortDescription = 'Predicted label. Defaults to \'defaultLabel\''
	builder.spec.description.output[1].shortDescription = 'Probabilities / score for each possible label.'

	builder.spec.description.trainingInput[0].shortDescription = 'Example input vector'
	builder.spec.description.trainingInput[1].shortDescription = 'Associated true label of each example vector'
	```
    
    ```{note}
	An empty `knn` model is updatable by default:
    ```
    
	```python
	# By default an empty knn model is updatable
	builder.is_updatable
	```
	
	```text
	True
	```

2. Confirm that the number of dimensions are set correctly:
    
	```python
	# Let's confirm the number of dimension is set correctly
	builder.number_of_dimensions
	```
	
	```text
	128
	```

## Set the Number of Neighbors Value

1. Verify the current number of neighbors value:
    
	```python
	# Let's check what the value of 'numberOfNeighbors' is
	builder.number_of_neighbors
	```
	
	```text
	3
	```
    
    ```{note}
    The number of neighbors is bounded by the default range:
    ```
    
	```python
	# The number of neighbors is bounded by the default range...
	builder.number_of_neighbors_allowed_range()
	```
	
	```text
	(1, 1000)
	```
    
	If you set the number of neighbors to a value outside of this default range, an `ValueError` will occur as shown in the **Out** tab:
    
	```python
	# If we try to set the number of neighbors to a value outside of this range
	builder.number_of_neighbors = 1001
	```
	
	```text
	---------------------------------------------------------------------------
	ValueError                                Traceback (most recent call last)
	<ipython-input-17-e8bea591e72c> in <module>
		  1 # If we try to set the number of neighbors to a value outside of this range
	----> 2 builder.number_of_neighbors = 1001

	~/eng/sources/coreml/coremltools/coremltools/models/nearest_neighbors/builder.py in number_of_neighbors(self, number_of_neighbors)
		312                 self.spec.kNearestNeighborsClassifier.numberOfNeighbors.defaultValue = number_of_neighbors
		313             else:
	--> 314                 raise ValueError('number_of_neighbors is not within range bounds')
		315         else:
		316             spec_values = self.spec.kNearestNeighborsClassifier.numberOfNeighbors.set.values

	ValueError: number_of_neighbors is not within range bounds
	```

2. Change the bounds for the number of neighbors. Individual values can be set for the `numberOfNeighbors` parameter:
    
	```python
	# Instead of a range, you can a set individual values that are valid for the numberOfNeighbors parameter.
	builder.set_number_of_neighbors_with_bounds(3, allowed_set={ 1, 3, 5 })
	```

3. Verify change using the `number_of_neighbors_allowed_set()` method.
    
	```python
	# Check out the results of the previous operation
	builder.number_of_neighbors_allowed_set()
	```
	
	```text
	{1, 3, 5}
	```

4. The number of neighbors value can now be set without an error:
    
	```python
	# And now if you attempt to set it to an invalid value...
	builder.number_of_neighbors = 4
	```
	
	```text
	---------------------------------------------------------------------------
	ValueError                                Traceback (most recent call last)
	<ipython-input-20-98c77c72c722> in <module>
		  1 # And now if you attempt to set it to an invalid value...
	----> 2 builder.number_of_neighbors = 4

	~/eng/sources/coreml/coremltools/coremltools/models/nearest_neighbors/builder.py in number_of_neighbors(self, number_of_neighbors)
		320                     self.spec.kNearestNeighborsClassifier.numberOfNeighbors.defaultValue = number_of_neighbors
		321                     return
	--> 322             raise ValueError('number_of_neighbors is not an allowed value')
		323 
		324     def set_number_of_neighbors_with_bounds(self, number_of_neighbors, allowed_range=None, allowed_set=None):

	ValueError: number_of_neighbors is not valid
	```

If desired, you can revert back to a valid range:

```python
# And of course you can go back to a valid range
builder.set_number_of_neighbors_with_bounds(3, allowed_range=(1, 30))
```

## Set the Index Type

1. Verify the current index type:
    
	```python
	# Let's see what the index type is
	builder.index_type
	```
	
	```text
	'linear'
	```

2. Set the index and leaf size:
    
	```python
	# Let's set the index to kd_tree with leaf size of 30
	builder.set_index_type('kd_tree', 30)
	builder.index_type
	```
	
	```text
	'kd_tree'
	```

3. Save the model:
    
	```python
	mlmodel_updatable_path = './UpdatableKNN.mlmodel'

	# Save the updated spec
	from coremltools.models import MLModel
	mlmodel_updatable = MLModel(builder.spec)
	mlmodel_updatable.save(mlmodel_updatable_path)
	```


