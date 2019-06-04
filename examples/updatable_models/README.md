Updatable Models Examples
=======================

In this set of notebooks examples, we show how to create different types of updatable models using coremltools.

- [Updatable Neural Network Classifier on MNIST Dataset](https://github.com/apple/coremltools/tree/master/examples/updatable_models/updatable_mnist.ipynb)  
This notebook demonstrates the process of creating a simple convolutional model on the MNIST dataset with Keras, converting it to a Core ML model, and making it updatable.
The updatable model has 2 updatable layers and uses Categorical Cross Entropy Loss and SGD Optimizer.

- [Updatable Tiny Drawing Classifier](https://github.com/apple/coremltools/tree/master/examples/updatable_models/updatable_tiny_drawing_classifier.ipynb)  
This notebook creates a model which can be used to train a simple drawing / sketch classifier based on user examples.  
The model is a pipeline composed of a drawing embedding model and an updatable nearest neighbor classifier. 

- [Updatable Nearest Neighbor Classifier](https://github.com/apple/coremltools/tree/master/examples/updatable_models/updatable_nearest_neighbor_classifier.ipynb)  
This notebook makes an empty updatable nearest neighbor classifier. Before updating with training examples it predicts 'defaultLabel' for all input. 
