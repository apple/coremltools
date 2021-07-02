NeuralNetworkRegressor
=======================


A neural network specialized as a regressor.


.. code-block:: proto

	message NeuralNetworkRegressor {

	    repeated NeuralNetworkLayer layers = 1;
	    repeated NeuralNetworkPreprocessing preprocessing = 2;

	    // use this enum value to determine the input tensor shapes to the neural network, for multiarray inputs
	    NeuralNetworkMultiArrayShapeMapping arrayInputShapeMapping = 5;

	    // use this enum value to determine the input tensor shapes to the neural network, for image inputs
	    NeuralNetworkImageShapeMapping imageInputShapeMapping = 6;

	    NetworkUpdateParameters updateParams = 10;

	}

