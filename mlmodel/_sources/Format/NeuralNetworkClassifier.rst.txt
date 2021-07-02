NeuralNetworkClassifier
=======================

A neural network specialized as a classifier.


.. code-block:: proto

	message NeuralNetworkClassifier {

	    repeated NeuralNetworkLayer layers = 1;
	    repeated NeuralNetworkPreprocessing preprocessing = 2;

	    // use this enum value to determine the input tensor shapes to the neural network, for multiarray inputs
	    NeuralNetworkMultiArrayShapeMapping arrayInputShapeMapping = 5;

	    // use this enum value to determine the input tensor shapes to the neural network, for image inputs
	    NeuralNetworkImageShapeMapping imageInputShapeMapping = 6;

	    NetworkUpdateParameters updateParams = 10;

	    // The set of labels for every possible class.
	    oneof ClassLabels {
	        StringVector stringClassLabels = 100;
	        Int64Vector int64ClassLabels = 101;
	    }

	    // The name of the output blob containing the probability of each class.
	    // In other words, the score vector. Must be a 1-D tensor with the same
	    // number and order of elements as ClassLabels.
	    string labelProbabilityLayerName = 200;
	}



