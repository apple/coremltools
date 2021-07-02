SupportVectorRegressor
=======================

A support vector regressor.


.. code-block:: proto

	message SupportVectorRegressor {
	    Kernel kernel = 1;
	
	    // Support vectors, either sparse or dense format
	    oneof supportVectors {
	        SparseSupportVectors sparseSupportVectors = 2;
	        DenseSupportVectors denseSupportVectors = 3;
	    }
	
	    // Coefficients, one for each support vector
	    Coefficients coefficients = 4;
	
	    double rho = 5;
	}




