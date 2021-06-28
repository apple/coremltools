SupportVectorClassifier
=======================


A support vector classifier


.. code-block:: proto

	message SupportVectorClassifier {
	    Kernel kernel = 1;
	
	    repeated int32 numberOfSupportVectorsPerClass = 2;
	
	    oneof supportVectors {
	        SparseSupportVectors sparseSupportVectors = 3;
	        DenseSupportVectors denseSupportVectors = 4;
	    }
	
	    repeated Coefficients coefficients = 5;
	
	    repeated double rho = 6;
	
	    repeated double probA = 7;
	    repeated double probB = 8;
	
	    oneof ClassLabels {
	        StringVector stringClassLabels = 100;
	        Int64Vector int64ClassLabels = 101;
	    }
	}



