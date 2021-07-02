Pipeline
=========


Pipeline
________________________________________________________________________________

A pipeline consisting of one or more models.


.. code-block:: proto

	message Pipeline {
	    repeated Model models = 1;
	
	    // Optional names given for each model
	    // If not supplied it defaults to ["model0",..., "model"(models.size()-1)]
	    // These names can be used to disambiguate the scope / domain of a parameter
	    repeated string names = 2;
	}






PipelineClassifier
________________________________________________________________________________

A classifier pipeline.


.. code-block:: proto

	message PipelineClassifier {
	    Pipeline pipeline = 1;
	}






PipelineRegressor
________________________________________________________________________________

A regressor pipeline.


.. code-block:: proto

	message PipelineRegressor {
	    Pipeline pipeline = 1;
	}




