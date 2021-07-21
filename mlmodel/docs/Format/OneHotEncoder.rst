OneHotEncoder
=============

Transforms a categorical feature into an array. The array will be all
zeros expect a single entry of one.

Each categorical value will map to an index, this mapping is given by
either the ``stringCategories`` parameter or the ``int64Categories``
parameter.



OneHotEncoder
________________________________________________________________________________


.. code-block:: proto

	message OneHotEncoder {
	    enum HandleUnknown {
	        ErrorOnUnknown = 0;
	        IgnoreUnknown = 1;   // Output will be all zeros for unknown values.
	    }
	
	    oneof CategoryType {
	        StringVector stringCategories = 1;
	        Int64Vector int64Categories = 2;
	    }
	
	    // Output can be a dictionary with only one entry, instead of an array.
	    bool outputSparse = 10;
	
	    HandleUnknown handleUnknown = 11;
	}




OneHotLayerParams
________________________________________________________________________________




.. code-block:: proto

	message OneHotLayerParams {

	    uint64 oneHotVectorSize = 1;
	    int64 axis = 2;
	    float onValue = 3;
	    float offValue = 4;
	}







OneHotEncoder.HandleUnknown
________________________________________________________________________________



.. code-block:: proto

	    enum HandleUnknown {
	        ErrorOnUnknown = 0;
	        IgnoreUnknown = 1;   // Output will be all zeros for unknown values.
	    }


