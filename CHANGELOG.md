## Changelog  
  
Changes of note, other than bug fixes  


### coremltools 4.0a6
  
- added ability for user to specify input and output names of mlmodels for pytorch model conversion.
- added ability for user to pass in shape tuples instead of tensors as input for pytorch model conversion.
  
### coremltools 4.0a5  
  
- updated requirements for coremltools install in setup.py file (packages such as attr, attrs, sympy, scipy) 
  
  
### coremltools 4.0a4  
  
- Added python 3.8 and 3.6 wheels to the PyPI upload  
  
- Updated the API for using the TensorFlow (new) and PyTorch converters from `coremltools.converters.nnv2.converter.convert()` to `coremltools.converters.convert()`.  
The former is modified to `coremltools.converters.nnv2.converter._convert()` since it is for internal use only.

### coremltools 4.0a1

- First alpha release of coremltools 4. Includes 2 new converters (TensorFlow and PyTorch) that can be accessed via the `coremltools.converters.nnv2.converter.convert()` API