# Typed Execution

A model’s compute precision impacts its performance and numerical accuracy, which may impact the user experience of apps using the model. Core ML models saved as [ML Programs](doc:ml-programs) or neural networks execute with either float 32 or float 16 precision. 

This page describes how the precision is determined by the runtime during execution of either type of model. The ability to choose this precision can give you more flexible control over computations and performance.

While its useful to understand how the system works, you do not need a complete understanding of the runtime to get your app working optimally with Core ML. In most cases you do not have to take any special action, either during model conversion with coremltools, or during model prediction in your app with the Core ML framework. The defaults picked by coremltools and the Core ML framework work well with most models and are picked to optimize the performance of your model.
