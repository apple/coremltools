```{eval-rst}
.. index:: 
    single: MLModelInspector
    single: MLModelValidator
    single: MLModelComparator
    single: TorchScriptMLModelComparator
    single: TorchExportMLModelComparator
    single: MLModelBenchmarker
    single: TorchMLModelBenchmarker
    single: Remote-Device
```

# Debugging And Performance Utilities

These utilities help identify and resolve both numerical and performance issues in exported Core ML models. When a model produces unexpected outputs-such as NaNs, infinities, or results that differ from the source model or exhibits performance bottlenecks, these tools assist in isolating the problematic operations. Once identified, targeted fixes can be applied to address and correct these issues, improving both the accuracy and efficiency of the model.

```{admonition} Experimental
These APIs are currently located under the experimental namespace, which means they may change or become incompatible with previous versions in future releases. They will remain in this namespace until they have been refined and are ready to be promoted to stable APIs.
```

## MLModelInspector

`MLModelInspector` is a utility class that retrieves intermediate outputs from a Core ML model by modifying the model to expose specified internal operations as the model outputs. `MLModelInspector` can be used to debug a model and is utilized by both `MLModelComparator` and `MLModelValidator`. 

For example, to retrieve output of a `convolution` operation identified by its output name `var_1`, you can use the following:


```python
import coremltools as ct
from coremltools.models.ml_program.experimental.debugging_utils import MLModelInspector

# Initialize the MLModelInspector.
inspector = MLModelInspector(
    model=model, 
    compute_units=ct.ComputeUnit.CPU_ONLY,
)
# Use the inspector to retrieve intermediate outputs from the model.
# `inputs` specifies the input data for the model, and `output_names` lists the internal operations
# (e.g., variables) whose outputs you want to inspect.
outputs = await inspector.retrieve_outputs(
    inputs={"input": np.array([1, 2, 3])},  # Input data for the model
    output_names=["var_1"],                 # Name of the intermediate variable to retrieve
)
# Print the retrieved output for the specified internal operation ("var_1").
print(outputs["var_1"])
```

## MLModelValidator

If an exported Core ML model produces unexpected outputs, such as infinities or NaNs, the `MLModelValidator` can assist in identifying and isolating the problematic operations within the ML program.

For example, if an exported Core ML model produces NaN values as output, the `find_failing_ops_with_nan_output` method can be used to identify the specific operations responsible for generating NaNs in the model's output.


```python
import coremltools as ct
from coremltools.models.ml_program.experimental.debugging_utils import MLModelValidator

# Initialize MLModelValidator
validator = MLModelValidator(
    model = model,
    compute_unit = ct.ComputeUnit.CPU_ONLY,
)
# Find the ops that are responsible for NaN output
failing_ops = await validator.find_failing_ops_with_nan_output(
    inputs={"input": np.array([1, 2, 3])} # Inputs that produce NaN output
)

print(failing_ops)
```

If the exported Core ML model produces infinity values as outputs, the `find_failing_ops_with_infinite_output` method can be used to identify the specific operations responsible for generating infinities in the model's output.


```python
import coremltools as ct
from coremltools.models.ml_program.experimental.debugging_utils import MLModelValidator

# Initialize MLModelValidator
validator = MLModelValidator(
    model = model,
    compute_unit = ct.ComputeUnit.CPU_ONLY,
)
# Find the ops that are responsible for NaN output
failing_ops = await validator.find_failing_ops_with_infinite_output(
    inputs={"input": np.array([1, 2, 3])} # Inputs that produce infinities in the output
)

print(failing_ops)
```

`MLModelValidator` also supports passing a custom validation function, enabling more tailored debugging for specific use cases.

```python
import coremltools as ct
import numpy as np

from coremltools.models.ml_program.experimental.debugging_utils import MLModelValidator
from coremltools import proto

# Initialize MLModelValidator
validator = MLModelValidator(
    model = model,
    compute_unit = ct.ComputeUnit.CPU_ONLY,
)

def validate_output(op: proto.MIL_pb2.Operation, value: np.array):
    # Check if the output is zero
    return np.all(value == 0)

# Find the ops that are responsible for unexpected output
failing_ops = await validator.find_failing_ops(
    validate_output=validatate_output,
    inputs={"input": np.array([1, 2, 3])} # Inputs that produce infinities in the output
)

print(failing_ops)
```

After identifying the problematic operations, the issue may stem from either division by zero or numerical overflow. For division by zero, the source model can be updated to address the problem directly. In cases of overflow, employing higher precision for the affected operations is often sufficient to resolve the issue.

Note: The process of identifying failing operations may be time-consuming, as the duration depends on the model's complexity.


## MLModelComparator

MLModelComparator is a utility designed to compare reference and target models derived from the same source model. It is particularly useful in scenarios where an exported Core ML model produces unexpected outputs on specific compute units or when using a particular precision (such as float16). By comparing the outputs of a reference model and a target model, MLModelComparator helps identify the operations responsible for these discrepancies.

For example, if an exported Core ML model produces correct outputs when using `float32` precision but generates unexpected outputs with `float16` precision, you can use `MLModelComparator` to identify the operations responsible for the discrepancies.


```python
import coremltools as ct
import numpy as np

from coremltools.models.ml_program.experimental.debugging_utils import MLModelComparator

# Initialize MLModelComparator to compare reference and target models
comparator = MLModelComparator(
    reference_model=reference_model,  # Model with expected behavior
    target_model=target_model,        # Model to be debugged
)

# Define a custom comparison function to evaluate output discrepancies
def compare_outputs(operation, reference_output, target_output):
    # Compare outputs with a tolerance of 1e-1
    # Return True if outputs are close, False otherwise
    return np.allclose(reference_output, target_output, atol=1e-1)

# Identify operations causing discrepancies between models
failing_ops = await comparator.find_failing_ops(
    inputs={"input": np.array([1, 2, 3])},  # Sample input for comparison
    compare_outputs=compare_outputs         # Custom comparison function
)

print(failing_ops)
```

After identifying the problematic operations, the issue might be related to the precision of those operations. In such cases, you can resolve the problem by using higher precision (e.g., float32) for the operations when exporting the model.

Note: The process of identifying failing operations may be time-consuming, as the duration depends on the model's complexity.


## TorchScriptMLModelComparator

`TorchScriptMLModelComparator` is a utility designed to compare the outputs of a torch module and its corresponding exported Core ML model.  It utilizes `torch.jit.trace` to convert the PyTorch model into a TorchScript representation, which is then converted into a Core ML model. This utility is useful for debugging cases where inconsistent outputs occur during the conversion process from PyTorch to Core ML using TorchScript. It helps to identify specific PyTorch modules that produce inconsistent results between the original torch model and the converted Core ML model. 

Before employing this utility, first verify if the `float32` precision model produces consistent results. If it does, it's preferable to use `MLModelComparator` with the `float32` model as the reference and the problematic model as the target. `TorchScriptMLModelComparator` operates at the module level, which may require additional steps to pinpoint specific problematic operations.

For example, to find the modules that produce inconsistent results, you can use the following:


```python
import coremltools as ct
import numpy as np
import torch

from coremltools.models.ml_program.experimental.torch.debugging_utils import TorchScriptMLModelComparator

# Define a simple PyTorch model
class Model(torch.nn.Module):
    def forward(self, x, y):
        return x + y

# Create an instance of the model
torch_model = Model()

# Prepare example inputs for the model
input1 = torch.full((1, 10), 1, dtype=torch.float)
input2 = torch.full((1, 10), 2, dtype=torch.float)
inputs = (input1, input2)

# Initialize the TorchScriptMLModelComparator
comparator = TorchScriptMLModelComparator(
    model=torch_model,
    example_inputs=inputs,  # Inputs used to trace the PyTorch model
    inputs=[
        # Define input tensor specifications for Core ML
        coremltools.TensorType(name="x", shape=inputs[0].shape, dtype=np.float32),
        coremltools.TensorType(name="y", shape=inputs[1].shape, dtype=np.float32),
    ],
    compute_unit = ct.ComputeUnit.CPU_ONLY,
)

# Define a custom comparison function
def compare_outputs(module, reference_output, target_output):
    # Compare outputs with a tolerance of 0.1
    return np.allclose(reference_output, target_output, atol=1e-1)

# Use the comparator to find failing modules
modules = await comparator.find_failing_modules(
    inputs=inputs,
    compare_outputs=compare_outputs
)

# Print the modules that failed the comparison
print(modules)
```

## TorchExportMLModelComparator

`TorchExportMLModelComparator` is a utility designed to compare the outputs of a torch module and its corresponding exported Core ML model.  It utilizes `torch.export.export` to convert the PyTorch model into an `ExportedProgram` , which is then converted into a Core ML model. This utility is useful for debugging cases where inconsistent outputs occur during the conversion process from PyTorch to Core ML using `torch.export.export`. It helps to identify specific PyTorch operations that produce inconsistent results between the original torch model and the converted Core ML model. 

Before employing this utility, first verify if the `float32` precision model produces consistent results. If it does, it's preferable to use `MLModelComparator` with the `float32` model as the reference and the problematic model as the target. 

For example, to find the modules that produce inconsistent results, you can use the following:


```python
import coremltools as ct
import numpy as np
import torch

from coremltools.models.ml_program.experimental.torch.debugging_utils import TorchExportMLModelComparator

# Define a simple PyTorch model
class Model(torch.nn.Module):
    def forward(self, x, y):
        return x + y

# Create an instance of the model
torch_model = Model()

# Prepare example inputs for the model
input1 = torch.full((1, 10), 1, dtype=torch.float)
input2 = torch.full((1, 10), 2, dtype=torch.float)
inputs = (input1, input2)
exported_program = torch.export.export(torch_model, inputs)

# Initialize the TorchExportMLModelComparator
comparator = TorchExportMLModelComparator(
    model=exported_program,
    inputs=[
        # Define input tensor specifications for Core ML
        ct.TensorType(name="x", shape=inputs[0].shape, dtype=np.float32),
        ct.TensorType(name="y", shape=inputs[1].shape, dtype=np.float32),
    ],
    compute_unit = ct.ComputeUnit.CPU_ONLY,
)

# Define a custom comparison function
def compare_outputs(operation, reference_output, target_output):
    # Compare outputs with a tolerance of 0.1
    return np.allclose(reference_output, target_output, atol=1e-1)

# Use the comparator to find failing operations
operations = await comparator.find_failing_ops(
    inputs=inputs,
    compare_outputs=compare_outputs
)

# Print the ops that failed the comparison
print(operations)
```

## MLModelBenchmarker

`MLModelBenchmarker` is a utility for analyzing the performance of Core ML models. It measures key metrics such as model loading time, prediction latency, and the execution times of individual operations. 

For example, to benchmark a model's load and prediction performance, you can use the following:

```python
from coremltools.models.ml_program.experimental.perf_utils import MLModelBenchmarker

# Initialize the MLModelBenchmarker with the Core ML model
benchmarker = MLModelBenchmarker(model=model)
# Benchmark the model's loading time over 5 iterations
# This measures how long it takes to load the model.
load_measurement = await benchmarker.benchmark_load(iterations=5)
# Print the median loading time from the benchmark results
print(load_measurement.statistics.median)

# Benchmark the model's prediction time over 5 iterations with a warmup phase
# The warmup ensures that any initialization overhead is excluded from the measurements.
predict_measurement = await benchmarker.benchmark_predict(iterations=5, warmup=True)
# Print the median prediction time from the benchmark results
print(predict_measurement.statistics.median)
```

To evaluate the execution performance of operations, you can use the following:

```python
from coremltools.models.ml_program.experimental.perf_utils import MLModelBenchmarker

# Initialize the MLModelBenchmarker with the Core ML model
benchmarker = MLModelBenchmarker(model=model)
# Benchmark operation execution times over 5 iterations with a warmup phase
# The warmup ensures that any initialization overhead is excluded from the measurements.
execution_time_measurements = benchmarker.benchmark_operation_execution(iterations=5, warmup=True)
# Print the median execution time of the most time-consuming operation
# The operations are sorted in descending order of execution time.
print(f"Median execution time of the slowest operation: {execution_time_measurements[0].statistics.median} seconds")
```

Note: `MLModelBenchmarker` utilizes the model's compute plan to estimate the execution time of individual operations within the model.


## TorchMLModelBenchmarker

`TorchMLModelBenchmarker` is a specialized benchmarking tool designed for PyTorch models. It extends the capabilities of `MLModelBenchmarker` to offer tailored performance analysis for PyTorch models. While retaining all the functionality of its parent class, `TorchMLModelBenchmarker` introduces additional methods to estimate execution times for individual torch nodes and modules.

For example, to benchmark the execution time of individual nodes in the PyTorch model, you can use the following:


```python
import coremltools as ct
import numpy as np
import torch

from coremltools.models.ml_program.experimental.torch.perf_utils as TorchMLModelBenchmarker

# Define a simple PyTorch model
class Model(torch.nn.Module):
    def forward(self, x, y):
        # Perform addition and subtraction on inputs x and y
        return (x + y, x - y)
# Create an instance of the PyTorch model
torch_model = Model()
# Prepare example inputs for the model
input1 = torch.full((1, 10), 1, dtype=torch.float)  # Tensor filled with ones
input2 = torch.full((1, 10), 2, dtype=torch.float)  # Tensor filled with twos

# Export the PyTorch model using torch.export or torch.jit.trace
traced_model = torch.export.export(torch_model, (input1, input2))  # For PyTorch >= 2.0
# traced_model = torch.jit.trace(torch_model, (input1, input2))   # For older versions of PyTorch

# Initialize the TorchMLModelBenchmarker for benchmarking the Torch model
benchmarker = TorchMLModelBenchmarker(
    model=traced_model,
    inputs=[
        ct.TensorType(name="x", shape=input1.shape, dtype=np.float16),  # Define input tensor x
        ct.TensorType(name="y", shape=input2.shape, dtype=np.float16),  # Define input tensor y
    ],
    minimum_deployment_target=ct.target.iOS16,  # Specify minimum deployment target (e.g., iOS16)
    compute_units=coremltools.ComputeUnit.ALL,  # Use all available compute units (CPU/GPU/Neural Engine)
)

# Benchmark node execution times in the model
# Perform 5 iterations with a warmup phase for stable measurements
node_execution_times = await benchmarker.benchmark_node_execution(iterations=5, warmup=True)
# Print the median execution time of the slowest PyTorch operation
print(f"Median execution time of the slowest operation: {node_execution_times[0].measurement.statistics.median} ms"
```

## Remote-Device

Remote-Device is a utility that allows you to run and analyze Core ML models on connected devices, offering tools for debugging and benchmarking issues specific to those devices. It utilizes `devicectl` to establish communication with the connected device, facilitating the deployment and execution of Core ML models. To leverage this utility, you must have Xcode and Xcode Command Line installed on your local system and have a development device. 

Make sure that the development device is connected and is unlocked. Running the following command will output the list of connected iPhone devices.


```python
from coremltools.models.ml_program.experimental.remote_device import (
    AppSigningCredentials, 
    Device,
    DeviceType,
)

# Get a list of connected iPhone devices
connected_devices = Device.get_connected_devices(device_type=DeviceType.IPHONE)
# This will display information about each connected iPhone, which may include device name, os version, and other relevant details
print(connected_devices)
```

The connected device should appear in the displayed information. The next step involves installing an application that coremltools uses to load and execute Core ML models on the device.

```python
connected_device = connected_devices[0]
# Define the app signing credentials
credentials = AppSigningCredentials(
    development_team="<TEAM-ID>",  # Your Apple Developer Team ID
    bundle_identifier="com.example.modelrunnerd",  # Unique identifier for your app
    provisioning_profile_uuid=None  # UUID of provisioning profile (if applicable)
)

# Prepare the device for model debugging
# This installs the application on the device
prepared_device = await connected_device.prepare_for_model_debugging(credentials=credentials)
```

In this example, we use the `Apple Developer Team ID`. `Xcode` will automatically create and manage a team provisioning profile associated with the specified Team ID. However, if you have a specific provisioning profile UUID, you can use it instead. Ensure that the `bundle_identifier` matches the one defined in the provisioning profile to avoid any conflicts.

`prepare_for_model_debugging` builds and installs the `ModelRunner` application on the device. The initial launch may take some time, but subsequent launches should be significantly faster. Once `prepare_for_model_debugging` completes, the `ModelRunner` application will be launched on the connected device.

You can now execute the model on the connected device.

```python
import coremltools as ct
import numpy as np
import torch

from coremltools.models.ml_program.experimental.async_wrapper import MLModelAsyncWrapper

# Define a simple PyTorch model
class Model(torch.nn.Module):
    def forward(self, x, y):
        # Perform element-wise addition and subtraction on inputs x and y
        return (x + y, x - y)
 
# Create example input tensors for the model
input1 = torch.randn(1, 100)  # Random tensor with shape (1, 100)
input2 = torch.randn(1, 100)  # Random tensor with shape (1, 100)
# Instantiate the PyTorch model and set it to evaluation mode
model = Model()
model.eval()
# Trace the PyTorch model to create a TorchScript representation
traced_model = torch.jit.trace(model, (input1, input2))
# Convert the TorchScript model to a Core ML model
ml_model = ct.convert(
    traced_model,
    inputs=[
        ct.TensorType(name="x", shape=input1.shape, dtype=np.float16),  # Define input tensor x
        ct.TensorType(name="y", shape=input2.shape, dtype=np.float16),  # Define input tensor y
    ],
    minimum_deployment_target=ct.target.iOS17, # Specify the minimum deployment target (iOS 17)
    compute_units=ct.ComputeUnit.ALL,          # Use all available compute units (CPU/GPU/Neural Engine)
)

# Wrap the Core ML model for remote execution on a connected device
remote_model = MLModelAsyncWrapper.from_spec_or_path(
    spec_or_path=ml_model.get_spec(),  # Provide the Core ML model specification
    weights_dir=ml_model.weights_dir,  # Specify the directory containing model weights
    device=prepared_device             # Target device for remote execution
)

# Prepare example inputs for prediction
x = np.full((1, 100), 1.0)  # Input tensor x filled with ones
y = np.full((1, 100), 2.0)  # Input tensor y filled with twos
# Perform prediction on the remote device and print the results
print(await remote_model.predict(inputs={"x": x, "y": y}))
```

The remote device can also be utilized with other tools, such as `MLModelBenchmarker`, `TorchMLModelBenchmarker`, `MLModelInspector`, `MLModelValidator`, and `MLModelComparator`, to perform benchmarking and debugging on the remote device.

For instance, `MLModelBenchmarker` can be used with a connected device to benchmark the model's performance directly on the device.


```python
from coremltools.models.ml_program.experimental.perf_utils import MLModelBenchmarker

# Initialize the MLModelBenchmarker with the Core ML model and the remote device.
benchmarker = MLModelBenchmarker(model=model, device=prepared_device)
# Benchmark operation execution times over 5 iterations with a warmup phase
# The warmup ensures that any initialization overhead is excluded from the measurements.
execution_time_measurements = benchmarker.benchmark_operation_execution(iterations=5, warmup=True)
# Print the median execution time of the most time-consuming operation
# The operations are sorted in descending order of execution time.
print(f"Median execution time of the slowest operation: {execution_time_measurements[0].statistics.median} seconds")
```
