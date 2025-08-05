# Converting a Natural Language Processing Model

```{eval-rst}
.. index:: 
    single: PyTorch; combine tracing and scripting
    single: PyTorch; convert natural language processing model
```

The following example demonstrates how you can combine [model tracing](model-tracing) and [model scripting](model-scripting) in order to properly convert a model that includes a data-dependent control flow, such as a loop or conditional. 

```{warning}
If you convert a scripted model, a warning appears explaining that support for scripted models is experimental. For details, see [model scripting](model-scripting).
```

You can apply a [mix of scripting and tracing](model-scripting.md#mix-tracing-and-scripting) to optimize which parts of the model you want to trace, and which part you want to script. You can trace those portions of the model that are free of the control flow, and then script the control flow. In this example, the model runs the body of the code a fixed number of times inside a control loop. You can therefore trace the body of the code separately, and apply scripting to this outer control loop.

In this example you do the following:

1. [Import libraries and set up the model](#import-libraries-and-set-up-the-model)
2. [Trace and script the model](#trace-and-script-the-model)
3. [Convert the model to Core ML](#convert-the-model-to-core-ml)
4. [Encode the sentence fragment as input](#encode-the-sentence-fragment-as-input)
5. [Run the PyTorch model](#run-the-pytorch-model)
6. [Run the converted Core ML model](#run-the-converted-core-ml-model)


## Requirements

This example requires macOS Monterey or newer versions, [PyTorch](https://pytorch.org/), and [Transformers](https://huggingface.co/transformers/index.html). Use the following commands:

```shell
pip install torch
pip install transformers
pip install -U coremltools
```


## The GPT-2 NLP Model

This example converts the PyTorch [GPT-2](https://huggingface.co/transformers/model_doc/gpt2.html) transformer-based natural language processing (NLP) model to Core ML.

GPT-2 was trained on a dataset of over eight million web pages, with a simple objective: predict the next word, given all of the previous words within some text. For example, if you input "The Manhattan bridge is", the model produces the rest of the sentence: "The Manhattan bridge is a major artery for the city's subway system, and the bridge is one of the busiest in the country."

## Import Libraries and Set Up the Model

Import the `torch`, `numpy`, and `coremltools` libraries, and `GPT2LMHeadModel` and `GPT2Tokenizer` from `transformers`. 

```python
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import coremltools as ct
```

The example code passes a sentence fragment, encoded as integer tokens, into the model, which predicts the next token in sequence. A partially constructed sentence is then fed into the model, which appends a new token. This process is repeated until the model predicts a special end-of-sentence (`eos`) token. 

The `FinishMySentence()` module inherits from [`torch.nn.Module` ](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) and contains attributes for the `eos` token, the `next_token_predictor` model, and the default token denoting the beginning of a sentence. In its `forward` method, the loop body takes a list of tokens and predicts the next one. The loop continues until the `eos` token is generated. When this happens, the sentence is returned.

```python
class FinishMySentence(torch.nn.Module):
    def __init__(self, model=None, eos=198):
        super(FinishMySentence, self).__init__()
        self.eos = torch.tensor([eos])
        self.next_token_predictor = model
        self.default_token = torch.tensor([0])
    
    def forward(self, x):
        sentence = x
        token = self.default_token
        while token != self.eos:
            predictions, _ = self.next_token_predictor(sentence)
            token = torch.argmax(predictions[-1, :], dim=0, keepdim=True)
            sentence = torch.cat((sentence, token), 0)
        
        return sentence
```

## Trace and Script the Model

Initialize the `token_predictor` from `GPT2LMHeadModel`, a GPT2 model transformer:

```python
token_predictor = GPT2LMHeadModel.from_pretrained("gpt2", torchscript=True).eval()
```

You can now use PyTorch's JIT tracer ([`torch.jit.trace`](https://pytorch.org/docs/stable/generated/torch.jit.trace.html)) to trace the loop body as it predicts the next token from a list of random tokens:

```python
random_tokens = torch.randint(10000, (5,))
traced_token_predictor = torch.jit.trace(token_predictor, random_tokens)
```

Tracing the loop body in this manner elicits a warning from JIT tracer that the trace might not generalize to other inputs, but you can ignore this warning.

With the bulk of the loop body traced, you can instantiate the model and apply PyTorch's [JIT script](https://pytorch.org/docs/stable/generated/torch.jit.script.html) to script the outer control loop:

```python
model = FinishMySentence(model=traced_token_predictor)
scripted_model = torch.jit.script(model)
```

For more about tracing and scripting, see [Model Tracing](model-tracing) and [Model Scripting](model-scripting).

## Convert the Model to Core ML

Convert the model `scripted_model` to Core ML using the [`convert()`](https://apple.github.io/coremltools/source/coremltools.converters.convert.html#module-coremltools.converters._converters_entry) method. Specify the `TensorType` as the required input, using as `shape` the range of `[1, 64]` for the sequence dimension, and `numpy.int32` for the `dtype` of the `TensorType`:

```python
mlmodel = ct.convert(
    scripted_model,
    # Range for the sequence dimension to be between [1, 64]
    inputs=[ct.TensorType(name="context", shape=(ct.RangeDim(1, 64),), dtype=np.int32)],
)
```

## Encode the Sentence Fragment as Input

To test the performance of the converted model, encode the sentence fragment (`"The Manhattan bridge is"`) using the `GPT2Tokenizer`, and  convert that list of tokens into a Torch tensor.

```python
sentence_fragment = "The Manhattan bridge is"

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
context = torch.tensor(tokenizer.encode(sentence_fragment))
```


## Run the PyTorch Model

Run the original PyTorch model with the tokenized sentence fragment as input, to establish the benchmark for the model's performance. The output appears below the code:

```python
torch_out = scripted_model(context)
generated_text_torch = tokenizer.decode(torch_out)
print("Fragment: {}".format(sentence_fragment))
print("Completed: {}".format(generated_text_torch))
```

```text Output
Fragment: The Manhattan bridge is
Completed: The Manhattan bridge is a major artery for the city's subway system, and the bridge is one of the busiest in the country.
```

## Run the Converted Core ML Model

Now run the converted Core ML version of the model with the same input:

```python
coreml_inputs = {"context": context.to(torch.int32).numpy()}
prediction_dict = mlmodel.predict(coreml_inputs)
generated_tensor = prediction_dict["sentence_2"]
generated_text = tokenizer.decode(generated_tensor)
print("Fragment: {}".format(sentence_fragment))
print("Completed: {}".format(generated_text))
```

```text Output
Fragment: The Manhattan bridge is
Completed: The Manhattan bridge is a major artery for the city's subway system, and the bridge is one of the busiest in the country.
```

As you can see, the converted Core ML model performs in the same manner as the original model.


