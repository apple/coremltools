# Converting an Open Efficient Language Model

The following example shows how to convert a PyTorch [OpenELM](https://huggingface.co/apple/OpenELM) model to Core ML. OpenELM is a family of Open Efficient Language Models. The small size variants are suitable for mobile and embedded language applications.

In this example you do the following:

1. [Import libraries and set up the model](#import-libraries-and-set-up-the-model)
2. [Export the model](#export-the-model)
3. [Convert the model to Core ML](#convert-the-model-to-core-ml)
4. [Tokenize the prompt](#tokenize-the-prompt)
5. [Run the PyTorch model](#run-the-pytorch-model)
6. [Run the converted Core ML model](#run-the-converted-core-ml-model)

## Requirements

This example requires macOS Monterey or newer versions, [PyTorch](https://pytorch.org/), [Transformers](https://huggingface.co/transformers/index.html), and Core ML Tools 8.0 or newer versions. Use the following commands:

```shell
pip install torch
pip install transformers
pip install coremltools
```

At the time of creating this example, the author environment is
```text Output
torch              2.5.1
transformers       4.46.3
coremltools        8.1
```

## Import Libraries and Set Up the Model

Import the `torch`, `numpy`, and `coremltools` libraries, and `AutoModelForCausalLM` and `AutoTokenizer` from `transformers`. 

```python
import torch
import numpy as np
import coremltools as ct
from transformers import AutoModelForCausalLM, AutoTokenizer
```

Initialize the language model along with its tokenizer. Here we choose the smallest variant [OpenELM-270M-Instruct](https://huggingface.co/apple/OpenELM-270M-Instruct), which uses the [Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b-hf) tokenizer

```python
torch_model = AutoModelForCausalLM.from_pretrained(
    "apple/OpenELM-270M-Instruct",
    torch_dtype=torch.float32,
    trust_remote_code=True,
    return_dict=False,
    use_cache=False,
)
torch_model.eval()

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
```

## Export the Model

You can now use PyTorch's exporter ([`torch.export.export`](https://pytorch.org/docs/stable/export.html#torch.export.export)) to export the language model:

```python
example_input_ids = torch.zeros((1, 32), dtype=torch.int32)
sequence_length = torch.export.Dim(name="sequence_length", min=1, max=128)
dynamic_shapes = {"input_ids": {1: sequence_length}}
exported_program = torch.export.export(
    torch_model,
    (example_input_ids,),
    dynamic_shapes=dynamic_shapes,
)
```

For more about exporting, see [Model Exporting](model-exporting).

## Convert the Model to Core ML

Convert the model `exported_program` to Core ML using the [`convert()`](https://apple.github.io/coremltools/source/coremltools.converters.convert.html#module-coremltools.converters._converters_entry) method:

```python
mlmodel = ct.convert(exported_program)
```

## Tokenize the Prompt

To test the performance of the converted model, tokenize the prompt (`"Once upon a time there was"`), and convert that list of tokens into a Torch tensor.

```python
prompt = "Once upon a time there was"
tokenized_prompt = torch.tensor(tokenizer(prompt)["input_ids"])
# Since model takes input ids in batch,
# create a dummy batch dimension (i.e. size 1) for tokenized prompt
tokenized_prompt = tokenized_prompt.unsqueeze(0)
```

## Run the PyTorch Model

Run the original PyTorch model with the tokenized prompt as input, to establish the benchmark for the model's performance. The output appears below the code:

```python
max_sequence_length = 64

input_ids = tokenized_prompt
# extend sentence (sequence) word-by-word (token-by-token)
# until reach max sequence length
for i in range(max_sequence_length):
    logits = torch_model(input_ids)[0]
    # determine the next token by greedily choosing the one with highest logit (probability)
    output_id = torch.argmax(logits, -1)[:, -1 :]
    # append the next token to sequence
    input_ids = torch.cat((input_ids, output_id), axis=-1)
# decode tokens back to text
output_text = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
print("Output text from the original torch model:")
print(output_text)
```

```text Output
Output text from the original torch model:
Once upon a time there was a man named John Smith. John Smith was a hard-working farmer, a good husband, and a good father. He loved his family dearly, and he cherished every moment with them.
But one day, John's life changed forever.
John's wife, Mary, died suddenly of
```

## Run the Converted Core ML Model

Now run the converted Core ML version of the model with the same input:

```python
input_ids = np.int32(tokenized_prompt.detach().numpy())
# extend sentence (sequence) word-by-word (token-by-token)
# until reach max sequence length
for i in range(max_sequence_length):
    logits = list(mlmodel.predict({"input_ids": input_ids}).values())[0]
    # determine the next token by greedily choosing the one with highest logit (probability)
    output_id = np.argmax(logits, -1)[:, -1 :]
    # append the next token to sequence
    input_ids = np.concat((input_ids, output_id), dtype=np.int32, axis=-1)
# decode tokens back to text
output_text = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
print("Output text from the converted Core ML model:")
print(output_text)
```

```text Output
Output text from the converted Core ML model:
Once upon a time there was a man named John Smith. John Smith was a hard-working farmer, a good husband, and a good father. He loved his family dearly, and he cherished every moment with them.
But one day, John's life changed forever.
John's wife, Mary, died suddenly of
```

As you can see, the converted Core ML model performs in the same manner as the original model.
