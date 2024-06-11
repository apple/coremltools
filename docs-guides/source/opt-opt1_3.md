Optimizing OPT Model
====================

In this tutorial, we will explore how we can use Core ML Tools APIs
for compressing an [OPT-1.3B](https://huggingface.co/docs/transformers/en/model_doc/opt) model.
``OPT-1.3B`` is a decoder-only pretrained transformer model, with 1.3 billion parameters, which 
is about ``2.6 GB`` in size when stored in ``FP16``.

Let's first load the model from [HuggingFace](https://huggingface.co/docs/transformers/en/model_doc/opt):

```python
from transformers import OPTForCausalLM

model = OPTForCausalLM.from_pretrained(
    "facebook/opt-1.3b", torch_dtype=torch.float32, use_cache=False
)
```

We will use [perplexity](https://huggingface.co/docs/transformers/en/perplexity)
on [c4](https://huggingface.co/datasets/allenai/c4) and [wikitext2](https://huggingface.co/datasets/Salesforce/wikitext)
datasets to evaluate the models. A lower perplexity is better. 

We will also look at knowledge score, 
an average of 9 different question-answering tasks, such as [arc_challenge](https://allenai.org/data/arc), 
[arc_easy](https://allenai.org/data/arc), 
[lambada_openai](https://huggingface.co/datasets/EleutherAI/lambada_openai), 
[triviaqa](https://nlp.cs.washington.edu/triviaqa/), etc. A higher knowledge score is better.

We will look at two different compression modes to compress the model: 
[quantization](opt-quantization.rst) and [palettization](opt-palettization.rst).


## Quantization 

### Data-free compression

Let us use ``PostTrainingQuantizer``, which scales and rounds weights to the nearest integer in the specified dtype.

```python
from coremltools.optimize.torch.quantization import PostTrainingQuantizer, \ 
    PostTrainingQuantizerConfig

config = PostTrainingQuantizerConfig.from_dict(
    {
        "global_config": {
            "weight_dtype": "int4",
            "granularity": "per_block",
            "block_size": 32,
        },
    }
)

quantizer = PostTrainingQuantizer(model, config)
compressed_model = quantizer.compress()
```

In this example, we are using 4 bits to represent the weights, with block-wise quantization, where
for a linear layer, ``32`` elements along a row in the weight matrix share the same quantization scales. 
One could also use ``per_tensor`` and ``per_channel`` granularity, and ``int8`` dtype. 

In the above code snippet, we apply data free compression directly to the PyTorch model.
However, you can also use [ct.optimize.coreml.linear_quantize_weights](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.post_training_quantization.html#coremltools.optimize.coreml.linear_quantize_weights)
to apply quantization directly on the Core ML model.

### Results

Let's look at the model size vs accuracy trade-off for these different configurations:

| Config                              | Model Size (GB) | c4 Perplexity     | wikitext2 Perplexity     | Knowledge Score (% correct) |
| :---                                |    :----:       |             :---: |                    :---: | :---:                       |
| Uncompressed                        | 2.63            | 16.07             | 14.62                    | 45.81                       |
| 8 bits per tensor                   | 1.32            | 16.15             | 14.61                    | 45.71                       |
| 8 bits per channel                  | 1.32            | 16.08             | 14.61                    | 45.98                       |
| 4 bits per tensor                   | 0.66            | 18763             | 31087                    | 21.66                       |
| 4 bits per channel                  | 0.66            | 31.51             | 31.91                    | 38.78                       |
| 4 bits per block, block size 1024   | 0.668           | 23.12             | 22.63                    | 41.39                       |
| 4 bits per block, block size 32     | 0.747           | 17.02             | 15.28                    | 45.16                       |
| **4 bits per block, block size 16** | **0.829**       | **16.78**         | **14.9**                 | 45.26                       |
| 4 bits per block, block size 8      | 0.993           | 16.55             | 14.83                    | 45.60                       |
| 4 bits per block, block size 4      | 1.32            | 16.33             | 14.64                    | 45.63                       |

We see that there is barely any loss in accuracy with ``8-bit`` quantization (``per-channel`` or ``per-tensor``), 
with a ``2x``reduction in model size.

As we look to compress the model more with ``4`` bits instead, the ``per-tensor``/``per-channel`` modes break down, 
and even block size ``1024`` gives poor perplexity. As the block size is decreased, 
we use more granular quantization, where fewer elements share the same scales, thus reducing the 
quantization error. We can see that as quantization error reduces, the perplexity values improve. 
However, this comes with an increase in model size, because we now need to store
more quantization parameters per weight. This overhead of storing quantization parameters becomes 
significant when using very small block sizes. 

From the table above, we can see that with post training data free quantization, 
we can go down to about ``0.83 GB``,  which is about ``3.2x`` reduction in model size, 
while keeping the increase in perplexity within ```~5%``` of the uncompressed value, 
and an even smaller decrease (```~1.2%```)  in knowledge score (the row in **bold**).
With more granular compression (block size < ``16``), we can limit the perplexity 
increase further down to ``~3%``, with a compression ratio of about ``~2.6x``.

For ``per-channel`` quantization configs, the model will run efficiently on either Neural 
Engine (NE) or GPU, whereas using the ``per-block`` config is beneficial for models that are 
being deployed on a Mac and running on the GPU specifically. 

Runtime memory and latency should improve compared to the uncompressed model, 
with the gains dependent on the model type, device, macOS version.

### Algorithm Runtime

Since ``PostTrainingQuantizer`` is data free, it's very fast, and it only takes a few seconds 
to compress OPT-1.3B model.

## Palettization

### Data-free compression

For palettization in a data free manner, we will use ``PostTrainingPalettizer``. This algorithm performs k-means to compress
the model's weights.

```python
from coremltools.optimize.torch.palettization import PostTrainingPalettizer, PostTrainingPalettizerConfig

config = PostTrainingPalettizerConfig.from_dict(
    {
        "global_config": {
            "n_bits": 4,
            "granularity": "per_grouped_channel",
            "group_size": 32,
            "enable_per_channel_scale": False
        },
    }
)

palettizer = PostTrainingPalettizer(model, config)
compressed_model = palettizer.compress(num_kmeans_workers=32)
```

``per_grouped_channel`` granularity with ``group_size = 32`` allows each group of 32 rows in the weight 
matrix to share a look-up table. 

``enable_per_channel_scale`` option can be turned on to have an output 
scale for each row in the weight matrix, which may help offset some of the effects of outliers by 
allowing weight values to be scaled to be between ``[-1, 1]`` before palettization.

In the above code snippet, we apply data free compression directly to the PyTorch model.
However, you can also use [ct.optimize.coreml.palettize_weights](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.post_training_quantization.html#coremltools.optimize.coreml.palettize_weights)
to directly palettize a Core ML model.

Here, we are using ``32`` parallel processes to perform k-means, to speed up the computation. 
All k-means operations are performed independently of each other, and hence can be massively 
parallelized. ``num_kmeans_workers`` can be set to the number of CPU cores available. 

### Calibration data based compression

For palettization with calibration data, we will use ``SKMPalettizer``, which implements  the algorithm
described in [SqueezeLLM: Dense-and-Sparse Quantization](https://arxiv.org/pdf/2306.07629.pdf),
and performs weighted k-means to compress the model's weights. 

```python
from transformers import AutoTokenizer
from coremltools.optimize.torch.palettization import SKMPalettizer, SKMPalettizerConfig

config = SKMPalettizerConfig.from_dict(
    {
        "global_config": {
            "n_bits": 4,
            "granularity": "per_grouped_channel",
            "group_size": 32,
            "enable_per_channel_scale": False
        },
        "calibration_nsamples": 128
    }
)

def get_c4(nsamples, seqlen):
    """
    Function for loading a subset for c4 training dataset
    """
    train_data = load_dataset(...) # Load allenai/c4 dataset
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
    
    # tokenize train_data and chunk it into sequences of length 
    # nsamples sequences of length seqlen
    
    return tokenized_train_data

def loss_fn(model, data):
    """
    Perform forward pass on the model and compute loss 
    """
    lm_logits = model(data).logits
    shift_logits = lm_logits[:, :-1, :].contiguous()
    shift_labels = data[:, 1:]

    loss_fct = torch.nn.CrossEntropyLoss()
    return loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

palettizer = SKMPalettizer(model, config)
compressed_model = palettizer.compress(dataloader=get_c4(nsamples=128, seqlen=2048),
                                       loss_fn=loss_fn,
                                       num_kmeans_workers=32)
```

``SKMPalettizer`` is similar to ``PostTrainingPalettizer``, but it requires some calibration data
and a loss function as inputs to the ``compress`` method. These are used to compute gradients 
of the model's weights, which are in turn used to compute Fisher information matrices. These matrices
serve as importance weights used in weighted k-means operations performed by ``SKMPalettizer``. We only 
need a few data samples (128 sequences in this example) to compute a reliable estimate of Fisher information
matrices. 

### Results

As before, let us look at the model size vs accuracy trade-off for these different configurations and algorithms:

| Config                                    | Model Size (GB) | c4 Perplexity (``PostTrainingPalettizer``)     | c4 Perplexity (``SKMPalettizer``)                |
| :---                                      |    :----:       |             :---:                              |                    :---:                         |
| Uncompressed                              | 2.63            | 16.07                                          | 16.07                                            |
| 4 bits per tensor                         | 0.819           | 21.57                                          | 17.02                                            |
| 4 bits per grouped channel, group size 8  | 0.821           | 17.80                                          | 16.58                                            |
| *4 bits per grouped channel, group size 4*| *0.823*         | 17.41                                          | *16.51*                                          |
| 4 bits per grouped channel, group size 2  | 0.827           | 17.34                                          | 16.47                                            |
| **4 bits per channel**                    | **0.834**       | 17.07                                          | **16.41**                                        |
| 3 bits per channel                        | 0.676           | 93.35                                          | 18.24                                            |
| 2 bits per channel                        | 0.521           | 6704                                           | 456                                              |

With palettization, as group size is decreased, the overhead of storing extra look up tables doesn't increase as 
dramatically, as it does for quantization parameters. As a result, we are able to achieve a better accuracy-size trade-off
of ``~2%`` degradation in perplexity with ``3.2x`` reduction (config in **bold** in table above). We can also observe 
that calibration data based algorithm improves over the performance of the data free algorithm, and provides a better 
accuracy-size trade-off. The difference between the two is more pronounced when group size is large, 
when ``per-tensor`` is used or when the bits are below 4.

For ``SKMPalettizer``+``4 bits per grouped channel, group size 4`` config, the knowledge score is ``45.16``,
whereas for ``SKMPalettizer``+``4 bits per channel`` model, the score is ``45.68``, and thus is limited within 
``~1%`` of the uncompressed model. 

The model degrades considerably in sub-4 bit compression regime, and this degradation is even more stark for the 
data free algorithm. 

#### Effect of per channel scales

| Config                                    | Model Size (GB) | c4 Perplexity (``PostTrainingPalettizer``)     | c4 Perplexity Perplexity (``SKMPalettizer``)     |
| :---                                      |    :----:       |             :---:                              |                    :---:                         |
| Uncompressed                              | 2.63            | 16.07                                          | 16.07                                            |
| 4 bits per tensor                         | 0.820           | 18.71                                          | 16.78                                            |
| 4 bits per grouped channel, group size 8  | 0.822           | 17.43                                          | 16.58                                            |
| 4 bits per grouped channel, group size 4  | 0.824           | 17.43                                          | 16.53                                            |
| 4 bits per grouped channel, group size 2  | 0.828           | 17.23                                          | 16.48                                            |
| 4 bits per channel                        | 0.835           | 17.08                                          | 16.41                                            |
| 3 bits per channel                        | 0.677           | 93.35                                          | 18.24                                            |
| 2 bits per channel                        | 0.522           | 6703                                           | 456                                              |

For this model, using ``per-channel`` scales doesn't improve the results much for ``SKMPalettizer``, because it 
automatically preserves sensitive values. However, for ``PostTrainingPalettizer``, there is some benefit to using
``per-channel`` scales, especially when no grouping is used. At smaller group sizes, the improvement disappears 
for data free algorithm as well. 

### Impact on Latency

Now, let's take a look at the impact of palettization on model latency. The model latency only depends on 
the config used for palettization (number of bits, group size, etc.) and not on which algorithm
was used for compression. Hence, we only look at latency numbers for models produced by ``SKMPalettizer``.

We measure latency for generating next token for a prompt for length 64. The measurements
are made on iPhone 15 Pro, iOS18 seed build, for the model running on the Neural Engine (NE).
Models compressed with ``per grouped channel`` palettization run efficiently on NE.

| Config                                     | Model Latency (ms) |  Speed-up |
| :---                                       |    :----:          |  :---:    |
| Uncompressed                               | 92.15              | 1x        |
| 4 bits per tensor                          | 14.76              | 6.24x     |
| 4 bits per grouped channel, group size 8   | 15.49              | 5.95x     |
| *4 bits per grouped channel, group size 4* | *17.07*            | *5.4x*    |
| 4 bits per grouped channel, group size 2   | 30.72              | 3x        |
| 4 bits per channel                         | 59.11              | 1.6x      |

From the table above, we observe that for group size up to ``4``, we get almost ``~6x`` speed-up. 
This is because  the uncompressed model is very large, and most of the inference time is spent 
on reading model weights into the memory. Such a model is called weight **memory bound**.
For OPT-1.3B model, below group size ``4``, the overhead of de-compressing 
multiple look up tables for each weight at runtime starts to become significant, 
leading to a marked increase in latency. 

The relative speed up you see will depend on numerous factors such as the 
model architecture, how it's defined, values of various parameters (embedding size, sequence length etc.), 
how the model is exported (static or dynamic input shapes, stateful K-V cache or not, etc.).
In most cases, for transformer based models that are weight memory bound, you would see 
latency gains with palettization on NE, all the way from a few percentage points to 
considerable gains, as with this specific OPT model.

From this data, we can conclude that ``4 bits per grouped channel palettization`` with group size ``4`` using 
``SKMPalettizer`` gives the best trade-off between model size, model latency and model accuracy, achieving
``5.4x`` improvement in model runtime, with about ``~3%`` degradation in model's perplexity (*italicized* above). 

**Note:** The latency numbers above are for relative comparison between compressed and uncompressed only. 
These were obtained on an iPhone 15 Pro with iOS18 seed 1 build. The actual numbers may vary 
depending on the iOS version, state of the device, model authoring code etc.

### Algorithm Runtime

Depending on the type and number of GPUs used, and the number of calibration samples,
it can take several minutes to compute the Fisher information matrices
for ``SKMPalettizer``. For example, it takes about a minute to compute it, when using 
8 ``A100`` GPUs. Once importance scores are computed, both ``SKMPalettizer``
and ``PostTrainingPalettizer`` take about ``10 - 30`` mins to compress the model. The time 
taken is more when using larger smaller group size (more k-means operations need to be 
performed) and when the number of bits is large.  

## Conclusions

In this tutorial, we looked at data free and calibration data based algorithms for 
compressing a large model. We learnt that calibration data based algorithms tend to 
provide better trade-off between model size and latency vs model accuracy, than data free algorithms.
These algorithms run almost as fast as data free algorithms and require very little data, 
much less than fine-tuning based compression algorithms. 

Models with a large number of parameters, where each parameter is itself large,
are often weight memory bound, and thus reducing the model size correlates with 
reduction in model runtime latency. Model accuracy can be traded-off with model size 
and runtime latency by using higher granularity of compression and this often produces 
good quality models which run fast. 

