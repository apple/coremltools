Overview
========


This section covers optimization techniques that help you get a smaller model by compressing 
its weights and activations.
In particular, it will go over APIs for taking a model from 
`float` precision (16 or 32 bits per value) to <= 8 bits, 
while maintaining good accuracy.
 

Model compression can help reduce the 
memory footprint of your model, reduce inference latency, reduce  
power consumption, or reduce them all at once. For instance, you may be working with a large model (e.g. >1B parameters) and need 
to reduce its memory footprint to make it reasonably run on an iPhone, 
or you may want to enable a real-time experience for your user that
requires you to shave off a few milliseconds of the inference time, 
or you may be shipping several models in your single app whose disk size you 
want to keep in check. In this section, you will learn about several 
techniques to help achieve these goals. 
You will learn how you can try them quickly on your model 
to see the effect on latency and size, and then 
choose the right approach to trade off accuracy and the time/data needed to optimize, based on your model. 


You will learn about compression schemes that are possible with the 
Core ML runtime and Apple Silicon, which kind of hardware is best suited for 
different kinds of schemes, and how various algorithms implemented in `coremltools`
can help you achieve memory and latency savings on your model.

Visit [What’s New](opt-whats-new) to see what is available on different 
OS versions. Go over the [example overview section](opt-overview-examples) 
to get a quick idea of typical workflows and to get a tour of compression APIs applied 
on a standard convnet and a generative model.


## How to Compress

Given a pre-trained model, compression can be performed 
with different ingredients in varying quantities: 

- **Just the model**:
  This is referred to as data-free compression. 
  It is very quick to apply and can work really well for
  reducing the model size by half (8 bits) or, depending on the model,
  even up to 6 or 4 bits of representation, with only a slight decrease in accuracy. 
- **Model and a small amount of training data**: This is referred to as post-training 
  compression with data calibration. It can lead to better accuracy for higher compression ratios.
- **Model, training pipeline, and data**: Having access to the full training pipeline and data
  lets you fine-tune the model while compressing it. This allows for the best accuracy and compression ratio trade offs.

`coremltools.optimize` offers APIs for performing all these workflows. 
You can execute some workflows directly on a Core ML model (`mlpackage`). For other workflows 
requiring gradient computation, you can perform optimizations
on your PyTorch model prior to conversion. 
To find out more, see 
[Optimization Workflow](opt-workflow).

## Types of Compression

There are three primary ways to reduce the precision of weights and activations
of a model: 

- **Palettization**: Refers to discretizing weights using lookup tables. 
  Core ML supports representing weights with palettization to get them
  down to `nbits` precision, where `nbits` can be `{1, 2, 3, 4, 6, 8}`. 
  Read more in the [palettization section](opt-palettization).   
- **Linear quantization**: Refers to approximating weights with a quantization function.
  Core ML supports INT4 and INT8 quantization options for weights and INT8 for activations.
  Read more in the [quantization section](opt-quantization). 
- **Pruning**: refers to zeroing out values that are close to zero. Core ML supports sparse representations for weights. 
  Read more in the [pruning section](opt-pruning).

These techniques can be combined as well. For example, a joint sparse and palettized model or a joint sparse and quantized weights model can result in further compression and runtime performance gains. See [Combining Compression Types](opt-joint-compression) 
to learn how to do that. 
  
## Effect on Runtime performance on Apple Silicon

Typically gains from model compression could be observed in the form of 
runtime memory, latency, power consumption, or some combination of them.
However, these gains depend not only on the specific model 
(for example, whether it’s compute-bound or memory-bound) and compression type,
but also on the hardware and compute unit that the model is running on. 
This is because implementations for compressed kernels in the NE/GPU/CPU compilers vary. 

For instance, some compiler backends may choose to decompress the weights fully
before runtime, leading to a latency identical to that of the `float16` model. 
In other cases, decompression may happen “on the fly”, 
utilizing hardware capabilities and fused kernels, needing
less data movement of weights between DRAM and processor cache, 
leading to lower inference times. 

Because the decompression strategy varies per hardware and compute unit, is highly recommended to test 
on your specific model and Apple Silicon combination. 
As of `iOS18`/`macOS15`, here are a few high level recommendations 
that you may use to guide your experiments. For more detailed observations based on 
specific modes of optimization used, refer to the performance page in the 
quantization, palettization, and pruning sections. 

- Weight palettization (all bits from 1 to 8) typically works the best on the Neural Engine for 
  runtime memory and latency gains. On the GPU you may see runtime memory benefits.
- 8-bit activation plus weight quantization, also referred to as the W8A8 mode, 
  can lead to considerable latency benefits on the Neural Engine by leveraging the faster int8-int8 
  compute path supported in newer hardware (A17 pro, M4).
- INT4 `per-block` quantization of weights can work really well for models using the GPU on a Mac.
- Pruning the models can lead to latency gains on the Neural Engine and the CPU 
  compute units. 


In most cases, you do not lose much accuracy with 6 or 8 bits of palettization 
or 8 bits of weight-only quantization, which can be applied to your model in 
a matter of minutes. That is one minimum amount 
of compression that you should consider for 
all your models that are currently shipping in `float16` precision.  

## Availability of features 

To find which optimization options are available in which OS versions see [What’s New](opt-whats-new.md). 


```{tip}
You may also find it useful to view the presentation in 
[WWDC 2023](https://developer.apple.com/videos/play/wwdc2023/10047/) 
for an introduction to the optimizations, and the presentation 
in [WWDC 2024](https://developer.apple.com/videos/play/wwdc2024/10159/) for an overview 
of new features available from `iOS18`/`macOS15` and optimizations for 
large transformer models. 
```

