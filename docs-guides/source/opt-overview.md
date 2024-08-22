Overview
========


This section covers optimization techniques that help you get a smaller model by compressing 
its weights and activations.
In particular, it will go over APIs for taking a model from 
`float` precision (16 or 32 bits per value) to <= 8 bits, 
while maintaining good accuracy.
 

You want to consider compression to either reduce the 
memory footprint of your model, or reduce the inference latency or 
the power consumption, or all at once. For instance, you may be working with a large model (e.g. >1B parameters) and need 
to reduce its memory footprint to make it reasonably run on an iPhone, 
or you may want to enable a real time experience for your user that
requires you to shave off a few milli-seconds of the inference time, 
or you may be shipping several models in your single app whose disk size you 
want to keep in check. In this section, you will learn about several 
techniques to help achieve these goals. 
You will learn how you can try them quickly on your model 
to see the effect on latency and size, and then based on your model 
choose the right approach 
to trade off accuracy and the time/data needed to optimize. 


You will learn about compression schemes that are possible with the 
CoreML runtime and Apple Silicon, which kind of hardware is best suited for 
different kinds of schemes and how various algorithms implemented in coremltools
can help you achieve memory and latency savings on your model.

Visit the [whats new](opt-whats-new) page to see what is available on different 
OS versions,
and go over the [example overview section](opt-overview-examples) 
to get a quick idea of typical workflows, with a tour 
of compression APIs applied 
on a standard convnet and a generative model.


## How to Compress

Given a pre-trained model, compression can be performed 
with different amounts of ingredients:  

- **Just the model**:
  this is referred to as data free compression. 
  It is very quick to apply and can work really well for
  reducing the model size by half (8 bits) or depending on the model 
  even up to 6 or 4 bits of representation, with only a slight decrease in accuracy. 
- **Model and a small amount of training data**: this is referred to as post training 
  compression with data calibration. Can lead to better accuracy for higher compression ratios.
- **Model, training pipeline and data**: having access to the full training pipeline and data
  allows for fine-tuning the model while compressing it. This allows for the best accuracy and compression ratio trade offs.

`coremltools.optimize` offers APIs for performing all these workflows. 
Some of these can be done directly on a Core ML model (mlpackage), while 
for some others requiring gradient computation, you can perform optimizations
on your PyTorch model prior to conversion. 
To find out more see 
[Optimization Workflow](opt-workflow).

## Types of Compression

Primarily, there are 3 ways to reduce the precision of weights/activations
of a model. These are: 

- **Palettization**: refers to discretizing weights using look up tables. 
   Core ML supports representing weights with palettization, to get them
  down to `nbits` precision, where `nbits` can be `{1, 2, 3, 4, 6, 8}`. 
  Check out the [palettization section](opt-palettization) for details.   
- **Linear quantization**: refers to approximating weights with a quantization function.
   Core ML supports INT4 and INT8 quantization options for weights, and INT8 for activations.
   Go to the [quantization section](opt-quantization) to know all about it. 
- **Pruning**: refers to zero-ing out values that are close to 0. Core ML supports sparse representations for weights. Find out more about this 
   in the [pruning section](opt-pruning).

These techniques can be combined as well, for instance
to give joint sparse and palettized, joint sparse and quantized weights etc
for further compression and runtime performance gains. See [this page](opt-joint-compression) 
to learn how to do that. 
  
## Effect on Runtime performance on Apple Silicon

Typically gains could be observed in one or more of 
runtime memory or latency or power consumption for a compressed model.
However, it would depend on not only the specific model 
(say whether its compute bound or memory bound) 
and compression type,
but also on the hardware and compute unit that the model is running on,
as implementations for compressed kernels in the NE/GPU/CPU compilers vary. 

For instance, 
some compiler backends may choose to decompress the weights fully
before runtime, thereby leading to a latency identical to that of
the `float16` model. In other cases, decompression may happen "on the fly"
utilizing hardware capabilities and fused kernels, needing
less data movement of weights between DRAM and processor cache, 
leading to lower inference times. 

Hence it is highly recommended to test on your specific model and Apple Silicon 
combination. As of `iOS18`/`macOS15`, following are a few high level recommendations 
that you may use to guide your experiments. For more detailed observations based on 
specific modes of optimization used,
please refer to the performance page in each of the 
quantization/palettization/pruning sections. 

- Weight palettization (all bits from 1 to 8) typically works the best on the Neural engine for 
  runtime memory and latency gains. On the GPU you may see runtime memory benefits.
- 8 bit activation plus weight quantization (also referred to as the W8A8) mode 
  can lead to considerable latency benefits on the neural engine, 
  by leveraging the faster int8-int8 
  compute path supported in newer hardware (A17 pro, M4).
- INT4 `per-block` quantization of weights can work really well for models using the GPU on a Mac.
- Pruning the models can lead to latency gains on the neural engine and the CPU 
  compute units. 


In most cases, you do not lose much accuracy with 6 or 8 bits of palettization 
or 8 bits of weight only quantization, which can be applied to your model in 
a matter of minutes. Hence that is one minimum amount 
of compression that you should consider for 
all your models that are currently shipping in `float16` precision.  

## Availability of features 

To find which optimization options are available in which OS versions look
at the [whats new](opt-whats-new.md) page. 


```{tip}
You may also find it useful to view the presentation in 
[WWDC 2023](https://developer.apple.com/videos/play/wwdc2023/10047/), 
for an introduction to the optimizations and 
in [WWDC 2024](https://developer.apple.com/videos/play/wwdc2024/10159/), for an overview 
of new features available from `iOS18`/`macOS15` and optimizations for 
large transformer models. 
```

