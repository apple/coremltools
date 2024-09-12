Overview
===========

```{eval-rst}
.. index:: 
    single: pruning; overview
```

Pruning a model is the process of sparsifying the weight 
matrices of the model’s layers, thereby reducing its storage size by 
packing weights more efficiently. 
Pruning sets a fraction of the values in the model’s weight matrices to zero.

Pruned weights can be represented more efficiently 
using a _sparse_ representation rather than the typical _dense_ 
representation. In a sparse representation, 
only non-zero values are stored, along with a bit mask, that takes 
the value `1` at the indices of the non-zero values. For 
example, if the weight values are `{0, 7, 0, 0, 0, 0, 0, 56.3}`, 
the sparse representation contains a bit mask with `1`s in the 
locations where the value is non-zero: `01000001b`. 
This is accompanied by the non-zero data, which in the 
following example is a size-2 vector of value `{7, 56.3}`.

```{figure} images/sparse_weights.png
:alt: Creating a sparse representation
:align: center
:class: imgnoborder

Creating a sparse representation.
```

“Unstructured sparsity” is when weight values of lowest magnitude are set to 
zero from within the weight tensor as a whole, without adhering to 
any structure. On the other hand, certain constraints can 
be enforced while pruning, resulting in what is referred to as 
“structured sparsity”. A few of the common types include: 

- `block sparsity`: When weights are pruned in a grouped manner, 
  such as in blocks of size 2, 4, and so on.  
- `n:m sparsity`: When weights are divided into blocks of size `m` 
  and within each such block, `n` smallest values are chosen to be zero. 
  For example, `3:4` sparsity would imply a 75% sparse level in 
  which three values in each block of size 4 are set to `0`. 

While imposing more constraints, structured sparsity 
techniques could be advantageous for certain hardware in 
terms of latency performance.

Starting from `iOS18`/`macOS15`, the non-zero float values in a
sparse representation can be further compressed either using
palettization or linear quantization, thereby providing further 
memory savings and improved runtime performance.   


```{admonition} Feature Availability

Sparse weight representation for Core ML `mlprogram` 
models is available in `iOS16`/`macOS13`/`watchOS9`/`tvOS16` 
and newer deployment target formats.
```

