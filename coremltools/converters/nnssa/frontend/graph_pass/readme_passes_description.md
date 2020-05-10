Pass
====
This directory contains all the graph transformation pass implementations

```
1: insert `get_tuple`
2: Delete disconnected nodes: `delete_disconnected_nodes.py`
3: Functionalize Loops `functionalize_loops.py`
4: Constant Propagation. `constant_propagation.py`
5: Remove Variable Nodes. `variable_node_transform`
6: Type Inference. `type_inference.py`
7: Delete ancestors of constant nodes. `constant_propagation.py`
```
Insert Get Tuple
================
Tensorflow uses input "nodename:i" to denote "get tuple i" from "nodename".
Here we split it so that:

```
node1:i -> node2
```

gets transformed into

```
node1 -> get_tuple(i) --> node2
```

Takes a graph in "dict{str, ParsedTFNode}" form, and returns a new graph.


We do not do this for control flow nodes(Switch, Enter, Exit, Merge
LoopCond, NextIteration). Since they behave in interesting ways. For these nodes
we just convert

```
node1:i -> node2
```

to

```
node1 -> node2
```

This ensures that the graph is fully connected prior to the functionalize loop
transformation.

Functionalize Loop
==================
This is one of the most complicated graph transformations and there is not
really an easy way to describe what this does or how this works.

For a while loop of a single variable, Tensorflow generates the following
dataflow graph

```
while(lambda i: i<10, lambda i:i+1, [i])


            Less(10)----->LoopCond----------|
              ^                             |
              |                             v
i-->Enter-->Merge------------------------>Switch-->Exit--...>
              ^                             |
              |                             |
              --NextIteration<----Add(1)<----
```

For a loop of multiple variables, there is one Enter, Merge, Switch, Exit, 
NextIteration for each variable. For instance

```
while(lambda i,j: i<j, lambda i,j:(i+1,j), [i,j])


              --NextIteration<--Identity<---|
              |                             |
              v                             |
j-->Enter-->Merge------------------------>Switch-->Exit--...>
              |                             ^
              v                             | 
             Less-------->LoopCond----------|
              ^                             |
              |                             v
i-->Enter-->Merge------------------------>Switch-->Exit--..>
              ^                             |
              |                             |
              --NextIteration<----Add(1)<---|
```

There is one exception: Constants have an Enter, but do not have the rest of
the control flow nodes.

This code is complicated looking but it basically tries to find, for each
input variable, the corresponding Enter, Merge, NextIteration, Switch, Exit
nodes.

Find Loop Condition
-------------------
After which, we disconnect the edges `Enter->Merge, Merge->Switch, *->Merge, LoopCond->Switch`
(note there is only one LoopCond node).

In the graph above, we end up with

```
                NextIteration<--Identity<---|
                                            |
                                            |
j-->Enter   Merge                         Switch-->Exit--..>
              |
              v
             Less-------->LoopCond
              ^ 
              |
i-->Enter   Merge                         Switch-->Exit--..>
                                            |
                                            |
                NextIteration<----Add(1)<---|
```

At this point, the subgraph Merge-...LoopCond is the condition function.
Replacing Merge with its corresponding `get_tuple`, and LoopCond with a `return`,
we have the function definition.
```
 get_tuple
     |
     v
    Less-------->Return
     ^ 
     |
 get_tuple
```

Find Loop Body
--------------
To find the loop body, we next disconnect Switch from Exit

```
                NextIteration<--Identity<---|
                                            |
                                            |
j-->Enter   Merge                         Switch   Exit--..>
              |
              v
             Less-------->LoopCond
              ^ 
              |
i-->Enter   Merge                         Switch   Exit--..>
                                            |
                                            |
                NextIteration<----Add(1)<---|
```

Then we take the subgraph between Switch and NextIteration: 

```
NextIteration<--Identity<---|
                            |
                            |
                          Switch


                          Switch
                            |
                            |
NextIteration<----Add(1)<---|
```
Switch are then replaced by `get_tuple`, and NextIteration replaced with
a single `make_tuple`

```
             ---Identity<---|
             |              |
             |              |
             |           get_tuple
             ^
return<--make_tuple
             ^
             |           get_tuple
             |              |
             |              |
             -----Add(1)<---|
```


Create Functional While
-----------------------
The remaining nodes are now just Enter and Exits
```
j-->Enter   Exit--..>

i-->Enter   Exit--..>
```

Enter are replaced with `make_tuple`, Exit are replaced with `get_tuple`,
and a while loop referencing the condition function and the body function 
is inserted between them.

```
j-->Enter    make_tuple--..>
     |           ^
     v           |
  make_tuple-->while
     ^           |
     |           v
i-->Enter    make_tuple--..>
```

And we are done.


Constant Propagation
====================

There are 2 classes of constant propagation.

Tensorflow Constant Propagation
-------------------------------
The first class of constant propagation is done in constant\_propagation.py
and converts all nodes which have no Placeholder ancestors, to constant.

This is necessary since there may be operations of the sort:

```
X=Range(100)
# we might be doing something real with X later
# But we are going to use the shape of X for something else.
Y = tf.fill(shape(X), 1)
```

This may generate:
```
    Range --> Shape --> Fill -- ... -->
     |
     |
     .
     .
     .
Other Stuff
```
Generally, this is a useful coding pattern since it allows me to change the
shape of X in just one place, (like changing it to Range(200)) will 
automatically update the shape of all the rest of the code. However, this can
be problematic because it limits the shape inference downstream. ex: without
constant propagation, the output shape of Fill is unknown.

However, this can be solved by using Tensorflow to perform constant propagation.
constant\_propagation.py extracts subgraphs whose ancestors are only constant
nodes, and we rebuild a partial TF graph and let Tensorflow perform the
evaluation. This allows the above graph to be reduced to

```
    Range    Const --> Fill -- ... -->
     |
     |
     .
     .
     .
Other Stuff
```

Constant Propagation in Type Inference
--------------------------------------
However, there is a second common pattern in which constant propagation by
Tensorflow is insufficient and we need to perform some propagation on our own.

In particular, this is in settings where the Shape of a Tensor is used as a
value, and acted upon. For instance:

```
MakeTensorArray(Shape(X)[1])
```

This end up creating ops of the form

```
X --> Shape
       |    |-Const
       v    v
     StridedSlice <-- Const
       |    ^
       |    |
       |  Const
       |
       v
  TensorArrayV3
```

Now, if Shape(X) = [1,2,3] for instance, then we might be able remove
the StridedSlice, reducing this to:
```

X --> Shape

     Const(2)
       |
       v
  TensorArrayV3
```

However, if Shape(X) = [1,-1,3] then we have to maintain the entire graph.

As such, type\_inference.py performs limited value propagation. In particular
the Shape operator tries to fill in its own value. To differentiate
between the two cases above:
  - if Shape(X) is fully defined, we fill in the
`node.value` field, in which case we can delete some ancestor nodes. If Shape(X)
is not fully defined, 
  - If Shape(X) is not fully defined (has -1 values), we fill in
  `node.attr['incomplete_value']` and node deletion does not occur. But 
  downstream nodes which recognize the incomplete value can still act on them.


Delete Disconnected Nodes
=========================
This is a very simple pass. It simply deletes all nodes with no inputs
or outputs.


Variable Nodes
==============
Variable nodes are not horribly complicated.

There are Variable nodes which don't really do much on their own

To initialize, there is an additional Assign op which is just dangling away
on one side which assigns from `Variable/initial_value`

```
[Variable] --> Assign <-- Const (VariableName/initial_value)
     |
     | ... rest of graph ...
     v
... Assign <---- New Values
... etc
```

Reads of the variable go through an Identity node with the name 
`VariableName/read`, and has attribute `_class:loc:@VariableName`.

Writes of the variable go through an Assign nodes which take as input 
one Variable and one value, and has attribute `_class:loc:@VariableName`.
Assign also returns the new value of the variable.



 - We transform Variable to a function attribute
 - We transform Assign ops to just `set_global` with attribute variable:VariableName
 - We transform Read ops to just `get_global` with attribute variable:VariableName


TensorArray
============
A TensorArray is essentially a runtime vector<Tensor> with the following properties:

 - an optional requirement `infer_shape` (True by default) that all Tensors
   stored within the vector have the same size/shape (inferred by the
   first element stored into the tensor)
 - an optional `element_shape` which requires all elements to have this
   exact shape.
 - an optional `clear_after_read` (True by default) where read of an index 
   is destructive. (It doesn't *really* destroy, but just enables a particular
   optimization where the tensor memory can be reused).
 - An optional `dynamic_size` (False by default) where the vector is resized 
   automatically at runtime

The way it works is rather odd. To enforce "control dependency" constraints,
a single float (flow) variable is passed between operations that write/read
the TensorArray. Additionally, a "Resource" variable is also passed along
which contains the actual handle to the TensorArray.

The TensorArray can therefore also be passed around as as argument to while
loops.  Thus unlike a global "Variable", this really is better thought of as
an additional type, a list[tensor]. 

See:

<https://github.com/tensorflow/tensorflow/blob/r1.6/tensorflow/python/ops/tensor_array_ops.py>

<https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/tensor_array.h>

<https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/tensor_array.cc>

<https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/ops/data_flow_ops.cc>

<https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/tensor_array_ops.cc>

The way we transform it is to introduce a new type. list[tensor]
The flow variable is the list[tensor] since that is consistently passed through
every operation.  The 'resource' node then gets passed as void. This transform
is performed inside of `type_inference.py`


Type Inference
==============
Type inference tries to define a type for each vertex. Tensorflow sometimes
gives us enough type information, but frequently we have to infer and propagate
the rules on our own.


