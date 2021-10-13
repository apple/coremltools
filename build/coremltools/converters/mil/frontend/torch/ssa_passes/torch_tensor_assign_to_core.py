#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause


from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil import Builder as mb

@register_pass(namespace="torch")
def torch_tensor_assign_to_core(prog):
    """
    Try to map Torch dialect ops `torch_tensor_assign` into core op set if compatible.
	We only support the single index selection + whole dimension slicing + stride 1

	Example 1:
	x[0] = 0

	Example 2:
	x[:,:,0,:] = [[[0],[0]]]

	Currently, we tranform the torch_tensor_assign op into transposes + scatter + expand_dims

	Given:
        %output = torch_tensor_assign(data=%data, updates=%updates)

    Result:
        %data_transpose_1 = transpose(%data)
        %updates_expand = expand_dims(%updates)
        %scatter = scatter(data=%data_transpose_1, updates=%updates_expand)
        %output = transpose(%scatter)
        ...

	"""
    for f in prog.functions.values():
        _torch_tensor_assign_to_core_block(f)

def _torch_tensor_assign_to_core_block(block):
    for op in block.operations[:]:
        for b in op.blocks:
            _torch_tensor_assign_to_core_block(b)

        if op.op_type in ["torch_tensor_assign"]:
        	with block:
        		_transform_tensor_assign(op, block)

def _transform_tensor_assign(op, block):
	begin = op.begin.val
	end = op.end.val
	strides = op.stride.val

	begin_mask = op.begin_mask.val
	end_mask = op.end_mask.val
	squeeze_mask = op.squeeze_mask.val

	# check for the pattern is supported
	if any([stride != 1 for stride in strides]):
		raise NotImplementedError("Only tensor assignment with stride 1 is supported.")

	if sum(squeeze_mask) != 1:
		raise NotImplementedError("Only tensor assignment with exactly 1 pure dimension selection is supported")

	for i in range(len(squeeze_mask)):
		if not squeeze_mask[i]:
			if not (begin_mask[i] or begin[i] == 0) or not end_mask[i]:
				raise NotImplementedError("Non supported tensor assignment pattern detected.")

	# put the select dimension in front
	# for instance, x[:,0] = ...
	# we transpose the tensor to make the assignment be x[0] = ... instead
	data = op.data
	updates = op.updates
	out_name = op.outputs[0].name

	select_dim = squeeze_mask.tolist().index(True)
	perm = [select_dim] + [i for i in range(len(squeeze_mask)) if i != select_dim]
	data = mb.transpose(x=data, perm=perm, before_op=op)
	updates = mb.expand_dims(x=updates, axes=[0], before_op=op)
	select_idx = begin[select_dim]
	data = mb.scatter(data=data, indices=[select_idx], updates=updates, axis=0, mode="update", before_op=op)
	perm_back = [perm.index(i) for i in range(len(perm))]
	data = mb.transpose(x=data, perm=perm_back, name=out_name, before_op=op)

	op.enclosing_block.replace_uses_of_var_after_op(
	    anchor_op=op, old_var=op.outputs[0], new_var=data
	)
	# Remove all the ops at once
	block.remove_ops([op])
