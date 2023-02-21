#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

def fuse_conv_bias(paddle_program):
    for block in paddle_program.blocks:
        cnt = 0
        op_num = len(block.ops)
        for ind in range(op_num):
            i = ind + cnt
            op = block.ops[i]
            if op.type == "elementwise_add" and i > 0:
                last_op = block.ops[i-1]
                if last_op.type == "conv2d" and not last_op.input("Bias"):
                    block._insert_op(i+1, 
                                    type = "conv2d", 
                                    inputs={
                                        'Input': last_op.input("Input")[0],
                                        'Filter': last_op.input("Filter")[0],
                                        'Bias': op.input("Y")[0]
                                    },
                                    outputs={
                                        'Output': op.output("Out")[0]
                                    },
                                    attrs={
                                        'dilations': last_op.desc.attr('dilations'),
                                        'groups': last_op.desc.attr('groups'),
                                        'paddings': last_op.desc.attr('paddings'),
                                        'strides': last_op.desc.attr('strides')
                                    }
                                    )
                    block._remove_op(i-1)
                    block._remove_op(i-1)
                    cnt -= 1

    
            
