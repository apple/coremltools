# -*- coding: utf-8 -*-
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _

import pytest
from coremltools import models
import numpy as np
from coremltools.converters.nnv2.nnv2_program.ops import CoremlBuilder as cb
from coremltools.converters.nnv2 import converter
import logging

np.random.seed(0)

def test_single_layer_example():
    batch_size, input_dim, output_dim = 2, 4, 2
    @cb.program(input_specs=[
        cb.TensorSpec(shape=(batch_size, input_dim)),
        ])
    def prog(x):
        # Weight
        W_val = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                ).reshape(input_dim, output_dim).T.astype(np.float32)
        W = cb.const(val=W_val, mode="file_value", name="const_W")

        # bias
        b_val = np.array([-0.5, 0.5]).astype(np.float32)
        b = cb.const(val=b_val, mode="file_value", name="const_b")

        return cb.linear(x=x, weight=W, bias=b, name="lin")

    logging.info('prog:\n', prog)

    proto = converter._convert(prog,
                convert_from="NitroSSA",
                convert_to="nnv1_proto")

    feed_dict = {
            'x': np.random.rand(batch_size, input_dim).astype(np.float32),
            }
    model = models.MLModel(proto)
    assert model is not None
    prediction = model.predict(feed_dict)
    assert len(prediction) == 1

@pytest.mark.xfail # to be fixed in eng/PR-59740053 PR
def test_conv_example():
    batch, C_in, C_out, H, W = 2, 2, 3, 7, 10
    kH, kW = 3, 5
    img_shape, seq_shape = (batch, C_in, H, W), (batch, C_in, H)

    @cb.program(input_specs=[
        cb.TensorSpec(shape=img_shape),
        cb.TensorSpec(shape=seq_shape),
        ])
    def prog(img, seq):
        ## 2D convolution
        # Weight
        W_2d = np.random.rand(C_out, C_in, kH, kW).astype(np.float32)
        W_2d = cb.const(val=W_2d, mode="file_value", name="const_W")

        # Test 1: provide only required arguments.
        conv1 = cb.conv(x=img, W=W_2d, pad_type="valid")
        logging.info("conv1 shape: {}".format(conv1.shape))

        # Test 2: stride > 1
        conv2 = cb.conv(x=img, W=W_2d, pad_type="valid", strides=[2,3])
        logging.info("conv2 shape: {}".format(conv2.shape))

        # Test 3: same padding
        conv3 = cb.conv(x=img, W=W_2d, pad_type="same", strides=[2,3])
        logging.info("conv3 shape: {}".format(conv3.shape))

        # Test max_pool
        pool1 = cb.max_pool(x=img, kernel_sizes=[kH, kW],
                pad_type="valid", strides=[2,3])
        logging.info("pool1 shape: {}".format(pool1.shape))

        # Test max_pool
        pool2 = cb.max_pool(x=img, kernel_sizes=[kH, kW],
                pad_type="same", strides=[2,3])
        logging.info("pool2 shape: {}".format(pool2.shape))

        ## 1D convolution
        W_1d = np.random.rand(C_out, C_in, kH).astype(np.float32)
        W_1d = cb.const(val=W_1d, mode="file_value", name="const_W_1d")
        logging.info("W_1d val: {}".format(W_1d.val))

        # Test 4: provide only required arguments for 1D.
        conv4 = cb.conv(x=seq, W=W_1d, pad_type="valid")
        logging.info("conv4 shape: {}".format(conv4.shape))

        return conv1, conv2, conv3, pool1, pool2, conv4

    proto = converter._convert(prog,
                convert_from="NitroSSA",
                convert_to="nnv1_proto")

    feed_dict = {
            'img': np.random.rand(*img_shape).astype(np.float32),
            'seq': np.random.rand(*seq_shape).astype(np.float32),
            }
    model = models.MLModel(proto)
    assert model is not None
    prediction = model.predict(feed_dict)
    assert len(prediction) == 2

def test_while_example():
    def body(a, b):
        return cb.add(x=a, y=b), b

    def cond(a, b):
        a_mean = cb.reduce_mean(x=a, axes=[0, 1])
        b_mean = cb.reduce_mean(x=b, axes=[0, 1])
        return cb.less(x=a_mean, y=b_mean)

    @cb.program(input_specs=[
        cb.TensorSpec(shape=(1, 2)),
        cb.TensorSpec(shape=(1, 2)),
        ])
    def prog(a, b):
        return cb.while_loop(_cond=cond, _body=body, loop_vars=(a, b))

    logging.info('prog:\n', prog)

    proto = converter._convert(prog,
                convert_from="NitroSSA",
                convert_to="nnv1_proto")

    feed_dict = {
            'a': np.random.rand(1, 2).astype(np.float32),
            'b': np.random.rand(1, 2).astype(np.float32),
            }
    model = models.MLModel(proto)
    assert model is not None
    prediction = model.predict(feed_dict)
    assert len(prediction) == 2

