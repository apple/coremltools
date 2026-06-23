#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np
from PIL import Image

from coremltools import proto


def random_gen_input_feature_type(input_desc):
    if input_desc.type.WhichOneof("Type") == "multiArrayType":
        shape = [s for s in input_desc.type.multiArrayType.shape]
        if (
            input_desc.type.multiArrayType.dataType
            == proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT32
        ):
            dtype = np.float32
        elif (
            input_desc.type.multiArrayType.dataType
            == proto.FeatureTypes_pb2.ArrayFeatureType.INT32
        ):
            dtype = np.int32
        elif (
            input_desc.type.multiArrayType.dataType
            == proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT16
        ):
            dtype = np.float16
        elif (
            input_desc.type.multiArrayType.dataType
            == proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT64
        ):
            dtype = np.float64
        else:
            raise ValueError("unsupported type")
        return np.random.rand(*shape).astype(dtype)
    elif input_desc.type.WhichOneof("Type") == "imageType":
        if input_desc.type.imageType.colorSpace in (
            proto.FeatureTypes_pb2.ImageFeatureType.BGR,
            proto.FeatureTypes_pb2.ImageFeatureType.RGB,
        ):
            shape = [3, input_desc.type.imageType.height, input_desc.type.imageType.width]
            x = np.random.randint(low=0, high=256, size=shape)
            return Image.fromarray(np.transpose(x, [1, 2, 0]).astype(np.uint8))
        elif (
            input_desc.type.imageType.colorSpace
            == proto.FeatureTypes_pb2.ImageFeatureType.GRAYSCALE
        ):
            shape = [input_desc.type.imageType.height, input_desc.type.imageType.width]
            x = np.random.randint(low=0, high=256, size=shape)
            return Image.fromarray(x.astype(np.uint8), "L")
        elif (
            input_desc.type.imageType.colorSpace
            == proto.FeatureTypes_pb2.ImageFeatureType.GRAYSCALE_FLOAT16
        ):
            shape = (input_desc.type.imageType.height, input_desc.type.imageType.width)
            x = np.random.rand(*shape)
            return Image.fromarray(x.astype(np.float32), "F")
        else:
            raise ValueError("unrecognized image type")
    else:
        raise ValueError("unsupported type")


def compute_snr_and_psnr(x, y):
    assert len(x) == len(y)
    eps = 1e-5
    eps2 = 1e-10
    noise = x - y
    noise_var = np.sum(noise**2) / len(noise)
    signal_energy = np.sum(y**2) / len(y)
    max_signal_energy = np.amax(y**2)
    snr = 10 * np.log10((signal_energy + eps) / (noise_var + eps2))
    psnr = 10 * np.log10((max_signal_energy + eps) / (noise_var + eps2))
    return snr, psnr
