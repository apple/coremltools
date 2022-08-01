# Copyright (c) 2022, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np

from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.input_type import (
    BoolInputType,
    FloatInputType,
    InputSpec,
    ScalarOrTensorInputType,
    StringInputType,
    TensorInputType,
)
from coremltools.converters.mil._deployment_compatibility import AvailableTarget as target
from coremltools.converters.mil.mil.operation import Operation
from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op
from coremltools.converters.mil.mil.ops.defs.iOS15.image_resizing import resample as _resample_iOS15
from coremltools.converters.mil.mil.ops.defs.iOS16 import _IOS16_TARGET

@register_op(opset_version=_IOS16_TARGET)
class resample(_resample_iOS15):
    """
    iOS16 version of resample supports float16 coordinates
    """
    input_spec = InputSpec(
        x=TensorInputType(),
        coordinates=ScalarOrTensorInputType(type_domain=(np.int32, np.float32, np.float16)),
        sampling_mode=StringInputType(const=True),
        padding_mode=StringInputType(const=True),
        padding_value=FloatInputType(const=True),
        coordinates_mode=StringInputType(const=True),
        align_corners=BoolInputType(const=True),
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def type_inference(self):
        return super().type_inference()
