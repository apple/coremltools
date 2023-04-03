#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import pytest

from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.types import type_mapping


class TestTypeMapping:
    def test_promote_dtypes_basic(self):
        assert type_mapping.promote_dtypes([types.int32, types.int32]) == types.int32
        assert type_mapping.promote_dtypes([types.int32, types.int64, types.int16]) == types.int64
        assert type_mapping.promote_dtypes([types.fp16, types.fp32, types.fp64]) == types.fp64
        assert type_mapping.promote_dtypes([types.fp16, types.int32, types.int64]) == types.fp16

    @pytest.mark.parametrize(
        "input_size",
        [10, 10000],
    )
    def test_promote_dtypes_different_input_sizes(self, input_size):
        assert (
            type_mapping.promote_dtypes([types.int32, types.int64, types.int16] * input_size)
            == types.int64
        )
