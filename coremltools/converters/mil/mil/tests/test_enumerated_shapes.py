#  Copyright (c) 2026, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np
import pytest

from coremltools.converters.mil.input_types import EnumeratedShapes, RangeDim, Shape, TensorType
from coremltools.converters.mil.mil import get_new_symbol, types


@pytest.mark.parametrize("shape_factory", [Shape, list, tuple])
@pytest.mark.parametrize("default", [None, (1, 4)])
def test_enumerated_shapes_preserves_individual_shapes(shape_factory, default):
    first = shape_factory((1, 2))
    second = shape_factory((1, 4))
    enumerated = EnumeratedShapes([first, second], default=default)

    assert enumerated.shapes[0].symbolic_shape == [1, 2]
    assert enumerated.shapes[1].symbolic_shape == [1, 4]
    assert not enumerated.shapes[0].has_symbolic
    assert enumerated.symbolic_shape[0] == 1
    assert types.symbolic.is_symbolic(enumerated.symbolic_shape[1])
    assert tuple(enumerated.default) == ((1, 2) if default is None else default)
    if shape_factory is Shape:
        assert enumerated.shapes[0] is first
        assert first.to_list() == [1, 2]


def test_enumerated_shapes_reuses_shape_without_cross_contamination():
    shared = Shape((1, 2))
    first = EnumeratedShapes([shared, (1, 4)])
    first_symbolic_shape = first.symbolic_shape.copy()
    second = EnumeratedShapes([shared, (3, 2)])

    assert shared.to_list() == [1, 2]
    assert first.symbolic_shape == first_symbolic_shape
    assert first.symbolic_shape[0] == 1
    assert second.symbolic_shape[1] == 2
    assert types.symbolic.is_symbolic(second.symbolic_shape[0])
    assert first.symbolic_shape is not second.symbolic_shape
    # The original fixed shape must still reject an incompatible optional default.
    with pytest.raises(ValueError, match=r"default_value shape"):
        TensorType(shape=shared, default_value=np.zeros((1, 3), dtype=np.float32))


@pytest.mark.parametrize("symbolic_first", [False, True])
@pytest.mark.parametrize("use_range_dim", [False, True])
def test_enumerated_shapes_preserves_symbolic_inputs(symbolic_first, use_range_dim):
    dimension = RangeDim(2, 8, default=4) if use_range_dim else get_new_symbol()
    symbolic = Shape((dimension, 2))
    fixed = Shape((4, 3))
    symbol = symbolic.symbolic_shape[0]
    inputs = [symbolic, fixed] if symbolic_first else [fixed, symbolic]

    enumerated = EnumeratedShapes(inputs)

    assert symbolic.symbolic_shape == [symbol, 2]
    assert fixed.symbolic_shape == [4, 3]
    assert enumerated.symbolic_shape[0] is symbol
    assert types.symbolic.is_symbolic(enumerated.symbolic_shape[1])
    assert enumerated.default == inputs[0].default


def test_enumerated_shapes_repeated_shape_has_independent_symbolic_shape():
    shared = Shape((1, 2))
    enumerated = EnumeratedShapes([shared, shared])
    EnumeratedShapes([shared, (3, 4)])
    assert shared.to_list() == [1, 2]
    assert enumerated.symbolic_shape == [1, 2]
