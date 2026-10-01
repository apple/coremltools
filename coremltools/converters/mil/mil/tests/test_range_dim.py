#  Copyright (c) 2026, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import pytest

from coremltools.converters.mil.input_types import RangeDim, Shape


@pytest.mark.parametrize(
    "left_bounds, right_bounds, expected_bounds",
    [
        ((2, 10), (0, 5), (0, 10)),
        ((2, 10), (12, 20), (2, 20)),
        ((2, 10), (0, -1), (0, -1)),
        ((2, 10), (12, -1), (2, -1)),
        ((2, -1), (0, 10), (0, -1)),
        ((2, -1), (12, -1), (2, -1)),
        ((0, 0), (1, -1), (0, -1)),
    ],
)
@pytest.mark.parametrize("default_position", ["implicit", "lower", "middle", "upper"])
def test_rangedim_union_preserves_default(left_bounds, right_bounds, expected_bounds, default_position):
    lower, upper = left_bounds
    # An unbounded range may have a default above the other range's upper bound.
    finite_upper = upper if upper != -1 else 20
    default = {
        "implicit": None,
        "lower": lower,
        "middle": (lower + finite_upper) // 2,
        "upper": finite_upper,
    }[default_position]
    left = RangeDim(*left_bounds, default=default)
    right = RangeDim(*right_bounds)
    original = left
    original_symbol = left.symbol
    original_default = left.default
    right_state = (right.lower_bound, right.upper_bound, right.default, right.symbol)

    left |= right

    assert left is original
    assert left.symbol is original_symbol
    assert (left.lower_bound, left.upper_bound) == expected_bounds
    assert left.default == original_default
    assert Shape((1, left)).default == (1, original_default)
    assert (right.lower_bound, right.upper_bound, right.default, right.symbol) == right_state


@pytest.mark.parametrize("upper_bound", [10, -1])
def test_rangedim_union_with_itself_preserves_default(upper_bound):
    dim = RangeDim(lower_bound=2, upper_bound=upper_bound, default=7)
    dim |= dim
    assert (dim.lower_bound, dim.upper_bound, dim.default) == (2, upper_bound, 7)
