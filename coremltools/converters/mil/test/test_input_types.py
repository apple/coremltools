import pytest
from coremltools.converters.mil.input_types import RangeDim

def test_rangedim_default_within_bounds():
    dim = RangeDim(lower_bound=0, upper_bound=10, default=5)
    assert dim.default == 5

def test_rangedim_default_falls_back_to_lower_bound():
    dim = RangeDim(lower_bound=1, upper_bound=5)
    assert dim.default == 1

def test_rangedim_raises_if_default_below_lower():
    with pytest.raises(ValueError, match=r"less than minimum value"):
        RangeDim(lower_bound=3, upper_bound=10, default=2)

def test_rangedim_raises_if_default_above_upper():
    with pytest.raises(ValueError, match=r"greater than maximum value"):
        RangeDim(lower_bound=0, upper_bound=5, default=6)

def test_rangedim_ior_merges_bounds_and_adjusts_default():
    dim1 = RangeDim(lower_bound=0, upper_bound=10, default=5)
    dim2 = RangeDim(lower_bound=2, upper_bound=8, default=3)
    dim1 |= dim2
    assert dim1.lower_bound == 0     # keep this unless your __ior__ updates it
    assert dim1.upper_bound == 10    # same here unless logic changes
    assert dim1.default >= dim1.lower_bound
    assert dim1.default <= dim1.upper_bound
