#  Copyright (c) 2022, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np

from coremltools import _logger as logger

from .annotate import annotate, class_annotate, delay_type
from .type_bool import bool
from .type_spec import Type


def make_complex(width):
    delay_type_complex = getattr(delay_type, "complex" + str(width))

    @class_annotate()
    class complex:
        _width = width

        def __init__(self, v=0 + 0j):
            self._val: np.complexfloating = (
                np.complex64(v) if width == 64 else np.complex128(v)
            )

        @property
        def val(self):
            return self._val

        @val.setter
        def val(self, v):
            from .type_mapping import (
                builtin_to_string,
                nptype_from_builtin,
                numpy_type_to_builtin_type,
            )

            if not isinstance(v, np.generic):

                if isinstance(v, np.ndarray) and v.ndim == 0:
                    # Rank zero tensor case. Use as a scalar.
                    self._val = v.item()
                else:
                    raise ValueError(
                        f"Types should have zero-rank ndarray input, got {v} instead."
                    )

            elif isinstance(v, np.complexfloating):
                v_type = numpy_type_to_builtin_type(v.dtype)
                if v_type.get_bitwidth() <= self.get_bitwidth():
                    self._val = v
                else:
                    self._val = v.astype(nptype_from_builtin(self.__class__))
                    logger.warning(
                        "Saving value type of {} into a builtin type of {}, might lose precision!".format(
                            v.dtype, builtin_to_string(self.__class__)
                        )
                    )
            else:
                self._val = v.astype(nptype_from_builtin(self.__class__))
                logger.warning(
                    "Saving value type of {} into a builtin type of {}, might be incompatible or "
                    "loses precision!".format(
                        v.dtype, builtin_to_string(self.__class__)
                    )
                )

        @classmethod
        def __type_info__(cls):
            return Type("complex" + str(cls._width), python_class=cls)

        @classmethod
        def get_bitwidth(cls):
            return cls._width

        @annotate(delay_type_complex, other=delay_type_complex)
        def __add__(self, other):
            assert isinstance(other, complex)
            return complex(self.val + other.val)

        @annotate(delay_type_complex, other=delay_type_complex)
        def __sub__(self, other):
            assert isinstance(other, complex)
            return complex(self.val - other.val)

        @annotate(delay_type_complex, other=delay_type_complex)
        def __mul__(self, other):
            assert isinstance(other, complex)
            return complex(self.val * other.val)

        @annotate(delay_type_complex, other=delay_type_complex)
        def __div__(self, other):
            assert isinstance(other, complex)
            return complex(self.val / other.val)

        @annotate(delay_type_complex, other=delay_type_complex)
        def __mod__(self, other):
            raise ValueError("Can't mod complex numbers.")

        @annotate(delay_type.bool, other=delay_type_complex)
        def __lt__(self, other):
            return bool(self.val < other.val)

        @annotate(delay_type.bool, other=delay_type_complex)
        def __gt__(self, other):
            return bool(self.val > other.val)

        @annotate(delay_type.bool, other=delay_type_complex)
        def __le__(self, other):
            return bool(self.val <= other.val)

        @annotate(delay_type.bool, other=delay_type_complex)
        def __ge__(self, other):
            return bool(self.val >= other.val)

        @annotate(delay_type.bool, other=delay_type_complex)
        def __eq__(self, other):
            return bool(self.val == other.val)

        @annotate(delay_type.bool, other=delay_type_complex)
        def __ne__(self, other):
            return bool(self.val != other.val)

        @annotate(delay_type.bool)
        def __bool__(self):
            return self.val

        @annotate(delay_type.int)
        def __int__(self):
            logger.warning(
                "ComplexWarning: Casting complex to real discards the imaginary part."
            )
            return int(np.real(self.val))

        @annotate(delay_type_complex)
        def __complex__(self):
            return complex(self.val)

        @annotate(delay_type.str)
        def __str__(self):
            return str(self.val)

        @annotate(delay_type_complex)
        def __log__(self):
            # The `math.log` doesn't support complex numbers yet.
            return np.log(self.val)

        @annotate(delay_type_complex)
        def __exp__(self):
            return np.exp(self.val)

        @annotate(delay_type_complex)
        def __neg__(self):
            return complex(-self.val)

    complex.__name__ = "complex%d" % complex.get_bitwidth()
    return complex


# We keep consistent with PyTorch and Tensorflow:
# - complex64 consists of a fp32 real and a fp32 imag.
# - complex128 consists of a fp64 real and a fp64 imag.
complex64 = make_complex(64)
complex128 = make_complex(128)
complex = complex64


def is_complex(t):
    complex_types_set = (complex64, complex128)
    return (t in complex_types_set) or isinstance(t, complex_types_set)
