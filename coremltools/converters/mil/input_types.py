#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import logging
import numpy as np
from coremltools.converters.mil.mil.types.symbolic import is_symbolic
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.types.type_mapping import (
    numpy_type_to_builtin_type,
    is_builtin,
)


class ClassifierConfig(object):
    def __init__(
        self,
        class_labels,
        predicted_feature_name="classLabel",
        predicted_probabilities_output=None,
    ):
        """
        Configuration for classifier models.

        Parameters
        ----------
        class_labels: str / list of int / list of str
            If a ``list`` is given, the ``list`` maps the index of the output of a
            neural network to labels in a classifier.
            
            If a ``str`` is given, the ``str`` points to a file which maps the index
            to labels in a classifier.
        
        predicted_feature_name: str
            Name of the output feature for the class labels exposed in the
            Core ML neural network classifier. Default: ``'classLabel'``.
        
        predicted_probabilities_output: str
            If provided, then this is the name of the neural network blob which
            generates the probabilities for each class label (typically the output
            of a softmax layer). 
            
            If not provided, then the last output layer is assumed.
        """
        self.class_labels = class_labels
        self.predicted_feature_name = predicted_feature_name
        self.predicted_probabilities_output = predicted_probabilities_output


class InputType(object):
    def __init__(self, name=None, shape=None, dtype=types.fp32):
        """
        The Input Type for inputs fed into the model.

        Parameters
        ----------
        name: (str)
            The name of the input.
        
        shape: list, tuple, Shape object, EnumeratedShapes object, or None
            The shape(s) that are valid for this input.
            
            If set to ``None``, the shape will be infered from the model itself.
        """

        self.name = name
        if shape is not None:
            self.shape = _get_shaping_class(shape)
        else:
            self.shape = None
        self.dtype = dtype


class ImageType(InputType):
    def __init__(
        self,
        name=None,
        shape=None,
        scale=1.0,
        bias=None,
        color_layout="RGB",
        channel_first=None,
    ):
        """
        Configuration class used for image inputs in CoreML.

        Parameters
        ----------
        scale: (float)
            The scaling factor for all values in the image channels.
        
        bias: float or list of float
            If ``color_layout`` is ``'G'``, bias would be a ``float``.
            
            If `color_layout` is ``'RGB'`` or ``'BGR'``, bias would be a list of ``float``.
        
        color_layout: string
            Color layout of the image.
            
            Valid values:
                * ``'G'``: Grayscale
                * ``'RGB'``: [Red, Green, Blue]
                * ``'BGR'``: [Blue, Green, Red]
            
        channel_first: (bool) or None
            Set to ``True`` if input format is channel first.
            
            Default format:
            	For TensorFlow: channel last (``channel_first=False``).
            	
                For PyTorch: channel first (``channel_first=True``).
        """
        super(ImageType, self).__init__(name, shape)
        self.scale = scale
        if color_layout not in ["G", "RGB", "BGR"]:
            raise ValueError(
                "color_layout should be one of ['G', 'RGB', 'BGR'], got '{}' instead".format(
                    color_layout
                )
            )
        self.color_layout = color_layout

        if bias is None:
            self.bias = 0.0 if color_layout == "G" else [0.0, 0.0, 0.0]
        else:
            self.bias = bias
        self.channel_first = channel_first

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        str_repr = 'ImageType[name={}, shape={}, scale={}, bias={}, ' +\
                'color_layout={}, channel_first={}]'
        return str_repr.format(self.name, self.shape, self.scale, self.bias,
                self.color_layout, self.channel_first)


class TensorType(InputType):
    def __init__(self, name=None, shape=None, dtype=None,
        default_value=None):
        """
        Specify a (dense) tensor input.

        Parameters
        ----------
        name: str
            Input name. Must match an input name in the model (usually the
            Placeholder name for TensorFlow or the input name for PyTorch).
            
            The ``name`` is required except for a TensorFlow model in which there is
            exactly one input Placeholder.
        
        shape: (1) list of positive int or RangeDim, or (2) EnumeratedShapes
            The shape of the input.
            
            For TensorFlow:
              * The ``shape`` is optional. If omitted, the shape is inferred from
                TensorFlow graph's Placeholder shape.
            
            For PyTorch:
              * The ``shape`` is required.
        
        dtype: np.generic or mil.type type
            Numpy ``dtype`` (for example, ``np.int32``). Default is ``np.float32``.
        
        default_value: np.ndarray
            If provided, the input is considered optional. At runtime, if the
            input is not provided, ``default_value`` is used.
            
            Limitations:
              *  If ``default_value`` is ``np.ndarray``, all
                 elements are required to have the same value.

              * The ``default_value`` may not be specified if ``shape`` is
                ``EnumeratedShapes``.

        Examples
        --------
        * ``ct.TensorType(name="input", shape=(1, 2, 3))` implies `dtype ==
          np.float32``
        
        * ``ct.TensorType(name="input", shape=(1, 2, 3), dtype=np.int32)``
    	
        * ``ct.TensorType(name="input", shape=(1, 2, 3),
          dtype=ct.converters.mil.types.fp32)``
        """
        super(TensorType, self).__init__(name, shape)
        if dtype is None:
            self.dtype = types.fp32
        elif is_builtin(dtype):
            self.dtype = dtype
        else:
            # Assume dtype is numpy type
            try:
                self.dtype = numpy_type_to_builtin_type(dtype)
            except TypeError:
                raise TypeError("dtype={} is unsupported".format(dtype))

        if default_value is not None:
            if isinstance(shape, EnumeratedShapes):
                msg = 'TensorType input {} has EnumeratedShapes and ' +\
                    'may not be optional'
                raise ValueError(msg.format(name))
            if not isinstance(default_value, np.ndarray):
                msg = 'TensorType {} default_value is not np.ndarray'
                raise ValueError(msg.format(name))
            default_fill_val = default_value.flatten()[0]
            if not np.all(default_value == default_fill_val):
                msg = 'TensorType {} default_value can only have ' +\
                    'same entries'
                raise ValueError(msg.format(name))
            if not self.shape.has_symbolic and \
                list(default_value.shape) != list(self.shape.symbolic_shape):
                msg = 'TensorType {} default_value shape {} != ' +\
                    'TensorType.shape {}'
                raise ValueError(msg.format(name, default_value.shape,
                    self.shape.to_list()))
            if numpy_type_to_builtin_type(default_value.dtype) != self.dtype:
                msg = 'TensorType {} default_value dtype {} != ' +\
                    'TensorType.dtype {}'
                raise ValueError(msg.format(name, default_value.dtype,
                    self.dtype.__type_info__()))

        self.default_value = default_value

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'TensorType[name={}, shape={}, dtype={}]'.format(self.name,
                self.shape, self.dtype)


class RangeDim(object):
    def __init__(self, lower_bound=1, upper_bound=-1, default=None,
            symbol=None):
        """
        A class that can be used to give a range of accepted shapes.

        Parameters
        ----------
        lower_bound: (int)
            The minimum valid value for the shape.
        
        upper_bound: (int)
            The maximum valid value for the shape.
            
            Set to ``-1`` if there's no upper limit.
        
        default: (int) or None
            The default value that is used for initiating the model, and set in

            input shape field of the model file.
            
            If set to ``None``, ``lower_bound`` would be used as default.

        symbol: (str)
            Optional symbol name for the dim. Autogenerate a symbol name if
            not specified.
        """
        if symbol is None:
            from coremltools.converters.mil.mil import get_new_symbol
            self.symbol = get_new_symbol()
        else:
            from coremltools.converters.mil.mil import Symbol
            self.symbol = Symbol(symbol)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        if default is None:
            self.default = lower_bound
        else:
            if default < lower_bound:
                raise ValueError(
                    "Default value {} is less than minimum value ({}) for range".format(
                        default, lower_bound
                    )
                )
            if upper_bound > 0 and default > upper_bound:
                raise ValueError(
                    "Default value {} is greater than maximum value ({}) for range".format(
                        default, upper_bound
                    )
                )
            self.default = default

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'RangeDim(lower_bound={}, upper_bound={}, default={}, symbol="{}")'.format(
            self.lower_bound, self.upper_bound, self.default, self.symbol)


class Shape(object):
    def __init__(self, shape, default=None):
        """
        The basic shape class to be set in InputType.

        Parameters
        ----------
        shape: list of (int), symbolic values, RangeDim object
            The valid shape of the input.
        
        default: tuple of int or None
            The default shape that is used for initiating the model, and set in
            the metadata of the model file.
            
            If None, then ``shape`` is used.
        """
        from coremltools.converters.mil.mil import get_new_symbol

        if not isinstance(shape, (list, tuple)):
            msg = "Shape should be list or tuple, got type {} instead"
            raise ValueError(msg.format(type(shape)))
        self.symbolic_shape = []
        shape = list(shape)
        for idx, s in enumerate(shape):
            if s is None or s == -1:
                msg = 'Dimension cannot be None or -1. Use ' +\
                        'ct.RangeDim for runtime determined dimension. ' +\
                        'Dim {}: {} ' +\
                        'See https://coremltools.readme.io/docs/flexible-inputs'
                raise ValueError(msg.format(idx, s))
            if isinstance(s, RangeDim):
                sym = s.symbol
                self.symbolic_shape.append(sym)
            elif isinstance(s, (np.generic, int)) or is_symbolic(s):
                self.symbolic_shape.append(s)
            else:
                raise ValueError(
                    "Unknown type {} to build symbolic shape.".format(type(s))
                )

        self.shape = tuple(shape)
        if default is not None:
            if not isinstance(default, (list, tuple)):
                raise ValueError(
                    "Default shape should be list or tuple, got type {} instead".format(
                        type(default)
                    )
                )
            for idx, s in enumerate(default):
                if not isinstance(
                    s, (np.generic, int)
                ) and not is_symbolic(s):
                    raise ValueError(
                        "Default shape invalid, got error at index {} which is {}".format(
                            idx, s
                        )
                    )
        else:
            default = []
            for idx, s in enumerate(self.shape):
                if isinstance(s, RangeDim):
                    default.append(s.default)
                elif s is None or s == -1:
                    default.append(self.symbolic_shape[idx])
                else:
                    default.append(s)
        self.default = tuple(default)

    @property
    def has_symbolic(self):
        return any(is_symbolic(s) for s in self.symbolic_shape)

    def to_list(self, allow_symbolic=False):
      if not allow_symbolic and self.has_symbolic:
          return None
      return self.symbolic_shape


class EnumeratedShapes(object):
    def __init__(self, shapes, default=None):
        """
        A shape class that is used for setting multiple valid shape in InputType.

        Parameters
        ----------
        shapes: list of Shape objects, or Shape-compatible lists.
            The valid shapes of the inputs.
            
            If input provided is not Shape object, but can be converted to Shape,
            the Shape object would be stored in ``shapes`` instead.
        
        default: tuple of int or None
            The default shape that is used for initiating the model, and set in
            the metadata of the model file.
            
            If None, then the first element in ``shapes`` is used.
        """
        from coremltools.converters.mil.mil import get_new_symbol

        if not isinstance(shapes, (list, tuple)):
            raise ValueError(
                "EnumeratedShapes should be list or tuple of shape, got type {} instead".format(
                    type(shapes)
                )
            )
        if len(shapes) < 2:
            raise ValueError(
                "EnumeratedShapes should be take a list or tuple with len >= 2, got {} instead".format(
                    len(shapes)
                )
            )

        self.shapes = []
        for idx, s in enumerate(shapes):
            if isinstance(s, Shape):
                self.shapes.append(s)
            else:
                self.shapes.append(Shape(s))

        self.symbolic_shape = self.shapes[0].symbolic_shape
        for shape in self.shapes:
            for idx, s in enumerate(shape.symbolic_shape):
                if is_symbolic(self.symbolic_shape[idx]):
                    continue
                elif is_symbolic(s):
                    self.symbolic_shape[idx] = s
                elif s != self.symbolic_shape[idx]:
                    self.symbolic_shape[idx] = get_new_symbol()

        if default is not None:
            if not isinstance(default, (list, tuple)):
                raise ValueError(
                    "Default shape should be list or tuple, got type {} instead".format(
                        type(default)
                    )
                )
            for idx, s in enumerate(default):
                if not isinstance(
                    s, (np.generic, int)
                ) and not is_symbolic(s):
                    raise ValueError(
                        "Default shape invalid, got error at index {} which is {}".format(
                            idx, s
                        )
                    )
        else:
            default = self.shapes[0].default
        self.default = default


def _get_shaping_class(shape):
    """
        Returns a Shape class or EnumeratedShapes class for `shape`
        where `shape` could be lists/tuple/Shape/EnumeratedShapes/etc.
    """
    if isinstance(shape, (Shape, EnumeratedShapes)):
        return shape

    try:
        enum_shape = EnumeratedShapes(shape)
        return enum_shape
    except ValueError:
        pass
    try:
        shape = Shape(shape)
        return shape
    except ValueError:
        pass
    raise ValueError("Can't convert to CoreML shaping class from {}.".format(shape))
