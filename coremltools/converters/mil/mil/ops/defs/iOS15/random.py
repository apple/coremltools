#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil import (get_new_symbol,
                                            get_new_variadic_symbol, types)
from coremltools.converters.mil.mil.input_type import (DefaultInputs,
                                                       InputSpec,
                                                       TensorInputType)
from coremltools.converters.mil.mil.operation import Operation
from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op
from coremltools.converters.mil.mil.types.symbolic import any_symbolic


class RandomDistribution(Operation):
    """
    Random Op Superclass
    """
    input_spec = InputSpec(
        shape=TensorInputType(type_domain=types.int32),
    )
    out_dtype = types.fp32

    def type_inference(self):
        if any_symbolic(self.shape.shape):
            # We can't infer any shape if shape has variable length.
            return types.tensor(self.out_dtype, (get_new_variadic_symbol(),))

        # shape has fixed length here.
        if self.shape.sym_val is None:
            shape = tuple([get_new_symbol() for _ in range(self.shape.shape[0])])
            return types.tensor(self.out_dtype, shape)

        return types.tensor(self.out_dtype, tuple(self.shape.sym_val.tolist()))


"""
Random Op Implementation(s)
"""


@register_op
class random_bernoulli(RandomDistribution):
    r"""
    Returns a tensor with the specified shape, with random values from a Bernoulli
    distribution.

    .. math::
       f(k) = \begin{cases}1-p  &\text{if } k = 0\\
                        p    &\text{if } k = 1\end{cases}

    for :math:`k` in :math:`\{0, 1\}`.

    Parameters
    ----------
    shape: <K, i32> (Required)
        * Target output tensor shape.
        * ``K`` is the rank of the output tensor.
          ``shape[k] > 0`` for ``k = 0,..., K-1``.
    prob: const<T> (Optional)
        * The probability of sampling ``1``. Defaults to ``0.5``.
    seed: const<i32> (Optional)
        * Seed to create a reproducible sequence of values across multiple invokes.

    Returns
    -------
    <\\*, T>
        * A tensor of the given target output shape filled with random values.

    Attributes
    ----------
    T: fp16, fp32

    See Also
    --------
    random_categorical, random_normal, random_uniform
    """

    input_spec = (
        InputSpec(
            shape=TensorInputType(type_domain=types.int32),
            prob=TensorInputType(const=True, optional=True, type_domain="T"),
            seed=TensorInputType(const=True, optional=True, type_domain=types.int32),
        )
        + RandomDistribution.input_spec
    )

    type_domains = {
        "T": (types.fp16, types.fp32),
    }

    def default_inputs(self):
        return super().default_inputs() + \
            DefaultInputs(
                seed=-1,
                prob=0.5,
            )

    def type_inference(self):
        self.out_dtype = self.prob.dtype
        return super().type_inference()


@register_op
class random_categorical(Operation):
    """
    Returns random values from a categorical distribution.

    Parameters
    ----------
    x: <\\*D_in, T>
        * N-dimensional tensor which represents ``logits`` (event log-probabilities) or ``probs``
          (event probabilities) depending on ``mode``. The first ``N - 1`` dimensions specifies
          distributions, and the last dimension represents a vector of probabilities.

    mode: const<str> (Optional)
        One of ``['logits', 'probs']``. Defaults to ``logits``.
        When set to ``probs``, an element-wise log layer will be added to calculate logits.

    size: const<i32> (Optional)
        Number of samples to draw. Defaults to ``1``.
        When set as ``1``, it's categorical distribution.
        When set larger than ``1``, it's actually multinomial distribution by drawing with
        replacement. It means that when a sample index is drawn, it can be drawn again.
        The categorical distribution is a special case of the multinomial distribution, giving
        the probabilities of potential outcomes of a single drawing rather than multiple drawings.

    seed: const<i32> (Optional)
        Seed to create a reproducible sequence of values across multiple invokes.

    Returns
    -------
    <\\*D_in[:-1] + [size], T>
        * A tensor of the given target output shape filled with random values.

    Attributes
    ----------
    T: fp16, fp32

    See Also
    --------
    random_bernoulli, random_normal, random_uniform
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        mode=TensorInputType(const=True, optional=True, type_domain=types.str),
        size=TensorInputType(const=True, optional=True, type_domain=types.int32),
        seed=TensorInputType(const=True, optional=True, type_domain=types.int32),
    )

    type_domains = {
        "T": (types.fp16, types.fp32),
    }

    def default_inputs(self):
        return DefaultInputs(
            mode="logits",
            size=1,
            seed=-1,
        )

    def type_inference(self):
        self.out_dtype = self.x.dtype
        output_shape = self.x.shape[:-1] + (self.size.val,)
        return types.tensor(self.out_dtype, output_shape)


@register_op
class random_normal(RandomDistribution):
    r"""
    Returns a tensor with the specified shape, with random values from a normal
    distribution.

    Parameters
    ----------
    shape: <K, i32> (Required)
        * Target output tensor shape.
        * ``K`` is the rank of the output tensor.
          ``shape[k] > 0`` for ``k = 0,..., K-1``.
    mean: const<T> (Optional)
        The mean (center) of the normal distribution. Defaults to 0.0.
    stddev: const<T> (Optional)
        The standard deviation (width) of the normal distribution. Defaults to ``1.0``.
    seed: const<i32> (Optional)
        Seed to create a reproducible sequence of values across multiple invokes.

    Returns
    -------
    <\\*, T>
        * A tensor of the given target output shape filled with random values.

    Attributes
    ----------
    T: fp16, fp32

    See Also
    --------
    random_categorical, random_bernoulli, random_uniform
    """

    input_spec = (
        InputSpec(
            shape=TensorInputType(type_domain=types.int32),
            mean=TensorInputType(const=True, optional=True, type_domain="T"),
            stddev=TensorInputType(const=True, optional=True, type_domain="T"),
            seed=TensorInputType(const=True, optional=True, type_domain=types.int32),
        )
        + RandomDistribution.input_spec
    )

    type_domains = {
        "T": (types.fp16, types.fp32),
    }

    def default_inputs(self):
        return super().default_inputs() + \
            DefaultInputs(
                mean=0.,
                stddev=1.,
                seed=-1,
            )

    def type_inference(self):
        if self.mean.dtype != self.stddev.dtype:
            raise ValueError("Incompatible primitive types in random_normal operation")
        self.out_dtype = self.mean.dtype
        return super().type_inference()


@register_op
class random_uniform(RandomDistribution):
    r"""
    Returns a tensor with the specified shape with random values from a uniform
    distribution. Samples are uniformly distributed over the half-open interval
    ``[low, high)`` (includes low, but excludes high).

    .. math::
       p(x) = \frac{1}{high - low}

    For a real number :math:`x`.

    When ``high == low``, values of ``low`` will be returned. If ``high < low``,
    the results are officially undefined and may eventually raise an error.

    Parameters
    ----------
    shape: <K, i32> (Required)
        * Target output tensor shape.
        * ``K`` is the rank of the output tensor.
          ``shape[k] > 0`` for ``k = 0,..., K-1``.
    low: const<T> (Optional)
        * Lower boundary of the output interval (inclusive). Defaults to ``0.0``.
    high: const<T> (Optional)
        * Upper boundary of the output interval (exclusive). Defaults to ``1.0``.
    seed: const<i32> (Optional)
        * Seed to create a reproducible sequence of values across multiple invokes.

    Returns
    -------
    <\\*, T>
        * A tensor of the given target output shape filled with random values.

    Attributes
    ----------
    T: fp16, fp32

    See Also
    --------
    random_categorical, random_bernoulli, random_normal
    """

    input_spec = (
        InputSpec(
            shape=TensorInputType(type_domain=types.int32),
            low=TensorInputType(const=True, optional=True, type_domain="T"),
            high=TensorInputType(const=True, optional=True, type_domain="T"),
            seed=TensorInputType(const=True, optional=True, type_domain=types.int32),
        )
        + RandomDistribution.input_spec
    )

    type_domains = {
        "T": (types.fp16, types.fp32),
    }

    def default_inputs(self):
        return super().default_inputs() + \
            DefaultInputs(
                low=0.,
                high=1.,
                seed=-1,
            )

    def type_inference(self):
        if self.low.dtype != self.high.dtype:
            raise ValueError("Incompatible primitive types in random_uniform operation")
        self.out_dtype = self.low.dtype
        return super().type_inference()
