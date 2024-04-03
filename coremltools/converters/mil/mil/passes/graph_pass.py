#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Text, Union

from coremltools.converters.mil import Operation, Program
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.scope import ScopeInfo, ScopeSource


class PassOption:
    """
    Option that will be applied in a graph pass.

    Each graph pass need to have their own implementation to support the corresponding option.
    Available options are documented in each pass's docstring.
    """

    # The Callable option_val is for op_selector backward compatibility only.
    def __init__(self, option_name: Text, option_val: Union[Text, Callable[[Operation], bool]]):
        if not isinstance(option_name, Text):
            raise ValueError(f"The option name should be text, but got {type(option_name)}")
        self._option_name = option_name
        self._option_val = option_val

    def __str__(self):
        return f"{self.option_name}: {self.option_val}"

    @property
    def option_name(self):
        return self._option_name

    @property
    def option_val(self):
        return self._option_val


class AbstractGraphPass(ABC):
    """
    Base class for a graph pass.

    Each graph pass should be a subclass of this and implement the `apply` method.
    Each graph pass can also implement their own supported options.
    See examples of `skip_ops_by_type` in `add_fp16_cast` and `skip_const_by_size` in
    `const_elimination` about how to support new options in each pass.
    """

    def __call__(self, prog: Program):
        if not prog.skip_all_passes:
            # we use the scope context manager to populate the graph pass information to the ops
            # constructed by the pass.
            with mb.scope(ScopeInfo(source=ScopeSource.COREMLTOOLS_GRAPH_PASS, data=[str(self)])):
                self.apply(prog)

    def __str__(self):
        return type(self).__name__

    @abstractmethod
    def apply(self, prog: Program):
        pass

    def set_options(self, pass_options: Optional[List[PassOption]] = None):
        """Set pass options."""
        if pass_options is not None:
            for pass_option in pass_options:
                option_name = pass_option.option_name
                if not hasattr(self, option_name):
                    raise NotImplementedError(
                        f"The graph pass `{self}` doesn't support option `{option_name}`."
                    )
                setattr(self, option_name, pass_option.option_val)
