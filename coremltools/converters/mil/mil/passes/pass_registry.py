#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import inspect
from typing import Dict, Optional, Text, Type

from coremltools import _logger as logger
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass


class PassRegistry:
    def __init__(self):
        """
        Store the pass class instead of instance to avoid the same instance got modified by several
        callers.
        """
        self.passes: Dict[Text, Type[AbstractGraphPass]] = {}

    def __getitem__(self, pass_id: Text) -> AbstractGraphPass:
        """
        pass_id: namespace::func_name (e.g., 'common::const_elimination')
        """
        if pass_id not in self.passes:
            raise KeyError(f"Pass {pass_id} not found")
        current_pass = self.passes[pass_id]
        # The current_pass could be a PassContainer instance if registered by register_generic_pass.
        return current_pass() if inspect.isclass(current_pass) else current_pass

    def __contains__(self, pass_id: Text) -> bool:
        return pass_id in self.passes

    def add(
        self,
        namespace: Text,
        pass_cls: Type[AbstractGraphPass],
        override: bool,
        name: Optional[Text],
    ):
        cls_name = pass_cls.__name__ if name is None else name
        pass_id = namespace + "::" + cls_name
        logger.debug(f"Registering pass {pass_id}")
        if pass_id in self.passes and not override:
            raise KeyError(f"Pass {pass_id} already registered.")
        self.passes[pass_id] = pass_cls


PASS_REGISTRY = PassRegistry()


def register_pass(namespace: Text, override: bool = False, name: Optional[Text] = None):
    """
    namespaces like {'common', 'nn_backend', <other-backends>, <other-frontends>}

    Params:
        override: indicate the graph pass can override an existing pass with the same name.
        name: name of the graph pass. Default to class name if not provided
    """

    def class_wrapper(pass_cls: Type[AbstractGraphPass]):
        PASS_REGISTRY.add(namespace, pass_cls, override, name)
        return pass_cls

    return class_wrapper
