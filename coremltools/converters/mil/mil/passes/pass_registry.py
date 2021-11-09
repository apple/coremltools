#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import logging


class PassRegistry:
    def __init__(self):
        # str -> an AbstractGraphPass instance, which has an 'apply' method that takes
        # program as input and modifies it in place
        self.passes = {}

    def __getitem__(self, pass_id):
        """
        pass_id (str): namespace::func_name (e.g., 'common::const_elimination')
        """
        if pass_id not in self.passes:
            raise KeyError("Pass {} not found".format(pass_id))
        return self.passes[pass_id]

    def __contains__(self, pass_id):
        return pass_id in self.passes

    def add(self, namespace, pass_cls, override, name):
        cls_name = pass_cls.__name__ if name is None else name
        pass_id = namespace + "::" + cls_name
        logging.debug("Registering pass {}".format(pass_id))
        if pass_id in self.passes and not override:
            msg = "Pass {} already registered."
            raise KeyError(msg.format(pass_id))
        self.passes[pass_id] = pass_cls()


PASS_REGISTRY = PassRegistry()


def register_pass(namespace, override=False,  name=None):
    """
    namespaces like {'common', 'nn_backend', <other-backends>,
    <other-frontends>}
    override: indicate the graph pass can override an existing pass with the same name
    name: name of the graph pass. Default to class name if not provided
    """

    def class_wrapper(pass_cls):
        PASS_REGISTRY.add(namespace, pass_cls, override, name)
        return pass_cls

    return class_wrapper
