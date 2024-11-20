#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import copy
from collections import defaultdict
from enum import Enum
from typing import Dict, List, Union

from attrs import define, field, validators


class ScopeSource(Enum):
    """
    Pre-defined scope source enum:

    # Torch script related:
    TORCHSCRIPT_MODULE_TYPE:
        * Torchscript module type of a scope, which usually corresponds to the submodule object class type.
        * If provided as str, it denotes a single scope, and cannot be an empty str.
        * Nested scopes are represented by a list of str.

    TORCHSCRIPT_MODULE_NAME:
        * Unique torchscript identifier for a scope, which usually corresponds to the submodule object name.
        * If provided as str, it denotes a single scope.
        * Nested scopes are represented by a list of str.

    # Core ML converter graph passes related:
    COREMLTOOLS_GRAPH_PASS:
        * This scope traces the graph transformations (graph passes) applied on the program.
        * For instance, operations constructed under the "fuse_conv_batchnorm" pass, is going to have
          the scopes attribute of ``{COREMLTOOLS_GRAPH_PASS: ["fuse_conv_batchnorm"]}``.
        * If the op went through multiple graph pass transformations, it is represetned by a list of str.
          For instance: ["fuse_conv_batchnorm", "add_fp16_cast"] means the op is created by "fuse_conv_batchnorm"
          and then undergoes "add_fp16_cast".

    # Torch export related:
    EXIR_STACK_TRACE:
        * The ``stack_trace`` metadata inherited from torch.fx.Node.meta in EXIR
        * This metadata traces the MIL op back to original python source code

    EXIR_DEBUG_HANDLE:
        * The ``debug_handle`` metadata inherited from torch.fx.Node.meta in EXIR
        * This metadata enables post-run analysis in ExecuTorch integration
        * ExecuTorch uses integer as debug handle. When a MIL op can be traced back to ExecuTorch
          (e.g. translated from torch op), we inherit the integer value
        * If a MIL op cannot be traced back to ExecuTorch (e.g. created by graph pass),
          then we use None to denote "no debug handle"


    Examples
    --------
    Here is an example of torchscript related scope enum:

    .. sourcecode:: python

        class SubModule(torch.nn.Module):
            pass


        class MainModule(torch.nn.Module):
            def __init__(self):
                self.submodule_1 = SubModule()

            def forward(self, x):
                node = self.submodule_1(x)
                return node


        my_model = MainModule()

    when the above model is translated into pymil, the Operation corresponding to ``node`` would have:

        * TORCHSCRIPT_MODULE_TYPE: ["SubModule", ...]
        * TORCHSCRIPT_MODULE_NAME: ["submodule_1", ...]

    in their scope attributes.
    """

    TORCHSCRIPT_MODULE_TYPE = 0
    TORCHSCRIPT_MODULE_NAME = 1
    COREMLTOOLS_GRAPH_PASS = 2
    EXIR_STACK_TRACE = 3  # no serialization for such debug info should be allowed yet
    EXIR_DEBUG_HANDLE = 4


class ScopeStack(defaultdict):
    """
    A utility class to handle the scope context manager
    """

    def __init__(self):
        super().__init__(list)

    def get_curr_scopes(self) -> Dict[ScopeSource, List[str]]:
        """
        Returns the current scope information as a dictionary.
        """
        res = defaultdict(list)
        for key, val in self.items():
            if len(val) == 0:
                continue
            scope_for_one_source = []
            for v in val:
                scope_for_one_source.extend(v.data)
            res[key] = scope_for_one_source
        return res


SCOPE_STACK = ScopeStack()
VALID_OPS_TO_COPY_SCOPE_INFO = []


def add_graph_pass_scope(
    src_scopes: Dict[ScopeSource, List[str]], graph_pass_scopes: Dict[ScopeSource, List[str]]
) -> Dict[ScopeSource, List[str]]:
    res = {}
    """
    Construct a scope by adding graph pass scopes from ``graph_pass_scopes`` to ``src_scopes``.

    The rules are the following:

    (1) We append the COREMLTOOLS_GRAPH_PASS ScopeSource in ``graph_pass_scopes`` to the ``src_scopes``.
        This will allow us to keep tracking the history of transformation.
        For instance:

        Input:

            src_scopes = {
                ScopeSource.COREMLTOOLS_GRAPH_PASS: ["pass_1"],
            }
            graph_pass_scopes = {
                ScopeSource.COREMLTOOLS_GRAPH_PASS: ["pass_2", "pass_3"],
            }

        Output:

            res = {
                ScopeSource.COREMLTOOLS_GRAPH_PASS: ["pass_1", "pass_2", "pass_3"],
            }

    (2) Only COREMLTOOLS_GRAPH_PASS ScopeSource is allowed in ``graph_pass_scopes``.

    (3) Other ScopeSource will be passed down from ``src_scopes``.

        Input:

            src_scopes = {
                ScopeSource.TORCHSCRIPT_MODULE_TYPE: ["module_1"],
                ScopeSource.TORCHSCRIPT_MODULE_NAME: ["a1"],
                ScScopeSource.COREMLTOOLS_GRAPH_PASS: ["pass_1"],
            }
            graph_pass_scopes = {
                ScScopeSource.COREMLTOOLS_GRAPH_PASS: ["pass_2", "pass_3"],
            }

        Output:

            res = {
                ScopeSource.TORCHSCRIPT_MODULE_TYPE: ["module_1"],
                ScopeSource.TORCHSCRIPT_MODULE_NAME: ["a1"],
                ScScopeSource.COREMLTOOLS_GRAPH_PASS: ["pass_1", "pass_2", "pass_3"],
            }
    """
    res = defaultdict(list)
    for scope_source_key in ScopeSource:
        if scope_source_key in graph_pass_scopes:
            assert (
                scope_source_key == ScopeSource.COREMLTOOLS_GRAPH_PASS
            ), "Only ScopeSource.COREMLTOOLS_GRAPH_PASS is allowed in the graph_pass_scopes."
            if ScopeSource.COREMLTOOLS_GRAPH_PASS in src_scopes:
                old_graph_pass_data = copy.copy(src_scopes[ScopeSource.COREMLTOOLS_GRAPH_PASS])
            else:
                old_graph_pass_data = []
            new_graph_pass_data = copy.copy(graph_pass_scopes[ScopeSource.COREMLTOOLS_GRAPH_PASS])
            res[ScopeSource.COREMLTOOLS_GRAPH_PASS] = old_graph_pass_data + new_graph_pass_data
        elif scope_source_key in src_scopes:
            res[scope_source_key] = copy.copy(src_scopes[scope_source_key])

    return res


@define
class ScopeInfo:
    """
    Parameters
    ----------
    source: str
        * Source of the scope. For instance, it could be a frontend framework like torchsccript, or a converter graph pass, etc.
        * Must be type of ScopeSource Enum.

    data: Union[str, List[str]]
        * Scope data.
        * It could be type of str or List[str].

    Examples
    --------
    Here are examples of creating a ScopeInfo:

    .. sourcecode:: python
        # A scope for a single torchscript module type
        scope_info = ScopeInfo(
            source=ScopeSource.TORCHSCRIPT_MODULE_TYPE,
            data="Module_1",
        )

        # A scope for a two layers torchscript model hierarchy type
        scope_info = ScopeInfo(
            source=ScopeSource.TORCHSCRIPT_MODULE_TYPE,
            data=["Module_1", "Module_2"],
        )
    """

    source: str = field(validator=validators.instance_of(ScopeSource))
    data: Union[str, List[str]] = field(validator=validators.instance_of((str, list)))

    def __attrs_post_init__(self):
        # cleanup scope info
        if self.source in (
            ScopeSource.TORCHSCRIPT_MODULE_NAME,
            ScopeSource.TORCHSCRIPT_MODULE_TYPE,
            ScopeSource.COREMLTOOLS_GRAPH_PASS,
        ):
            if not isinstance(self.data, list):
                self.data = [self.data]
            for i, val in enumerate(self.data):
                if not isinstance(val, str):
                    raise ValueError(
                        f"Scope must be type of List[str]. Got element {val} with type {type(val)}."
                    )
                self.data[i] = val.replace(" ", "")
        elif self.source == ScopeSource.EXIR_DEBUG_HANDLE:
            if not isinstance(self.data, list):
                self.data = [self.data]
            for val in self.data:
                if val is not None and not isinstance(val, int):
                    raise ValueError(
                        f"Scope must be None or type of List[int]. Got element {val} with type {type(val)}."
                    )

        if self.source == ScopeSource.COREMLTOOLS_GRAPH_PASS:
            if len(self.data) > 1:
                raise ValueError(
                    f"COREMLTOOLS_GRAPH_PASS scope cannot have len > 1. Got {self.data}."
                )

        if self.source == ScopeSource.TORCHSCRIPT_MODULE_TYPE:
            if "" in self.data:
                raise ValueError(
                    "TORCHSCRIPT_MODULE_TYPE scope info cannot contains empty string."
                )

        if self.source == ScopeSource.EXIR_DEBUG_HANDLE:
            if len(self.data) > 1:
                raise ValueError(f"EXIR_DEBUG_HANDLE scope cannot have len > 1. Got {self.data}.")


class ScopeContextManager:
    def __init__(
        self,
        *scopes: List[ScopeInfo],
    ):
        """
        A context manager pushes/pops the scope information, which makes the
        operations created within it have the corresponding scope information.

        Parameters
        ----------
        scopes: Optional[List[ScopeInfo]] (Optional)
            * A list of ScopeInfo under the context manager.
            * The source in each ScopeInfo cannot be duplicated.
            * If not provided, this context manager does no affects.

        Examples
        --------
        Here is an example of creating a scope for torchscript module heirarchy with type and name information.

        .. sourcecode:: python

            @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
            def prog(x):
                with mb.scope(
                    ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_TYPE, data=["Module1"]),
                    ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_NAME, data=["module_1"]),
                ):
                    return mb.add(x=x, y=4.3, name="add_1")


        In the above example, the "add_1" op will have two scope attributes, for torchscipt module type and name:
            * TORCHSCRIPT_MODULE_TYPE: ["Module1"]
            * TORCHSCRIPT_MODULE_NAME: ["module_1"]

        Here is an example of creating nested scopes:

        .. sourcecode:: python

            @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
            def prog(x):
                with mb.scope(
                    ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_TYPE, data=["Module1"]),
                ):
                    x = mb.add(x=x, y=4.3, name="add_1")
                    with mb.scope(
                        ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_TYPE, data=["Module2"]),
                        ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_NAME, data=["module_2"]),
                    ):
                        return mb.add(x=x, y=3.2, name="add_2")

        In the above example, the "add_1" op would have a scope attribute:
            * TORCHSCRIPT_MODULE_TYPE: ["Module1"]

        while the "add_2" op would have scope attributes:
            * TORCHSCRIPT_MODULE_TYPE: ["Module1", "Module2"]
            * TORCHSCRIPT_MODULE_NAME: ["module_2"]
        """
        self.scopes = scopes
        # Validate scopes are type of ScopeInfo
        for scope in self.scopes:
            if not isinstance(scope, ScopeInfo):
                raise ValueError(
                    f"mb.scope only accepts inputs of type ScopeInfo. Got {type(scope)}."
                )

        # validate there is no duplicated scope source
        visited_scope_sources = set()
        for scope in self.scopes:
            if scope.source in visited_scope_sources:
                raise ValueError(f"Scope source {scope.source} duplicated.")
            visited_scope_sources.add(scope.source)

    def __enter__(self):
        for scope in self.scopes:
            SCOPE_STACK[scope.source].append(scope)
            if scope.source == ScopeSource.COREMLTOOLS_GRAPH_PASS:
                VALID_OPS_TO_COPY_SCOPE_INFO.append(set())

    def __exit__(self, type, value, traceback):
        for scope in self.scopes:
            SCOPE_STACK[scope.source].pop()
            if scope.source == ScopeSource.COREMLTOOLS_GRAPH_PASS:
                VALID_OPS_TO_COPY_SCOPE_INFO.pop()
