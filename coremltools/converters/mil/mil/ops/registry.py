#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from collections import defaultdict

from coremltools import _logger as logger
from coremltools.converters.mil._deployment_compatibility import \
    AvailableTarget as target
from coremltools.converters.mil.mil.block import curr_opset_version

from ..builder import Builder
from .helper import _get_version_of_op


class SSAOpRegistry:

    """
    There are three kinds of operations that we could register:

    (1) core_ops:  dict[str, dict[Operation]]
        - These are the core ops in PyMIL, which have a direct mapping to the backend in neural_network or mlprogram
        - The registered op is considered a core op if the namespace is not provided
        - coreml_ops[op_type] is a dict that tracks different opset versions for an op. For instance
            - ``core_ops[op_1] = {
                    ct.target.iOS13: op_1_iOS13,
                    ct.target.iOS14: op_1_iOS13,
                    ct.target.iOS15: op_1_iOS13,
                    ct.target.iOS16: op_1_iOS13,
               }``
                . Only one version of op type ``op_1`` is registered, and it is defined in iOS13, which both
                  neural_network and mlprogram backend support
            - ``core_ops[op_2] = {
                    ct.target.iOS13: op_2_iOS13,
                    ct.target.iOS14: op_2_iOS13,
                    ct.target.iOS15: op_2_iOS13,
                    ct.target.iOS16: op_2_iOS16,
               }``
                . Two versions of op type ``op_2`` are registered, one each for iOS13, iOS16.
                . The builder picks up correct version of the op according to curr_opset_version(), which returns the opset version of
                  the current function.
                    -- If ``curr_opset_version()`` is ``None`` (the version of the function is not set), ``mb.op_2`` would call the oldest version of the op by default, which is ``op_2_ios13``
                    -- Otherwise, the builder would pick up core_ops[op_2][curr_opset_version()]
        - In the highest level, users can choose the desired version by specifying the ``minum_deployment_target`` argument in ``coremltools.convert``
        - The default ``opset_version`` for the core ops would be set to iOS13, for which neural_network backend supports

    (2) dialect_ops: dict[str, Operation]
        - These are the ops that are created for specific frontend framework, for instance: ``tf_lstm_block, torch_upsample_nearest_neighbor``
        - A graph pass must be customized by the developer to translate a dialect_ops into core ops

    (3) custom_ops: dict[str, Operation]
        - These are the custom ops, in which an additional ``bindings`` which should be specified in operator
    """
    SUPPORTED_OPSET_VERSIONS = (
        target.iOS13,
        target.iOS14,
        target.iOS15,
        target.iOS16,
        target.iOS17,
        target.iOS18,
    )
    core_ops = defaultdict(dict)
    dialect_ops = {}
    custom_ops = {}

    @staticmethod
    def _get_core_op_cls(op_type=None):
        """
        A utility function that retrieves an op cls using the curr_opset_version
        """
        if op_type not in SSAOpRegistry.core_ops:
            raise ValueError("op {} not registered.".format(op_type))
        candidate_ops = SSAOpRegistry.core_ops[op_type]
        return _get_version_of_op(candidate_ops, curr_opset_version())

    @staticmethod
    def register_op(_cls=None, is_custom_op=False, namespace=None, opset_version=target.iOS13, allow_override=False):
        """
        Registration routine for MIL Program operators

        Parameters
        ----------
        is_custom_op: boolean
            - If ``True``, maps current operator to ``custom_op``. ``custom_op`` requires additional ``bindings`` which should be specified in operator
            - Default ``False``

        namespace: str
            - If provided, the op is registered as a dialect op
            - Otherwise is considered as a core op

        opset_version: int
            - Specify the minimum spec version that supports this op
            - Default to ``ct.target.iOS13``, which is for the neural_network backend

        allow_override: boolean
            - If True, it is allowed for an operation to override the previous operation with the same registered name
            - Default ``False``
        """
        def class_wrapper(op_cls):
            op_type = op_cls.__name__

            # debug message
            op_msg = "op"
            is_dialect_op = (namespace is not None)
            if is_custom_op:
                op_msg = "Custom op"
            elif is_dialect_op:
                op_msg = "Dialect op"
            logger.debug("Registering {} {}".format(op_msg, op_type))

            # pick the right dict for registration
            if is_custom_op:
                op_reg = SSAOpRegistry.custom_ops
            elif is_dialect_op:
                op_reg = SSAOpRegistry.dialect_ops
                # Check that op_type is prefixed with namespace
                if op_type[: len(namespace)] != namespace:
                    msg = (
                        "Dialect op type {} registered under {} namespace must " + "prefix with {}"
                    )
                    raise ValueError(msg.format(op_type, namespace, namespace))
                op_cls._dialect_namespace = namespace
            else:
                op_reg = SSAOpRegistry.core_ops

            # verify that the op have not been registered before if allow_override = False
            msg = "SSA {} {} already registered.".format(op_msg, op_type)
            if is_custom_op or is_dialect_op:
                if op_type in op_reg and not allow_override:
                    raise ValueError(msg)
            else:
                if opset_version in op_reg[op_type] and not allow_override:
                    if opset_version - 1 not in op_reg[op_type] or (op_reg[op_type][opset_version - 1] != op_reg[op_type][opset_version]):
                        raise ValueError(msg)

            # add the op to op_reg
            if is_custom_op or is_dialect_op:
                op_reg[op_type] = op_cls
            else:
                # The older version of the op must be registered first, or it will override the
                # newer version. For example, assuming an op has two versions: IOS13 and IOS15. If
                # the IOS15 is registered first, the op_reg[op_type] will have that op class for
                # IOS15/16/..., and when IOS13 is registered, it will override all op classes for
                # IOS13/14/15/16/... where IOS15 op class will get lost. So we error out early
                # instead of keep registering when this happens.
                if opset_version in op_reg[op_type]:
                    old_op_cls = op_reg[op_type][opset_version]
                    for i in range(opset_version, SSAOpRegistry.SUPPORTED_OPSET_VERSIONS[-1] + 1):
                        if op_reg[op_type][i] != old_op_cls:
                            raise ValueError(
                                f"Older version of op {op_type} must be registered "
                                f"before a newer version."
                            )
                idx = SSAOpRegistry.SUPPORTED_OPSET_VERSIONS.index(opset_version)
                for i in range(idx, len(SSAOpRegistry.SUPPORTED_OPSET_VERSIONS)):
                    op_reg[op_type][SSAOpRegistry.SUPPORTED_OPSET_VERSIONS[i]] = op_cls

            # add the version information to the op cls
            op_cls._op_variants = op_reg[op_type]

            @classmethod
            def add_op(cls, **kwargs):
                """
                An utility function that help the builder to pickup the correct op class when calling ``mb.op``

                There are two cases:

                (1) custom op / dialect op:
                    If the op is a custom op or a dialect op, we could directly pick up the op class through
                    ``SSAOpRegistry.custom_ops[op_type]`` or  ``SSAOpRegistry.dialect_ops[op_type]``

                (2) core op:
                    For the core op, the builder would pick up the correct version according to ``curr_opset_version()``
                """
                op_cls_to_add = None
                is_core_op = (op_reg == SSAOpRegistry.core_ops)
                if is_core_op:
                    op_cls_to_add = SSAOpRegistry._get_core_op_cls(op_type)
                else:
                    op_cls_to_add = op_reg[op_type]

                return cls._add_op(op_cls_to_add, **kwargs)

            setattr(Builder, op_type, add_op)
            return op_cls

        if _cls is None:
            return class_wrapper

        return class_wrapper(_cls)
