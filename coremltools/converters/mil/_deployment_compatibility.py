# Copyright (c) 2021, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from enum import IntEnum

from coremltools import (
    _SPECIFICATION_VERSION_IOS_13,
    _SPECIFICATION_VERSION_IOS_14,
    _SPECIFICATION_VERSION_IOS_15,
    _SPECIFICATION_VERSION_IOS_16,
    _SPECIFICATION_VERSION_IOS_17,
)


class AvailableTarget(IntEnum):
    # iOS versions
    iOS13 = _SPECIFICATION_VERSION_IOS_13
    iOS14 = _SPECIFICATION_VERSION_IOS_14
    iOS15 = _SPECIFICATION_VERSION_IOS_15
    iOS16 = _SPECIFICATION_VERSION_IOS_16
    iOS17 = _SPECIFICATION_VERSION_IOS_17

    # macOS versions (aliases of iOS versions)
    macOS10_15 = _SPECIFICATION_VERSION_IOS_13
    macOS10_16 = _SPECIFICATION_VERSION_IOS_14
    macOS11 = _SPECIFICATION_VERSION_IOS_14
    macOS12 = _SPECIFICATION_VERSION_IOS_15
    macOS13 = _SPECIFICATION_VERSION_IOS_16
    macOS14 = _SPECIFICATION_VERSION_IOS_17

    # watchOS versions (aliases of iOS versions)
    watchOS6 = _SPECIFICATION_VERSION_IOS_13
    watchOS7 = _SPECIFICATION_VERSION_IOS_14
    watchOS8 = _SPECIFICATION_VERSION_IOS_15
    watchOS9 = _SPECIFICATION_VERSION_IOS_16
    watchOS10 = _SPECIFICATION_VERSION_IOS_17

    # tvOS versions (aliases of iOS versions)
    tvOS13 = _SPECIFICATION_VERSION_IOS_13
    tvOS14 = _SPECIFICATION_VERSION_IOS_14
    tvOS15 = _SPECIFICATION_VERSION_IOS_15
    tvOS16 = _SPECIFICATION_VERSION_IOS_16
    tvOS17 = _SPECIFICATION_VERSION_IOS_17

    # customized __str__
    def __str__(self):
        original_str = super().__str__()
        new_str = original_str.replace(type(self).__name__, "coremltools.target")
        return new_str


_get_features_associated_with = {}


def register_with(name):
    def decorator(func):
        if name not in _get_features_associated_with:
            _get_features_associated_with[name] = func
        else:
            raise ValueError("Function is already registered with {}".format(name))
        return func

    return decorator


@register_with(AvailableTarget.iOS14)
def iOS14Features(spec):
    features_list = []

    if spec.WhichOneof("Type") == "neuralNetwork":
        nn_spec = spec.neuralNetwork
    elif spec.WhichOneof("Type") in "neuralNetworkClassifier":
        nn_spec = spec.neuralNetworkClassifier
    elif spec.WhichOneof("Type") in "neuralNetworkRegressor":
        nn_spec = spec.neuralNetworkRegressor
    else:
        raise ValueError("Invalid neural network specification for the model")

    # Non-zero default optional values
    for idx, input in enumerate(spec.description.input):
        value = 0
        if input.type.isOptional:
            value = max(value, input.type.multiArrayType.floatDefaultValue)
            value = max(value, input.type.multiArrayType.doubleDefaultValue)
            value = max(value, input.type.multiArrayType.intDefaultValue)

        if value != 0:
            msg = "Support of non-zero default optional values for inputs."
            features_list.append(msg)
            break

    # Layers or modifications introduced in iOS14
    new_layers = [
        "oneHot",
        "cumSum",
        "clampedReLU",
        "argSort",
        "pooling3d",
        "convolution3d",
        "globalPooling3d",
    ]
    for layer in nn_spec.layers:
        layer_type = layer.WhichOneof("layer")

        msg = ""

        if layer_type in new_layers:
            msg = "{} {}".format(layer_type.capitalize(), "operation")

        if layer_type == "tile" and len(layer.input) == 2:
            msg = "Dynamic Tile operation"

        if layer_type == "upsample" and layer.upsample.linearUpsampleMode in [1, 2]:
            msg = "Upsample operation with Align Corners mode"

        if layer_type == "reorganizeData" and layer.reorganizeData.mode == 2:
            msg = "Pixel Shuffle operation"

        if layer_type == "sliceDynamic" and layer.sliceDynamic.squeezeMasks:
            msg = "Squeeze mask for dynamic slice operation"

        if layer_type == "sliceStatic" and layer.sliceDynamic.squeezeMasks:
            msg = "Squeeze mask for static slice operation"

        if layer_type == "concatND" and layer.concatND.interleave:
            msg = "Concat layer with interleave operation"

        if msg != "" and (msg not in features_list):
            features_list.append(msg)

    return features_list


def check_deployment_compatibility(spec, representation, deployment_target):

    if not isinstance(deployment_target, AvailableTarget):
        raise TypeError(
            "Argument for deployment_target must be an enumeration from Enum class AvailableTarget"
        )

    for any_target in AvailableTarget:

        if any_target > deployment_target and any_target in _get_features_associated_with:
            missing_features = _get_features_associated_with[any_target](spec)

            if missing_features:
                msg = (
                    "Provided minimum deployment target requires model to be of version {} but converted model "
                    "uses following features which are available from version {} onwards. Please use a higher "
                    "minimum deployment target to convert. \n ".format(
                        deployment_target.value, any_target.value
                    )
                )

                for i, feature in enumerate(missing_features):
                    msg += "   {}. {}\n".format(i + 1, feature)

                raise ValueError(msg)

    # Default exception throwing if not able to find the reason behind spec version bump
    if spec.specificationVersion > deployment_target.value:
        msg = (
            "Provided deployment target requires model to be of version {} but converted model has version {} "
            "suitable for later releases".format(
                deployment_target.value, spec.specificationVersion,
            )
        )
        raise ValueError(msg)
