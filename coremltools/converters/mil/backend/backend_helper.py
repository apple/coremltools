#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.input_types import ColorLayout
from coremltools.converters.mil.mil.passes.name_sanitization_utils import \
    NameSanitizer
from coremltools.proto import FeatureTypes_pb2 as ft


def _get_probability_var_for_classifier(prog, classifier_config):
    '''
    Return the var which will be used to construct the dictionary for the classifier.
    :param prog: mil program
    :param classifier_config: an instance of coremltools.ClassifierConfig class
    :return: var
    '''
    block = prog.functions["main"]
    probability_var = None
    if classifier_config.predicted_probabilities_output is None \
            or classifier_config.predicted_probabilities_output == "":
        # user has not indicated which tensor in the program to use as probabilities
        # (i.e which tensor to link to the classifier output)
        # in this case, attach the last non const op to the classify op
        for op in reversed(block.operations):
            if op.op_type != 'const' and len(op.outputs) == 1:
                probability_var = op.outputs[0]
                break
        if probability_var is None:
            raise ValueError("Unable to determine the tensor in the graph "
                             "that corresponds to the probabilities for the classifier output")
    else:
        # user has indicated which tensor in the program to use as probabilities
        # (i.e which tensor to link to the classifier output)
        # Verify that it corresponds to a var produced in the program
        predicted_probabilities_output = NameSanitizer().sanitize_name(classifier_config.predicted_probabilities_output)
        for op in block.operations:
            for out in op.outputs:
                if out.name == predicted_probabilities_output:
                    probability_var = out
                    break
        if probability_var is None:
            msg = "'predicted_probabilities_output', '{}', provided in 'ClassifierConfig', does not exist in the MIL program."
            raise ValueError(msg.format(predicted_probabilities_output))
    return probability_var


def _get_colorspace_enum(color_layout):
    if color_layout == ColorLayout.GRAYSCALE:
        return ft.ImageFeatureType.ColorSpace.GRAYSCALE
    elif color_layout == ColorLayout.GRAYSCALE_FLOAT16:
        return ft.ImageFeatureType.ColorSpace.GRAYSCALE_FLOAT16
    elif color_layout == ColorLayout.BGR:
        return ft.ImageFeatureType.ColorSpace.BGR
    else:
        return ft.ImageFeatureType.ColorSpace.RGB

def _validate_image_input_output_shapes(color_layout, shape, name, is_input=True):
    io_str = "input" if is_input else "output"
    if len(shape) != 4:
        raise ValueError("Image {}, '{}', must have rank 4. Instead it has rank {}".
                         format(io_str, name, len(shape)))
    if color_layout in (ColorLayout.BGR, ColorLayout.RGB):
        if shape[1] != 3 or shape[0] != 1:
            raise ValueError("Shape of the RGB/BGR image {}, '{}', must be of kind (1, 3, H, W), "
                             "i.e., first two dimensions must be (1, 3), instead they are: {}".
                             format(io_str, name, shape[:2]))
    elif color_layout in (ColorLayout.GRAYSCALE, ColorLayout.GRAYSCALE_FLOAT16):
        if shape[1] != 1 or shape[0] != 1:
            raise ValueError("Shape of the Grayscale image {}, '{}', must be of kind (1, 1, H, W), "
                             "i.e., first two dimensions must be (1, 1), instead they are: {}".
                             format(io_str, name, shape[:2]))
    else:
        raise KeyError("Unrecognized color_layout {}".format(color_layout))