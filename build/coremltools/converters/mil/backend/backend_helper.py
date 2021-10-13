#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil.passes.name_sanitization_utils import NameSanitizer

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