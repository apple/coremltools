#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np

from coremltools.converters.mil.mil import Operation, types
from coremltools.converters.mil.mil.input_type import (InputSpec,
                                                       ListInputType,
                                                       TensorInputType)
from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op
from coremltools.converters.mil.mil.types.symbolic import any_symbolic


@register_op
class classify(Operation):
    """
    The presence of this op indicates that the model is of type classifier. The op
    constructs the model output accordingly; that is, the predicted class label
    and the output probability dictionary. The parameters of this op are set
    based on the attributes set for the 
    `coremltools.ClassifierConfig <https://apple.github.io/coremltools/source/coremltools.converters.mil.input_types.html#classifierconfig>`_ class
    by the user. The outputs of this op cannot be used by another op.

    Parameters
    ----------
    probabilities: tensor<[\* , ProbT]> (Required)
        A tensor in the graph, which is used to compute the classifier output(s). This
        is the tensor whose values are mapped to the class labels and used for constructing
        the predicted class label and the output dictionary of class names and values.

    classes: list<\*, ClassT> (Required)
        List of classes.

    Returns
    -------
    <classT>
    Dict[classT, probT]


    Attributes
    ----------
    ProbT: fp32
    ClassT: i64, str
    """

    input_spec = InputSpec(
        probabilities=TensorInputType(type_domain=types.fp32),
        classes=ListInputType(const=True),
    )

    def type_inference(self):
        # check the type of "classes"
        if not types.is_list(self.classes.sym_type):
            msg = "'classes' in the op 'classify' must be of type list. Instead it is {}."
            raise ValueError(msg.format(self.classes.sym_type.__type_info__()))

        # check the type of "probabilities"
        if self.probabilities.dtype != types.fp32:
            msg = "classify op: input probabilities must be of type fp32. Instead it is of type {}"
            raise TypeError(msg.format(self.probabilities.sym_type.get_primitive().__type_info__()))

        classes_elem_type = self.classes.elem_type
        if classes_elem_type not in {types.str, types.int64}:
            msg = "Type of elements in 'classes' in the op 'classify' must be either str or int64. Instead it is {}."
            raise ValueError(msg.format(classes_elem_type.__type_info__()))

        # check that the size of "classes" is compatible with the size of "probabilities"
        if not any_symbolic(self.probabilities.shape):
            size = np.prod(self.probabilities.shape)
            if len(self.classes.val) != size:
                msg = "In op 'classify', number of classes must match the size of the tensor corresponding to 'probabilities'."
                raise ValueError(msg)

        return classes_elem_type, types.dict(classes_elem_type, types.double)
