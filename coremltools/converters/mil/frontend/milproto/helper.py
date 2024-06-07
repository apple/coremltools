#  Copyright (c) 2022, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause


from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.program import get_new_symbol


def get_proto_dim(dim):
    if dim.WhichOneof("dimension") == "constant":
        return dim.constant.size
    else:
        if not dim.unknown.variadic:
            return get_new_symbol()
        raise NotImplementedError("Variadic dimensions not yet implemented.")


def proto_to_types(valuetype):
    """
    A helper function that maps the proto value type to PyMIL types.
    """
    if valuetype.WhichOneof("type") == "tensorType":
        tensortype = valuetype.tensorType
        dtype = types.PROTO_TO_BUILTIN_TYPE[tensortype.dataType]

        if tensortype.rank < 0:
            raise ValueError("Negative or Dynamic ranks not supported")
        if tensortype.rank != len(tensortype.dimensions):
            raise ValueError("Rank doesn't match the number of dimensions")
        if tensortype.attributes != {}:
            raise ValueError("Attributes on tensorType not supported")

        shape = []
        for i in range(tensortype.rank):
            shape.append(get_proto_dim(tensortype.dimensions[i]))

        # For the zero rank tensor, we always convert it back to scalar in PyMIL first
        if tensortype.rank == 0:
            return dtype

        return types.tensor(dtype, shape)

    elif valuetype.WhichOneof("type") == "listType":
        listtype = valuetype.listType
        elem_type = proto_to_types(listtype.type)

        if listtype.length.unknown:
            init_length = None
        else:
            init_length = listtype.length.constant.size

        # In the MIL proto, there is no such thing of "dynamic_length", hence we set it to True when
        # converting back to PyMIL
        return types.list(elem_type, init_length, dynamic_length=True)

    elif valuetype.WhichOneof("type") == "dictionaryType":
        dicttype = valuetype.dictionaryType
        keytype = proto_to_types(dicttype.keyType)
        valuetype = proto_to_types(dicttype.valueType)

        return types.dict(keytype, valuetype)

    elif valuetype.WhichOneof("type") == "stateType":
        wrapped_type = proto_to_types(valuetype.stateType.wrappedType)

        return types.state(wrapped_type)
    else:
        raise NotImplementedError("Types {} not yet implemented".format(valuetype.WhichOneof("type")))
