#ifndef __MIL_ENUMS_H
#define __MIL_ENUMS_H
enum MLDataType: int {
    MLDataTypeUNUSED_TYPE = 0,
    MLDataTypeBOOL = 1,
    MLDataTypeSTRING = 2,
    MLDataTypeFLOAT16 = 10,
    MLDataTypeFLOAT32 = 11,
    MLDataTypeFLOAT64 = 12,
    MLDataTypeBFLOAT16 = 13,
    MLDataTypeINT8 = 21,
    MLDataTypeINT16 = 22,
    MLDataTypeINT32 = 23,
    MLDataTypeINT64 = 24,
    MLDataTypeUINT8 = 31,
    MLDataTypeUINT16 = 32,
    MLDataTypeUINT32 = 33,
    MLDataTypeUINT64 = 34,
};

enum MLBindingbinding: int {
    MLBindingbinding_name = 1,
    MLBindingbinding_value = 2,
    MLBindingbinding_NOT_SET = 0,
};

__attribute__((__unused__))
static const char * MLBindingbinding_Name(MLBindingbinding x) {
    switch (x) {
        case MLBindingbinding_name:
            return "MLBindingbinding_name";
        case MLBindingbinding_value:
            return "MLBindingbinding_value";
        case MLBindingbinding_NOT_SET:
            return "INVALID";
    }
    return "INVALID";
}

enum MLValueTypetype: int {
    MLValueTypetype_tensorType = 1,
    MLValueTypetype_listType = 2,
    MLValueTypetype_tupleType = 3,
    MLValueTypetype_dictionaryType = 4,
    MLValueTypetype_NOT_SET = 0,
};

__attribute__((__unused__))
static const char * MLValueTypetype_Name(MLValueTypetype x) {
    switch (x) {
        case MLValueTypetype_tensorType:
            return "MLValueTypetype_tensorType";
        case MLValueTypetype_listType:
            return "MLValueTypetype_listType";
        case MLValueTypetype_tupleType:
            return "MLValueTypetype_tupleType";
        case MLValueTypetype_dictionaryType:
            return "MLValueTypetype_dictionaryType";
        case MLValueTypetype_NOT_SET:
            return "INVALID";
    }
    return "INVALID";
}

enum MLDimensiondimension: int {
    MLDimensiondimension_constant = 1,
    MLDimensiondimension_unknown = 2,
    MLDimensiondimension_NOT_SET = 0,
};

__attribute__((__unused__))
static const char * MLDimensiondimension_Name(MLDimensiondimension x) {
    switch (x) {
        case MLDimensiondimension_constant:
            return "MLDimensiondimension_constant";
        case MLDimensiondimension_unknown:
            return "MLDimensiondimension_unknown";
        case MLDimensiondimension_NOT_SET:
            return "INVALID";
    }
    return "INVALID";
}

enum MLValuevalue: int {
    MLValuevalue_immediateValue = 3,
    MLValuevalue_blobFileValue = 5,
    MLValuevalue_NOT_SET = 0,
};

__attribute__((__unused__))
static const char * MLValuevalue_Name(MLValuevalue x) {
    switch (x) {
        case MLValuevalue_immediateValue:
            return "MLValuevalue_immediateValue";
        case MLValuevalue_blobFileValue:
            return "MLValuevalue_blobFileValue";
        case MLValuevalue_NOT_SET:
            return "INVALID";
    }
    return "INVALID";
}

enum MLImmediateValuevalue: int {
    MLImmediateValuevalue_tensor = 1,
    MLImmediateValuevalue_tuple = 2,
    MLImmediateValuevalue_list = 3,
    MLImmediateValuevalue_dictionary = 4,
    MLImmediateValuevalue_NOT_SET = 0,
};

__attribute__((__unused__))
static const char * MLImmediateValuevalue_Name(MLImmediateValuevalue x) {
    switch (x) {
        case MLImmediateValuevalue_tensor:
            return "MLImmediateValuevalue_tensor";
        case MLImmediateValuevalue_tuple:
            return "MLImmediateValuevalue_tuple";
        case MLImmediateValuevalue_list:
            return "MLImmediateValuevalue_list";
        case MLImmediateValuevalue_dictionary:
            return "MLImmediateValuevalue_dictionary";
        case MLImmediateValuevalue_NOT_SET:
            return "INVALID";
    }
    return "INVALID";
}

enum MLTensorValuevalue: int {
    MLTensorValuevalue_floats = 1,
    MLTensorValuevalue_ints = 2,
    MLTensorValuevalue_bools = 3,
    MLTensorValuevalue_strings = 4,
    MLTensorValuevalue_longInts = 5,
    MLTensorValuevalue_doubles = 6,
    MLTensorValuevalue_bytes = 7,
    MLTensorValuevalue_NOT_SET = 0,
};

__attribute__((__unused__))
static const char * MLTensorValuevalue_Name(MLTensorValuevalue x) {
    switch (x) {
        case MLTensorValuevalue_floats:
            return "MLTensorValuevalue_floats";
        case MLTensorValuevalue_ints:
            return "MLTensorValuevalue_ints";
        case MLTensorValuevalue_bools:
            return "MLTensorValuevalue_bools";
        case MLTensorValuevalue_strings:
            return "MLTensorValuevalue_strings";
        case MLTensorValuevalue_longInts:
            return "MLTensorValuevalue_longInts";
        case MLTensorValuevalue_doubles:
            return "MLTensorValuevalue_doubles";
        case MLTensorValuevalue_bytes:
            return "MLTensorValuevalue_bytes";
        case MLTensorValuevalue_NOT_SET:
            return "INVALID";
    }
    return "INVALID";
}

#endif
