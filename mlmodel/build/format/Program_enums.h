#ifndef __PROGRAM_ENUMS_H
#define __PROGRAM_ENUMS_H
enum MLScalarType: int {
    MLScalarTypeDYNAMIC = 0,
    MLScalarTypeBOOL = 1,
    MLScalarTypeSTRING = 2,
    MLScalarTypeFLOAT16 = 10,
    MLScalarTypeFLOAT32 = 11,
    MLScalarTypeFLOAT64 = 12,
    MLScalarTypeBFLOAT16 = 13,
    MLScalarTypeINT4 = 20,
    MLScalarTypeINT8 = 21,
    MLScalarTypeINT16 = 22,
    MLScalarTypeINT32 = 23,
    MLScalarTypeINT64 = 24,
    MLScalarTypeUINT4 = 30,
    MLScalarTypeUINT8 = 31,
    MLScalarTypeUINT16 = 32,
    MLScalarTypeUINT32 = 33,
    MLScalarTypeUINT64 = 34,
};

enum MLValueTypetype: int {
    MLValueTypetype_scalarType = 1,
    MLValueTypetype_tensorType = 2,
    MLValueTypetype_listType = 3,
    MLValueTypetype_tupleType = 4,
    MLValueTypetype_NOT_SET = 0,
};

__attribute__((__unused__))
static const char * MLValueTypetype_Name(MLValueTypetype x) {
    switch (x) {
        case MLValueTypetype_scalarType:
            return "MLValueTypetype_scalarType";
        case MLValueTypetype_tensorType:
            return "MLValueTypetype_tensorType";
        case MLValueTypetype_listType:
            return "MLValueTypetype_listType";
        case MLValueTypetype_tupleType:
            return "MLValueTypetype_tupleType";
        case MLValueTypetype_NOT_SET:
            return "INVALID";
    }
    return "INVALID";
}

enum MLDimensiondimension: int {
    MLDimensiondimension_symbol = 1,
    MLDimensiondimension_size = 2,
    MLDimensiondimension_NOT_SET = 0,
};

__attribute__((__unused__))
static const char * MLDimensiondimension_Name(MLDimensiondimension x) {
    switch (x) {
        case MLDimensiondimension_symbol:
            return "MLDimensiondimension_symbol";
        case MLDimensiondimension_size:
            return "MLDimensiondimension_size";
        case MLDimensiondimension_NOT_SET:
            return "INVALID";
    }
    return "INVALID";
}

enum MLValuevalue: int {
    MLValuevalue_immediateValue = 3,
    MLValuevalue_fileValue = 4,
    MLValuevalue_NOT_SET = 0,
};

__attribute__((__unused__))
static const char * MLValuevalue_Name(MLValuevalue x) {
    switch (x) {
        case MLValuevalue_immediateValue:
            return "MLValuevalue_immediateValue";
        case MLValuevalue_fileValue:
            return "MLValuevalue_fileValue";
        case MLValuevalue_NOT_SET:
            return "INVALID";
    }
    return "INVALID";
}

enum MLImmediateValuevalue: int {
    MLImmediateValuevalue_f = 1,
    MLImmediateValuevalue_i = 2,
    MLImmediateValuevalue_b = 3,
    MLImmediateValuevalue_s = 4,
    MLImmediateValuevalue_tensor = 5,
    MLImmediateValuevalue_tuple = 6,
    MLImmediateValuevalue_NOT_SET = 0,
};

__attribute__((__unused__))
static const char * MLImmediateValuevalue_Name(MLImmediateValuevalue x) {
    switch (x) {
        case MLImmediateValuevalue_f:
            return "MLImmediateValuevalue_f";
        case MLImmediateValuevalue_i:
            return "MLImmediateValuevalue_i";
        case MLImmediateValuevalue_b:
            return "MLImmediateValuevalue_b";
        case MLImmediateValuevalue_s:
            return "MLImmediateValuevalue_s";
        case MLImmediateValuevalue_tensor:
            return "MLImmediateValuevalue_tensor";
        case MLImmediateValuevalue_tuple:
            return "MLImmediateValuevalue_tuple";
        case MLImmediateValuevalue_NOT_SET:
            return "INVALID";
    }
    return "INVALID";
}

#endif
