// Copyright (c) 2025, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

#import "CoreMLPythonArray.h"

@implementation PybindCompatibleArray

+ (MLMultiArrayDataType)dataTypeOf:(py::array)array {
    const auto& dt = array.dtype();
    char kind = dt.kind();
    size_t itemsize = dt.itemsize();


    if(kind == 'i' && itemsize == 4) {
        return MLMultiArrayDataTypeInt32;
    }
#if BUILT_WITH_MACOS26_SDK
    else if (kind == 'i' && itemsize == 1) {
        return MLMultiArrayDataTypeInt8;
    }
#endif
    else if(kind == 'f' && itemsize == 4) {
        return MLMultiArrayDataTypeFloat32;
    } else if( (kind == 'f' || kind == 'd') && itemsize == 8) {
        return MLMultiArrayDataTypeDouble;
    }

    throw std::runtime_error("Unsupported array type: " + std::to_string(kind) + " with itemsize = " + std::to_string(itemsize));
}

+ (NSArray<NSNumber *> *)shapeOf:(py::array)array {
    NSMutableArray<NSNumber *> *ret = [[NSMutableArray alloc] init];
    for (size_t i=0; i<array.ndim(); i++) {
        [ret addObject:[NSNumber numberWithUnsignedLongLong:array.shape(i)]];
    }
    return ret;
}

+ (NSArray<NSNumber *> *)stridesOf:(py::array)array {
    // numpy strides is in bytes.
    // this type must return number of ELEMENTS! (as per mlkit)

    NSMutableArray<NSNumber *> *ret = [[NSMutableArray alloc] init];
    for (size_t i=0; i<array.ndim(); i++) {
        size_t stride = array.strides(i) / array.itemsize();
        [ret addObject:[NSNumber numberWithUnsignedLongLong:stride]];
    }
    return ret;
}

- (PybindCompatibleArray *)initWithArray:(py::array)array {

    self = [super initWithDataPointer:array.mutable_data()
                                shape:[self.class shapeOf:array]
                             dataType:[self.class dataTypeOf:array]
                              strides:[self.class stridesOf:array]
                          deallocator:nil
                                error:nil];

    if (self) {
        m_array = array;
    }
    return self;
}

- (void)dealloc {
    // `m_array` owns a reference to a Python object, but the lifetime of this
    // Objective-C object is controlled by Core ML, not by Python. Core ML does
    // not necessarily release the input feature values before
    // `-predictionFromFeatures:...` returns: the MLE5 engine keeps its input
    // ports bound after a prediction and unbinds them asynchronously on one of
    // its own dispatch queues (`-[MLE5ExecutionStream resetAfterLingering:]`
    // running on com.apple.coreml.MLE5ExecutionStream.resetQueue).
    //
    // On that path the compiler generated `.cxx_destruct` would destroy
    // `m_array` -- and therefore `Py_DECREF` the numpy array -- from a thread
    // that holds neither the GIL nor a Python thread state, which races with
    // the interpreter and corrupts the Python heap. So drop the reference here,
    // with the GIL held, and leave `.cxx_destruct` a no-op.
    PyObject *array = m_array.release().ptr();
    if (array == NULL) {
        return;
    }

    if (!Py_IsInitialized()) {
        // The interpreter is already gone; everything it owned went with it.
        return;
    }

    PyGILState_STATE gilState = PyGILState_Ensure();
    Py_DECREF(array);
    PyGILState_Release(gilState);
}

@end
