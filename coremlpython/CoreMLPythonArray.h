// Copyright (c) 2021, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wexit-time-destructors"
#pragma clang diagnostic ignored "-Wdocumentation"
#pragma clang diagnostic ignored "-Wrange-loop-analysis"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

#ifdef check
#define __old_check check
#undef check
#endif

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef __old_check
#define check __old_check
#endif

#pragma clang diagnostic pop

#import <CoreML/CoreML.h>

#include "CoreMLPythonUtils.h"

#include <cstring>
#include <sstream>
#include <vector>

namespace py = pybind11;

@interface PybindCompatibleArray : MLMultiArray {
    // Holding reference to underlying memory
    py::array m_array;
}

- (PybindCompatibleArray *)initWithArray:(py::array)array;

@end
