#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wexit-time-destructors"
#pragma clang diagnostic ignored "-Wdocumentation"
#pragma clang diagnostic ignored "-Wrange-loop-analysis"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#pragma clang diagnostic pop

#import <CoreML/CoreML.h>

#include "CoreMLPythonUtils.h"

#include <cstring>
#include <sstream>
#include <vector>

namespace py = pybind11;

@interface PybindCompatibleArray : MLMultiArray

- (PybindCompatibleArray *)initWithArray:(py::array)array;

@end
