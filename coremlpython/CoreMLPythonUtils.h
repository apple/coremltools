#include <string>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wexit-time-destructors"
#pragma clang diagnostic ignored "-Wdocumentation"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <os/availability.h>
#pragma clang diagnostic pop

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import "LayerShapeConstraints.hpp"

namespace py = pybind11;

namespace CoreML {
    namespace Python {
        namespace Utils {

            NSURL * stringToNSURL(const std::string& str);
            void handleError(NSError *error);
            
            // python -> objc
            MLDictionaryFeatureProvider * dictToFeatures(const py::dict& dict, NSError **error);
            MLFeatureValue * convertValueToObjC(const py::handle& handle);
            
            // objc -> cpp
            std::vector<size_t> convertNSArrayToCpp(NSArray<NSNumber *> *array);
            NSArray<NSNumber *>* convertCppArrayToObjC(const std::vector<size_t>& array);
            
            // objc -> python
            py::dict featuresToDict(id<MLFeatureProvider> features);
            py::object convertValueToPython(MLFeatureValue *value);
            py::object convertArrayValueToPython(MLMultiArray *value);
            py::object convertDictionaryValueToPython(NSDictionary<NSObject *,NSNumber *> * value);
            py::object convertImageValueToPython(CVPixelBufferRef value);
            py::object convertSequenceValueToPython(MLSequence *seq) API_AVAILABLE(macos(10.14));
            py::dict shapeConstraintToPyDict(const ShapeConstraint& constraint);

        }
    }
}
