// Copyright (c) 2021, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
#import <CoreML/CoreML.h>
#import "CoreMLPythonArray.h"
#import "CoreMLPython.h"
#import "CoreMLPythonUtils.h"
#import "Globals.hpp"
#import "Utils.hpp"
#import <fstream>
#import <vector>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-prototypes"

#if ! __has_feature(objc_arc)
#error "ARC is off"
#endif

namespace py = pybind11;

using namespace CoreML::Python;

Model::~Model() {
    @autoreleasepool {
        NSFileManager *fileManager = [NSFileManager defaultManager];
        if (compiledUrl != nil) {
            [fileManager removeItemAtURL:compiledUrl error:NULL];
        }
    }
}

Model::Model(const std::string& urlStr, const std::string& computeUnits) {
    @autoreleasepool {

        // Compile the model
        NSError *error = nil;
        NSURL *specUrl = Utils::stringToNSURL(urlStr);

        // Swallow output for the very verbose coremlcompiler
        int stdoutBack = dup(STDOUT_FILENO);
        int devnull = open("/dev/null", O_WRONLY);
        dup2(devnull, STDOUT_FILENO);

        // Compile the model
        compiledUrl = [MLModel compileModelAtURL:specUrl error:&error];

        // Close all the file descriptors and revert back to normal
        dup2(stdoutBack, STDOUT_FILENO);
        close(devnull);
        close(stdoutBack);

        // Translate into a type that pybind11 can bridge to Python
        if (error != nil) {
            std::stringstream errmsg;
            errmsg << "Error compiling model: \"";
            errmsg << error.localizedDescription.UTF8String;
            errmsg << "\".";
            throw std::runtime_error(errmsg.str());
        }

        if (@available(macOS 10.14, *)) {
            MLModelConfiguration *configuration = [MLModelConfiguration new];
            if (computeUnits == "CPU_ONLY") {
                configuration.computeUnits = MLComputeUnitsCPUOnly;
            } else if (computeUnits == "CPU_AND_GPU") {
                configuration.computeUnits = MLComputeUnitsCPUAndGPU;
            } else {
                assert(computeUnits == "ALL");
                configuration.computeUnits = MLComputeUnitsAll;
            }
            m_model = [MLModel modelWithContentsOfURL:compiledUrl configuration:configuration error:&error];
        } else {
            m_model = [MLModel modelWithContentsOfURL:compiledUrl error:&error];
        }
        Utils::handleError(error);
    }
}

py::dict Model::predict(const py::dict& input) {
    @autoreleasepool {
        NSError *error = nil;
        MLDictionaryFeatureProvider *inFeatures = Utils::dictToFeatures(input, &error);
        Utils::handleError(error);
        id<MLFeatureProvider> outFeatures = [m_model predictionFromFeatures:static_cast<MLDictionaryFeatureProvider * _Nonnull>(inFeatures)
                                                                      error:&error];
        Utils::handleError(error);
        return Utils::featuresToDict(outFeatures);
    }
}

py::bytes Model::autoSetSpecificationVersion(const py::bytes& modelBytes) {

    CoreML::Specification::Model model;
    std::istringstream modelIn(static_cast<std::string>(modelBytes), std::ios::binary);
    CoreML::loadSpecification<Specification::Model>(model, modelIn);
    model.set_specificationversion(CoreML::MLMODEL_SPECIFICATION_VERSION_NEWEST);
    // always try to downgrade the specification version to the
    // minimal version that supports everything in this mlmodel
    CoreML::downgradeSpecificationVersion(&model);
    std::ostringstream modelOut;
    saveSpecification(model, modelOut);
    return static_cast<py::bytes>(modelOut.str());

}

int32_t Model::maximumSupportedSpecificationVersion() {
    return CoreML::MLMODEL_SPECIFICATION_VERSION_NEWEST;
}


/*
 *
 * bindings
 *
 */

PYBIND11_PLUGIN(libcoremlpython) {
    py::module m("libcoremlpython", "CoreML.Framework Python bindings");

    py::class_<Model>(m, "_MLModelProxy")
        .def(py::init<const std::string&, const std::string&>())
        .def("predict", &Model::predict)
        .def_static("auto_set_specification_version", &Model::autoSetSpecificationVersion)
        .def_static("maximum_supported_specification_version", &Model::maximumSupportedSpecificationVersion);

    return m.ptr();
}

#pragma clang diagnostic pop
