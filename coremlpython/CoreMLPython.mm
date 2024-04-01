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
#import <AvailabilityMacros.h>
#import <fstream>
#import <vector>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-prototypes"

#if ! __has_feature(objc_arc)
#error "ARC is off"
#endif

#ifndef BUILT_WITH_MACOS13_SDK
#define BUILT_WITH_MACOS13_SDK (MAC_OS_X_VERSION_MAX_ALLOWED >= 130000)
#endif

namespace py = pybind11;

using namespace CoreML::Python;

bool usingMacOS13OrHigher() {
    // MLProgram class was introduced in macOS 13.
    return (NSProtocolFromString(@"MLProgram") != nil);
}

Model::~Model() {
    @autoreleasepool {
        NSFileManager *fileManager = [NSFileManager defaultManager];
        if (compiledUrl != nil and m_deleteCompiledModelOnExit) {
            [fileManager removeItemAtURL:compiledUrl error:NULL];
        }
    }
}

Model::Model(const std::string& urlStr, const std::string& computeUnits) {
    @autoreleasepool {
        NSError *error = nil;

        if (! Utils::isCompiledModelPath(urlStr)) {
            // Compile the model
            NSURL *specUrl = Utils::stringToNSURL(urlStr);

            // Swallow output for the very verbose coremlcompiler
            int stdoutBack = dup(STDOUT_FILENO);
            int devnull = open("/dev/null", O_WRONLY);
            dup2(devnull, STDOUT_FILENO);

            // Compile the model
            compiledUrl = [MLModel compileModelAtURL:specUrl error:&error];
            m_deleteCompiledModelOnExit = true;

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
        } else {
            m_deleteCompiledModelOnExit = false;  // Don't delete user specified file
            compiledUrl = Utils::stringToNSURL(urlStr);
        }

        MLModelConfiguration *configuration = [MLModelConfiguration new];
        setComputeUnit(configuration, computeUnits);

        // Create MLModel
        m_model = [MLModel modelWithContentsOfURL:compiledUrl configuration:configuration error:&error];
        Utils::handleError(error);
    }
}


Model::Model(MLModel* mlModel, NSURL* compiledUrl, bool deleteCompiledModelOnExit)
    : m_model(mlModel),
      compiledUrl(compiledUrl),
      m_deleteCompiledModelOnExit(deleteCompiledModelOnExit)
{
}

py::dict Model::predict(const py::dict& input) const {
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


void Model::setComputeUnit(MLModelConfiguration *configuration, const std::string& computeUnits) {
    if (computeUnits == "CPU_ONLY") {
        configuration.computeUnits = MLComputeUnitsCPUOnly;
    } else if (computeUnits == "CPU_AND_GPU") {
        configuration.computeUnits = MLComputeUnitsCPUAndGPU;
    } else if (computeUnits == "CPU_AND_NE") {
        if (usingMacOS13OrHigher()) {
#if BUILT_WITH_MACOS13_SDK
            configuration.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
#endif // BUILT_WITH_MACOS13_SDK
        } else {
            throw std::runtime_error("CPU_AND_NE is only available on macOS >= 13.0");
        }
    } else {
        assert(computeUnits == "ALL");
        configuration.computeUnits = MLComputeUnitsAll;
    }
}


py::list Model::batchPredict(const py::list& batch) const {
  @autoreleasepool {
      NSError* error = nil;

      // Convert input to a BatchProvider
      NSMutableArray* array = [[NSMutableArray alloc] initWithCapacity: batch.size()];
      for(int i = 0; i < batch.size(); i++) {
        MLDictionaryFeatureProvider* cur = Utils::dictToFeatures(batch[i], &error);
        Utils::handleError(error);
        [array addObject: cur];
      }
      MLArrayBatchProvider* batchProvider = [[MLArrayBatchProvider alloc] initWithFeatureProviderArray: array];

      // Get predictions
      MLArrayBatchProvider* predictions = (MLArrayBatchProvider*)[m_model predictionsFromBatch:batchProvider
                                                                                         error:&error];
      Utils::handleError(error);

      // Convert predictions to output
      py::list ret;
      for (int i = 0; i < predictions.array.count; i++) {
        ret.append(Utils::featuresToDict(predictions.array[i]));
      }
      return ret;
  }
}


py::str Model::getCompiledModelPath() const {
    if (this->compiledUrl == nil) {
        return nil;
    }
    return [this->compiledUrl.path UTF8String];
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


py::str Model::compileModel(const std::string& urlStr) {
    @autoreleasepool {
        NSError* error = nil;

        NSURL* specUrl = Utils::stringToNSURL(urlStr);
        NSURL* compiledUrl = [MLModel compileModelAtURL:specUrl error:&error];

        Utils::handleError(error);
        return [compiledUrl.path UTF8String];
    }
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
        .def("batchPredict", &Model::batchPredict)
        .def("get_compiled_model_path", &Model::getCompiledModelPath)
        .def_static("auto_set_specification_version", &Model::autoSetSpecificationVersion)
        .def_static("maximum_supported_specification_version", &Model::maximumSupportedSpecificationVersion)
        .def_static("compileModel", &Model::compileModel);

    return m.ptr();
}

#pragma clang diagnostic pop
