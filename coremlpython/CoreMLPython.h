// Copyright (c) 2021, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause


// Disable a few warnings and include pybind first, then re-enable warnings
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wexit-time-destructors"
#pragma clang diagnostic ignored "-Wdocumentation"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#pragma clang diagnostic pop

#import <CoreML/CoreML.h>


#ifndef BUILT_WITH_MACOS15_SDK
#define BUILT_WITH_MACOS15_SDK \
  !(TARGET_OS_OSX && (!defined(__MAC_15_0) || __MAC_OS_X_VERSION_MAX_ALLOWED < __MAC_15_0))
#endif

// Print BUILT_WITH_MACOS15_SDK value
#if BUILT_WITH_MACOS15_SDK
#pragma message ("Building with macOS 15+ SDK")
#else
#pragma message ("Building without macOS 15 SDK")
#endif


namespace py = pybind11;

namespace CoreML {
    namespace Python {


        struct State {
#if BUILT_WITH_MACOS15_SDK
            // MLState must be wrapped in a C++ class for PyBind.
            MLState* m_state = nil;
#endif
        };

        class Model {
        private:
            MLModel *m_model = nil;
            NSURL *compiledUrl = nil;
            bool m_deleteCompiledModelOnExit;

        public:
            static py::bytes autoSetSpecificationVersion(const py::bytes& modelBytes);
            static py::str compileModel(const std::string& urlStr);
            static int32_t maximumSupportedSpecificationVersion();
            static void setComputeUnit(MLModelConfiguration *configuration, const std::string& computeUnits);

            Model(const Model&) = delete;
            Model& operator=(const Model&) = delete;
            ~Model();
            explicit Model(const std::string& urlStr, const std::string& computeUnits, const std::string& functionName);
            explicit Model(MLModel* m_model, NSURL* compiledUrl, bool deleteCompiledModelOnExit);

            py::list batchPredict(const py::list& batch) const;

            py::str getCompiledModelPath() const;

            py::dict predict(const py::dict& input, State* state=NULL) const;

#if BUILT_WITH_MACOS15_SDK
            State newState() const;
#endif

        };
    }
}
