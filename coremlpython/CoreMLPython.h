// Copyright (c) 2021, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wexit-time-destructors"
#pragma clang diagnostic ignored "-Wdocumentation"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#pragma clang diagnostic pop

#import <CoreML/CoreML.h>

namespace py = pybind11;

namespace CoreML {
    namespace Python {

        class Model {
        private:
            MLModel *m_model = nil;
            NSURL *compiledUrl = nil;
        public:
            Model(const Model&) = delete;
            Model& operator=(const Model&) = delete;
            ~Model();
            explicit Model(const std::string& urlStr, const std::string& computeUnits);
            py::dict predict(const py::dict& input, bool useCPUOnly);
            static py::bytes autoSetSpecificationVersion(const py::bytes& modelBytes);
            static int32_t maximumSupportedSpecificationVersion();
            std::string toString() const;
        };
    }
}
