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
#import <Availability.h>

#import <vector>

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

#if !defined(ML_COMPUTE_PLAN_IS_AVAILABLE) && __has_include(<CoreML/MLComputePlan.h>)
#define ML_COMPUTE_PLAN_IS_AVAILABLE 1
#endif

#if !defined(ML_MODEL_STRUCTURE_IS_AVAILABLE) && __has_include(<CoreML/MLModelStructure.h>)
#define ML_MODEL_STRUCTURE_IS_AVAILABLE 1
#endif

#if !defined(ML_COMPUTE_DEVICE_IS_AVAILABLE) && __has_include(<CoreML/MLComputeDeviceProtocol.h>)
#define ML_COMPUTE_DEVICE_IS_AVAILABLE 1
#endif

#if !defined(ML_MODEL_ASSET_IS_AVAILABLE) && __has_include(<CoreML/MLModelAsset.h>)
#define ML_MODEL_ASSET_IS_AVAILABLE 1
#endif

#if !defined(ML_STATE_IS_AVAILABLE) && __has_include(<CoreML/MLState.h>)
#define ML_STATE_IS_AVAILABLE 1
#endif


namespace py = pybind11;

namespace CoreML {
    namespace Python {
        struct State {
            // MLState must be wrapped in a C++ class for PyBind.
            inline State(id impl):
            m_impl(impl) {}

#if ML_STATE_IS_AVAILABLE
            API_AVAILABLE(macos(15.0))
            inline MLState *getImpl() const {
                return (MLState *)m_impl;
            }
#endif

            private:
             // Type erase `m_impl` otherwise it will result in a compiler warning.
            id m_impl = nil;
        };

        struct CPUComputeDevice {
            // MLCPUComputeDevice must be wrapped in a C++ class for PyBind.
            inline CPUComputeDevice(id impl):
            m_impl(impl) {}

#if ML_COMPUTE_DEVICE_IS_AVAILABLE
            API_AVAILABLE(macos(14.0))
            inline MLCPUComputeDevice *getImpl() const {
                return (MLCPUComputeDevice *)m_impl;
            }
#endif

            private:
            // Type erase `m_impl` otherwise it will result in a compiler warning.
            id m_impl = nil;
        };

        struct GPUComputeDevice {
            // MLGPUComputeDevice must be wrapped in a C++ class for PyBind.
            inline GPUComputeDevice(id impl):
            m_impl(impl) {}

#if ML_COMPUTE_DEVICE_IS_AVAILABLE
            API_AVAILABLE(macos(14.0))
            inline MLGPUComputeDevice *getImpl() const {
                return (MLGPUComputeDevice *)m_impl;
            }
#endif

            private:
            // Type erase `m_impl` otherwise it will result in a compiler warning.
            id m_impl = nil;
        };

        struct NeuralEngineComputeDevice {
            // MLNeuralEngineComputeDevice must be wrapped in a C++ class for PyBind.
            inline NeuralEngineComputeDevice(id impl):
            m_impl(impl) {}

#if ML_COMPUTE_DEVICE_IS_AVAILABLE
            API_AVAILABLE(macos(14.0))
            inline MLNeuralEngineComputeDevice *getImpl() const {
                return (MLNeuralEngineComputeDevice *)m_impl;
            }
#endif

            int getTotalCoreCount() const;

            private:
            // Type erase `m_impl` otherwise it will result in a compiler warning.
            id m_impl = nil;
        };

        struct ModelStructureProgramOperation {
            // MLModelStructureProgramOperation must be wrapped in a C++ class for PyBind.
            inline ModelStructureProgramOperation(id impl):
            m_impl(impl) {}

#if ML_MODEL_STRUCTURE_IS_AVAILABLE
            API_AVAILABLE(macos(14.4))
            inline MLModelStructureProgramOperation *getImpl() const {
                return (MLModelStructureProgramOperation *)m_impl;
            }
#endif

            private:
            // Type erase `m_impl` otherwise it will result in a compiler warning.
            __weak id m_impl = nil;
        };

        struct ModelStructureNeuralNetworkLayer {
            // ModelStructureNeuralNetworkLayer must be wrapped in a C++ class for PyBind.
            inline ModelStructureNeuralNetworkLayer(id impl):
            m_impl(impl) {}

#if ML_MODEL_STRUCTURE_IS_AVAILABLE
            API_AVAILABLE(macos(14.4))
            inline MLModelStructureNeuralNetworkLayer *getImpl() const {
                return (MLModelStructureNeuralNetworkLayer *)m_impl;
            }
#endif

            private:
            // Type erase `m_impl` otherwise it will result in a compiler warning.
            __weak id m_impl = nil;
        };

        struct ComputePlan {
            // MLComputePlan must be wrapped in a C++ class for PyBind.
            inline ComputePlan(id impl, py::object modelStructure):
            m_impl(impl),
            m_modelStructure(modelStructure) {}

            inline py::object getModelStructure() const {
                return m_modelStructure;
            }

#if ML_COMPUTE_PLAN_IS_AVAILABLE
            API_AVAILABLE(macos(14.4))
            inline MLComputePlan *getImpl() const {
                return (MLComputePlan *)m_impl;
            }

            py::object getComputeDeviceUsageForMLProgramOperation(py::object operation);
            py::object getComputeDeviceUsageForNeuralNetworkLayer(py::object layer);
            py::object getEstimatedCostForMLProgramOperation(py::object operation);
#endif

            private:
            id m_impl = nil;
            py::object m_modelStructure;
        };

        struct ModelAsset {
            // MLModelAsset must be wrapped in a C++ class for PyBind.
            inline ModelAsset(id impl, std::vector<py::bytes> datas):
            m_impl(impl),
            m_datas(std::move(datas)) {}

            API_AVAILABLE(macos(13.0))
            inline MLModelAsset *getImpl() const {
                return (MLModelAsset *)m_impl;
            }

            id m_impl = nil;
            std::vector<py::bytes> m_datas;
        };

        class Model {
        private:
            MLModel *m_model = nil;
            NSURL *compiledUrl = nil;
            bool m_deleteCompiledModelOnExit = false;

        public:
            static py::bytes autoSetSpecificationVersion(const py::bytes& modelBytes);
            static py::str compileModel(const std::string& urlStr);
            static int32_t maximumSupportedSpecificationVersion();
            static void setComputeUnit(MLModelConfiguration *configuration, const std::string& computeUnits);

            Model(const Model&) = delete;
            Model& operator=(const Model&) = delete;
            ~Model();
            explicit Model(const std::string& urlStr,
                           const std::string& computeUnits,
                           const std::string& functionName,
                           const py::dict& optimizationHints,
                           const py::object& asset);

            explicit Model(const std::string& urlStr,
                           const std::string& computeUnits,
                           const std::string& functionName,
                           const py::dict& optimizationHints);


            explicit Model(MLModel* m_model, NSURL* compiledUrl, bool deleteCompiledModelOnExit);

            py::list batchPredict(const py::list& batch) const;

            py::str getCompiledModelPath() const;

            py::dict predict(const py::dict& input, State* state=NULL) const;

#if BUILT_WITH_MACOS15_SDK
            static void setOptimizationHints(MLModelConfiguration *configuration, const py::dict& optimizationHints);
            State newState() const;
#endif

            static py::object createModelAssetFromPath(const std::string& path);
            static py::object createModelAssetFromMemory(const py::bytes& specData, const py::dict& blobMapping);
            static py::object getModelStructure(const std::string& modelPath);
            static py::list getAvailableComputeDevices();
            static py::list getAllComputeDevices();
            static py::object getComputePlan(const std::string& modelPath, const std::string& computeUnits);
        };
    }
}
