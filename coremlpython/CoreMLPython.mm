// Copyright (c) 2021, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

#import <CoreML/CoreML.h>
#import <mach/mach_time.h>
#import "CoreMLPythonArray.h"
#import "CoreMLPython.h"
#import "CoreMLPythonUtils.h"
#import "Globals.hpp"
#import "Utils.hpp"
#import <AvailabilityMacros.h>
#import <fstream>
#import <vector>

const NSErrorDomain CoreMLPythonErrorDomain = @"com.apple.coremlpython";

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-prototypes"

#if ! __has_feature(objc_arc)
#error "ARC is off"
#endif

#ifndef BUILT_WITH_MACOS13_SDK
#define BUILT_WITH_MACOS13_SDK (MAC_OS_X_VERSION_MAX_ALLOWED >= 130000)
#endif

#ifndef BUILT_WITH_MACOS26_SDK
#define BUILT_WITH_MACOS26_SDK (MAC_OS_X_VERSION_MAX_ALLOWED >= 140000)
#endif

namespace py = pybind11;

using namespace CoreML::Python;

namespace {
    inline py::object getPythonClass(Class cls,const py::module_& module) {
        return module.attr(NSStringFromClass(cls).UTF8String);
    }

    inline py::module_ getComputePlanModule() {
        return py::module_::import("coremltools.models.compute_plan");
    }

    inline py::module_ getComputeDeviceModule() {
        return py::module_::import("coremltools.models.compute_device");
    }

    inline py::module_ getMLModelModule() {
        return py::module_::import("coremltools.models.model");
    }

    inline NSData *toNSDataWithoutCopy(const py::bytes& bytes) {
        char* buffer;
        ssize_t length;
        PYBIND11_BYTES_AS_STRING_AND_SIZE(bytes.ptr(), &buffer, &length);
        return [NSData dataWithBytesNoCopy:static_cast<void *>(buffer) length:static_cast<NSUInteger>(length) freeWhenDone:NO];
    }

    inline py::list toPythonList(NSArray<NSString *> *values) {
        py::list result;
        for (NSString *value in values) {
            result.append(value.UTF8String);
        }
        return result;
    }

#if ML_COMPUTE_DEVICE_IS_AVAILABLE
    API_AVAILABLE(macos(14.0))
    py::object toPythonObject(id<MLComputeDeviceProtocol> computeDevice) {
        if (@available(macOS 14.0, *)) {
            auto module = getComputeDeviceModule();
            if ([computeDevice isKindOfClass:MLCPUComputeDevice.class]) {
                CPUComputeDevice cpuComputeDevice(computeDevice);
                auto cls = getPythonClass(MLCPUComputeDevice.class, module);
                return cls(cpuComputeDevice);
            } else if ([computeDevice isKindOfClass:MLGPUComputeDevice.class]) {
                GPUComputeDevice gpuComputeDevice(computeDevice);
                auto cls = getPythonClass(MLGPUComputeDevice.class, module);
                return cls(gpuComputeDevice);
            } else if ([computeDevice isKindOfClass:MLNeuralEngineComputeDevice.class]) {
                NeuralEngineComputeDevice neuralEngineComputeDevice(computeDevice);
                auto cls = getPythonClass(MLNeuralEngineComputeDevice.class, module);
                return cls(neuralEngineComputeDevice);
            } else {
                return py::none();
            }
        } else {
            throw std::runtime_error("MLComputeDevice is only available on macOS >= 14.0");
            return py::none();
        }
    }

    API_AVAILABLE(macos(14.0))
    py::list toPythonList(NSArray<id<MLComputeDeviceProtocol>> *computeDevices) {
        if (@available(macOS 14.0, *)) {
            py::list result;
            for (id<MLComputeDeviceProtocol> computeDevice in computeDevices) {
                auto pyComputeDevice = toPythonObject(computeDevice);
                if (!pyComputeDevice.is_none()) {
                    result.append(pyComputeDevice);
                }
            }

            return result;
        } else {
            throw std::runtime_error("MLComputeDevice is only available on macOS >= 14.0");
            return py::none();
        }
    }

#endif

#if ML_MODEL_STRUCTURE_IS_AVAILABLE
    API_AVAILABLE(macos(14.4))
    py::object toPythonObject(MLModelStructureNeuralNetworkLayer *neuralNetworkLayer) {
        if (@available(macOS 14.4, *)) {
            py::str name(neuralNetworkLayer.name.UTF8String);
            py::str type(neuralNetworkLayer.type.UTF8String);
            py::list inputNames(toPythonList(neuralNetworkLayer.inputNames));
            py::list outputNames(toPythonList(neuralNetworkLayer.outputNames));
            py::object cls = getPythonClass(MLModelStructureNeuralNetworkLayer.class, getComputePlanModule());
            auto proxy = ModelStructureNeuralNetworkLayer(neuralNetworkLayer);
            return cls(name, type, inputNames, outputNames, proxy);
        } else {
            throw std::runtime_error("MLModelStructure is only available on macOS >= 14.4");
            return py::none();
        }
    }

    API_AVAILABLE(macos(14.4))
    py::object toPythonObject(MLModelStructureNeuralNetwork *neuralNetwork) {
        if (@available(macOS 14.4, *)) {
            py::list layers;
            for (MLModelStructureNeuralNetworkLayer *neuralNetworkLayer in neuralNetwork.layers) {
                layers.append(toPythonObject(neuralNetworkLayer));
            }
            py::object cls = getPythonClass(MLModelStructureNeuralNetwork.class, getComputePlanModule());
            return cls(layers);
        } else {
            throw std::runtime_error("MLModelStructure is only available on macOS >= 14.4");
            return py::none();
        }
    }

    API_AVAILABLE(macos(14.4))
    py::object toPythonObject(MLModelStructureProgramValueType *valueType) {
        if (@available(macOS 14.4, *)) {
            py::object cls = getPythonClass(MLModelStructureProgramValueType.class, getComputePlanModule());
            return cls();
        } else {
            throw std::runtime_error("MLModelStructure is only available on macOS >= 14.4");
            return py::none();
        }
    }

    API_AVAILABLE(macos(14.4))
    py::object toPythonObject(MLModelStructureProgramNamedValueType *namedValueType) {
        if (@available(macOS 14.4, *)) {
            py::str name(namedValueType.name.UTF8String);
            py::object type = toPythonObject(namedValueType.type);
            py::object cls = getPythonClass(MLModelStructureProgramNamedValueType.class, getComputePlanModule());
            return cls(name, type);
        } else {
            throw std::runtime_error("MLModelStructure is only available on macOS >= 14.4");
            return py::none();
        }
    }

    API_AVAILABLE(macos(14.4))
    py::object toPythonObject(MLModelStructureProgramValue *value) {
        if (@available(macOS 14.4, *)) {
            py::object cls = getPythonClass(MLModelStructureProgramValue.class, getComputePlanModule());
            return cls();
        } else {
            throw std::runtime_error("MLModelStructure is only available on macOS >= 14.4");
            return py::none();
        }
    }

    API_AVAILABLE(macos(14.4))
    py::object toPythonObject(MLModelStructureProgramBinding *binding) {
        if (@available(macOS 14.4, *)) {
            py::object name = py::none();
            if (binding.name) {
                name = py::str(binding.name.UTF8String);
            }

            py::object value = py::none();
            if (binding.value) {
                value = toPythonObject(binding.value);
            }

            py::object cls = getPythonClass(MLModelStructureProgramBinding.class, getComputePlanModule());
            return cls(name, value);
        } else {
            throw std::runtime_error("MLModelStructure is only available on macOS >= 14.4");
            return py::none();
        }
    }

    API_AVAILABLE(macos(14.4))
    py::object toPythonObject(MLModelStructureProgramArgument *argument) {
        if (@available(macOS 14.4, *)) {
            py::list bindings;
            for (MLModelStructureProgramBinding *binding in argument.bindings) {
                bindings.append(toPythonObject(binding));
            }

            py::object cls = getPythonClass(MLModelStructureProgramArgument.class, getComputePlanModule());
            return cls(bindings);
        } else {
            throw std::runtime_error("MLModelStructure is only available on macOS >= 14.4");
            return py::none();
        }
    }

    py::object toPythonObject(MLModelStructureProgramBlock *block) API_AVAILABLE(macos(14.4));

    API_AVAILABLE(macos(14.4))
    py::object toPythonObject(MLModelStructureProgramOperation *operation) {
        if (@available(macOS 14.4, *)) {
            py::dict inputs;
            [operation.inputs enumerateKeysAndObjectsUsingBlock:^(NSString *name, MLModelStructureProgramArgument *argument, BOOL *stop) {
                inputs[py::str(name.UTF8String)] = toPythonObject(argument);
            }];

            py::str operatorName(operation.operatorName.UTF8String);

            py::list outputs;
            for (MLModelStructureProgramNamedValueType *output in operation.outputs) {
                outputs.append(toPythonObject(output));
            }

            py::list blocks;
            for (MLModelStructureProgramBlock *block in operation.blocks) {
                blocks.append(toPythonObject(block));
            }

            py::object cls = getPythonClass(MLModelStructureProgramOperation.class, getComputePlanModule());
            auto proxy = ModelStructureProgramOperation(operation);
            return cls(inputs, operatorName, outputs, blocks, proxy);
        }  else {
            throw std::runtime_error("MLModelStructure is only available on macOS >= 14.4");
            return py::none();
        }
    }

    API_AVAILABLE(macos(14.4))
    py::object toPythonObject(MLModelStructureProgramBlock *block) {
        if (@available(macOS 14.4, *)) {
            py::list inputs;
            for (MLModelStructureProgramNamedValueType *input in block.inputs) {
                inputs.append(toPythonObject(input));
            }

            py::list outputNames = toPythonList(block.outputNames);

            py::list operations;
            for (MLModelStructureProgramOperation *operation in block.operations) {
                operations.append(toPythonObject(operation));
            }

            py::object cls = getPythonClass(MLModelStructureProgramBlock.class, getComputePlanModule());
            return cls(inputs, operations, outputNames);
        } else {
            throw std::runtime_error("MLModelStructure is only available on macOS >= 14.4");
            return py::none();
        }
    }

    API_AVAILABLE(macos(14.4))
    py::object toPythonObject(MLModelStructureProgramFunction *function) {
        if (@available(macOS 14.4, *)) {
            py::list inputs;
            for (MLModelStructureProgramNamedValueType *input in function.inputs) {
                inputs.append(toPythonObject(input));
            }
            py::object block = toPythonObject(function.block);
            py::object cls = getPythonClass(MLModelStructureProgramFunction.class, getComputePlanModule());
            return cls(inputs, block);
        } else {
            throw std::runtime_error("MLModelStructure is only available on macOS >= 14.4");
            return py::none();
        }
    }


    API_AVAILABLE(macos(14.4))
    py::object toPythonObject(MLModelStructureProgram *program) {
        if (@available(macOS 14.4, *)) {
            py::dict functions;
            [program.functions enumerateKeysAndObjectsUsingBlock:^(NSString *name, MLModelStructureProgramFunction *function, BOOL *stop) {
                functions[py::str(name.UTF8String)] = toPythonObject(function);
            }];
            py::object cls = getPythonClass(MLModelStructureProgram.class, getComputePlanModule());
            return cls(functions);
        } else {
            throw std::runtime_error("MLModelStructure is only available on macOS >= 14.4");
            return py::none();
        }
    }

    py::object toPythonObject(MLModelStructure *modelStructure) API_AVAILABLE(macos(14.4));

    API_AVAILABLE(macos(14.4))
    py::object toPythonObject(MLModelStructurePipeline *pipeline) {
        if (@available(macOS 14.4, *)) {
            py::list submodels;
            NSEnumerator<NSString *> *names = [pipeline.subModelNames objectEnumerator];
            for (MLModelStructure *subModel in pipeline.subModels) {
                NSString *name = [names nextObject];
                py::tuple tuple = py::make_tuple(py::str(name.UTF8String), toPythonObject(subModel));
                submodels.append(tuple);
            }

            py::object cls = getPythonClass(MLModelStructurePipeline.class, getComputePlanModule());
            return cls(submodels);
        } else {
            throw std::runtime_error("MLModelStructure is only available on macOS >= 14.4");
            return py::none();
        }
    }

    API_AVAILABLE(macos(14.4))
    py::object toPythonObject(MLModelStructure *modelStructure) {
        if (@available(macOS 14.4, *)) {
            py::object cls = getPythonClass(MLModelStructure.class, getComputePlanModule());
            if (modelStructure.neuralNetwork) {
                py::object neuralNetwork = toPythonObject(modelStructure.neuralNetwork);
                return cls(neuralNetwork, py::none(), py::none());
            } else if (modelStructure.program) {
                py::object program = toPythonObject(modelStructure.program);
                return cls(py::none(), program, py::none());
            } else if (modelStructure.pipeline) {
                py::object pipeline = toPythonObject(modelStructure.pipeline);
                return cls(py::none(), py::none(), pipeline);
            } else {
                return cls(py::none(), py::none(), py::none());
            }
        } else {
            throw std::runtime_error("MLModelStructure is only available on macOS >= 14.4");
            return py::none();
        }
    }
#endif

#if ML_COMPUTE_PLAN_IS_AVAILABLE
    API_AVAILABLE(macos(14.4))
    py::object toPythonObject(MLComputePlanDeviceUsage *computeDeviceUsage) {
        if (@available(macOS 14.4, *)) {
            py::list supportedComputeDevices;
            for (id<MLComputeDeviceProtocol> computeDevice in computeDeviceUsage.supportedComputeDevices) {
                auto pyComputeDevice = toPythonObject(computeDevice);
                if (!pyComputeDevice.is_none()) {
                    supportedComputeDevices.append(pyComputeDevice);
                }
            }

            py::object preferredComputeDevice = toPythonObject(computeDeviceUsage.preferredComputeDevice);
            py::object cls = getPythonClass(MLComputePlanDeviceUsage.class, getComputePlanModule());
            return cls(preferredComputeDevice, supportedComputeDevices);
        } else {
            throw std::runtime_error("MLComputePlan is only available on macOS >= 14.4");
            return py::none();
        }
    }

    API_AVAILABLE(macos(14.4))
    py::object toPythonObject(MLComputePlanCost *estimatedCost) {
        if (@available(macOS 14.4, *)) {
            py::object cls = getPythonClass(MLComputePlanCost.class, getComputePlanModule());
            return cls(estimatedCost.weight);
        } else {
            throw std::runtime_error("MLComputePlan is only available on macOS >= 14.4");
            return py::none();
        }
    }
#endif

uint64_t convertMachTimeToNanoSeconds(uint64_t time) {
    static dispatch_once_t once;
    static mach_timebase_info_data_t timebase;
    dispatch_once(&once, ^{
        mach_timebase_info(&timebase);
    });
    uint64_t result = (time * timebase.numer) / timebase.denom;
    return result;
}

#if ML_MODEL_ASSET_IS_AVAILABLE
    API_AVAILABLE(macos(13.0))
    std::pair<MLModel *, uint64_t> createModelFromModelAsset(
        MLModelAsset *modelAsset,
        MLModelConfiguration *configuration,
        NSError * __autoreleasing *error
    ) {
        if (@available(macOS 13.0, *)) {
            dispatch_semaphore_t sem = dispatch_semaphore_create(0);
            __block MLModel *result = nil;
            __block NSError *lError = nil;
            uint64_t loadStartTime = mach_absolute_time();
            __block uint64_t loadEndTime = loadStartTime;
            [MLModel loadModelAsset:modelAsset
                    configuration:configuration
                completionHandler:^(MLModel * _Nullable model, NSError * _Nullable loadError){
                    result = model;
                    lError = loadError;
                    loadEndTime = mach_absolute_time();
                    dispatch_semaphore_signal(sem);
            }];

            dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);

            if (error) {
                *error = lError;
            }

            uint64_t loadDurationInNanoSeconds = convertMachTimeToNanoSeconds(loadEndTime - loadStartTime);
            return {result, loadDurationInNanoSeconds};
        } else {
            throw std::runtime_error("MLModelAsset is only available on macOS >= 13.0");
            return {nil, 0};
        }
    }
#endif
}

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

Model::Model(
    const std::string& urlStr,
    const std::string& computeUnits,
    const std::string& functionName,
    const py::dict& optimizationHints,
    const py::object& asset) {
    @autoreleasepool {
        NSError *error = nil;
        if (asset.is_none()) {
            if (Utils::isCompiledModelPath(urlStr)) {
               compiledUrl = Utils::stringToNSURL(urlStr);
            } else {
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
            }
        }

        MLModelConfiguration *configuration = [MLModelConfiguration new];
        setComputeUnit(configuration, computeUnits);

#if BUILT_WITH_MACOS15_SDK
        setOptimizationHints(configuration, optimizationHints);
        if (!functionName.empty()) {
            if (@available(macOS 15.0, *)) {
                configuration.functionName = [NSString stringWithUTF8String:functionName.c_str()];
            } else {
                throw std::runtime_error("The 'functionName' property can only be set on macOS >= 15.0");
            }
        }
#endif
        uint64_t loadDurationInNanoSeconds = 0;
        // Create MLModel
        if (asset.is_none()) {
            uint64_t loadStartTime = mach_absolute_time();
            m_model = [MLModel modelWithContentsOfURL:compiledUrl configuration:configuration error:&error];
            uint64_t loadEndTime = mach_absolute_time();
            loadDurationInNanoSeconds = convertMachTimeToNanoSeconds(loadEndTime - loadStartTime);
        }
        #if ML_MODEL_ASSET_IS_AVAILABLE
        else if (@available(macOS 13.0, *)) {
            auto pair = createModelFromModelAsset(py::cast<ModelAsset>(asset).getImpl(), configuration, &error);
            m_model = pair.first;
            loadDurationInNanoSeconds = pair.second;
        }
        #endif
        else {
            throw std::runtime_error("MLModelAsset is only available on macOS >= 13.0");
        }

        Utils::handleError(error);
        m_loadDurationInNanoSeconds = loadDurationInNanoSeconds;
    }
}

Model::Model(
    const std::string& urlStr,
    const std::string& computeUnits,
    const std::string& functionName,
    const py::dict& optimizationHints) :
    Model(urlStr, computeUnits, functionName, optimizationHints, py::none()) {}

Model::Model(MLModel* mlModel, NSURL* compiledUrl, bool deleteCompiledModelOnExit)
    : m_model(mlModel),
      compiledUrl(compiledUrl),
      m_deleteCompiledModelOnExit(deleteCompiledModelOnExit)
{
}


py::dict Model::predict(const py::dict& input, State* state) {
    @autoreleasepool {
        NSError *error = nil;
        MLDictionaryFeatureProvider *inFeatures = Utils::dictToFeatures(input, &error);
        Utils::handleError(error);

        id<MLFeatureProvider> outFeatures;
        uint64_t predictStartTime = mach_absolute_time();
        if (state == NULL) {
          outFeatures = [m_model predictionFromFeatures:static_cast<MLDictionaryFeatureProvider * _Nonnull>(inFeatures)
                                                            error:&error];
        }
        #if BUILT_WITH_MACOS15_SDK
        else if (@available(macOS 15.0, *)) {
           outFeatures = [m_model predictionFromFeatures:static_cast<MLDictionaryFeatureProvider * _Nonnull>(inFeatures)
                                              usingState:state->getImpl()
                                                   error:&error];
        }
        #endif
        else {
            throw std::runtime_error("Stateful predictions using MLState are only supported on macOS >= 15.0");
        }

        uint64_t predictEndTime = mach_absolute_time();
        Utils::handleError(error);

        m_lastPredictDurationInNanoSeconds = convertMachTimeToNanoSeconds(predictEndTime - predictStartTime);
        return Utils::featuresToDict(outFeatures);
    }
}


void Model::setComputeUnit(MLModelConfiguration *configuration, const std::string& computeUnits) {
    if (computeUnits == "CPU_ONLY") {
        configuration.computeUnits = MLComputeUnitsCPUOnly;
    } else if (computeUnits == "CPU_AND_GPU") {
        configuration.computeUnits = MLComputeUnitsCPUAndGPU;
    } else if (computeUnits == "CPU_AND_NE") {
        #if BUILT_WITH_MACOS13_SDK
        if (usingMacOS13OrHigher()) {
            if (@available(macOS 13.0, *)) {
                configuration.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
            }
        } else {
            throw std::runtime_error("MLComputeUnitsCPUAndNeuralEngine is only available on macOS >= 13.0");
        }
        #else
        throw std::runtime_error("MLComputeUnitsCPUAndNeuralEngine is only available on macOS >= 13.0");
        #endif
    } else {
        assert(computeUnits == "ALL");
        configuration.computeUnits = MLComputeUnitsAll;
    }
}


#if BUILT_WITH_MACOS15_SDK
void Model::setOptimizationHints(MLModelConfiguration *configuration, const py::dict& optimizationHints) {
    // This function does minimal validation. It assumes Python layer has already validated.

    // Reshape frequency optimization hint
    if (optimizationHints.contains("reshapeFrequency")) {
        const std::string val = optimizationHints["reshapeFrequency"].cast<std::string>();
        if (@available(macOS 15.0, *)) {
            if (val == "Frequent") {
                configuration.optimizationHints.reshapeFrequency = MLReshapeFrequencyHintFrequent;
            } else {
                assert(val == "Infrequent");
                configuration.optimizationHints.reshapeFrequency = MLReshapeFrequencyHintInfrequent;
            }
        } else {
            throw std::runtime_error("Setting 'reshapeFrequency' is only available on macOS >= 15.0");
        }
    }

    // Specialization strategy optimization hint
    if (optimizationHints.contains("specializationStrategy")) {
        const std::string val = optimizationHints["specializationStrategy"].cast<std::string>();
        if (@available(macOS 15.0, *)) {
            if (val == "Default") {
                configuration.optimizationHints.specializationStrategy = MLSpecializationStrategyDefault;
            } else {
                assert(val == "FastPrediction");
                configuration.optimizationHints.specializationStrategy = MLSpecializationStrategyFastPrediction;
            }
        } else {
            throw std::runtime_error("Setting 'specializationStrategy' is only available on macOS >= 15.0");
        }
    }

    // Allow low precision accumulation on GPU optimization hint
    if (optimizationHints.contains("allowLowPrecisionAccumulationOnGPU")) {
        const std::string val = optimizationHints["allowLowPrecisionAccumulationOnGPU"].cast<std::string>();
        if (@available(macOS 15.0, *)) {
            if (val == "True") {
                configuration.allowLowPrecisionAccumulationOnGPU = true;
            } else {
                assert(val == "False");
                configuration.allowLowPrecisionAccumulationOnGPU = false;
            }
        } else {
            throw std::runtime_error("Setting 'allowLowPrecisionAccumulationOnGPU' is only available on macOS >= 15.0");
        }
    }
}
#endif

py::list Model::batchPredict(const py::list& batch) {
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

      uint64_t predictStartTime = mach_absolute_time();
      // Get predictions
      MLArrayBatchProvider* predictions = (MLArrayBatchProvider*)[m_model predictionsFromBatch:batchProvider
                                                                                         error:&error];
      uint64_t predictEndTime = mach_absolute_time();
      Utils::handleError(error);

      m_lastPredictDurationInNanoSeconds = convertMachTimeToNanoSeconds(predictEndTime - predictStartTime);
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


#if BUILT_WITH_MACOS15_SDK
State Model::newState() const {
    if (@available(macOS 15.0, *)) {
        State state = State([m_model newState]);
        return state;
    }

    throw std::runtime_error("Stateful model is only available on macOS >= 15.0");
    return nil;
}
#endif

py::object State::readState(const std::string& stateName) const {
    __block py::object value = py::none();
    #if ML_STATE_IS_AVAILABLE
    if (@available(macOS 15.0, *)) {
        @try {
            [getImpl() getMultiArrayForStateNamed:@(stateName.c_str()) handler:^(MLMultiArray *buffer){
                value = Utils::convertArrayValueToPython(buffer);
            }];
        } @catch (NSException *exception) {
            NSString *reason = exception.reason ?: @"Unknown exception when reading state";
            throw std::runtime_error(reason.UTF8String);
        }
    } else {
        throw std::runtime_error("Reading state is only available on macOS >= 15.0");
    }
    #else
    throw std::runtime_error("Reading state is only available on macOS >= 15.0");
    #endif
    return value;
}

void State::writeState(const std::string& stateName, py::object value) {
    #if ML_STATE_IS_AVAILABLE
    if (@available(macOS 15.0, *)) {
        MLFeatureValue *featureValue = Utils::convertValueToObjC(value);
        if (!featureValue.multiArrayValue) {
            throw std::runtime_error("Failed to write state, only multi-array value is supported");
        }
        @try {
            [getImpl() getMultiArrayForStateNamed:@(stateName.c_str()) handler:^(MLMultiArray *buffer){
                [featureValue.multiArrayValue transferToMultiArray:buffer];
            }];
        } @catch (NSException *exception) {
            NSString *reason = exception.reason ?: @"Unknown exception when writing state";
            throw std::runtime_error(reason.UTF8String);
        }
    }
    else {
        throw std::runtime_error("Writing state is only available on macOS >= 15.0");
    }
    #else
    throw std::runtime_error("Writing state is only available on macOS >= 15.0");
    #endif
}

py::object Model::getModelStructure(const std::string& compiledModelPath) {
#if ML_MODEL_STRUCTURE_IS_AVAILABLE
    if (@available(macOS 14.4, *)) {
        @autoreleasepool {
            NSURL *compiledModelURL = Utils::stringToNSURL(compiledModelPath);
            dispatch_semaphore_t sem = dispatch_semaphore_create(0);
            __block py::object result = py::none();
            __block NSError *error = nil;
            __block MLModelStructure *modelStructure = nil;
            [MLModelStructure loadContentsOfURL:compiledModelURL
                                completionHandler:^(MLModelStructure * _Nullable lModelStructure,
                                                    NSError * _Nullable lError) {
                error  = lError;
                modelStructure = lModelStructure;
                dispatch_semaphore_signal(sem);
            }];

            dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
            if (modelStructure) {
                try {
                    result = toPythonObject(modelStructure);
                } catch(std::exception& ex) {
                    NSString *message = [NSString stringWithFormat:@"Failed to load model structure, with error=%s.", ex.what()];
                    error = [NSError errorWithDomain:CoreMLPythonErrorDomain code:0 userInfo:@{NSLocalizedDescriptionKey :message}];
                } catch(...) {
                    NSString *message = @"Failed to load model structure, with unknown error.";
                    error = [NSError errorWithDomain:CoreMLPythonErrorDomain code:0 userInfo:@{NSLocalizedDescriptionKey :message}];
                }
            }

            Utils::handleError(error);
            return result;
        }
    } else {
        throw std::runtime_error("MLModelStructure is only available on macOS >= 14.4");
        return py::none();
    }
#else
   throw std::runtime_error("MLModelStructure is only available on macOS >= 14.4");
   return py::none();
#endif
}

py::list Model::getAvailableComputeDevices() {
#if ML_COMPUTE_DEVICE_IS_AVAILABLE
    if (@available(macOS 14.0, *)) {
        @autoreleasepool {
            return toPythonList(MLModel.availableComputeDevices);
        }
    } else {
        throw std::runtime_error("MLComputeDevice is only available on macOS >= 14.0");
        return py::none();
    }
#else
   throw std::runtime_error("MLComputeDevice is only available on macOS >= 14.0");
   return py::none();
#endif
}

py::list Model::getAllComputeDevices() {
#if ML_COMPUTE_DEVICE_IS_AVAILABLE
    if (@available(macOS 14.0, *)) {
        @autoreleasepool {
            return toPythonList(MLAllComputeDevices());
        }
    } else {
        throw std::runtime_error("MLComputeDevice is only available on macOS >= 14.0");
        return py::none();
    }
#else
   throw std::runtime_error("MLComputeDevice is only available on macOS >= 14.0");
   return py::none();
#endif
}

int NeuralEngineComputeDevice::getTotalCoreCount() const {
#if ML_COMPUTE_DEVICE_IS_AVAILABLE
    if (@available(macOS 14.0, *)) {
        return getImpl().totalCoreCount;
    } else {
        throw std::runtime_error("MLComputeDevice is only available on macOS >= 14.0");
        return 0;
    }
#else
    throw std::runtime_error("MLNeuralEngineComputeDevice is only available on macOS >= 14.0");
    return 0;
#endif
}

py::object Model::getComputePlan(const std::string& compiledModelPath, const std::string& computeUnits) {
#if ML_COMPUTE_PLAN_IS_AVAILABLE
    if (@available(macOS 14.4, *)) {
        @autoreleasepool {
            NSURL *compiledModelURL = Utils::stringToNSURL(compiledModelPath);
            dispatch_semaphore_t sem = dispatch_semaphore_create(0);
            __block py::object result = py::none();
            __block NSError *error = nil;
            __block MLComputePlan *computePlan = nil;
            MLModelConfiguration *configuration = [[MLModelConfiguration alloc] init];
            setComputeUnit(configuration, computeUnits);
            [MLComputePlan loadContentsOfURL:compiledModelURL
                                configuration:configuration
                            completionHandler:^(MLComputePlan * _Nullable lComputePlan, NSError * _Nullable lError) {
                computePlan = lComputePlan;
                error = lError;
                dispatch_semaphore_signal(sem);
            }];

            dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
            if (computePlan) {
                try {
                    auto modelStructure = toPythonObject(computePlan.modelStructure);
                    auto proxy = ComputePlan(computePlan, modelStructure);
                    auto cls = getPythonClass(MLComputePlan.class, getComputePlanModule());
                    result = cls(proxy);
                } catch(std::exception& ex) {
                    NSString *message = [NSString stringWithFormat:@"Failed to load compute plan, with error=%s.", ex.what()];
                    error = [NSError errorWithDomain:CoreMLPythonErrorDomain code:0 userInfo:@{NSLocalizedDescriptionKey :message}];
                } catch(...) {
                    NSString *message = @"Failed to load compute plan, with unknown error.";
                    error = [NSError errorWithDomain:CoreMLPythonErrorDomain code:0 userInfo:@{NSLocalizedDescriptionKey :message}];
                }
            }

            Utils::handleError(error);
            return result;
        }
    } else {
        throw std::runtime_error("MLComputePlan is only available on macOS >= 14.4");
        return py::none();
    }
#else
    throw std::runtime_error("MLComputePlan is only available on macOS >= 14.4");
    return py::none();
#endif
}

#if ML_COMPUTE_PLAN_IS_AVAILABLE
py::object ComputePlan::getComputeDeviceUsageForMLProgramOperation(py::object operation) {
    if (@available(macOS 14.4, *)) {
        @autoreleasepool {
            auto proxy = py::cast<ModelStructureProgramOperation>(operation.attr("__proxy__"));
            MLModelStructureProgramOperation *operationImpl = proxy.getImpl();
            if (operationImpl == nil) {
                return py::none();
            }
            MLComputePlanDeviceUsage *computeDeviceUsage = [getImpl() computeDeviceUsageForMLProgramOperation:operationImpl];
            if (computeDeviceUsage == nil) {
                return py::none();
            }
            auto result = toPythonObject(computeDeviceUsage);
            return result;
        }
    } else {
        throw std::runtime_error("MLComputePlan is only available on macOS >= 14.4");
        return py::none();
    }
}

py::object ComputePlan::getComputeDeviceUsageForNeuralNetworkLayer(py::object layer) {
    if (@available(macOS 14.4, *)) {
        @autoreleasepool {
            auto proxy = py::cast<ModelStructureNeuralNetworkLayer>(layer.attr("__proxy__"));
            MLModelStructureNeuralNetworkLayer *layerImpl = proxy.getImpl();
            if (layerImpl == nil) {
                return py::none();
            }
            MLComputePlanDeviceUsage *computeDeviceUsage = [getImpl() computeDeviceUsageForNeuralNetworkLayer:layerImpl];
            if (computeDeviceUsage == nil) {
                return py::none();
            }
            auto result = toPythonObject(computeDeviceUsage);
            return result;
        }
    } else {
        throw std::runtime_error("MLComputePlan is only available on macOS >= 14.4");
        return py::none();
    }
}

py::object ComputePlan::getEstimatedCostForMLProgramOperation(py::object operation) {
    if (@available(macOS 14.4, *)) {
        @autoreleasepool {
            auto proxy = py::cast<ModelStructureProgramOperation>(operation.attr("__proxy__"));
            MLModelStructureProgramOperation *operationImpl = proxy.getImpl();
            if (operationImpl == nil) {
                return py::none();
            }
            MLComputePlanCost *estimatedCost = [getImpl() estimatedCostOfMLProgramOperation:operationImpl];
            if (estimatedCost == nil) {
                return py::none();
            }
            auto result = toPythonObject(estimatedCost);
            return result;
        }
    } else {
        throw std::runtime_error("MLComputePlan is only available on macOS >= 14.4");
        return py::none();
    }
}
#endif

py::object Model::createModelAssetFromPath(const std::string& path) {
#if BUILT_WITH_MACOS15_SDK
    if (@available(macOS 15.0, *)) {
        NSError *error = nil;
        NSURL *url = Utils::stringToNSURL(path);
        MLModelAsset *asset = [MLModelAsset modelAssetWithURL:url error:&error];
        Utils::handleError(error);
        auto proxy = ModelAsset(asset, {});
        auto cls = getPythonClass(MLModelAsset.class, getMLModelModule());
        return cls(proxy);
    } else {
        throw std::runtime_error("Creating an MLModelAsset from a file path is only available on macOS >= 15.0");
        return py::none();
    }
#else
    throw std::runtime_error("Creating an MLModelAsset from a file path is only available on macOS >= 15.0");
    return py::none();
#endif
}


py::object Model::createModelAssetFromMemory(const py::bytes& pySpecData, const py::dict& pyBlobMapping) {
#if ML_MODEL_ASSET_IS_AVAILABLE
    if (@available(macOS 13.0, *)) {
        NSError *error = nil;
        // We are creating `NSData` instances without copying the underlying data, retain
        // original data to prevent dangling references.
        std::vector<py::bytes> datas;

        NSData *specData = toNSDataWithoutCopy(pySpecData);
        datas.push_back(pySpecData);

        NSMutableDictionary<NSURL *, NSData *> *blobMapping = [NSMutableDictionary dictionary];
        for (const auto& pair : pyBlobMapping) {
            auto pyBlob = py::cast<py::bytes>(pair.second);
            NSString *blobName = @(py::cast<std::string>(pair.first).c_str());
            NSData *blobData = toNSDataWithoutCopy(pyBlob);
            NSURL *blobURL = [NSURL fileURLWithPath:blobName relativeToURL:nil];
            blobMapping[blobURL] = blobData;
            datas.push_back(std::move(pyBlob));
        }

        MLModelAsset *asset = nil;
        if (blobMapping.count > 0) {
            #if BUILT_WITH_MACOS15_SDK
            if (@available(macOS 15.0, *)) {
                asset = [MLModelAsset modelAssetWithSpecificationData:specData blobMapping:blobMapping error:&error];
            } else {
                throw std::runtime_error("Creating an MLModelAsset using blob mapping is only available on macOS >= 15.0");
                return py::none();
            }
            #else
            throw std::runtime_error("Creating an MLModelAsset using blob mapping is only available on macOS >= 15.0");
            return py::none();
            #endif
        } else {
            asset = [MLModelAsset modelAssetWithSpecificationData:specData error:&error];
        }
        Utils::handleError(error);
        auto proxy = ModelAsset(asset, std::move(datas));
        auto cls = getPythonClass(MLModelAsset.class, getMLModelModule());
        return cls(proxy);
    } else {
        throw std::runtime_error("MLModelAsset is only available on macOS >= 13.0");
    }
#else
    throw std::runtime_error("Creating an MLModelAsset from memory is only available on macOS >= 14.0");
    return py::none();
#endif
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

py::object Model::getLoadDurationInNanoSeconds() const {
    if (m_loadDurationInNanoSeconds) {
        return py::cast(m_loadDurationInNanoSeconds.value());
    }

    return py::none();
}

py::object Model::getLastPredictDurationInNanoSeconds() const {
    if (m_lastPredictDurationInNanoSeconds) {
        return py::cast(m_lastPredictDurationInNanoSeconds.value());
    }

    return py::none();
}

/*
 *
 * bindings
 *
 */

PYBIND11_PLUGIN(libcoremlpython) {
    py::module m("libcoremlpython", "CoreML.Framework Python bindings");

    py::class_<Model>(m, "_MLModelProxy")
        .def(py::init<const std::string&, const std::string&, const std::string&, const py::dict&>())
        .def(py::init<const std::string&, const std::string&, const std::string&, const py::dict&, const py::object&>())
        .def("predict", &Model::predict)
        .def("batchPredict", &Model::batchPredict)
        .def("get_compiled_model_path", &Model::getCompiledModelPath)
        .def("get_load_duration_in_nano_seconds", &Model::getLoadDurationInNanoSeconds)
        .def("get_last_predict_duration_in_nano_seconds", &Model::getLastPredictDurationInNanoSeconds)
        .def_static("auto_set_specification_version", &Model::autoSetSpecificationVersion)
        .def_static("maximum_supported_specification_version", &Model::maximumSupportedSpecificationVersion)
#if BUILT_WITH_MACOS15_SDK
        .def("newState", &Model::newState)
#endif
        .def_static("create_model_asset_from_path", &Model::createModelAssetFromPath)
        .def_static("create_model_asset_from_memory", &Model::createModelAssetFromMemory)
        .def_static("get_model_structure", &Model::getModelStructure)
        .def_static("get_available_compute_devices", &Model::getAvailableComputeDevices)
        .def_static("get_all_compute_devices", &Model::getAllComputeDevices)
        .def_static("get_compute_plan", &Model::getComputePlan)
        .def_static("compileModel", &Model::compileModel);

    py::class_<State>(m, "_State", py::module_local())
        .def("read_state", &State::readState)
        .def("write_state", &State::writeState);

#if ML_COMPUTE_DEVICE_IS_AVAILABLE
    py::class_<CPUComputeDevice>(m, "_MLCPUComputeDeviceProxy", py::module_local())
       .def(py::init());
    py::class_<GPUComputeDevice>(m, "_MLGPUComputeDeviceProxy", py::module_local())
       .def(py::init());
    py::class_<NeuralEngineComputeDevice>(m, "_MLNeuralEngineComputeDeviceProxy", py::module_local())
        .def(py::init())
        .def("get_total_core_count", &NeuralEngineComputeDevice::getTotalCoreCount);
#endif

#if ML_COMPUTE_PLAN_IS_AVAILABLE
    py::class_<ComputePlan>(m, "_MLComputePlanProxy", py::module_local())
        .def(py::init())
        .def_property_readonly("model_structure", &ComputePlan::getModelStructure)
        .def("get_compute_device_usage_for_mlprogram_operation", &ComputePlan::getComputeDeviceUsageForMLProgramOperation)
        .def("get_compute_device_usage_for_neuralnetwork_layer", &ComputePlan::getComputeDeviceUsageForNeuralNetworkLayer)
        .def("get_estimated_cost_for_mlprogram_operation", &ComputePlan::getEstimatedCostForMLProgramOperation);
#endif

#if ML_MODEL_STRUCTURE_IS_AVAILABLE
    py::class_<ModelStructureProgramOperation>(m, "_MLModelStructureProgramOperationProxy", py::module_local());
    py::class_<ModelStructureNeuralNetworkLayer>(m, "_MLModelStructureNeuralNetworkLayerProxy", py::module_local());
#endif

#if ML_MODEL_ASSET_IS_AVAILABLE
    py::class_<ModelAsset>(m, "_MLModelAssetProxy", py::module_local());
#endif

    return m.ptr();
}

#pragma clang diagnostic pop
