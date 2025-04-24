//
//  Copyright (c) 2025, Apple Inc. All rights reserved.
//
//  Use of this source code is governed by a BSD-3-clause license that can be
//  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import CoreML
import Darwin


struct LoadedModel {
    let id: String
    let name: String
    let mlModel: MLModel
    let state: Any?
    let assetURL: URL
}

func measure<T>(_ body: () throws -> T) rethrows -> (result: T, elapsedTime: UInt64) {
    var info = mach_timebase_info(numer: 0, denom: 0)
    mach_timebase_info(&info)

    let startTime = mach_absolute_time()
    let result = try body()
    let endTime = mach_absolute_time()
    let elapsed = endTime - startTime

    return (result, elapsed * UInt64(info.numer) /  UInt64(info.denom))
}

extension LoadedModel {
    /// Represents errors that can occur during the operation of the service.
    enum Error: Swift.Error, CustomStringConvertible {
        /// Indicates that the model being used is not supported by the current implementation.
        ///
        /// This error occurs when the service encounters a model with incompatible output format
        /// or other characteristics that are not supported by the current system.
        case unsupportedOutput(details: String)

        /// A human-readable description of the error.
        var description: String {
            switch self {
            case .unsupportedOutput(let details):
                return "Unsupported output: \(details)"
            }
        }
    }

    fileprivate func predict(inputs: MLFeatureProvider) throws -> MLFeatureProvider {
        if #available(iOS 18.0, macOS 15.0, watchOS 11.0, tvOS 18.0, *) {
            if let mlState = state.map({ $0 as! MLState}) {
                return try mlModel.prediction(from: inputs, using: mlState)
            }
        }

        return try mlModel.prediction(from: inputs)
    }

    fileprivate func predict(inputs: [String: TensorDescriptor], storage: Data) throws -> (outputs: [String: TensorDescriptor], storage: Data, duration: UInt64) {
        var outputs = [String : TensorDescriptor]()
        var buffer = Data()
        var duration = UInt64(0)
        try storage.withUnsafeBytes { (bufferPointer: UnsafeRawBufferPointer) in
            guard let rawPtr = bufferPointer.baseAddress, !bufferPointer.isEmpty else {
                throw Model.Error.invalidResource(details: "The backing store for the model inputs is invalid or empty.")
            }

            var modelInputs = [String : MLFeatureValue]()
            for (name, descriptor) in inputs {
                let multiArray = try MLMultiArray(descriptor: descriptor, rawPtr: rawPtr)
                modelInputs[name] = MLFeatureValue(multiArray: multiArray)
            }

            let (modelOutputs, predictDuration) = try measure {
                try predict(inputs: try MLDictionaryFeatureProvider(dictionary: modelInputs))
            }

            duration = predictDuration

            var segment = TensorStorage()
            for name in modelOutputs.featureNames {
                guard let multiArrayValue = modelOutputs.featureValue(for: name)?.multiArrayValue else {
                    throw Error.unsupportedOutput(details:"Incompatible output format detected for model ID '\(id)'.")
                }

                multiArrayValue.withUnsafeBytes { ptr in
                    segment.size += ptr.count
                    buffer.append(contentsOf: ptr)
                }
                let descriptor = TensorDescriptor(dataType: multiArrayValue.dataType.representation,
                                                  shape: multiArrayValue.shape.map { $0.intValue },
                                                  strides: multiArrayValue.strides.map { $0.intValue },
                                                  storage: segment)
                outputs[name] = descriptor
                segment.offset = segment.size
            }
        }

        return (outputs: outputs, storage: buffer, duration: duration)
    }
}

struct TensorStorage: Codable & Sendable  {
    var offset: Int = 0
    var size: Int = 0

    var range: Range<Int> {
        (offset..<offset+size)
    }
}

struct TensorDescriptor: Codable & Sendable {
    let dataType: String
    let shape: [Int]
    let strides: [Int]
    let storage: TensorStorage
}

extension MLMultiArrayDataType {
    /// Creates an `MLMultiArrayDataType` instance from a string representation.
    init?(representation: String) {
        switch representation {
        case "Float64":
            self = .float64
        case "Float32":
            self = .float32
        case "Float16":
            self = .float16
        case "Int32":
            self = .int32
        default:
            return nil
        }
    }

    var representation: String {
        switch self {
        case .double:
            return "Float64"
        case .float:
            return "Float32"
        case .float16:
            return "Float16"
        case .int32:
            return "Int32"
        default:
            fatalError("Unknown MLMultiArrayDataType: \(self)")
        }
    }
}

/// Represents the hierarchical path to a component within an `MLModelStructure`.
///
/// This struct provides a way to navigate and identify a specific component in `MLModelStructure`.
struct ModelStructurePath: Codable & Sendable {
    /// Represents different components within an ML Program structure.
    struct Program: Codable & Sendable {
        /// Defines an operation within a program.
        struct Operation: Codable & Sendable {
            /// Names of an output produced by this operation.
            let outputName: String
        }

        /// Represents a block within a program operation.
        struct Block: Codable & Sendable {
            /// The index of this block within its parent operation.
            let index: Int
        }

        /// Represents a function within a program.
        struct Function: Codable & Sendable {
            /// The name of the function.
            let name: String
        }
    }

    /// Represents different components within a NeuralNetwork structure.
    struct NeuralNetwork: Codable & Sendable {
        struct Layer: Codable & Sendable {
            let name: String
        }
    }

    /// Represents different components within a Pipeline structure.
    struct Pipeline: Codable & Sendable {
        struct Model: Codable & Sendable {
            let name: String
        }
    }

    enum Component: Codable & Sendable {
        case neuralNetwork
        case neuralNetworkLayer(data:NeuralNetwork.Layer)
        case program
        case programFunction(data:Program.Function)
        case programOperation(data:Program.Operation)
        case programBlock(data:Program.Block)
        case pipeline
        case pipelineModel(data:Pipeline.Model)
    }

    let components: [Component]
}

/// Represent a compute device.
enum ComputeDevice: Codable & Sendable {
    case cpu
    case gpu
    case neuralEngine(totalCoreCount: Int)
}

enum Model {
    struct Empty: Codable & Sendable {}

    enum Request: Sendable, Identifiable {
        /// Represents a request to load a model.
        struct Load: Codable & Sendable {
            /// Unique identifier for the model.
            let modelID: String
            /// Specifies the compute units to be used (e.g., "CPU_ONLY", "CPU_AND_GPU").
            let computeUnits: String
            /// Name of the model to be loaded.
            let name: String
            /// The function name.
            let functionName: String?
            /// The optimization hints.
            let optimizationHints: [String : String]?
        }

        /// Represents a request to get the compute plan of a model.
        struct ComputePlan: Codable & Sendable {
            /// Unique identifier for the model.
            let modelID: String
        }

        /// Represents a request to unload a model.
        struct Unload: Codable & Sendable {
            /// Unique identifier for the model.
            let modelID: String
            /// Specifies the compute units to be used (e.g., "CPU_ONLY", "CPU_AND_GPU").
            let computeUnits: String
        }

        /// Represents a request to perform a prediction using a loaded model.
        struct Prediction: Codable & Sendable {
            /// Unique identifier for the model.
            let modelID: String
            /// Specifies the compute units to be used for prediction.
            let computeUnits: String
            /// Dictionary of input tensors, keyed by name.
            let inputs: [String: TensorDescriptor]
        }

        /// A request to load a model.
        ///
        /// - Parameters:
        ///   - details: The details of the load operation, including model identifier and compute units.
        ///   - requestID: The request ID.
        ///   - resource: An optional URL pointing to the model file or resource to be loaded.
        case load(details: Load, requestID: String, resource: URL?)

        /// A request to unload a previously loaded model.
        ///
        /// - Parameters:
        ///   - details: The details of the unload operation, primarily the model identifier.
        ///   - requestID: The request ID.
        case unload(details: Unload, requestID: String)

        /// A request to perform a prediction using a loaded model.
        ///
        /// - Parameters:
        ///   - details: The details of the prediction request, including input data and compute units.
        ///   - requestID: The request ID.
        ///   - resource: An optional URL pointing to additional resources needed for the prediction.
        case prediction(details: Prediction, requestID: String, resource: URL?)

        /// A request to get the compute plan of a model.
        ///
        /// - Parameters:
        ///   - details: The details of the compute plan operation, including model identifier and compute units.
        ///   - requestID: The request ID.
        case computePlan(details: ComputePlan, requestID: String)

        var id: String {
            switch self {
            case let .load(_, requestID, _):
                return requestID

            case let .unload(_, requestID):
                return requestID

            case let .computePlan(_, requestID):
                return requestID

            case let .prediction(_, requestID, _):
                return requestID
            }
        }
    }

    enum Response: Sendable, Identifiable {
        /// Represents the result of a model loading operation.
        struct Load: Codable & Sendable {
            /// Unique identifier for the model.
            let modelID: String
            /// Duration of the load operation in nanoseconds.
            let duration: UInt64
            /// Indicates whether the model is cached.
            let isCached: Bool
        }

        /// Represents the result of a model unloading operation.
        struct Unload: Codable & Sendable {
            /// Unique identifier for the model.
            let modelID: String
        }

        /// Represents the result of a prediction operation.
        struct Prediction: Codable & Sendable {
            /// Unique identifier for the model.
            let modelID: String
            /// Dictionary of output tensors, keyed by name.
            let outputs: [String: TensorDescriptor]
            /// Duration of the prediction operation in nano seconds.
            let duration: UInt64
            /// The resource name.
            let resource: String
        }

        /// Represents the compute plan of a model.
        struct ComputePlan: Codable & Sendable {
            /// Specifies the compute device usages for an operation/layer execution.
            struct DeviceUsage: Codable & Sendable {
                /// An array of devices that can support the operation/layer execution.
                let supported: [ComputeDevice]
                /// The preferred device for an operation/layer execution.
                let preferred: ComputeDevice
            }

            struct OperationOrLayerInfo: Codable & Sendable {
                let deviceUsage: DeviceUsage
                let estimatedCost: Double?
                let path: ModelStructurePath
            }

            /// Unique identifier for the model.
            let modelID: String
            /// Details of the compute plan.
            let infos: [OperationOrLayerInfo]
        }

        /// Response for a successful model load operation.
        ///
        /// - Parameters:
        ///   - details: The details of the load operation result.
        ///   - responseID: The response ID.
        case load(details: Load, responseID: String)

        /// Response for a successful model unload operation.
        /// - Parameters:
        ///   - details: The details of the unload operation.
        ///   - responseID: The response ID.
        case unload(details: Unload, responseID: String)

        /// Response for a successful model predict operation.
        /// - Parameters:
        ///   - details: The details of the prediction operation.
        ///   - responseID: The response ID.
        case prediction(details: Prediction, responseID: String, resource: URL?)

        /// Response for a successful compute plan operation.
        /// - Parameters:
        ///   - details: The details of the compute plan operation.
        ///   - responseID: The response ID.
        case computePlan(details: ComputePlan, responseID: String)

        var id: String {
            switch self {
            case let .load(_, responseID):
                return responseID

            case let .unload(_, responseID):
                return responseID

            case let .prediction(_, responseID, _):
                return responseID

            case let .computePlan(_, responseID):
                return responseID
            }
        }
    }

    /// Represents errors that can occur during the operation of the service.
    enum Error: LocalizedError, CustomStringConvertible {
        /// Indicates that the received request is malformed or invalid.
        ///
        /// - Parameter details: A string providing more information about why the request is considered malformed.
        case malformedRequest(details: String)


        /// Indicates that a required resource is missing.
        ///
        /// This error occurs when the service cannot find or access a resource
        /// that is necessary for completing the requested operation.
        /// - Parameter details: A string providing more information about the missing resource and its impact.
        case missingResource(details: String)


        /// Indicates that a provided resource is invalid or unusable.
        ///
        /// This error occurs when the service can access a resource, but finds it to be
        /// in an invalid format, corrupted, or otherwise unsuitable for the intended use.
        /// - Parameter details: A string providing more information about the invalid resource and its impact.
        case invalidResource(details: String)

        /// Indicates that the model being used is not supported by the current implementation.
        ///
        /// This error occurs when the service encounters a model with incompatible output format
        /// or other characteristics that are not supported by the current system.
        case unsupportedModel(details: String)

        /// Indicates that the request is not supported by the current system.
        ///
        /// This error occurs when the service encounters a request which is not supported by the current system.
        case unsupportedRequest(details: String)

        /// A human-readable description of the error.
        var description: String {
            switch self {
            case .malformedRequest(let details):
                return "Malformed request: \(details)"
            case .missingResource(let details):
                return "Missing resource: \(details)"
            case .invalidResource(let details):
                return "Invalid resource: \(details)"
            case .unsupportedModel(let details):
                return "Unsupported model: \(details)"
            case .unsupportedRequest(let details):
                return "Unsupported request: \(details)"
            }
        }

        var errorDescription: String? {
            self.description
        }
    }
}

@available(macOS 14.4, iOS 17.4, watchOS 10.4, tvOS 17.4, *)
extension ComputeDevice {
    init(from computeDevice: MLComputeDevice) {
        switch computeDevice {
        case .cpu:
            self = .cpu
        case .gpu:
            self = .gpu
        case let .neuralEngine(device):
            self = .neuralEngine(totalCoreCount: device.totalCoreCount)
        @unknown default:
            fatalError("Unknown compute device=\(computeDevice)")
        }
    }
}

@available(macOS 14.4, iOS 17.4, watchOS 10.4, tvOS 17.4, *)
extension Model.Response.ComputePlan.DeviceUsage {
    init(from computeDeviceUsage: MLComputePlan.DeviceUsage) {
        let preferred = ComputeDevice(from: computeDeviceUsage.preferred)
        let supported = computeDeviceUsage.supported.map { ComputeDevice(from: $0) }
        self = Model.Response.ComputePlan.DeviceUsage(
            supported: supported,
            preferred: preferred
        )
    }
}

@available(macOS 14.4, iOS 17.4, watchOS 10.4, tvOS 17.4, *)
extension Model.Response.ComputePlan {
    fileprivate static func details(
        of model: MLModelStructure,
        plan: MLComputePlan,
        path: ModelStructurePath
    ) -> [Model.Response.ComputePlan.OperationOrLayerInfo] {
        switch model {
        case let .neuralNetwork(neuralNetwork):
            return details(of: neuralNetwork, plan: plan, path: path)
        case let .program(program):
            return details(of: program, plan: plan, path: path)
        case let .pipeline(pipeline):
            return details(of: pipeline, plan: plan, path: path)
        case .unsupported:
            return []

        @unknown default:
            return []
        }
    }

    fileprivate static func details(
        of neuralNetwork: MLModelStructure.NeuralNetwork,
        plan: MLComputePlan,
        path: ModelStructurePath
    ) -> [Model.Response.ComputePlan.OperationOrLayerInfo] {
        var result = [Model.Response.ComputePlan.OperationOrLayerInfo]()
        for layer in neuralNetwork.layers {
            guard let computeDeviceUsage = plan.deviceUsage(for: layer) else {
                continue
            }

            var components = path.components
            components.append(.neuralNetwork)
            components.append(.neuralNetworkLayer(data: ModelStructurePath.NeuralNetwork.Layer(name: layer.name)))
            let deviceUsage = Model.Response.ComputePlan.DeviceUsage(from: computeDeviceUsage)
            let detail = Model.Response.ComputePlan.OperationOrLayerInfo(
                deviceUsage: deviceUsage,
                estimatedCost: nil,
                path: ModelStructurePath(components: components)
            )

            result.append(detail)
        }

        return result
    }

    fileprivate static func details(
        of block: MLModelStructure.Program.Block,
        plan: MLComputePlan,
        path: ModelStructurePath
    ) -> [Model.Response.ComputePlan.OperationOrLayerInfo] {
        var result = [Model.Response.ComputePlan.OperationOrLayerInfo]()
        for operation in block.operations {
            guard let computeDeviceUsage = plan.deviceUsage(for: operation) else {
                continue
            }

            var components = path.components
            components.append(
                .programOperation(data: ModelStructurePath.Program.Operation(
                    outputName: operation.outputs.map({$0.name}).first ?? ""
                ))
            )

            let deviceUsage = Model.Response.ComputePlan.DeviceUsage(from: computeDeviceUsage)
            let estimatedCost = plan.estimatedCost(of: operation)
            let detail = Model.Response.ComputePlan.OperationOrLayerInfo(
                deviceUsage: deviceUsage,
                estimatedCost: estimatedCost?.weight,
                path: ModelStructurePath(components: components)
            )

            result.append(detail)
            for (index, block) in operation.blocks.enumerated() {
                components.append(.programBlock(data: ModelStructurePath.Program.Block(index: index)))
                result.append(contentsOf: details(
                    of: block,
                    plan: plan,
                    path: ModelStructurePath(components: components))
                )
            }
        }

        return result
    }

    fileprivate static func details(
        of program: MLModelStructure.Program,
        plan: MLComputePlan,
        path: ModelStructurePath
    ) -> [Model.Response.ComputePlan.OperationOrLayerInfo] {
        var result = [Model.Response.ComputePlan.OperationOrLayerInfo]()
        for (name, function) in program.functions {
            var components = path.components
            components.append(.program)
            components.append(.programFunction(data: ModelStructurePath.Program.Function(name: name)))
            components.append(.programBlock(data: ModelStructurePath.Program.Block(index: -1)))
            result.append(contentsOf: details(
                of: function.block,
                plan: plan,
                path: ModelStructurePath(components: components))
            )
        }

        return result
    }

    fileprivate static func details(
        of pipeline: MLModelStructure.Pipeline,
        plan: MLComputePlan,
        path: ModelStructurePath
    ) -> [Model.Response.ComputePlan.OperationOrLayerInfo] {
        var result = [Model.Response.ComputePlan.OperationOrLayerInfo]()
        for (name, model) in zip(pipeline.subModelNames, pipeline.subModels) {
            var components = path.components
            components.append(.pipeline)
            components.append(.pipelineModel(data: ModelStructurePath.Pipeline.Model(name: name)))
            result.append(contentsOf: details(
                of: model,
                plan: plan,
                path: ModelStructurePath(components: components))
            )
        }

        return result
    }

    fileprivate static func details(
        at url: URL,
        configuration: MLModelConfiguration
    ) async throws -> [Model.Response.ComputePlan.OperationOrLayerInfo] {
        let plan = try await MLComputePlan.load(contentsOf: url, configuration: configuration)
        return details(
            of: plan.modelStructure,
            plan: plan,
            path: ModelStructurePath(components: [])
        )
    }
}

extension MLMultiArray {
    fileprivate convenience init(
        descriptor: TensorDescriptor,
        rawPtr: UnsafeRawPointer
    ) throws {
        let dataType = MLMultiArrayDataType(representation: descriptor.dataType)
        guard let dataType else {
            throw Model.Error.malformedRequest(
                details:
                """
                Unsupported data type in TensorDescriptor: \(descriptor.dataType).
                Please check the data type representation.
                """
            )
        }

        let mutableRawPtr = UnsafeMutableRawPointer(mutating: rawPtr)
        try self.init(dataPointer: mutableRawPtr.advanced(by: descriptor.storage.offset),
                      shape: descriptor.shape.map { NSNumber(value: $0) },
                      dataType: dataType,
                      strides: descriptor.strides.map { NSNumber(value: $0) },
                      deallocator: nil)
    }
}

extension MLComputeUnits {
    /// Creates an `MLComputeUnits` instance from a string representation.
    init?(representation: String) {
        switch representation {
        case "CPU_ONLY":
            self = .cpuOnly
        case "CPU_AND_GPU":
            self = .cpuAndGPU
        case "CPU_AND_NE":
            self = .cpuAndNeuralEngine
        case "ALL":
            self = .all
        default:
            return nil
        }
    }
}

@available(macOS 14.4, iOS 17.4, watchOS 17.4, tvOS 17.4, *)
extension MLOptimizationHints.ReshapeFrequency {
    init(representation: String) throws {
        switch representation{
        case "Frequent":
            self = .frequent
        case "Infrequent":
            self = .infrequent
        default:
            throw Model.Error.malformedRequest(
                details: """
                Invalid ReshapeFrequency representation: \(representation).
                Expected either 'Frequent' or 'InFrequent'.
                """
            )
        }
    }
}

@available(macOS 15.0, iOS 18.0, watchOS 11.0, tvOS 18.0, *)
extension MLOptimizationHints.SpecializationStrategy {
    init(representation: String) throws {
        switch representation{
        case "Default":
            self = .default
        case "FastPrediction":
            self = .fastPrediction
        default:
            throw Model.Error.malformedRequest(
                details: """
                Invalid SpecializationStrategy representation: \(representation).
                Expected either 'Default' or 'FastPrediction.
                """
            )
        }
    }
}

@available(macOS 14.4, iOS 17.4, watchOS 10.4, tvOS 17.4, *)
extension MLOptimizationHints {
    init(representation: [String : String]) throws {
        self.init()
        if let value = representation["reshapeFrequency"] {
            self.reshapeFrequency = try .init(representation: value)
        }

        if let value = representation["specializationStrategy"] {
            if #available(macOS 15.0, iOS 18.0, watchOS 11.0, tvOS 18.0, *) {
                self.specializationStrategy = try .init(representation: value)
            } else {
                throw Model.Error.malformedRequest(
                    details:
                    """
                    Specialization Strategy are only supported on macOS 15.0+, iOS 18.0+, watchOS 11.0+, and tvOS 18.0+. \
                    Your current OS version does not support this feature.
                    """
                )
            }
        }
    }
}

extension MLModelConfiguration {
    fileprivate convenience init(computeUnits: String, functionName: String?) throws {
        guard let computeUnits = MLComputeUnits(representation: computeUnits) else {
            throw Model.Error.malformedRequest(
                details:
                """
                Invalid compute unit '\(computeUnits)'. Expected one of: CPU_ONLY, CPU_AND_GPU, CPU_AND_NE, or ALL.
                """
            )
        }

        self.init()
        self.computeUnits = computeUnits
        if let functionName {
            if #available(macOS 15.0, iOS 18.0, watchOS 11.0, tvOS 18.0, *) {
                self.functionName = functionName
            }
        }
    }
}

extension FileManager {
    fileprivate func validateModelContainer(at url: URL) throws {
        var isDirectory: ObjCBool = false
        let exists = fileExists(atPath: url.path, isDirectory: &isDirectory)

        if !exists {
            throw Model.Error.missingResource(
                details: "The model directory does not exist at path: \(url.path)"
            )
        } else if !isDirectory.boolValue {
            throw Model.Error.invalidResource(
                details: "Expected a directory for the mlmodelc, but found a file at path: \(url.path)"
            )
        } else if url.pathExtension != "mlmodelc" {
            throw Model.Error.invalidResource(
                details: "Expected an mlmodelc directory, but found: \(url.lastPathComponent)"
            )
        }
    }
}

actor ModelRequestProcessor: AsyncRequestProcessor {
    /// An instance of `FileManager` used for file system operations.
    private let fm: FileManager = FileManager()

    /// The loaded models.
    private var loadedModels: [LoadedModel] = []

    func process(
        details: Model.Request.Load,
        resource: URL?
    ) async throws -> Model.Response.Load {
        if loadedModels.contains(where: {$0.id == details.modelID &&
            MLComputeUnits(representation: details.computeUnits) == $0.mlModel.configuration.computeUnits}
        ) {
            return Model.Response.Load(modelID: details.modelID, duration: 0, isCached: true)
        }

        guard let resource else {
            throw Model.Error.missingResource(
                details:
                """
                Model file is required but was not provided.
                Please ensure a valid model file URL is specified.
                """
            )
        }

        try fm.validateModelContainer(at: resource)
        let configuration = try MLModelConfiguration(
            computeUnits: details.computeUnits,
            functionName: details.functionName
        )

        if let optimizations = details.optimizationHints {
            if #available(macOS 14.4, iOS 17.4, watchOS 17.4, tvOS 17.4, *) {
                configuration.optimizationHints = try MLOptimizationHints(representation: optimizations)
            } else {
                throw Model.Error.malformedRequest(
                    details:
                    """
                    Optimization hints are only supported on macOS 14.4+, iOS 17.4+, watchOS 17.4+, and tvOS 17.4+. \
                    Your current OS version does not support this feature.
                    """
                )
            }
        }

        let (mlModel, duration) = try measure {
            try MLModel(
                contentsOf: resource,
                configuration: configuration
            )
        }

        var state: Any? = nil
        if #available(macOS 15.0, iOS 18.0, watchOS 11.0, tvOS 18.0,  *) {
            if mlModel.modelDescription.stateDescriptionsByName.count > 0 {
                state = mlModel.makeState()
            }
        }

        loadedModels.append(
            LoadedModel(
                id: details.modelID,
                name: details.name,
                mlModel: mlModel,
                state: state,
                assetURL: resource
            )
        )
        return Model.Response.Load(modelID: details.modelID, duration: duration, isCached: false)
    }

    func process(
        details: Model.Request.Unload
    ) async throws -> Model.Response.Unload {
        let index = loadedModels.firstIndex(where: {
            $0.id == details.modelID &&
            MLComputeUnits(representation: details.computeUnits) == $0.mlModel.configuration.computeUnits
        })
        guard let index else {
            throw Model.Error.missingResource(details:
                """
                The model with ID '\(details.modelID)' and '\(details.computeUnits)' is not currently loaded.
                Please ensure that the model has been loaded before attempting to unload it.
                """
            )
        }

        let _ = autoreleasepool {
            loadedModels.remove(at: index)
        }

        return  Model.Response.Unload(modelID: details.modelID)
    }

    func process(
        details: Model.Request.Prediction,
        resource: URL?
    ) async throws -> (response: Model.Response.Prediction, resource: URL) {
        guard let resource = resource else {
            throw Model.Error.missingResource(
                details:
                """
                The TensorDescriptor requires a data file, but none was provided.
                Please ensure that a valid data file is passed.
                """
            )
        }

        let index = loadedModels.firstIndex(where: {
            $0.id == details.modelID &&
            MLComputeUnits(representation: details.computeUnits) == $0.mlModel.configuration.computeUnits
        })
        guard let index else {
            throw Model.Error.missingResource(details:
                """
                The model with ID '\(details.modelID)' and '\(details.computeUnits)' is not currently loaded.
                Please ensure that the model has been loaded before running prediction.
                """
            )
        }

        let loadedModel = loadedModels[index]

        let (outputs, storage, duration) = try loadedModel.predict(inputs: details.inputs,
                                                                   storage: try Data(contentsOf: resource, options: .mappedIfSafe))

        let storageURL = fm.temporaryDirectory.appending(path: "\(UUID().uuidString).bin")
        try storage.write(to: storageURL, options: .atomic)

        let response = Model.Response.Prediction(
            modelID: details.modelID,
            outputs: outputs,
            duration: duration,
            resource: storageURL.lastPathComponent
        )
        return (response: response, resource: storageURL)
    }

    func process(details: Model.Request.ComputePlan) async throws -> Model.Response.ComputePlan {
        let index = loadedModels.firstIndex(where: {$0.id == details.modelID })
        guard let index else {
            throw Model.Error.missingResource(details:
                """
                The model with ID '\(details.modelID)' is not currently loaded.
                Please ensure that the model has been loaded before retrieving its compute plan.
                """
            )
        }

        let loadedModel = loadedModels[index]
        let asset = loadedModel.assetURL
        let configuration = loadedModel.mlModel.configuration

        if #available(macOS 14.4, iOS 17.4, watchOS 10.4, tvOS 17.4, *) {
            let computePlanDetails = try await Model.Response.ComputePlan.details(at: asset, configuration: configuration)
            return Model.Response.ComputePlan(modelID: details.modelID, infos: computePlanDetails)
        } else {
            throw Model.Error.unsupportedRequest(
                details: "MLComputePlan is not supported on this platform."
            )
        }
    }

    func process(request: Model.Request) async throws -> Model.Response {
        switch request {
        case let .load(details, requestID, resource):
            return Model.Response.load(details: try await process(details: details, resource: resource), responseID: requestID)

        case let .unload(details, requestID):
            return Model.Response.unload(details: try await process(details: details), responseID: requestID)

        case let .prediction(details, requestID, resource):
            let (response, resource) = try await process(details:details, resource: resource)
            return Model.Response.prediction(details: response, responseID: requestID, resource: resource)

        case let .computePlan(details, requestID):
            let response = try await process(details:details)
            return Model.Response.computePlan(details: response, responseID: requestID)
        }
    }
}

extension Model.Response: DeviceCtlResponseEncodable {
    func convert() throws -> DeviceCtlSocket.Response {
        let encoder = JSONEncoder()

        func convert<T>(details: T, responseID: String, resource: URL?) throws -> DeviceCtlSocket.Response where T: Codable {
            let jsonResponse = JSONRPCResponse<T>(
                jsonrpc: "2.0",
                id: responseID,
                result: details,
                resource: resource?.lastPathComponent,
                error: nil
            )
            let jsonData = try encoder.encode(jsonResponse)
            let deviceCtlResponse = DeviceCtlSocket.Response(id: jsonResponse.id, original: jsonData, resource: resource)
            return deviceCtlResponse
        }

        switch self {
        case let .load(details, responseID):
            return try convert(details: details, responseID: responseID, resource: nil)
        case let .unload(details, responseID):
            return try convert(details: details, responseID: responseID, resource: nil)
        case let .computePlan(details, responseID):
            return try convert(details: details, responseID: responseID, resource: nil)
        case let .prediction(details, responseID, resource):
            return try convert(details: details, responseID: responseID, resource: resource)
        }
    }
}

extension Model.Request: DeviceCtlRequestDecodable {
    init(request: DeviceCtlSocket.Request) throws {
        let decoder = JSONDecoder()
        func decode<T>(_ request: DeviceCtlSocket.Request) throws -> T where T: Codable {
            let jsonRequest = try decoder.decode(JSONRPCRequest<T>.self, from: request.payload)
            guard let params = jsonRequest.params else {
                throw Model.Error.malformedRequest(
                    details:
                    """
                    Missing parameters for '\(request.name)'.
                    The ModelService requires non-nil parameters for this request type.
                    """
                )
            }

            return params
        }

        switch request.name {
        case "MLModelService.Load":
            let details: Load = try decode(request)
            self = .load(details: details, requestID: request.id, resource: request.resource)

        case "MLModelService.Unload":
            let details: Unload = try decode(request)
            self = .unload(details: details, requestID: request.id)

        case "MLModelService.Prediction":
            let details: Prediction = try decode(request)
            self = .prediction(details: details, requestID: request.id, resource: request.resource)

        case "MLModelService.ComputePlan":
            let details: ComputePlan = try decode(request)
            self = .computePlan(details: details, requestID: request.id)

        default:
            throw Model.Error.invalidResource(
                details:
                """
                Missing parameters for '\(request.name)'.
                The ModelService requires non-nil parameters for this request type.
                """
            )
        }
    }
}
