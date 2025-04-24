//
//  Copyright (c) 2024, Apple Inc. All rights reserved.
//
//  Use of this source code is governed by a BSD-3-clause license that can be
//  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import XCTest
import CoreML

extension AsyncStream {
    var first: Element? {
        get async {
            for await value in self {
                return value
            }

            return nil
        }
    }
}

struct Message<T>: Codable & Sendable & NamedRequest where T: Codable & Sendable {
    let value: T
    let name: String
}

extension Encodable {
    func jsonData() throws -> Data {
        try JSONEncoder().encode(self)
    }
}

actor CounterService: Service {
    typealias SocketType = PassThroughSocket<Message<Int>, Message<Int>>

    let socket:SocketType

    var task: Task<Void, Never>? = nil

    func process(request: Message<Int>) async -> Message<Int> {
        return Message(value: request.value + 1, name: request.name)
    }

    let name: String = "CounterService"

    func getTask() async -> Task<Void, Never>? {
        return task
    }

    func setTask(_ task: Task<Void, Never>?) async {
        self.task = task
    }

    init(socket: SocketType) {
        self.socket = socket
    }
}

extension DeviceCtlSocket {
    func push<T>(_ params: T, method: String, id: String) async throws where T: Codable & Sendable  {
        let request = JSONRPCRequest<T>(jsonrpc: "2.0",  id: id, method: method, params: params)
        let messageDirectory = messagesDirectoryURL.appending(path: request.id)
        let fm = FileManager()
        try fm.createDirectory(at: messageDirectory, withIntermediateDirectories: false)
        try request.jsonData().write(to: messageDirectory.appending(path: "message.json"))
    }
}

final class ModelRunnerTests: XCTestCase {
    func testSocketCreation() async throws {
        let socket = PassThroughSocket<Int, Int>()
        try await socket.send(value: 1)
    }

    func testSocketSendAndReceive() async throws {
        let socket = PassThroughSocket<Int, Int>()
        let stream = try await socket.receive()
        try await socket.push(value: 1)
        let value = await stream.first
        XCTAssert(value == 1)
    }

    func testAdapterSocketReceive() async throws {
        let socket = PassThroughSocket<Int, Int>()
        let adapterSocket = try await AdapterSocket<String, String>(socket: socket, receive: { value in
            String(value)
        }, send: { value in
            Int(value) ?? 0
        })

        let stream = try await adapterSocket.receive()
        try await socket.push(value: 1)
        let value = await stream.first
        XCTAssert(value == "1")
    }

    func testAdapterSocketSend() async throws {
        let socket = PassThroughSocket<Int, Int>()
        let adapterSocket = try await AdapterSocket<String, String>(socket: socket, receive: { value in
            String(value)
        }, send: { value in
            Int(value) ?? 0
        })

        let stream = socket.responseStream
        try await adapterSocket.send(value: "1")
        let value = await stream.first
        XCTAssert(value == 1)
    }

    func testDirectoryMonitor() async throws {
        let fm = FileManager()
        let directoryURL = fm.temporaryDirectory.appending(component: UUID().uuidString)
        try fm.createDirectory(at: directoryURL, withIntermediateDirectories: false)
        let monitor = try DirectoryMonitor(url: directoryURL)

        do {
            let data = NSData()
            try data.write(to: directoryURL.appending(path: "message1.json"), options: .atomic)
        }

        let stream = try await monitor.listen()
        guard let initialValue = await stream.first else {
            XCTFail("Expected a valid value but received a nil value.")
            return
        }

        switch initialValue {
        case let .initial(urls):
            XCTAssert(urls.count == 1, "Expected exactly one URL in initial change, but found \(urls.count)")
            XCTAssert(urls.first?.lastPathComponent == "message1.json")
        default:
            XCTFail("Unexpected change type received. Expected: .initial Received: \(initialValue)")
        }

        do {
            let data = NSData()
            try data.write(to: directoryURL.appending(path: "message2.json"), options: .atomic)
        }

        guard let nextValue = await stream.first else {
            XCTFail("Expected a valid value but received a nil value.")
            return
        }

        switch nextValue {
        case let .added(urls):
            XCTAssert(urls.count == 1, "Expected exactly one URL in change, but found \(urls.count)")
            XCTAssert(urls.first?.lastPathComponent == "message2.json")
        default:
            XCTFail("Unexpected change type received. Expected: .added Received: \(nextValue)")
        }

        try fm.removeItem(at: directoryURL.appending(path: "message1.json"))

        guard let nextNextValue = await stream.first else {
            XCTFail("Expected a valid value but received a nil value.")
            return
        }

        switch nextNextValue {
        case let .removed(urls):
            XCTAssert(urls.count == 1, "Expected exactly one URL in change, but found \(urls.count)")
            XCTAssert(urls.first?.lastPathComponent == "message1.json")
        default:
            XCTFail("Unexpected change type received. Expected: .added Received: \(nextValue)")
        }

        try? fm.removeItem(at: directoryURL)
        await monitor.shutdown()
    }

    func testDeviceCtlSocket() async throws {
        let fm = FileManager()
        let directoryURL = fm.temporaryDirectory.appending(component: UUID().uuidString)
        let socket = try await DeviceCtlSocket(directoryURL: directoryURL)
        let stream = try await socket.receive()
        let monitor = try DirectoryMonitor(url: socket.responsesDirectoryURL)
        let responseStream = try await monitor.listen()

        guard let _ = await responseStream.first else {
            XCTFail("Expected a valid value but received a nil value.")
            return
        }

        let params = Message<String>(value: "xxx", name: "")
        try await socket.push(params, method: "Echo", id: "1")

        guard let value = await stream.first else {
            XCTFail("Expected a valid value but received a nil value.")
            return
        }

        XCTAssert(value.id == "1")
        XCTAssert(value.name == "Echo")

        do {
            let result = Message(value: "xxx", name: "")
            let response = JSONRPCResponse<Message<String>>(
                jsonrpc: "2.0",
                id: "1",
                result: result,
                resource: nil,
                error: nil
            )
            try await socket.send(value: DeviceCtlSocket.Response(id: response.id, original: try response.jsonData(), resource: nil))
        }

        guard let nextValue = await responseStream.first else {
            XCTFail("Expected a valid value but received a nil value.")
            return
        }

        switch nextValue {
        case let .added(urls):
            XCTAssert(urls.count == 1, "Expected exactly one URL in change, but found \(urls.count)")
            XCTAssert(urls.first?.lastPathComponent == "1")
        default:
            XCTFail("Unexpected change type received. Expected: .added Received: \(nextValue)")
        }

        await monitor.shutdown()
        await socket.shutdown()
    }

    func testCounterService() async throws {
        let socket = PassThroughSocket<Message<Int>, Message<Int>>()
        var service = CounterService(socket: socket)
        try await service.resume()
        let stream = socket.responseStream
        for i in (0...10) {
            try await socket.push(value: Message<Int>(value: i, name: "CounterService" ))
            guard let message = await stream.first else {
                XCTFail("Expected one response")
                return
            }

            XCTAssert(message.value == i + 1)
        }
    }

    func testModelLoadAndUnload() async throws {
        let processor = ModelRequestProcessor()
        let testBundle = Bundle(for: type(of: self))
        guard let assetURL = testBundle.url(forResource: "add_model", withExtension: "mlmodelc") else {
            XCTFail("Failed to load 'add_model.mlmodelc' asset")
            return
        }

        let loadRequest = Model.Request.Load(
            modelID: "xyz",
            computeUnits: "CPU_ONLY",
            name: "test",
            functionName: nil,
            optimizationHints: nil
        )

        let loadResponse = try await processor.process(details: loadRequest, resource: assetURL)
        XCTAssert(loadResponse.modelID == loadRequest.modelID)

        let unloadRequest = Model.Request.Unload(modelID: "xyz", computeUnits: "CPU_ONLY")
        let unloadResponse = try await processor.process(details: unloadRequest)
        XCTAssert(unloadResponse.modelID == unloadRequest.modelID)
    }

    func testComputePlan() async throws {
        guard #available(macOS 14.4, iOS 17.4, tvOS 17.4, watchOS 10.4, *) else {
            throw XCTSkip("Test requires iOS 17.4+, macOS 14.4+, tvOS 17.4+, or watchOS 10.4+.")
        }

        let processor = ModelRequestProcessor()
        let testBundle = Bundle(for: type(of: self))
        guard let assetURL = testBundle.url(forResource: "add_model", withExtension: "mlmodelc") else {
            XCTFail("Failed to load 'add_model.mlmodelc' asset")
            return
        }

        let loadRequest = Model.Request.Load(
            modelID: "xyz",
            computeUnits: "ALL",
            name: "test",
            functionName: nil,
            optimizationHints: nil
        )

        let loadResponse = try await processor.process(details: loadRequest, resource: assetURL)
        XCTAssert(loadResponse.modelID == loadRequest.modelID)

        let computePlanRequest = Model.Request.ComputePlan(
            modelID: "xyz"
        )

        let computePlanResponse = try await processor.process(details: computePlanRequest)
        XCTAssert(computePlanResponse.modelID == computePlanResponse.modelID)
    }

    func testOptimizationHints() async throws {
        guard #available(macOS 15.0, iOS 18.0,tvOS 18.0, watchOS 11.0, *) else {
            throw XCTSkip("Test requires macOS 15.0+, iOS 18.0+, tvOS 18.0+, or watchOS 11.0+.")
        }

        let processor = ModelRequestProcessor()
        let testBundle = Bundle(for: type(of: self))
        guard let assetURL = testBundle.url(forResource: "add_model", withExtension: "mlmodelc") else {
            XCTFail("Failed to load 'add_model.mlmodelc' asset")
            return
        }

        do {
            // Test valid
            let loadRequest = Model.Request.Load(
                modelID: "xyz",
                computeUnits: "ALL",
                name: "test",
                functionName: nil,
                optimizationHints: ["reshapeFrequency" : "Frequent", "specializationStrategy" : "Default"]
            )

            let loadResponse = try await processor.process(details: loadRequest, resource: assetURL)
            XCTAssert(loadResponse.modelID == loadRequest.modelID)
        }

        do {
            // Test invalid
            let loadRequest = Model.Request.Load(
                modelID: "abc",
                computeUnits: "ALL",
                name: "test",
                functionName: nil,
                optimizationHints: ["reshapeFrequency" : "Frequently", "specializationStrategy" : "Default"]
            )

            do {
                let _ = try await processor.process(details: loadRequest, resource: assetURL)
                XCTFail("Expected error")
            } catch let error as Model.Error {
                switch error {
                case .malformedRequest:
                    break
                default:
                    XCTFail("Unexpected error: \(error)")
                }
            }
        }

    }
}
