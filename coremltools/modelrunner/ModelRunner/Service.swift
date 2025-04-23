//
//  Copyright (c) 2025, Apple Inc. All rights reserved.
//
//  Use of this source code is governed by a BSD-3-clause license that can be
//  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// A protocol defining a service that manages a connection and processes requests asynchronously.
///
/// Types conforming to this protocol are responsible for handling asynchronous tasks related to a specific socket type.
protocol Service {
    /// The type of socket used by the service.
    associatedtype SocketType: Socket where SocketType.Request: NamedRequest

    /// The socket associated with the service.
    var socket: SocketType { get }

    /// The unique identifier for the request type that this service can process.
    var name: String { get }

    /// Retrieves the current asynchronous task, if any.
    ///
    /// - Returns: An optional `Task` representing ongoing asynchronous work.
    func getTask() async -> Task<Void, Never>?

    /// Sets or updates the current asynchronous task.
    ///
    /// - Parameter task: The new task to set, or `nil` to clear the current task.
    func setTask(_ task: Task<Void, Never>?) async

    /// Resumes the service, starting or continuing its asynchronous operations.
    ///
    /// This method is responsible for initiating the listening process on the connection and handling incoming requests.
    ///
    /// - Throws: An error if the service fails to resume, such as issues with starting the underlying socket's receive.
    mutating func resume() async throws

    /// Suspends the service, pausing its asynchronous operations.
    ///
    /// This method cancels any ongoing tasks associated with the service
    func suspend() async

    /// Processes an incoming request and produces a response asynchronously.
    ///
    /// - Parameter request: The request to be processed.
    /// - Returns: The response generated from processing the request.
    func process(request: SocketType.Request) async -> SocketType.Response
}

extension Service {
    private func _resume() async throws -> Task<Void, Never> {
        let serviceName = self.name
        let requests = try await socket.receive()
        return Task {
            for await request in requests {
                if Task.isCancelled {
                    return
                }

                if !request.name.hasPrefix(serviceName) {
                    continue
                }

                let response = await process(request: request)
                try? await socket.send(value: response)
            }
        }
    }

    mutating func resume() async throws {
        await setTask(try await _resume())
    }

    func suspend() async {
        let task = await getTask()
        task?.cancel()
    }
}
